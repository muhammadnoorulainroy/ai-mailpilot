"""
Deterministic corpus clustering for MailPilot discovery.

Stateless ML sidecar: given a batch of email embedding vectors (bge-m3, unit-normalized), cluster them
into a small set of categories and return, per cluster, a centroid + several sub-prototypes + a few
representative rows. Core then names each cluster (from the representatives) and assigns the whole
inbox against the centroids/prototypes. The clusters ARE the categories, so a high-volume group can
never be missed the way a sample-then-name taxonomy misses it.

Pipeline: L2-normalize -> spectral-init UMAP (nonlinear, breaks the mega-cluster) -> HDBSCAN (density
clusters + noise, no fixed K) -> agglomerative merge of the fine clusters down to a target count. We
cluster in UMAP space and return ORIGINAL-space centroids; we never assign the corpus by a single
blurry centroid here (that collapses centroid-similar mail), which is why each cluster also carries
its fine sub-clusters as PROTOTYPES for Core's multi-prototype, nearest-prototype matching.

Deterministic for a fixed seed and the pinned library versions in requirements.txt. Privacy: numbers
in, numbers out; never touches the database or plaintext email.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Quiet unrelated TensorFlow/oneDNN chatter some environments emit on import; clustering never uses it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np


# Defaults tuned from validating on a real 48k inbox (ml/README.md).
DEFAULT_SEED = 42
DEFAULT_TARGET_CLUSTERS = 15
DEFAULT_CAP = 20000  # cluster at most this many rows: UMAP is only deterministic at subset scale
# (pynndescent's approximate NN is non-deterministic on larger inputs). Core assigns the rest via the
# returned prototypes, so capping here costs coverage of the *definition* set, not of the inbox.
DEFAULT_REPS_PER_CLUSTER = 8
UMAP_COMPONENTS = 15
UMAP_NEIGHBORS = 15


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Unit-normalize each row so cosine == dot product and centroids are well-defined."""
    x = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-9)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return (vector / norm if norm > 0 else vector).astype(np.float32)


def _reduce(x: np.ndarray, seed: int) -> np.ndarray:
    """UMAP to UMAP_COMPONENTS dims with seeded spectral init: deterministic, high quality, finite."""
    import umap

    return np.asarray(
        umap.UMAP(
            n_components=UMAP_COMPONENTS,
            n_neighbors=UMAP_NEIGHBORS,
            min_dist=0.0,
            metric="cosine",
            init="spectral",
            random_state=seed,
        ).fit_transform(x)
    )


def _merge(embedding: np.ndarray, fine: np.ndarray, target: int) -> tuple[np.ndarray, dict[int, int]]:
    """
    Collapse HDBSCAN's fine clusters to `target` by agglomerating their centroids. Returns per-point
    merged labels (0..K-1, or -1 noise) and the {fine label -> merged label} map (kept so each merged
    cluster can expose its fine clusters as prototypes). Fewer fine clusters than target are left as-is.
    """
    uniq = sorted(int(c) for c in set(fine.tolist()) if c != -1)
    if not uniq:
        return fine.astype(int), {}
    if len(uniq) > target:
        from sklearn.cluster import AgglomerativeClustering

        centroids = np.array([embedding[fine == c].mean(axis=0) for c in uniq])
        grouped = AgglomerativeClustering(n_clusters=target, linkage="average").fit_predict(centroids)
        fine_to_merged = {c: int(grouped[i]) for i, c in enumerate(uniq)}
    else:
        fine_to_merged = {c: i for i, c in enumerate(uniq)}
    merged = np.array([fine_to_merged.get(int(f), -1) for f in fine], dtype=int)
    return merged, fine_to_merged


def _metrics(embedding: np.ndarray, labels: np.ndarray, seed: int) -> dict:
    """UMAP-space quality on the clustered rows: noise share, biggest-cluster dominance, silhouette."""
    from sklearn.metrics import silhouette_score

    n = labels.shape[0]
    sizes = {int(c): int((labels == c).sum()) for c in set(labels.tolist()) if c != -1}
    noise = int((labels == -1).sum())
    biggest = max(sizes.values()) / n if sizes else 0.0
    silhouette = None
    nz = np.where(labels != -1)[0]
    if len(nz) > 50 and len(sizes) > 1:
        sample = np.random.RandomState(seed + 1).choice(nz, min(5000, len(nz)), replace=False)
        silhouette = float(silhouette_score(embedding[sample], labels[sample]))
    return {"noiseFraction": noise / n, "biggestClusterFraction": biggest, "silhouette": silhouette}


def cluster(
    vectors: np.ndarray,
    seed: int = DEFAULT_SEED,
    target_clusters: int = DEFAULT_TARGET_CLUSTERS,
    min_cluster_size: int | None = None,
    cap: int = DEFAULT_CAP,
    reps_per_cluster: int = DEFAULT_REPS_PER_CLUSTER,
) -> dict:
    """
    Cluster `vectors` (N x D) into at most `target_clusters` categories. Returns a JSON-serializable
    dict with `clusters`, each: {id, size, representatives (row indices into `vectors`, for naming),
    centroid (unit, original space), prototypes (unit sub-centroids, for nearest-prototype matching)}.
    Deterministic for a fixed seed and pinned numpy/scikit-learn/umap versions.
    """
    from sklearn.cluster import HDBSCAN

    n = int(vectors.shape[0])
    if n == 0:
        return {"count": 0, "sampledCount": 0, "seed": seed, "clusterCount": 0, "clusters": [],
                "metrics": {"noiseFraction": 0.0, "biggestClusterFraction": 0.0, "silhouette": None}}

    x = _l2_normalize(vectors)
    # Cluster up to `cap` rows (this defines the categories); Core assigns the rest via the centroids.
    idx = (np.sort(np.random.RandomState(seed).choice(n, cap, replace=False)) if n > cap
           else np.arange(n))
    xs = x[idx]
    m = xs.shape[0]
    if min_cluster_size is None:
        min_cluster_size = max(25, m // 500)

    embedding = _reduce(xs, seed) if m > UMAP_COMPONENTS * 3 else xs
    finite = np.isfinite(embedding).all(axis=1)
    fine = np.full(m, -1)
    if int(finite.sum()) >= 2:
        mcs = min(min_cluster_size, max(2, int(finite.sum()) // 4))
        fine[np.where(finite)[0]] = HDBSCAN(min_cluster_size=mcs).fit_predict(embedding[finite])
    merged, fine_to_merged = _merge(embedding, fine, target_clusters)

    clusters = []
    for c in sorted(int(v) for v in set(merged.tolist()) if v != -1):
        pos = np.where(merged == c)[0]
        centroid_emb = embedding[pos].mean(axis=0)
        order = pos[np.argsort(np.linalg.norm(embedding[pos] - centroid_emb, axis=1), kind="stable")]
        prototypes = [
            _unit(xs[np.where(fine == f)[0]].mean(axis=0))
            for f in sorted(fine_to_merged)
            if fine_to_merged[f] == c and int((fine == f).sum()) > 0
        ]
        clusters.append({
            "id": c,
            "size": int(pos.size),
            "representatives": [int(idx[p]) for p in order[:reps_per_cluster]],
            "centroid": [float(v) for v in _unit(xs[pos].mean(axis=0))],
            "prototypes": [[float(v) for v in p] for p in prototypes],
        })

    return {
        "count": n,
        "sampledCount": m,
        "seed": seed,
        "clusterCount": len(clusters),
        "clusters": clusters,
        "metrics": _metrics(embedding, merged, seed),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Deterministic corpus clustering for MailPilot discovery.")
    ap.add_argument("--input", required=True, help="float32 binary file of shape (count, dim)")
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--target-clusters", type=int, default=DEFAULT_TARGET_CLUSTERS)
    ap.add_argument("--min-cluster-size", type=int, default=None)
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP)
    ap.add_argument("--output", default=None, help="write JSON here; default stdout")
    a = ap.parse_args(argv)

    vectors = np.fromfile(a.input, dtype=np.float32).reshape(a.count, a.dim)
    result = cluster(
        vectors,
        seed=a.seed,
        target_clusters=a.target_clusters,
        min_cluster_size=a.min_cluster_size,
        cap=a.cap,
    )
    payload = json.dumps(result)
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            f.write(payload)
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
