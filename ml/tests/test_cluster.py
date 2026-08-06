"""
Automated checks for the clustering sidecar. The load-bearing one is test_determinism: same input and
seed must yield byte-identical output, since the whole point is reproducible discovery. The rest check
that it recovers well-separated groups with pure representatives, exposes sub-prototypes for Core's
multi-prototype matching, honours the target ceiling, and survives the edges (empty, tiny).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cluster as cl  # noqa: E402


def make_blobs(n_blobs: int, per: int, dim: int = 32, spread: float = 0.03, seed: int = 0):
    """n_blobs tight, near-orthogonal unit blobs; returns (vectors, ground-truth blob id per row)."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_blobs, dim)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    pts, truth = [], []
    for i, c in enumerate(centers):
        pts.append(c + spread * rng.randn(per, dim))
        truth += [i] * per
    return np.vstack(pts).astype(np.float32), np.array(truth)


@pytest.fixture(scope="module")
def six_blobs():
    return make_blobs(6, 100)


def test_determinism(six_blobs):
    x, _ = six_blobs
    a = cl.cluster(x, seed=42)
    b = cl.cluster(x, seed=42)
    assert a["clusterCount"] == b["clusterCount"]
    assert [c["representatives"] for c in a["clusters"]] == [c["representatives"] for c in b["clusters"]]
    assert [c["centroid"] for c in a["clusters"]] == [c["centroid"] for c in b["clusters"]]  # byte-identical


def test_recovers_groups_with_pure_reps(six_blobs):
    x, truth = six_blobs
    res = cl.cluster(x, seed=42, target_clusters=6)
    assert 4 <= res["clusterCount"] <= 6
    for c in res["clusters"]:
        reps = c["representatives"]
        assert len(reps) > 0
        # A cluster's representatives should overwhelmingly come from one ground-truth blob.
        blobs = [int(truth[i]) for i in reps]
        top = max(blobs.count(b) for b in set(blobs))
        assert top / len(reps) >= 0.75
        assert abs(np.linalg.norm(c["centroid"]) - 1.0) < 1e-3  # unit centroid


def test_every_cluster_has_prototypes(six_blobs):
    x, _ = six_blobs
    res = cl.cluster(x, seed=42)
    for c in res["clusters"]:
        assert len(c["prototypes"]) >= 1  # at least one sub-prototype for multi-prototype matching
        assert all(abs(np.linalg.norm(p) - 1.0) < 1e-3 for p in c["prototypes"])  # unit prototypes


def test_merges_down_to_target():
    x, _ = make_blobs(12, 60)
    res = cl.cluster(x, seed=42, target_clusters=5)
    assert 0 < res["clusterCount"] <= 5


def test_empty_input():
    res = cl.cluster(np.zeros((0, 32), dtype=np.float32))
    assert res["clusterCount"] == 0
    assert res["clusters"] == []


def test_tiny_input_does_not_crash():
    x, _ = make_blobs(2, 6, spread=0.01)
    res = cl.cluster(x, seed=42, target_clusters=2, min_cluster_size=3)
    assert res["count"] == 12
    assert isinstance(res["clusters"], list)
