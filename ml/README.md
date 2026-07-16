# ml — clustering sidecar

Stateless Python service that clusters email embedding vectors for corpus-first discovery. Core hands
it a batch of vectors and gets back one cluster label per vector plus per-cluster representatives; it
then names each cluster (LLM) and builds the category centroid from the cluster's real members. The
clusters **are** the categories, so a high-volume group can never be dropped the way a
sample-then-name taxonomy drops it.

It never touches the database or plaintext email — only numbers in, numbers out.

## Pipeline
`L2-normalize -> seeded spectral-init UMAP (15d) -> HDBSCAN -> agglomerative merge to a target count`.
Deterministic for a fixed seed and the pinned library versions in `requirements.txt`.

## Setup
```
python -m pip install -r ml/requirements.txt
```

## Run (Phase 1 CLI; Core will call this per discovery)
```
python ml/cluster.py --input vectors.f32 --count 48244 --dim 1024 --seed 42 --output labels.json
```
`--input` is a raw float32 buffer of shape `(count, dim)`. Output JSON: `labels`, `sizes`,
`representatives`, `metrics`.

## Test
```
python -m pytest ml/tests -q
```
`test_determinism` is the guard: same input + seed must give byte-identical labels. Re-run it after
any dependency bump.
