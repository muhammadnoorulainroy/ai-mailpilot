# Optional Python ML Tools

The shipped TypeScript Core does not require Python. This directory contains two independent tools
for experiments and opt-in retrieval quality improvements.

## Cross-Encoder Reranker

`rerank_server.py` is integrated with Core behind the `crossEncoderRerank` experimental feature
flag. When enabled, Core starts the process lazily, keeps the model warm, and sends a query plus a
small set of candidate snippets over newline-delimited JSON on standard input. The sidecar returns
one relevance score per candidate.

If Python, the model, or the sidecar is unavailable, chat keeps the existing fused retrieval order.
The optional feature therefore degrades without breaking an answer.

Install the reranking dependency:

```bash
# Install an appropriate PyTorch build first if required by your platform.
python -m pip install -r ml/requirements-rerank.txt
```

The default model is `BAAI/bge-reranker-v2-m3` and is downloaded by
`sentence-transformers` on first use. Override it with `MAILPILOT_RERANK_MODEL`; override the Python
executable with `MAILPILOT_PYTHON`.

Enable **Cross-encoder chat reranking** in AI MailPilot Settings, then restart Core so the optional
client is present in the composition root.

## UMAP/HDBSCAN Clustering Experiment

`cluster.py` is a standalone evaluation CLI. It is **not called by the current Core runtime**. The
production residual-discovery path uses deterministic bounded leader clustering in TypeScript.

The experimental pipeline is:

```text
L2 normalization
  -> seeded UMAP dimensionality reduction
  -> HDBSCAN fine clusters and noise
  -> agglomerative merge to a target cluster count
  -> original-space aggregate and sub-prototype vectors
```

It accepts only numeric embedding vectors and does not open the AI MailPilot database or read
plaintext email. The caller remains responsible for protecting the input file because embeddings
can still reveal information about their source data.

Install the pinned clustering environment:

```bash
python -m pip install -r ml/requirements.txt
```

Run the CLI:

```bash
python ml/cluster.py \
  --input vectors.f32 \
  --count 48244 \
  --dim 1024 \
  --seed 42 \
  --output labels.json
```

`--input` is a raw float32 matrix with shape `(count, dim)`. The output contains sampled counts,
cluster sizes, representative row indexes, aggregate centroids, sub-prototypes, and quality metrics.
The default clustering definition set is capped at 20,000 vectors.

## Tests

```bash
python -m pytest ml/tests -q
```

The clustering dependencies are pinned because the determinism test expects identical output for a
fixed input and seed. Run the Python suite after changing a dependency, numeric default, or output
schema.
