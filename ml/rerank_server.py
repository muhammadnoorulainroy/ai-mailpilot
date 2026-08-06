"""
Persistent cross-encoder reranking sidecar for local RAG.

Loads a multilingual cross-encoder ONCE, then serves newline-delimited JSON requests over stdin and
writes one JSON response line per request on stdout:

    {"query": "...", "documents": ["...", "..."]}  ->  {"scores": [0.98, 0.02]}

The parent (Core) keeps this process warm so the model load is amortized across all chat queries
rather than paid per query. A cross-encoder scores the (query, document) pair jointly, which ranks
far more accurately than bi-encoder similarity or a weak local LLM acting as a listwise reranker.

Privacy-preserving: only the query and candidate snippets cross the boundary, and no network call is
made once the model is cached locally. Optional: if sentence-transformers or the model is missing, it
prints a not-ready line and exits, and Core falls back to fusion order.

Default model: BAAI/bge-reranker-v2-m3 (568M, multilingual, standard architecture, no remote code).
Override with MAILPILOT_RERANK_MODEL.
"""

import json
import os
import sys


def _emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    model_id = os.environ.get("MAILPILOT_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_id, max_length=512)
    except Exception as e:  # noqa: BLE001 - report any load failure and let Core fall back
        _emit({"ready": False, "error": f"{type(e).__name__}: {e}"})
        return 1

    _emit({"ready": True, "model": model_id})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            query = req["query"]
            documents = req["documents"]
            if not documents:
                _emit({"scores": []})
                continue
            pairs = [[query, d] for d in documents]
            scores = model.predict(pairs)
            _emit({"scores": [float(s) for s in scores]})
        except Exception as e:  # noqa: BLE001 - one bad request must not kill the warm server
            _emit({"error": f"{type(e).__name__}: {e}"})

    return 0


if __name__ == "__main__":
    sys.exit(main())
