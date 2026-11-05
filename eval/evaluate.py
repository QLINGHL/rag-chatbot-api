"""Offline retrieval evaluation for the RAG API.

Usage (fully offline, no API key needed)::

    python eval/evaluate.py
    python eval/evaluate.py --top-k 2 --output eval/results.json
    python eval/evaluate.py --hybrid --output eval/results-hybrid.json

What it measures
-----------------
The script uploads one document (default: the sample handbook), asks every
question in ``eval/qa_set.json`` through the real ``/chat`` endpoint, and
scores retrieval at *chunk* level: a question is a hit when the chunk that
contains its ``expected`` answer phrase shows up in the retrieved sources.

Reported metrics (over all questions):

* ``hit@1``  - expected chunk ranked first.
* ``hit@k``  - expected chunk anywhere in the top-k retrieved sources.
  With the sample handbook there are only 3 chunks, so ``hit@k`` for large
  k is close to trivial; ``hit@1`` and MRR are the discriminating metrics.
* ``mrr``    - mean reciprocal rank of the first chunk containing the
  expected phrase (0 when it is never retrieved).

The scoring functions (``load_qa_set``, ``first_hit_rank``,
``score_results``) are pure and take no app dependency, so they are unit
tested without spinning up a store. ``run_evaluation`` takes any client
with the same ``post`` interface as ``fastapi.testclient.TestClient``.
The CLI imports the app lazily, only after the evaluation store path has
been decided, so importing this module never creates a Qdrant store.
"""

import argparse
import json
import os
import sys
import uuid

DEFAULT_QA_SET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qa_set.json")
DEFAULT_DOCUMENT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sample_docs",
    "northwind_handbook.txt",
)
_CONTENT_TYPES = {".txt": "text/plain", ".md": "text/markdown", ".pdf": "application/pdf"}


def load_qa_set(path):
    """Load and validate a QA set: list of {id, question, expected}."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list) or not data:
        raise ValueError(f"QA set {path!r} must be a non-empty JSON list.")
    seen = set()
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"QA entry {i} is not an object.")
        for field in ("id", "question", "expected"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"QA entry {i} has a missing/empty {field!r}.")
        if entry["id"] in seen:
            raise ValueError(f"Duplicate QA id {entry['id']!r}.")
        seen.add(entry["id"])
    return data


def first_hit_rank(sources, expected):
    """1-based rank of the first source containing ``expected`` (case
    insensitive), or ``None`` when no retrieved source contains it."""
    needle = expected.lower()
    for rank, source in enumerate(sources, start=1):
        if needle in (source.get("text") or "").lower():
            return rank
    return None


def score_results(per_question, top_k):
    """Aggregate per-question ranks into hit@1 / hit@k / MRR."""
    total = len(per_question)
    if total == 0:
        raise ValueError("Cannot score an empty result list.")
    hits_at_1 = sum(1 for r in per_question if r["rank"] == 1)
    hits_at_k = sum(1 for r in per_question if r["rank"] is not None and r["rank"] <= top_k)
    mrr = sum(1.0 / r["rank"] for r in per_question if r["rank"] is not None) / total
    return {
        "total": total,
        "hit@1": hits_at_1 / total,
        "hit@k": hits_at_k / total,
        "mrr": mrr,
        "misses": total - hits_at_k,
    }


def run_evaluation(client, qa_set, document_path, top_k=4, hybrid=False):
    """Upload ``document_path`` for a fresh tenant, ask every question,
    and return ``{"summary": ..., "results": [...]}``."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    tenant = "eval-" + uuid.uuid4().hex[:12]
    filename = os.path.basename(document_path)
    suffix = os.path.splitext(filename)[1].lower()
    with open(document_path, "rb") as fh:
        upload = client.post(
            f"/tenants/{tenant}/documents",
            files={"file": (filename, fh, _CONTENT_TYPES.get(suffix, "application/octet-stream"))},
        )
    if upload.status_code != 200:
        raise RuntimeError(f"Document upload failed: {upload.status_code} {upload.text}")

    results = []
    for entry in qa_set:
        chat = client.post(
            f"/tenants/{tenant}/chat",
            json={"question": entry["question"], "top_k": top_k, "hybrid": hybrid},
        )
        if chat.status_code != 200:
            results.append({**entry, "rank": None, "error": chat.text})
            continue
        sources = chat.json().get("sources", [])
        rank = first_hit_rank(sources, entry["expected"])
        results.append(
            {
                "id": entry["id"],
                "question": entry["question"],
                "expected": entry["expected"],
                "rank": rank,
                "hit": rank is not None,
                "top_score": sources[0]["score"] if sources else None,
            }
        )
    return {"summary": score_results(results, top_k), "results": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval on a QA set.")
    parser.add_argument("--qa-set", default=DEFAULT_QA_SET)
    parser.add_argument("--document", default=DEFAULT_DOCUMENT)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--hybrid", action="store_true",
                        help="Ask with hybrid (vector + BM25, RRF) retrieval.")
    parser.add_argument("--output", default=None, help="Write the full JSON report here.")
    args = parser.parse_args(argv)

    qa_set = load_qa_set(args.qa_set)
    if not os.path.exists(args.document):
        raise SystemExit(f"Document not found: {args.document}")

    # A throwaway store unless the caller pinned one, and the app module is
    # imported only after the environment is set.
    import tempfile

    os.environ.setdefault("QDRANT_PATH", tempfile.mkdtemp(prefix="rag-eval-"))
    os.environ.pop("QDRANT_URL", None)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    from fastapi.testclient import TestClient

    from app.main import app

    health = TestClient(app).get("/health").json()
    report = run_evaluation(
        TestClient(app), qa_set, args.document, top_k=args.top_k, hybrid=args.hybrid
    )
    report["top_k"] = args.top_k
    report["hybrid"] = args.hybrid
    report["embedder"] = health.get("embedder")
    report["document"] = os.path.basename(args.document)

    summary = report["summary"]
    print(f"document: {report['document']}  embedder: {report['embedder']}  "
          f"top_k: {args.top_k}  hybrid: {args.hybrid}")
    print(f"questions: {summary['total']}  hit@1: {summary['hit@1']:.2f}  "
          f"hit@{args.top_k}: {summary['hit@k']:.2f}  mrr: {summary['mrr']:.3f}  "
          f"misses: {summary['misses']}")
    for item in report["results"]:
        rank = item["rank"] if item["rank"] is not None else "-"
        print(f"  {item['id']} rank={rank}  {item['question']}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
