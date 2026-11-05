"""Hybrid retrieval tests: BM25 scoring, RRF fusion (unit level) and the
``hybrid`` switch on the chat endpoint (integration level).

The rescue behaviour — a chunk the vector ranking missed surfacing via its
BM25 rank — is asserted deterministically at the fusion level, where the
rankings are inputs. At the API level the hashing embedder is itself
keyword-driven, so the integration tests assert wiring (flag flows through,
per-side scores are reported, known-good ranking is preserved) rather than
a contrived rescue.
"""

import tempfile

import pytest

from app.embeddings import HashingEmbedder
from app.hybrid import bm25_scores, rrf_fuse, tokenize
from app.rag import RagStore
from tests.conftest import SAMPLE_TXT

REFUND_QUESTION = "How many days do customers have to request a refund?"


# ------------------------------------------------------------------ tokenize --

def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("How many days do customers have to request a refund?") == [
        "day", "customer", "request", "refund",
    ]


def test_tokenize_keeps_cjk_single_chars():
    assert tokenize("退款政策说明") == ["退", "款", "政", "策", "说", "明"]


# --------------------------------------------------------------------- bm25 --

def test_bm25_ranks_repeated_term_first_and_short_doc_over_long():
    docs = [
        {"id": "short", "text": "refund refund refund days"},
        {"id": "long", "text": "refund " + "filler " * 40 + "days"},
        {"id": "other", "text": "shipping takes five business days"},
    ]
    scores = bm25_scores(tokenize("refund"), docs)
    assert scores["short"] > scores["long"] > 0
    assert "other" not in scores


def test_bm25_empty_query_or_corpus_scores_nothing():
    docs = [{"id": "a", "text": "refund policy"}]
    assert bm25_scores([], docs) == {}
    assert bm25_scores(tokenize("refund"), []) == {}
    assert bm25_scores(tokenize("quokka"), docs) == {}


def test_bm25_rare_term_beats_common_term():
    docs = [
        {"id": "a", "text": "refund policy refund policy refund policy"},
        {"id": "b", "text": "refund policy refund policy refund policy"},
        {"id": "c", "text": "refund policy quokka"},
    ]
    scores = bm25_scores(tokenize("refund quokka"), docs)
    assert scores["c"] > scores["a"]


# ---------------------------------------------------------------------- rrf --

def test_rrf_fuse_keyword_only_chunk_is_rescued():
    # C ranks last by vector but first by BM25: fusion must lift it to #1.
    fused = rrf_fuse(["a", "b", "c"], ["c"], top_k=3)
    assert [entry["id"] for entry in fused] == ["c", "a", "b"]
    assert fused[0]["score"] > fused[1]["score"]


def test_rrf_fuse_chunk_in_one_ranking_only_still_appears():
    fused = rrf_fuse(["a"], ["b"], top_k=2)
    assert {entry["id"] for entry in fused} == {"a", "b"}


def test_rrf_fuse_respects_top_k_and_rejects_bad_top_k():
    fused = rrf_fuse(["a", "b", "c"], ["c", "b", "a"], top_k=2)
    assert len(fused) == 2
    with pytest.raises(ValueError):
        rrf_fuse(["a"], ["a"], top_k=0)


# -------------------------------------------------------------------- store --

def _make_store():
    return RagStore(
        embedder=HashingEmbedder(), qdrant_path=tempfile.mkdtemp(prefix="rag-hybrid-")
    )


def test_store_hybrid_search_reports_both_scores():
    store = _make_store()
    store.add_document(
        "t1", "handbook.txt",
        "Refunds and returns. Customers may request a full refund within "
        "30 days of purchase for any unused product.",
        chunk_size=600, overlap=0,
    )
    store.add_document(
        "t1", "shipping.txt",
        "Standard shipping takes 3 to 5 business days within Europe.",
        chunk_size=600, overlap=0,
    )
    hits = store.search("t1", "refund within 30 days", top_k=2, hybrid=True)
    assert len(hits) == 2
    assert "30 days" in hits[0]["text"]
    assert hits[0]["vector_score"] is not None
    assert hits[0]["bm25_score"] is not None
    assert hits[0]["bm25_score"] > 0


def test_store_vector_search_unchanged_without_hybrid():
    store = _make_store()
    store.add_document(
        "t1", "handbook.txt", "Refunds within 30 days of purchase.",
        chunk_size=600, overlap=0,
    )
    hits = store.search("t1", "refund", top_k=1)
    assert len(hits) == 1
    assert set(hits[0]) == {"text", "source", "chunk_index", "score"}


# ---------------------------------------------------------------------- api --

def test_chat_hybrid_flag_and_retrieval_field(client, new_tenant, upload_file):
    tenant = new_tenant()
    resp = upload_file(tenant, SAMPLE_TXT, "northwind_handbook.txt", "text/plain")
    assert resp.status_code == 200, resp.text

    chat = client.post(
        f"/tenants/{tenant}/chat",
        json={"question": REFUND_QUESTION, "hybrid": True},
    )
    assert chat.status_code == 200, chat.text
    body = chat.json()
    assert body["retrieval"] == "hybrid"
    assert body["sources"], "expected at least one retrieved chunk"
    top = body["sources"][0]
    # Hybrid must not break the known-good case: the refund chunk stays #1.
    assert "30 days" in top["text"]
    assert "vector_score" in top and "bm25_score" in top


def test_chat_defaults_to_vector_and_explicit_false_overrides(
    client, new_tenant, upload_file
):
    tenant = new_tenant()
    assert upload_file(
        tenant, SAMPLE_TXT, "northwind_handbook.txt", "text/plain"
    ).status_code == 200

    default = client.post(
        f"/tenants/{tenant}/chat", json={"question": REFUND_QUESTION}
    ).json()
    assert default["retrieval"] == "vector"
    assert "vector_score" not in default["sources"][0]

    explicit = client.post(
        f"/tenants/{tenant}/chat",
        json={"question": REFUND_QUESTION, "hybrid": False},
    ).json()
    assert explicit["retrieval"] == "vector"


def test_chat_hybrid_on_empty_tenant_returns_nothing(client, new_tenant):
    tenant = new_tenant()
    chat = client.post(
        f"/tenants/{tenant}/chat",
        json={"question": REFUND_QUESTION, "hybrid": True},
    )
    assert chat.status_code == 200
    assert chat.json()["sources"] == []
    assert chat.json()["retrieval"] == "hybrid"
    assert chat.json()["mode"] == "no-context"


def test_health_reports_hybrid_default(client):
    body = client.get("/health").json()
    assert body["hybrid_retrieval"] is False
