"""Hybrid retrieval helpers: BM25 keyword ranking + Reciprocal Rank Fusion.

This mirrors the hybrid search in eaap-agent-platform (``app/knowledge.py``,
FR-601-lite/602/603/605/607) so both services rank the same way: the same
tokenizer (lowercase, stopword removal, light stemming, CJK single
characters), classic BM25 (``k1=1.5``, ``b=0.75``) and RRF fusion with
``K=60``.

The vector side lives in Qdrant (see :meth:`RagStore.search`); this module
only handles the keyword side and the fusion of two ranked id lists.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "do", "does",
    "did", "have", "has", "had", "how", "what", "when", "where", "who", "why",
    "we", "i", "you", "he", "she", "it", "they", "to", "of", "in", "on", "at",
    "for", "with", "by", "from", "and", "or", "not", "this", "that", "there",
    "their", "our", "your", "his", "her", "its", "as", "if", "then", "than",
    "many", "much", "can", "could", "should", "would", "will", "shall",
}

# BM25 parameters (same as EAAP) and the RRF constant.
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60


def _stem(tok: str) -> str:
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str) -> List[str]:
    """Lowercase, split on word boundaries, drop stopwords, light stem."""
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower())
            if t not in _STOPWORDS]


def bm25_scores(query_tokens: Sequence[str], docs: Sequence[Dict]) -> Dict[str, float]:
    """Classic BM25 over a chunk corpus.

    ``docs`` is a list of ``{"id": ..., "text": ...}`` mappings; returns
    ``{doc_id: score}`` containing only docs with a positive score.
    An empty query or an empty corpus scores nothing.
    """
    docs = list(docs)
    n = len(docs)
    if n == 0 or not query_tokens:
        return {}
    doc_freq: Counter = Counter()
    counters: List[Counter] = []
    lengths: List[int] = []
    for doc in docs:
        toks = tokenize(doc["text"])
        counters.append(Counter(toks))
        lengths.append(len(toks))
        for t in set(toks):
            doc_freq[t] += 1
    avgdl = (sum(lengths) / n) if n else 1.0
    scores: Dict[str, float] = {}
    for doc, counter, dl in zip(docs, counters, lengths):
        score = 0.0
        for t in set(query_tokens):
            if t not in counter:
                continue
            idf = math.log(1 + (n - doc_freq[t] + 0.5) / (doc_freq[t] + 0.5))
            tf = counter[t]
            score += idf * (tf * (BM25_K1 + 1)) / (
                tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / avgdl)
            )
        if score > 0:
            scores[str(doc["id"])] = score
    return scores


def _ranks(ranked_ids: Sequence[str]) -> Dict[str, int]:
    """1-based rank per id, in the order given."""
    return {cid: i + 1 for i, cid in enumerate(ranked_ids)}


def rrf_fuse(
    vector_ranked_ids: Sequence[str],
    bm25_ranked_ids: Sequence[str],
    top_k: int,
    k: int = RRF_K,
) -> List[Dict[str, float]]:
    """Fuse two ranked id lists with Reciprocal Rank Fusion.

    Returns ``[{"id": ..., "score": ...}]`` sorted by fused score,
    truncated to ``top_k``. An id present in only one ranking still
    appears, scored from that ranking alone — that is the point of
    fusion: a keyword-only match can rescue a chunk the vector
    ranking missed (and vice versa).
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    vec_ranks = _ranks(vector_ranked_ids)
    bm_ranks = _ranks(bm25_ranked_ids)
    fused: Dict[str, float] = {}
    for cid in set(vec_ranks) | set(bm_ranks):
        fused[cid] = (1 / (k + vec_ranks[cid]) if cid in vec_ranks else 0) + (
            1 / (k + bm_ranks[cid]) if cid in bm_ranks else 0
        )
    return [
        {"id": cid, "score": score}
        for cid, score in sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
    ]
