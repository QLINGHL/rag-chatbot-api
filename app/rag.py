"""Retrieval layer: chunking + a multi-tenant Qdrant store.

Multi-tenancy model: every tenant gets its own Qdrant collection named
``tenant_<tenant_id>``. That gives hard isolation between tenants (no payload
filtering mistakes can leak documents across tenants) and makes per-tenant
deletion a single drop-collection call.

By default Qdrant runs in local embedded mode (path-based storage), so no
server or Docker is required to run the demo. If ``QDRANT_URL`` is set,
the store connects to a Qdrant server instead (see docker-compose.yml).
"""

import re
import uuid
from typing import Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .hybrid import bm25_scores, rrf_fuse, tokenize

_TENANT_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_tenant_id(tenant_id: str) -> str:
    if not _TENANT_RE.match(tenant_id):
        raise ValueError(
            "tenant_id must be 1-64 chars of letters, digits, '-' or '_'"
        )
    return tenant_id


def collection_name(tenant_id: str) -> str:
    return f"tenant_{tenant_id.lower()}"


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 120) -> List[str]:
    """Split text into overlapping chunks.

    Paragraphs are kept together where possible; oversized paragraphs are
    windowed with overlap so facts near a boundary still appear in a chunk.

    Raises ``ValueError`` for invalid windows: ``chunk_size`` must be at
    least 1 and ``overlap`` must satisfy ``0 <= overlap < chunk_size``
    (an overlap at or above the window size would never advance and
    would loop forever on an oversized paragraph).
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size.")
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    chunks: List[str] = []
    current = ""

    def flush(buf: str) -> None:
        if buf:
            chunks.append(buf)

    for block in blocks:
        if len(block) > chunk_size:
            flush(current)
            current = ""
            start = 0
            while start < len(block):
                chunks.append(block[start : start + chunk_size])
                if start + chunk_size >= len(block):
                    break
                start += chunk_size - overlap
        elif len(current) + len(block) + 1 <= chunk_size:
            current = f"{current}\n{block}" if current else block
        else:
            flush(current)
            current = block
    flush(current)
    return chunks


class RagStore:
    """Thin wrapper over Qdrant + an embedder.

    Connects to a Qdrant server when ``qdrant_url`` is given, otherwise
    uses local embedded mode persisted at ``qdrant_path``.
    """

    def __init__(self, embedder, qdrant_url: str = "", qdrant_path: str = ""):
        if qdrant_url:
            self._client = QdrantClient(url=qdrant_url)
        else:
            self._client = QdrantClient(path=qdrant_path)
        self._embedder = embedder

    @property
    def embedder_name(self) -> str:
        return getattr(self._embedder, "name", "unknown")

    def ensure_collection(self, tenant_id: str) -> str:
        name = collection_name(tenant_id)
        if not self._client.collection_exists(name):
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self._embedder.dim, distance=Distance.COSINE
                ),
            )
        return name

    def add_document(
        self, tenant_id: str, source: str, text: str, chunk_size: int, overlap: int
    ) -> int:
        name = self.ensure_collection(tenant_id)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return 0
        vectors = self._embedder.embed(chunks)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={"text": chunk, "source": source, "chunk_index": i},
            )
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        self._client.upsert(collection_name=name, points=points)
        return len(points)

    def _vector_search(self, name: str, vector, limit: int) -> List[Dict]:
        """Vector search; returns hits with their Qdrant point id attached."""
        try:  # qdrant-client >= 1.10
            result = self._client.query_points(
                collection_name=name, query=vector, limit=limit
            )
            hits = result.points
        except AttributeError:  # older clients
            hits = self._client.search(
                collection_name=name, query_vector=vector, limit=limit
            )
        return [
            {
                "id": str(hit.id),
                "text": hit.payload.get("text", ""),
                "source": hit.payload.get("source"),
                "chunk_index": hit.payload.get("chunk_index"),
                "score": float(hit.score),
            }
            for hit in hits
        ]

    def _scroll_corpus(self, name: str) -> List[Dict]:
        """All chunks of a collection (id + payload), for BM25 scoring."""
        corpus: List[Dict] = []
        offset = None
        while True:
            records, offset = self._client.scroll(
                collection_name=name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                corpus.append(
                    {
                        "id": str(record.id),
                        "text": payload.get("text", ""),
                        "source": payload.get("source"),
                        "chunk_index": payload.get("chunk_index"),
                    }
                )
            if offset is None:
                break
        return corpus

    def search(
        self, tenant_id: str, query: str, top_k: int, hybrid: bool = False
    ) -> List[Dict]:
        """Retrieve the top-k chunks for a query.

        Vector-only by default (Qdrant cosine search). With ``hybrid=True``,
        BM25 keyword scores over the tenant's whole chunk corpus are fused
        with the vector ranking via Reciprocal Rank Fusion (K=60, same as
        eaap-agent-platform), so an exact-keyword match can surface a chunk
        the vector ranking missed. Hybrid hits additionally carry
        ``vector_score`` / ``bm25_score`` (``None`` when a chunk only
        ranked on the other side).
        """
        name = self.ensure_collection(tenant_id)
        vector = self._embedder.embed([query])[0].tolist()
        if not hybrid:
            return [
                {k: v for k, v in hit.items() if k != "id"}
                for hit in self._vector_search(name, vector, top_k)
            ]

        # Candidate pool wider than top_k: fusion needs both rankings to
        # see chunks the other side would have cut off.
        candidate_limit = max(top_k * 2, 20)
        vec_hits = self._vector_search(name, vector, candidate_limit)
        corpus = self._scroll_corpus(name)
        bm25 = bm25_scores(tokenize(query), corpus)
        bm25_ranked = [
            cid for cid, _ in sorted(bm25.items(), key=lambda kv: -kv[1])
        ]
        fused = rrf_fuse(
            [hit["id"] for hit in vec_hits], bm25_ranked, top_k=top_k
        )
        by_id = {hit["id"]: hit for hit in vec_hits}
        by_id.update(
            {doc["id"]: doc for doc in corpus if doc["id"] not in by_id}
        )
        vec_scores = {hit["id"]: hit["score"] for hit in vec_hits}
        out = []
        for entry in fused:
            hit = by_id[entry["id"]]
            out.append(
                {
                    "text": hit["text"],
                    "source": hit.get("source"),
                    "chunk_index": hit.get("chunk_index"),
                    "score": round(entry["score"], 6),
                    "vector_score": (
                        round(vec_scores[entry["id"]], 6)
                        if entry["id"] in vec_scores
                        else None
                    ),
                    "bm25_score": (
                        round(bm25[entry["id"]], 4)
                        if entry["id"] in bm25
                        else None
                    ),
                }
            )
        return out

    def count(self, tenant_id: str) -> int:
        name = self.ensure_collection(tenant_id)
        info = self._client.get_collection(name)
        return int(info.points_count or 0)

    def list_documents(self, tenant_id: str) -> List[Dict]:
        """List documents uploaded to a tenant, aggregated from chunk payloads.

        No separate registry is kept: each point's payload carries its
        ``source`` filename, so the listing works retroactively for
        documents indexed before this endpoint existed.
        """
        name = self.ensure_collection(tenant_id)
        docs: Dict[str, Dict] = {}
        offset = None
        while True:
            records, offset = self._client.scroll(
                collection_name=name,
                limit=1000,
                offset=offset,
                with_payload=["source"],
                with_vectors=False,
            )
            for record in records:
                source = (record.payload or {}).get("source")
                if source is None:
                    continue
                entry = docs.setdefault(source, {"document": source, "chunks": 0})
                entry["chunks"] += 1
            if offset is None:
                break
        return sorted(docs.values(), key=lambda d: d["document"])

    def delete_document(self, tenant_id: str, filename: str) -> int:
        """Delete every chunk of one document; returns the number removed."""
        name = self.ensure_collection(tenant_id)
        flt = Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=filename))]
        )
        result = self._client.count(collection_name=name, count_filter=flt)
        removed = int(result.count or 0)
        if removed:
            self._client.delete(collection_name=name, points_selector=flt)
        return removed

    def drop_tenant(self, tenant_id: str) -> None:
        name = collection_name(tenant_id)
        if self._client.collection_exists(name):
            self._client.delete_collection(name)
