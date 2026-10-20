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
from qdrant_client.models import Distance, PointStruct, VectorParams

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
    """
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

    def search(self, tenant_id: str, query: str, top_k: int) -> List[Dict]:
        name = self.ensure_collection(tenant_id)
        vector = self._embedder.embed([query])[0]
        try:  # qdrant-client >= 1.10
            result = self._client.query_points(
                collection_name=name, query=vector.tolist(), limit=top_k
            )
            hits = result.points
        except AttributeError:  # older clients
            hits = self._client.search(
                collection_name=name, query_vector=vector.tolist(), limit=top_k
            )
        return [
            {
                "text": hit.payload.get("text", ""),
                "source": hit.payload.get("source"),
                "chunk_index": hit.payload.get("chunk_index"),
                "score": float(hit.score),
            }
            for hit in hits
        ]

    def count(self, tenant_id: str) -> int:
        name = self.ensure_collection(tenant_id)
        info = self._client.get_collection(name)
        return int(info.points_count or 0)

    def drop_tenant(self, tenant_id: str) -> None:
        name = collection_name(tenant_id)
        if self._client.collection_exists(name):
            self._client.delete_collection(name)
