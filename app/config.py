"""Runtime configuration, loaded from environment variables (and an optional
local .env file). No third-party config library needed."""

import os
from dataclasses import dataclass


def _load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


@dataclass
class Settings:
    # OpenAI-compatible endpoints. If openai_api_key is empty, the service
    # runs fully offline: local hashing embeddings + extractive answers.
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Qdrant connection. If qdrant_url is set (e.g. http://qdrant:6333 in
    # Docker Compose), the app talks to a Qdrant server; otherwise it runs
    # in local embedded mode, persisted under qdrant_path.
    qdrant_url: str = ""
    qdrant_path: str = "./data/qdrant"

    # Retrieval defaults.
    top_k: int = 4
    chunk_size: int = 600
    chunk_overlap: int = 120
    # Hybrid retrieval: fuse Qdrant vector search with BM25 keyword search
    # via Reciprocal Rank Fusion (same scheme as eaap-agent-platform).
    # Off by default; can also be enabled per chat request.
    hybrid_retrieval: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        return cls(
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            openai_base_url=os.environ.get(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            ),
            chat_model=os.environ.get("CHAT_MODEL", "gpt-4o-mini"),
            embedding_model=os.environ.get(
                "EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            qdrant_url=os.environ.get("QDRANT_URL", ""),
            qdrant_path=os.environ.get("QDRANT_PATH", "./data/qdrant"),
            top_k=int(os.environ.get("TOP_K", "4")),
            chunk_size=int(os.environ.get("CHUNK_SIZE", "600")),
            chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "120")),
            hybrid_retrieval=os.environ.get("HYBRID_RETRIEVAL", "").lower()
            in ("1", "true", "yes"),
        )


settings = Settings.from_env()
