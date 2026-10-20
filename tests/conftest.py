"""Pytest configuration: run the whole suite offline, against a throwaway
store, with no API keys.

Environment is set BEFORE the app is imported: no ``OPENAI_API_KEY`` (so
the local hashing embedder + extractive answers are used) and an isolated
temporary Qdrant path (so tests never touch real data).
"""

import os
import sys
import tempfile
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["QDRANT_PATH"] = tempfile.mkdtemp(prefix="rag-tests-")
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("QDRANT_URL", None)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_TXT = os.path.join(PROJECT_ROOT, "sample_docs", "northwind_handbook.txt")
SAMPLE_PDF = os.path.join(PROJECT_ROOT, "sample_docs", "aurora_x200_guide.pdf")


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture()
def new_tenant():
    """Factory for fresh tenant ids: one tenant == one isolated collection,
    so a unique id per test keeps tests independent of each other."""

    def _make() -> str:
        return "t-" + uuid.uuid4().hex[:12]

    return _make


@pytest.fixture()
def upload_file(client):
    """Upload a file from disk to a tenant; returns the raw response."""

    def _upload(tenant: str, path: str, filename: str, content_type: str):
        with open(path, "rb") as fh:
            return client.post(
                f"/tenants/{tenant}/documents",
                files={"file": (filename, fh, content_type)},
            )

    return _upload
