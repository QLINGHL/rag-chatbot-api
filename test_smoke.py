"""End-to-end smoke test using the offline embedder (no API keys needed).

Run:  python test_smoke.py
Exits non-zero on the first failed assertion.
"""

import os
import tempfile

# Configure BEFORE importing the app: isolated storage, offline mode.
os.environ["QDRANT_PATH"] = tempfile.mkdtemp(prefix="rag-smoke-")
os.environ.pop("OPENAI_API_KEY", None)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

SAMPLE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sample_docs",
    "northwind_handbook.txt",
)


def main() -> None:
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200, health.text
    body = health.json()
    assert body["status"] == "ok", body
    assert body["embedder"] == "local-hashing", body
    print("health:", body)

    with open(SAMPLE, "rb") as fh:
        up = client.post(
            "/tenants/acme/documents",
            files={"file": ("northwind_handbook.txt", fh, "text/plain")},
        )
    assert up.status_code == 200, up.text
    up_body = up.json()
    assert up_body["chunks_stored"] >= 3, up_body
    print("upload:", up_body)

    chat = client.post(
        "/tenants/acme/chat",
        json={"question": "How many days do customers have to request a refund?"},
    )
    assert chat.status_code == 200, chat.text
    chat_body = chat.json()
    assert chat_body["sources"], chat_body
    top = chat_body["sources"][0]
    print("top retrieved chunk (score=%.3f): %s..." % (top["score"], top["text"][:90]))
    assert "30 days" in top["text"], "expected the refund chunk to rank first"
    assert "30 days" in chat_body["answer"], chat_body["answer"]
    assert chat_body["mode"] == "extractive-fallback", chat_body["mode"]

    # Tenant isolation: another tenant must not see acme's documents.
    other = client.post(
        "/tenants/other/chat",
        json={"question": "How many days do customers have to request a refund?"},
    )
    assert other.status_code == 200, other.text
    assert other.json()["sources"] == [], other.json()
    print("tenant isolation: OK (other tenant retrieved 0 chunks)")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
