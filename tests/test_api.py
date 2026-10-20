"""API behaviour tests: health, text upload + chat retrieval, tenant
isolation, request validation. All run offline (hashing embedder)."""

from tests.conftest import SAMPLE_TXT

REFUND_QUESTION = "How many days do customers have to request a refund?"


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    # Offline mode: local embedder, no LLM configured.
    assert body["embedder"] == "local-hashing"
    assert body["llm_configured"] is False


def test_upload_and_chat_txt(client, new_tenant, upload_file):
    tenant = new_tenant()
    resp = upload_file(tenant, SAMPLE_TXT, "northwind_handbook.txt", "text/plain")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["chunks_stored"] >= 3
    assert body["total_chunks"] == body["chunks_stored"]

    chat = client.post(
        f"/tenants/{tenant}/chat", json={"question": REFUND_QUESTION}
    )
    assert chat.status_code == 200, chat.text
    body = chat.json()
    assert body["mode"] == "extractive-fallback"
    assert body["sources"], "expected at least one retrieved chunk"
    top = body["sources"][0]
    # The refund paragraph must rank first and ground the answer.
    assert "30 days" in top["text"]
    assert top["source"] == "northwind_handbook.txt"
    assert "30 days" in body["answer"]


def test_tenant_isolation(client, new_tenant, upload_file):
    tenant_a, tenant_b = new_tenant(), new_tenant()
    up = upload_file(tenant_a, SAMPLE_TXT, "northwind_handbook.txt", "text/plain")
    assert up.status_code == 200, up.text

    # Tenant B has no documents: retrieval is empty, nothing leaks from A.
    chat_b = client.post(
        f"/tenants/{tenant_b}/chat", json={"question": REFUND_QUESTION}
    )
    assert chat_b.status_code == 200
    assert chat_b.json()["sources"] == []
    assert chat_b.json()["mode"] == "no-context"

    # Stats reflect the isolation too.
    stats_a = client.get(f"/tenants/{tenant_a}/stats").json()
    stats_b = client.get(f"/tenants/{tenant_b}/stats").json()
    assert stats_a["chunks"] >= 3
    assert stats_b["chunks"] == 0

    # A only ever retrieves its own sources.
    chat_a = client.post(
        f"/tenants/{tenant_a}/chat", json={"question": REFUND_QUESTION}
    )
    assert chat_a.json()["sources"]
    assert all(
        s["source"] == "northwind_handbook.txt" for s in chat_a.json()["sources"]
    )


def test_invalid_tenant_id_rejected(client):
    resp = client.get("/tenants/bad tenant!/stats")
    assert resp.status_code == 400


def test_chat_on_empty_tenant(client, new_tenant):
    tenant = new_tenant()
    resp = client.post(
        f"/tenants/{tenant}/chat", json={"question": "Anything at all?"}
    )
    assert resp.status_code == 200
    assert resp.json()["sources"] == []
    assert resp.json()["mode"] == "no-context"
