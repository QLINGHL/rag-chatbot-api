"""Document-management tests: list indexed documents per tenant and delete
a single document's chunks. All run offline (hashing embedder)."""

from tests.conftest import SAMPLE_TXT


def _upload_two_docs(client, upload_file, tenant):
    a = upload_file(tenant, SAMPLE_TXT, "doc-a.txt", "text/plain")
    b = upload_file(tenant, SAMPLE_TXT, "doc-b.txt", "text/plain")
    assert a.status_code == 200, a.text
    assert b.status_code == 200, b.text
    return a.json()["chunks_stored"], b.json()["chunks_stored"]


def test_list_documents(client, new_tenant, upload_file):
    tenant = new_tenant()
    # Empty tenant: no documents yet.
    empty = client.get(f"/tenants/{tenant}/documents")
    assert empty.status_code == 200
    assert empty.json()["documents"] == []

    chunks_a, chunks_b = _upload_two_docs(client, upload_file, tenant)

    listing = client.get(f"/tenants/{tenant}/documents")
    assert listing.status_code == 200
    docs = {d["document"]: d["chunks"] for d in listing.json()["documents"]}
    assert docs == {"doc-a.txt": chunks_a, "doc-b.txt": chunks_b}

    # Tenant-scoped: another tenant sees nothing.
    other = client.get(f"/tenants/{new_tenant()}/documents")
    assert other.json()["documents"] == []


def test_delete_document(client, new_tenant, upload_file):
    tenant = new_tenant()
    chunks_a, chunks_b = _upload_two_docs(client, upload_file, tenant)

    resp = client.delete(f"/tenants/{tenant}/documents/doc-a.txt")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["document"] == "doc-a.txt"
    assert body["chunks_deleted"] == chunks_a

    docs = {d["document"] for d in client.get(f"/tenants/{tenant}/documents").json()["documents"]}
    assert docs == {"doc-b.txt"}

    # The removed document's chunks are gone; the other file still answers.
    stats = client.get(f"/tenants/{tenant}/stats").json()
    assert stats["chunks"] == chunks_b
    chat = client.post(f"/tenants/{tenant}/chat", json={"question": "anything"})
    assert chat.status_code == 200
    assert chat.json()["sources"]
    assert all(s["source"] == "doc-b.txt" for s in chat.json()["sources"])

    # Deleting it again is a 404 now.
    again = client.delete(f"/tenants/{tenant}/documents/doc-a.txt")
    assert again.status_code == 404


def test_delete_missing_document_404(client, new_tenant):
    tenant = new_tenant()
    resp = client.delete(f"/tenants/{tenant}/documents/nope.txt")
    assert resp.status_code == 404


def test_document_name_validation(client, new_tenant):
    tenant = new_tenant()
    # URL-encoded traversal reaches the handler and must be rejected.
    resp = client.delete(f"/tenants/{tenant}/documents/..%2Fsecret.txt")
    assert resp.status_code == 400, resp.status_code
    resp = client.delete(f"/tenants/{tenant}/documents/..%5Csecret.txt")
    assert resp.status_code == 400, resp.status_code


def test_invalid_tenant_id_rejected(client):
    assert client.get("/tenants/bad tenant!/documents").status_code == 400
