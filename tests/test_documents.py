"""Document-ingestion tests: PDF extraction + retrieval, and rejection of
unsupported or unreadable uploads."""

from tests.conftest import SAMPLE_PDF

WARRANTY_QUESTION = "How long is the extended warranty for the Aurora X200 drone?"


def test_pdf_upload_and_retrieval(client, new_tenant, upload_file):
    tenant = new_tenant()
    resp = upload_file(tenant, SAMPLE_PDF, "aurora_x200_guide.pdf", "application/pdf")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["document"] == "aurora_x200_guide.pdf"
    assert body["chunks_stored"] >= 1

    chat = client.post(
        f"/tenants/{tenant}/chat", json={"question": WARRANTY_QUESTION}
    )
    assert chat.status_code == 200, chat.text
    body = chat.json()
    assert body["sources"], "expected at least one retrieved chunk"
    top = body["sources"][0]
    # Text really came out of the PDF: the warranty fact ranks first.
    assert "36 months" in top["text"]
    assert top["source"] == "aurora_x200_guide.pdf"
    assert "36 months" in body["answer"]


def test_invalid_file_type_rejected(client, new_tenant):
    tenant = new_tenant()
    resp = client.post(
        f"/tenants/{tenant}/documents",
        files={"file": ("photo.png", b"\x89PNG\r\n\x1a\n not really a png", "image/png")},
    )
    assert resp.status_code == 400
    assert "supported" in resp.json()["detail"]


def test_corrupt_pdf_rejected(client, new_tenant):
    tenant = new_tenant()
    resp = client.post(
        f"/tenants/{tenant}/documents",
        files={"file": ("broken.pdf", b"this is not a pdf at all", "application/pdf")},
    )
    assert resp.status_code == 400


def test_empty_text_file_rejected(client, new_tenant):
    tenant = new_tenant()
    resp = client.post(
        f"/tenants/{tenant}/documents",
        files={"file": ("empty.txt", b"   \n  ", "text/plain")},
    )
    assert resp.status_code == 400
