"""Chunking overlap tests: unit-level windowing behaviour, per-upload
``chunk_size`` / ``chunk_overlap`` query parameters, and the CHUNK_SIZE /
CHUNK_OVERLAP environment settings."""

import pytest

from app.config import Settings
from app.rag import chunk_text
from tests.conftest import SAMPLE_TXT

LONG_PARAGRAPH = " ".join(f"word{i}" for i in range(200))


def test_overlap_shares_tail_between_consecutive_chunks():
    chunks = chunk_text(LONG_PARAGRAPH, chunk_size=100, overlap=20)
    assert len(chunks) >= 3
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt[:20] == prev[-20:]
    # No content is lost: the windows tile the paragraph end to end.
    assert chunks[0].startswith("word0")
    assert chunks[-1].endswith("word199")


def test_zero_overlap_gives_disjoint_chunks():
    chunks = chunk_text(LONG_PARAGRAPH, chunk_size=100, overlap=0)
    assert "".join(chunks) == LONG_PARAGRAPH


def test_larger_overlap_produces_more_chunks():
    no_overlap = chunk_text(LONG_PARAGRAPH, chunk_size=100, overlap=0)
    with_overlap = chunk_text(LONG_PARAGRAPH, chunk_size=100, overlap=80)
    assert len(with_overlap) > len(no_overlap)


def test_empty_text_chunks_to_nothing():
    assert chunk_text("   \n  ", chunk_size=100, overlap=10) == []


@pytest.mark.parametrize(
    "chunk_size,overlap",
    [(0, 0), (100, 100), (100, 150), (100, -1)],
)
def test_invalid_window_rejected(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text(LONG_PARAGRAPH, chunk_size=chunk_size, overlap=overlap)


def test_settings_read_chunk_env_vars(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "400")
    monkeypatch.setenv("CHUNK_OVERLAP", "50")
    settings = Settings.from_env()
    assert settings.chunk_size == 400
    assert settings.chunk_overlap == 50


def _upload_with_params(client, tenant, chunk_size, chunk_overlap):
    with open(SAMPLE_TXT, "rb") as fh:
        return client.post(
            f"/tenants/{tenant}/documents"
            f"?chunk_size={chunk_size}&chunk_overlap={chunk_overlap}",
            files={"file": ("northwind_handbook.txt", fh, "text/plain")},
        )


def test_upload_accepts_chunk_params_and_echoes_them(
    client, new_tenant
):
    tenant = new_tenant()
    resp = _upload_with_params(client, tenant, 100, 20)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["chunk_size"] == 100
    assert body["chunk_overlap"] == 20
    assert body["chunks_stored"] >= 1


def test_upload_larger_overlap_stores_more_chunks(client, new_tenant):
    tenant_plain, tenant_overlap = new_tenant(), new_tenant()
    plain = _upload_with_params(client, tenant_plain, 100, 0)
    overlapped = _upload_with_params(client, tenant_overlap, 100, 80)
    assert plain.status_code == 200, plain.text
    assert overlapped.status_code == 200, overlapped.text
    assert overlapped.json()["chunks_stored"] > plain.json()["chunks_stored"]


def test_upload_rejects_overlap_at_or_above_chunk_size(client, new_tenant):
    tenant = new_tenant()
    resp = _upload_with_params(client, tenant, 100, 100)
    assert resp.status_code == 400
    assert "chunk_overlap" in resp.json()["detail"]
