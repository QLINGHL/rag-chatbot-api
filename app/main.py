"""RAG Document Q&A API.

Multi-tenant retrieval-augmented generation over uploaded text documents:

* POST /tenants/{tenant_id}/documents - upload a .txt/.md/.pdf file; its
  text is extracted, chunked, embedded and stored in that tenant's own
  Qdrant collection.
* POST /tenants/{tenant_id}/chat      - ask a question; the top-k chunks from
  the tenant's collection are retrieved and used to build the LLM prompt.
* GET  /tenants/{tenant_id}/stats     - chunk count for a tenant.
* DELETE /tenants/{tenant_id}         - drop a tenant's collection.
* GET  /health                        - liveness + active embedding mode.

With OPENAI_API_KEY set, embeddings and answers come from any
OpenAI-compatible API. Without it, the service runs fully offline: local
hashing embeddings and an extractive answer built from the top chunk.
"""

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .config import settings
from .documents import extract_text, is_supported
from .embeddings import get_embedder
from .rag import RagStore, validate_tenant_id

app = FastAPI(
    title="RAG Document Q&A API",
    description="Multi-tenant RAG chat over uploaded documents (FastAPI + Qdrant).",
    version="1.0.0",
)

store = RagStore(
    embedder=get_embedder(settings),
    qdrant_url=settings.qdrant_url,
    qdrant_path=settings.qdrant_path,
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=settings.top_k, ge=1, le=20)


def build_prompt(question: str, contexts) -> tuple:
    context_text = "\n\n".join(
        f"[{i + 1}] (source: {c['source']})\n{c['text']}"
        for i, c in enumerate(contexts)
    )
    system = (
        "You are a helpful assistant. Answer the user's question using only "
        "the context below. If the answer is not in the context, say you "
        "don't know. Cite sources like [1], [2] where relevant."
    )
    user = f"Context:\n{context_text}\n\nQuestion: {question}"
    return system, user


def generate_answer(question: str, contexts) -> tuple:
    """Return (answer, mode). Uses the chat API when a key is configured,
    otherwise falls back to an extractive answer from the top chunk."""
    if not contexts:
        return (
            "I don't have any documents for this tenant yet, so I can't "
            "answer that. Upload a document first.",
            "no-context",
        )
    if settings.openai_api_key:
        system, user = build_prompt(question, contexts)
        try:
            resp = httpx.post(
                f"{settings.openai_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={
                    "model": settings.chat_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.2,
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"]
            return answer, "llm"
        except Exception:
            pass  # fall through to the offline extractive answer
    top = contexts[0]["text"].strip()
    return (
        f"Based on the uploaded documents, the most relevant passage is:\n\n{top}",
        "extractive-fallback",
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "embedder": store.embedder_name,
        "llm_configured": bool(settings.openai_api_key),
        "chat_model": settings.chat_model if settings.openai_api_key else None,
    }


@app.post("/tenants/{tenant_id}/documents")
async def upload_document(tenant_id: str, file: UploadFile = File(...)):
    try:
        validate_tenant_id(tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    filename = file.filename or ""
    if not is_supported(filename):
        raise HTTPException(
            status_code=400, detail="Only .txt, .md and .pdf files are supported."
        )
    raw = await file.read()
    try:
        text = extract_text(filename, raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty.")
    chunks = store.add_document(
        tenant_id=tenant_id,
        source=filename,
        text=text,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    return {
        "tenant_id": tenant_id,
        "document": filename,
        "chunks_stored": chunks,
        "total_chunks": store.count(tenant_id),
    }


@app.post("/tenants/{tenant_id}/chat")
def chat(tenant_id: str, req: ChatRequest):
    try:
        validate_tenant_id(tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    contexts = store.search(tenant_id, req.question, req.top_k)
    answer, mode = generate_answer(req.question, contexts)
    return {
        "tenant_id": tenant_id,
        "question": req.question,
        "answer": answer,
        "mode": mode,
        "sources": contexts,
    }


@app.get("/tenants/{tenant_id}/stats")
def stats(tenant_id: str):
    try:
        validate_tenant_id(tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"tenant_id": tenant_id, "chunks": store.count(tenant_id)}


@app.delete("/tenants/{tenant_id}")
def delete_tenant(tenant_id: str):
    try:
        validate_tenant_id(tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    store.drop_tenant(tenant_id)
    return {"tenant_id": tenant_id, "deleted": True}
