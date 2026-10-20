# RAG Document Q&A API

A multi-tenant **Retrieval-Augmented Generation (RAG)** service built with
**FastAPI** and **Qdrant**. Upload text documents for a tenant, then ask
questions — the API retrieves the most relevant chunks from that tenant's own
vector collection, builds a grounded prompt, and answers with an LLM.

Runs in two modes:

- **Online mode** — set `OPENAI_API_KEY` to use any OpenAI-compatible API
  (OpenAI, Azure, or a local gateway) for embeddings and chat answers.
- **Offline demo mode** — no keys at all. A pure Python/NumPy hashing
  embedder and an extractive answer fallback keep the full pipeline
  (upload → chunk → embed → retrieve → answer) working end-to-end.

## Features

- 📄 **Document ingestion** — upload `.txt` / `.md` files; automatic
  paragraph-aware chunking with overlap.
- 🏢 **Multi-tenant by design** — every tenant gets a dedicated Qdrant
  collection (`tenant_<id>`), so tenant data is physically isolated and can
  be dropped with one call.
- 🔎 **Semantic retrieval** — top-k cosine search per tenant, with source
  attribution (file name, chunk index, score) returned on every answer.
- 🔌 **Pluggable embeddings** — OpenAI-compatible endpoint, or the built-in
  offline hashing embedder (TF-IDF-style, deterministic).
- 🗄️ **Zero-infra vector store** — Qdrant in local embedded mode
  (path-based), no server or Docker needed to run the demo.
- ✅ **Tested end-to-end** — `test_smoke.py` exercises upload, retrieval,
  answer assembly and tenant isolation with FastAPI's TestClient.

## Tech Stack

| Layer        | Technology                                  |
|--------------|---------------------------------------------|
| API          | FastAPI, Pydantic, Uvicorn                  |
| Vector store | Qdrant (`qdrant-client`, embedded mode)     |
| Embeddings   | OpenAI-compatible API, or local NumPy hashing embedder |
| LLM          | Any OpenAI-compatible chat completions API  |
| Language     | Python 3.10+                                |

## Project Structure

```
rag-chatbot-api/
├── app/
│   ├── main.py        # FastAPI app: endpoints, prompt building, answer generation
│   ├── rag.py         # Chunking + multi-tenant Qdrant store
│   ├── embeddings.py  # OpenAI embedder + offline hashing embedder
│   └── config.py      # Env-based settings (with .env support)
├── sample_docs/
│   └── northwind_handbook.txt   # Sample document for the demo
├── demo.sh            # curl walkthrough against a running server
├── test_smoke.py      # Offline end-to-end smoke test
├── requirements.txt
└── .env.example
```

## Setup

```bash
git clone <this-repo>
cd rag-chatbot-api

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # optional: add your OPENAI_API_KEY
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

Interactive docs (Swagger UI) are then available at
<http://localhost:8000/docs>.

### Offline smoke test

```bash
python test_smoke.py
```

This spins the app up in-process, uploads the sample handbook, and verifies
that a refund question retrieves the refund chunk and that tenants are
isolated from each other.

### Demo with curl

With the server running:

```bash
bash demo.sh
```

## API

### `GET /health`

```json
{ "status": "ok", "embedder": "local-hashing", "llm_configured": false }
```

### `POST /tenants/{tenant_id}/documents`

Upload a `.txt` or `.md` file (multipart, field name `file`). The file is
chunked, embedded, and stored in the tenant's collection.

```bash
curl -X POST http://localhost:8000/tenants/acme/documents \
  -F "file=@sample_docs/northwind_handbook.txt"
```

```json
{
  "tenant_id": "acme",
  "document": "northwind_handbook.txt",
  "chunks_stored": 3,
  "total_chunks": 3
}
```

### `POST /tenants/{tenant_id}/chat`

```bash
curl -X POST http://localhost:8000/tenants/acme/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days do customers have to request a refund?"}'
```

```json
{
  "tenant_id": "acme",
  "question": "How many days do customers have to request a refund?",
  "answer": "Based on the uploaded documents, the most relevant passage is:\n\nRefunds and Returns\nCustomers may request a full refund within 30 days of purchase...",
  "mode": "extractive-fallback",
  "sources": [
    {
      "text": "Refunds and Returns\nCustomers may request a full refund within 30 days ...",
      "source": "northwind_handbook.txt",
      "chunk_index": 0,
      "score": 0.61
    }
  ]
}
```

`mode` is `llm` when a chat API key is configured, and
`extractive-fallback` in offline mode. `sources` always shows exactly which
chunks grounded the answer.

### `GET /tenants/{tenant_id}/stats` · `DELETE /tenants/{tenant_id}`

Chunk count for a tenant, and dropping a tenant's whole collection.

## Configuration

| Variable          | Default                      | Description                                   |
|-------------------|------------------------------|-----------------------------------------------|
| `OPENAI_API_KEY`  | *(empty = offline mode)*     | API key for embeddings + chat                 |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1`  | Any OpenAI-compatible base URL                |
| `CHAT_MODEL`      | `gpt-4o-mini`                | Chat model for answers                        |
| `EMBEDDING_MODEL` | `text-embedding-3-small`     | Embedding model (online mode)                 |
| `QDRANT_PATH`     | `./data/qdrant`              | Local Qdrant storage path                     |
| `TOP_K`           | `4`                          | Chunks retrieved per question                 |

> Note: a tenant's collection is created with the vector size of the
> embedder active at first upload. If you switch embedders (e.g. from the
> offline embedder to OpenAI), delete the tenant's collection and re-upload.

## Production Notes

- Swap embedded Qdrant for a Qdrant server/cluster by constructing
  `QdrantClient(url=...)` — the rest of the code is unchanged.
- Add auth (API keys / JWT) in front of the tenant endpoints and validate
  tenant ownership before serving requests.
- The offline hashing embedder is a demo/testing tool; use a real embedding
  model for production retrieval quality.

## License

MIT
