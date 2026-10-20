# RAG Document Q&A API

A multi-tenant **Retrieval-Augmented Generation (RAG)** service built with
**FastAPI** and **Qdrant**. Upload documents (`.txt`, `.md`, or `.pdf`) for a
tenant, then ask questions — the API retrieves the most relevant chunks from
that tenant's own vector collection, builds a grounded prompt, and answers
with an LLM.

Runs in two modes:

- **Online mode** — set `OPENAI_API_KEY` to use any OpenAI-compatible API
  (OpenAI, Azure, or a local gateway) for embeddings and chat answers.
- **Offline demo mode** — no keys at all. A pure Python/NumPy hashing
  embedder and an extractive answer fallback keep the full pipeline
  (upload → chunk → embed → retrieve → answer) working end-to-end.

## Features

- 📄 **Document ingestion** — upload `.txt` / `.md` / `.pdf` files (PDF text
  extracted with `pypdf`); automatic paragraph-aware chunking with overlap.
- 🏢 **Multi-tenant by design** — every tenant gets a dedicated Qdrant
  collection (`tenant_<id>`), so tenant data is physically isolated and can
  be dropped with one call.
- 🔎 **Semantic retrieval** — top-k cosine search per tenant, with source
  attribution (file name, chunk index, score) returned on every answer.
- 🔌 **Pluggable embeddings** — OpenAI-compatible endpoint, or the built-in
  offline hashing embedder (TF-IDF-style, deterministic).
- 🗄️ **Flexible vector store** — Qdrant in local embedded mode (path-based)
  for zero-infra demos, or a Qdrant server when `QDRANT_URL` is set.
- 🐳 **Docker deployment** — `docker compose up` runs the API plus a Qdrant
  server, wired together out of the box.
- ✅ **Tested** — a pytest suite (`tests/`) covers health, upload/retrieval
  correctness, PDF ingestion, tenant isolation, and input validation — all
  fully offline. `test_smoke.py` remains as a quick script-style check.

## Tech Stack

| Layer        | Technology                                  |
|--------------|---------------------------------------------|
| API          | FastAPI, Pydantic, Uvicorn                  |
| Vector store | Qdrant (`qdrant-client`, embedded or server mode) |
| PDF parsing  | `pypdf`                                     |
| Embeddings   | OpenAI-compatible API, or local NumPy hashing embedder |
| LLM          | Any OpenAI-compatible chat completions API  |
| Deployment   | Docker + Docker Compose                     |
| Language     | Python 3.10+                                |

## Project Structure

```
rag-chatbot-api/
├── app/
│   ├── main.py        # FastAPI app: endpoints, prompt building, answer generation
│   ├── documents.py   # Text extraction for .txt / .md / .pdf uploads
│   ├── rag.py         # Chunking + multi-tenant Qdrant store
│   ├── embeddings.py  # OpenAI embedder + offline hashing embedder
│   └── config.py      # Env-based settings (with .env support)
├── sample_docs/
│   ├── northwind_handbook.txt   # Sample text document
│   └── aurora_x200_guide.pdf    # Sample PDF document
├── tests/             # Pytest suite (offline, TestClient-based)
│   ├── conftest.py
│   ├── test_api.py
│   └── test_documents.py
├── demo.sh            # curl walkthrough against a running server
├── test_smoke.py      # Offline end-to-end smoke test (script form)
├── Dockerfile
├── docker-compose.yml # API + Qdrant server, one command
├── requirements.txt
└── .env.example
```

## Quickstart with Docker

The fastest way to run the full stack — API plus a real Qdrant server:

```bash
docker compose up --build
```

The API is then at <http://localhost:8000> (Swagger UI at `/docs`), and
Qdrant's dashboard/API at <http://localhost:6333>. Compose sets
`QDRANT_URL=http://qdrant:6333` for the API, so data persists in Qdrant's
own volume across restarts. Set `OPENAI_API_KEY` in your shell (or a `.env`
file) before `up` to enable online mode; without it the stack runs fully
offline.

## Setup (without Docker)

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

Without `QDRANT_URL`, Qdrant runs embedded and stores data under
`./data/qdrant`. Interactive docs (Swagger UI) are available at
<http://localhost:8000/docs>.

## Testing

```bash
python -m pytest tests/ -q
```

The suite runs fully offline against a throwaway store and covers: health,
`.txt` upload → retrieval correctness, tenant isolation, `.pdf` upload →
retrieval, and rejection of invalid file types / corrupt PDFs / bad tenant
ids.

A script-style smoke test is also included:

```bash
python test_smoke.py
```

It spins the app up in-process, uploads the sample handbook, and verifies
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

Upload a `.txt`, `.md`, or `.pdf` file (multipart, field name `file`). For
PDFs, text is extracted page by page first; from there the pipeline is the
same — the file is chunked, embedded, and stored in the tenant's collection.
Unsupported types, corrupt PDFs, and PDFs with no extractable text are
rejected with HTTP 400.

```bash
curl -X POST http://localhost:8000/tenants/acme/documents \
  -F "file=@sample_docs/northwind_handbook.txt"

curl -X POST http://localhost:8000/tenants/acme/documents \
  -F "file=@sample_docs/aurora_x200_guide.pdf"
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
| `QDRANT_URL`      | *(empty = embedded mode)*    | Qdrant server URL, e.g. `http://qdrant:6333`  |
| `QDRANT_PATH`     | `./data/qdrant`              | Local Qdrant storage path (embedded mode)     |
| `TOP_K`           | `4`                          | Chunks retrieved per question                 |

> Note: a tenant's collection is created with the vector size of the
> embedder active at first upload. If you switch embedders (e.g. from the
> offline embedder to OpenAI), delete the tenant's collection and re-upload.

## Production Notes

- For anything beyond a demo, run Qdrant as a server: set `QDRANT_URL`
  (the included `docker-compose.yml` does this) or point it at a Qdrant
  Cloud cluster — no code changes needed.
- Add auth (API keys / JWT) in front of the tenant endpoints and validate
  tenant ownership before serving requests.
- The offline hashing embedder is a demo/testing tool; use a real embedding
  model for production retrieval quality.
- Scanned/image-only PDFs have no embedded text and are rejected; add an
  OCR step (e.g. Tesseract) in front of ingestion if you need them.

## License

MIT
