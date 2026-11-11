# RAG Document Q&A API

[![CI](https://github.com/QLINGHL/rag-chatbot-api/actions/workflows/ci.yml/badge.svg)](https://github.com/QLINGHL/rag-chatbot-api/actions/workflows/ci.yml)

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
- 🔀 **Hybrid retrieval (optional)** — BM25 keyword search fused with the
  vector ranking via Reciprocal Rank Fusion (K=60), the same scheme as
  eaap-agent-platform's knowledge base. Enable per request with
  `"hybrid": true` on `/chat`, or globally with `HYBRID_RETRIEVAL=1`.
  Hybrid hits also report `vector_score` / `bm25_score`. On the sample
  handbook benchmark it lifts hit@1 from 0.92 to 1.00 (see
  `eval/evaluate.py --hybrid`).
- 🔌 **Pluggable embeddings** — OpenAI-compatible endpoint, or the built-in
  offline hashing embedder (TF-IDF-style, deterministic).
- 🗄️ **Flexible vector store** — Qdrant in local embedded mode (path-based)
  for zero-infra demos, or a Qdrant server when `QDRANT_URL` is set.
- 🐳 **Docker deployment** — `docker compose up` runs the API plus a Qdrant
  server, wired together out of the box.
- ✅ **Tested** — a pytest suite (`tests/`) covers health, upload/retrieval
  correctness, PDF ingestion, per-document management (list + delete),
  tenant isolation, hybrid retrieval (BM25 + RRF) and input validation —
  all fully offline.
  `test_smoke.py` remains as a quick script-style check.

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
│   ├── hybrid.py      # BM25 keyword scoring + RRF fusion (mirrors EAAP)
│   ├── embeddings.py  # OpenAI embedder + offline hashing embedder
│   └── config.py      # Env-based settings (with .env support)
├── sample_docs/
│   ├── northwind_handbook.txt   # Sample text document
│   └── aurora_x200_guide.pdf    # Sample PDF document
├── eval/                # Offline retrieval benchmark (qa_set.json + evaluate.py)
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

### Evaluation (offline retrieval)

`eval/` holds a small retrieval benchmark: 12 questions grounded in the
sample handbook, each with the answer phrase its chunk must contain.

```bash
python eval/evaluate.py
python eval/evaluate.py --top-k 2 --output eval/results.json
python eval/evaluate.py --hybrid --output eval/results-hybrid.json
```

It uploads the sample document, asks every question through the real
`/chat` endpoint, and reports chunk-level hit@1, hit@k and MRR. Baseline
with the offline hashing embedder (2026-11-01): 12 questions, hit@1 0.92,
hit@4 1.00, MRR 0.958 — only the password question ranked 2nd, the rest
ranked 1st. With `--hybrid` (2026-11-05): hit@1 1.00, hit@4 1.00,
MRR 1.000 — the password question now ranks 1st. The report file is a run
artifact and is not committed.

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

Chunking is configurable per upload with the `chunk_size` and
`chunk_overlap` query parameters (defaults `600` / `120`, also settable
via the `CHUNK_SIZE` / `CHUNK_OVERLAP` environment variables);
`chunk_overlap` must be smaller than `chunk_size`, and the response echoes
the values actually used. Overlap only applies when a single paragraph is
longer than `chunk_size` — short paragraphs are packed whole.

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
chunks grounded the answer. `retrieval` is `vector` by default, or `hybrid`
when hybrid retrieval is on for the request; hybrid hits additionally carry
`vector_score` and `bm25_score` (either may be `null` when the chunk only
ranked on the other side).

### `GET /tenants/{tenant_id}/documents` · `DELETE /tenants/{tenant_id}/documents/{filename}`

List the documents a tenant has indexed (each entry: filename + chunk
count — aggregated from chunk payloads, no separate registry), and remove
a single document's chunks. Deleting a missing document returns 404;
path-traversal style names (`/`, `..`, `\`) are rejected with 400.

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
| `HYBRID_RETRIEVAL`| *(empty = off)*              | `1`/`true` enables hybrid (vector + BM25, RRF) retrieval by default; per-request `"hybrid"` overrides it |

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
