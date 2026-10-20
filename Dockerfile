# RAG Document Q&A API - production image.
# Talks to a Qdrant server when QDRANT_URL is set (see docker-compose.yml);
# without it, falls back to embedded local storage under ./data/qdrant.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so the layer is cached unless requirements change.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY sample_docs ./sample_docs

# Run as an unprivileged user; ./data holds embedded-mode Qdrant storage.
RUN useradd --create-home appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
