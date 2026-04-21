# RAG Starter

A practical retrieval layer to feed a bring-your-own inference service.

## Stack

- **FastAPI** — HTTP surface
- **ChromaDB** — persisted vector store (file-backed; swap to remote Chroma later)
- **ImmuDB** — tamper-evident audit of every ingest and every query
- **S3-compatible** ingest — OCI Object Storage, MinIO, AWS S3
- **Pluggable inference** via a `doinference` HTTP contract

## Install

```bash
pip install fastapi 'uvicorn[standard]' pydantic \
            chromadb immudb-py boto3 \
            pypdf python-docx beautifulsoup4 pandas openpyxl \
            httpx python-multipart
```

## Run

```bash
# Start ImmuDB locally (optional but recommended)
docker run -d --name immudb -p 3322:3322 -p 9497:9497 codenotary/immudb:latest

# Start the API
uvicorn app:app --host 0.0.0.0 --port 8080
```

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `CHROMA_PATH` | `./chroma_data` | Where Chroma persists to disk |
| `CHROMA_COLLECTION` | `documents` | Collection name |
| `EMBED_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible embeddings endpoint |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model name |
| `INFERENCE_BASE_URL` | `http://localhost:8000` | Customer's inference service |
| `INFERENCE_API_KEY` | *(unset)* | Bearer token for inference, if required |
| `IMMUDB_HOST` / `IMMUDB_PORT` | `localhost` / `3322` | ImmuDB endpoint |
| `IMMUDB_USER` / `IMMUDB_PASSWORD` | `immudb` / `immudb` | ImmuDB credentials (change these) |
| `S3_ENDPOINT_URL` | *(AWS default)* | e.g. `https://<tenancy>.compat.objectstorage.<region>.oraclecloud.com` |
| `S3_ACCESS_KEY` / `S3_SECRET_KEY` | *(unset)* | S3-compatible credentials |

## The `doinference` contract

The customer's inference service must expose `POST /doinference` accepting:

```json
{
  "query": "string — the user's question",
  "contexts": [
    {
      "id": "doc-id::chunk::0001",
      "text": "retrieved chunk text",
      "metadata": {"doc_id": "...", "title": "...", "source": "..."},
      "score": 0.82
    }
  ]
}
```

And returning:

```json
{ "response": "string — the generated answer" }
```

If their contract differs, replace the body of `do_inference()` in `app.py` —
it's deliberately the only place that knows about their inference layer.

## Endpoints

- `GET  /health` — liveness + component status
- `POST /ingest/upload` — multipart upload of a single file
- `POST /ingest/s3` — pull and index objects under an S3 prefix
- `POST /query` — retrieve top-k, optionally run inference, audit everything

## What's intentionally missing

- **Auth** — put this behind your own gateway (OAuth2/JWT is ~10 lines with `fastapi.security`)
- **Rate limiting** — same answer
- **Reranking** — add a cross-encoder stage after the Chroma retrieval if recall@5 isn't enough
- **Token-precise chunking** — the paragraph-greedy chunker is a shippable default, not the final answer
