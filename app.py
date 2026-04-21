"""
RAG Starter — FastAPI + Chroma + ImmuDB + S3 ingest
===================================================

A practical, production-shaped starting point for a customer that has already
built their own inference layer and now needs a retrieval pipeline to feed it.

What this gives you
-------------------
1. FastAPI surface with three endpoints:
     POST /ingest/upload     — push a single file (pdf/docx/html/txt/csv/xlsx)
     POST /ingest/s3         — pull documents from an S3-compatible bucket
     POST /query             — retrieve top-k chunks and (optionally) run inference
2. ChromaDB as the persisted vector store (local, file-backed — swap the client
   for a remote Chroma server later without touching the rest of the code).
3. ImmuDB audit trail: every ingest and every inference call is written as a
   tamper-evident record. If the customer is in a regulated / sovereign context
   this is the bit the auditors actually care about.
4. A pluggable `do_inference` seam. The customer drops their own inference
   client in one place — HTTP call, gRPC, local model, whatever they've built.
5. Text extraction lifted and modernised from the Anvil pipeline (pypdf,
   python-docx, BeautifulSoup, pandas), but decoupled from Anvil's data tables.

Design notes
------------
• Embeddings are computed via a swappable function (`embed_texts`). Default
  wiring points at an OpenAI-compatible endpoint (which covers OCI Generative
  AI, vLLM, Ollama, and most self-hosted stacks). Replace with a direct call
  to your embedding model if preferred.
• Chunking is deliberately simple — paragraph splits with a token-ish budget.
  Good enough to demonstrate retrieval quality; tune later.
• The audit schema is minimal on purpose: actor, action, subject, payload
  hash, timestamp. Anything richer belongs in a structured log sink alongside.
• No auth middleware here — assume the customer puts this behind their own
  gateway. Adding OAuth2/JWT is a ten-minute job with fastapi.security.

Dependencies (pip)
------------------
    fastapi uvicorn[standard] pydantic
    chromadb immudb-py boto3
    pypdf python-docx beautifulsoup4 pandas openpyxl
    httpx python-multipart
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

import chromadb
import httpx
import pandas as pd
import pypdf
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

# Optional imports — the app still runs if these aren't configured, but the
# corresponding features will return 503.
try:
    import boto3  # type: ignore
    from botocore.client import Config as BotoConfig  # type: ignore
    _HAS_BOTO = True
except ImportError:
    _HAS_BOTO = False

try:
    from immudb import ImmudbClient  # type: ignore
    _HAS_IMMUDB = True
except ImportError:
    _HAS_IMMUDB = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Settings:
    """Env-driven config. Keep it boring."""

    # Chroma
    CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")
    CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "documents")

    # Embeddings — any OpenAI-compatible endpoint (OCI Gen AI, vLLM, Ollama, ...)
    EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:11434/v1")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
    EMBED_API_KEY = os.getenv("EMBED_API_KEY", "not-needed")

    # Inference — exposed as do_inference(); customer swaps this out
    INFERENCE_BASE_URL = os.getenv("INFERENCE_BASE_URL", "http://localhost:8000")
    INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY", "")

    # ImmuDB
    IMMUDB_HOST = os.getenv("IMMUDB_HOST", "localhost")
    IMMUDB_PORT = int(os.getenv("IMMUDB_PORT", "3322"))
    IMMUDB_USER = os.getenv("IMMUDB_USER", "immudb")
    IMMUDB_PASSWORD = os.getenv("IMMUDB_PASSWORD", "immudb")
    IMMUDB_DB = os.getenv("IMMUDB_DB", "defaultdb")

    # S3 (compatible — OCI Object Storage, MinIO, AWS, etc.)
    S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")  # None => AWS default
    S3_REGION = os.getenv("S3_REGION", "eu-west-1")
    S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
    S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")

    # Retrieval
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
    CHUNK_CHAR_BUDGET = int(os.getenv("CHUNK_CHAR_BUDGET", "1200"))


settings = Settings()
logger = logging.getLogger("rag_starter")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# ---------------------------------------------------------------------------
# Chroma — the one place vector storage lives
# ---------------------------------------------------------------------------

_chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
_collection = _chroma_client.get_or_create_collection(
    name=settings.CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"},
)


# ---------------------------------------------------------------------------
# ImmuDB audit — tamper-evident log of every material action
# ---------------------------------------------------------------------------

class Audit:
    """
    Thin wrapper around ImmuDB. Every call writes a single key/value pair
    where the key encodes the action and timestamp, and the value is the
    canonical JSON of the event. Retrieval and verification are out of scope
    for this starter — the customer will already have their own audit reader.
    """

    def __init__(self) -> None:
        self._client = None
        if not _HAS_IMMUDB:
            logger.warning("immudb-py not installed — audit writes will be no-ops")
            return
        try:
            self._client = ImmudbClient(f"{settings.IMMUDB_HOST}:{settings.IMMUDB_PORT}")
            self._client.login(settings.IMMUDB_USER, settings.IMMUDB_PASSWORD)
            self._client.useDatabase(settings.IMMUDB_DB)
            logger.info("ImmuDB audit ready on %s:%s", settings.IMMUDB_HOST, settings.IMMUDB_PORT)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ImmuDB unavailable (%s) — audit writes will be no-ops", exc)
            self._client = None

    def write(self, actor: str, action: str, subject: str, payload: dict[str, Any]) -> None:
        """Write one audit record. Never raises — audit must not break the hot path."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "action": action,
            "subject": subject,
            "payload_sha256": hashlib.sha256(
                json.dumps(payload, sort_keys=True, default=str).encode()
            ).hexdigest(),
            "payload": payload,
        }
        if self._client is None:
            logger.debug("AUDIT (no-op): %s", record)
            return
        try:
            key = f"{action}:{record['ts']}:{uuid.uuid4().hex[:8]}".encode()
            self._client.set(key, json.dumps(record, default=str).encode())
        except Exception as exc:  # noqa: BLE001
            logger.error("Audit write failed (record preserved in logs): %s | %s", exc, record)


audit = Audit()


# ---------------------------------------------------------------------------
# Text extraction — ported and tightened from your Anvil pipeline
# ---------------------------------------------------------------------------

_NUMBERED_PARA_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,5}|[A-Z]\))\s+")


def _normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n\s*Page\s+\d+\s+of\s+\d+\s*\n", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)  # de-hyphenate soft wraps
    return text


def extract_text(filename: str, data: bytes) -> str:
    """Dispatch on extension. Returns plaintext, normalised."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = [(p.extract_text() or "").strip() for p in reader.pages]
        return _normalise("\n\n".join(pages))

    if ext in ("docx", "doc"):
        d = DocxDocument(io.BytesIO(data))
        return _normalise("\n\n".join(p.text for p in d.paragraphs if p.text.strip()))

    if ext in ("html", "htm"):
        soup = BeautifulSoup(data.decode("utf-8", errors="replace"), "html.parser")
        for tag in soup(["style", "script", "xml"]):
            tag.decompose()
        return _normalise(soup.get_text(" ", strip=True))

    if ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(data), engine="openpyxl" if ext == "xlsx" else None)
        return _normalise(df.to_csv(index=False))

    if ext == "csv":
        return _normalise(data.decode("utf-8", errors="replace"))

    # txt and unknown — decode as text
    return _normalise(data.decode("utf-8", errors="replace"))


def chunk_text(text: str, budget: int = settings.CHUNK_CHAR_BUDGET) -> list[str]:
    """
    Paragraph-greedy chunker. Accumulates paragraphs until the budget is hit,
    then starts a new chunk. Not token-precise — good enough to ship, easy to
    replace with tiktoken-based or semantic splitting later.
    """
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for p in paras:
        if size + len(p) > budget and buf:
            chunks.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(p)
        size += len(p) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


# ---------------------------------------------------------------------------
# Embeddings + inference — both pluggable, both HTTP by default
# ---------------------------------------------------------------------------

async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    OpenAI-compatible embeddings call. Works against Ollama, vLLM, OCI Gen AI
    (via its compatibility endpoint), and anything else speaking the same API.
    Swap out wholesale if using a native SDK.
    """
    if not texts:
        return []
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{settings.EMBED_BASE_URL}/embeddings",
            headers={"Authorization": f"Bearer {settings.EMBED_API_KEY}"},
            json={"model": settings.EMBED_MODEL, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        return [row["embedding"] for row in data["data"]]


async def do_inference(query: str, contexts: list[dict[str, Any]]) -> str:
    """
    THE CUSTOMER'S SEAM.

    This is the function the customer replaces with a call to their own
    inference layer. It takes the user's query and the retrieved context
    documents and returns a string response.

    Default implementation: POST to {INFERENCE_BASE_URL}/doinference with a
    simple JSON body. Matches the 'doinference' contract we discussed.
    """
    payload = {
        "query": query,
        "contexts": contexts,  # list of {id, text, metadata, score}
    }
    headers = {}
    if settings.INFERENCE_API_KEY:
        headers["Authorization"] = f"Bearer {settings.INFERENCE_API_KEY}"

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{settings.INFERENCE_BASE_URL}/doinference",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        body = resp.json()
        # Expecting {"response": "..."} — adjust to customer's contract.
        return body.get("response", json.dumps(body))


# ---------------------------------------------------------------------------
# Indexing — the write path
# ---------------------------------------------------------------------------

async def index_document(
    *,
    doc_id: str,
    title: str,
    text: str,
    source: str,
    actor: str,
    extra_metadata: dict[str, Any] | None = None,
) -> int:
    """
    Chunk, embed, and upsert into Chroma. Returns the chunk count.
    Writes one audit record per successful indexing.
    """
    chunks = chunk_text(text)
    if not chunks:
        return 0

    embeddings = await embed_texts(chunks)

    ids = [f"{doc_id}::chunk::{i:04d}" for i in range(len(chunks))]
    metadatas = [
        {
            "doc_id": doc_id,
            "title": title,
            "source": source,
            "chunk_index": i,
            **(extra_metadata or {}),
        }
        for i in range(len(chunks))
    ]

    _collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    audit.write(
        actor=actor,
        action="ingest",
        subject=doc_id,
        payload={"title": title, "source": source, "chunks": len(chunks)},
    )
    return len(chunks)


# ---------------------------------------------------------------------------
# S3 ingest
# ---------------------------------------------------------------------------

def _s3_client():
    if not _HAS_BOTO:
        raise HTTPException(503, "boto3 not installed — S3 ingest unavailable")
    if not (settings.S3_ACCESS_KEY and settings.S3_SECRET_KEY):
        raise HTTPException(503, "S3 credentials not configured")
    return boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT_URL,
        region_name=settings.S3_REGION,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
        config=BotoConfig(signature_version="s3v4"),
    )


def _iter_s3_objects(bucket: str, prefix: str) -> Iterable[dict[str, Any]]:
    client = _s3_client()
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            yield obj


# ---------------------------------------------------------------------------
# FastAPI surface
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Starter",
    version="0.1.0",
    description="FastAPI + Chroma + ImmuDB + S3 retrieval layer with a pluggable inference seam.",
)


class S3IngestRequest(BaseModel):
    bucket: str
    prefix: str = ""
    actor: str = Field(default="system", description="Audit actor")
    max_objects: int = Field(default=100, ge=1, le=10000)


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=settings.DEFAULT_TOP_K, ge=1, le=50)
    run_inference: bool = True
    actor: str = "anonymous"
    metadata_filter: dict[str, Any] | None = None


class ChunkHit(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any]
    score: float


class QueryResponse(BaseModel):
    query: str
    hits: list[ChunkHit]
    response: str | None = None


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "chroma_count": _collection.count(),
        "immudb": audit._client is not None,
        "s3_configured": bool(settings.S3_ACCESS_KEY and settings.S3_SECRET_KEY),
    }


@app.post("/ingest/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    title: str = Form(...),
    actor: str = Form("anonymous"),
) -> dict[str, Any]:
    """Ingest a single uploaded file."""
    data = await file.read()
    try:
        text = extract_text(file.filename or "unknown", data)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Extraction failed: {exc}") from exc

    doc_id = f"upload-{uuid.uuid4().hex}"
    chunk_count = await index_document(
        doc_id=doc_id,
        title=title,
        text=text,
        source=f"upload:{file.filename}",
        actor=actor,
    )
    return {"doc_id": doc_id, "chunks": chunk_count}


@app.post("/ingest/s3")
async def ingest_s3(req: S3IngestRequest) -> dict[str, Any]:
    """
    Pull every object under s3://{bucket}/{prefix}, extract, chunk, and index.
    Bounded by max_objects to keep the first run finite.
    """
    client = _s3_client()
    count = 0
    total_chunks = 0
    errors: list[dict[str, str]] = []

    for obj in _iter_s3_objects(req.bucket, req.prefix):
        if count >= req.max_objects:
            break
        key = obj["Key"]
        try:
            body = client.get_object(Bucket=req.bucket, Key=key)["Body"].read()
            text = extract_text(key, body)
            doc_id = f"s3-{hashlib.sha1(f'{req.bucket}/{key}'.encode()).hexdigest()[:16]}"
            chunks = await index_document(
                doc_id=doc_id,
                title=key.rsplit("/", 1)[-1],
                text=text,
                source=f"s3://{req.bucket}/{key}",
                actor=req.actor,
                extra_metadata={"s3_bucket": req.bucket, "s3_key": key},
            )
            total_chunks += chunks
            count += 1
        except Exception as exc:  # noqa: BLE001
            errors.append({"key": key, "error": str(exc)})
            logger.warning("S3 ingest failed for %s: %s", key, exc)

    return {
        "objects_ingested": count,
        "chunks_written": total_chunks,
        "errors": errors,
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """
    Retrieve top-k chunks; optionally pass them to the customer's inference
    layer via do_inference(). Every call is audited.
    """
    query_embedding = (await embed_texts([req.query]))[0]

    chroma_result = _collection.query(
        query_embeddings=[query_embedding],
        n_results=req.top_k,
        where=req.metadata_filter or None,
    )

    # Chroma returns parallel lists inside a single-element outer list.
    ids = chroma_result["ids"][0]
    docs = chroma_result["documents"][0]
    metas = chroma_result["metadatas"][0]
    dists = chroma_result["distances"][0]

    hits = [
        ChunkHit(
            id=ids[i],
            text=docs[i],
            metadata=metas[i] or {},
            # Chroma returns cosine *distance*; convert to similarity for sanity.
            score=float(1.0 - dists[i]),
        )
        for i in range(len(ids))
    ]

    response_text: str | None = None
    if req.run_inference and hits:
        contexts = [h.model_dump() for h in hits]
        try:
            response_text = await do_inference(req.query, contexts)
        except Exception as exc:  # noqa: BLE001
            logger.error("Inference call failed: %s", exc)
            response_text = f"[inference unavailable: {exc}]"

    audit.write(
        actor=req.actor,
        action="query",
        subject=req.query[:200],
        payload={
            "top_k": req.top_k,
            "hit_ids": [h.id for h in hits],
            "inference_ran": req.run_inference,
            "response_sha256": (
                hashlib.sha256(response_text.encode()).hexdigest()
                if response_text
                else None
            ),
        },
    )

    return QueryResponse(query=req.query, hits=hits, response=response_text)


# ---------------------------------------------------------------------------
# Local dev entrypoint:   python app.py
# Production:             uvicorn app:app --host 0.0.0.0 --port 8080
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
