
"""REST API entry point for AstraRAG AI.

Exposes the Agent through FastAPI endpoints for model discovery
and document-based question answering.
"""

import asyncio
import hashlib
import json
import logging
import os
import tempfile
from functools import lru_cache
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from rag.agent import Agent
from rag.config import get_openai_embed, get_qdrant, settings

from .models import BaseModel, Messages


logger = logging.getLogger(__name__)


app = FastAPI(
    title="AstraRAG AI",
    description="Document-based question answering using RAG.",
    version="1.0.0",
)


class Data(BaseModel):
    """Request body for the chat endpoint."""

    model: str
    messages: Messages


@app.get("/api/models")
async def get_models() -> JSONResponse:
    """Return the configured Groq model."""
    return JSONResponse(
        {
            "data": [settings.groq_model]
        }
    )


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_UPLOAD_SUFFIXES = {".pdf"}
EMBEDDING_TOKEN_LIMIT = 480
CHUNK_TOKEN_LIMIT = 350


@lru_cache(maxsize=1)
def _get_embedding_tokenizer() -> Any:
    """Load the tokenizer used by the configured embedding model."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(settings.embedding_model)


def _truncate_embedding_text(
    text: str,
    tokenizer: Any,
    max_tokens: int = EMBEDDING_TOKEN_LIMIT,
) -> str:
    """Ensure final contextualized text fits safely within the embedding window."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    special_token_count = tokenizer.num_special_tokens_to_add(pair=False)
    content_token_limit = max_tokens - special_token_count

    if len(token_ids) + special_token_count <= max_tokens:
        return text
    if content_token_limit <= 0:
        raise ValueError("max_tokens must leave room for special tokens.")

    return tokenizer.decode(
        token_ids[:content_token_limit],
        skip_special_tokens=True,
    ).strip()


def _extract_chunk_pages(chunk: Any) -> list[int]:
    """Extract unique one-based PDF page numbers from Docling provenance."""
    pages: set[int] = set()
    metadata = getattr(chunk, "meta", None)
    for item in getattr(metadata, "doc_items", None) or []:
        for provenance in getattr(item, "prov", None) or []:
            page_number = getattr(provenance, "page_no", None)
            if isinstance(page_number, int) and page_number > 0:
                pages.add(page_number)
    return sorted(pages)


def _prepare_pdf(path: Path, filename: str) -> tuple[list[dict[str, Any]], list[list[float]]]:
    """Convert, chunk, and embed a PDF using local models."""
    from docling.document_converter import DocumentConverter
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

    converted = DocumentConverter().convert(str(path))
    # Context headings are added after chunking, so enforce the limit again
    # on the final text that will actually be sent to the embedding model.
    chunker = HybridChunker(max_tokens=CHUNK_TOKEN_LIMIT)
    tokenizer = _get_embedding_tokenizer()
    chunks: list[dict[str, Any]] = []
    for chunk in chunker.chunk(dl_doc=converted.document):
        text = chunker.contextualize(chunk=chunk).strip()
        if text:
            text = _truncate_embedding_text(text, tokenizer)
            if text:
                chunks.append({"content": text, "source": filename, "pages": _extract_chunk_pages(chunk)})

    if not chunks:
        raise ValueError("No readable text chunks were found in this PDF.")

    embedder = get_openai_embed().embedder
    vectors = [
        next(iter(embedder.embed([chunk["content"]]))).tolist()
        for chunk in chunks
    ]
    return chunks, vectors


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> JSONResponse:
    """Convert and index one PDF in the configured local Qdrant collection."""
    filename = Path(file.filename or "").name
    if Path(filename).suffix.lower() not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(status_code=415, detail="Only PDF files are supported.")

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds the 25 MB upload limit.")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary_file:
            temporary_file.write(contents)
            temporary_path = Path(temporary_file.name)

        document_hash = hashlib.sha256(contents).hexdigest()
        chunks, dense_vectors = await asyncio.to_thread(
            _prepare_pdf, temporary_path, filename
        )
        for chunk in chunks:
            chunk["document_hash"] = document_hash

        indexed_count = await get_qdrant().upsert_chunks(chunks, dense_vectors)
        return JSONResponse(
            {
                "filename": filename,
                "chunks_indexed": indexed_count,
                "collection": settings.qdrant_collection,
                "message": "PDF indexed successfully.",
            },
            status_code=201,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not index PDF: {exc}",
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        await file.close()


@app.post("/api/chat")
async def send_messages(data: Data, stream: bool = False):
    """Generate an answer using the AstraRAG agent.

    The system prompt is automatically added if it is not
    already included in the incoming conversation.

    Args:
        data: Request containing the model and conversation messages.
        stream: Whether to stream the response using SSE.

    Returns:
        A JSON response or a server-sent event stream.
    """

    if data.model != settings.groq_model:
        raise HTTPException(
            status_code=400,
            detail="The requested model is not configured for this application.",
        )

    if not settings.groq_api_key:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured. Add it to your local .env file.",
        )

    if not data.messages.root:
        raise HTTPException(
            status_code=422,
            detail="At least one conversation message is required.",
        )

    agent = Agent(model=data.model)

    # Convert the incoming Pydantic messages to a Python list.
    messages = data.messages.model_dump(exclude_none=True)

    # Ensure the AstraRAG system prompt is included.
    if not messages or messages[0].get("role") != "system":
        messages.insert(
            0,
            {
                "role": "system",
                "content": agent.system,
            },
        )

    # Generate the response through the RAG agent.
    response = agent.generate(messages)

    if stream:

        async def stream_response() -> AsyncGenerator[str, None]:
            try:
                async for chunk in response:
                    yield f"data: {json.dumps(chunk)}\\n\\n"
            except Exception:
                logger.exception("Chat response stream failed.")
                yield (
                    "event: error\\n"
                    'data: {"error": "The response stream encountered an error. Please try again."}\\n\\n'
                )

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
        )

    buffer = ""

    async for chunk in response:
        buffer += chunk

    return JSONResponse(
        {
            "response": buffer
        }
    )


def api() -> None:
    """Run the FastAPI application using Uvicorn."""

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))

    uvicorn.run(
        app,
        host=host,
        port=port,
    )
