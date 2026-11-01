# AstraRAG AI

A local-first document question-answering application built with retrieval-augmented generation (RAG). It indexes documents into Qdrant and answers questions using retrieved passages, with source and chunk citations.

> **Development status:** Core retrieval, chat, Markdown rendering, and browser-based PDF ingestion are implemented. Production deployment and broader automated integration coverage are not yet complete.

## Current capabilities

- FastAPI chat API with streaming Server-Sent Events.
- Groq chat completions through the OpenAI-compatible API.
- Local dense embeddings using FastEmbed and `BAAI/bge-small-en-v1.5` (384 dimensions).
- Local persistent Qdrant storage.
- Dense semantic search, BM25 keyword search, and hybrid retrieval with Reciprocal Rank Fusion.
- PDF conversion and chunking with Docling through the browser upload API or Marimo notebook.
- Source filename and Qdrant chunk ID included in retrieved context.
- React + Vite chat interface with Markdown-rendered answers.
- Browser-based PDF upload and local ingestion into the configured Qdrant collection (25 MB per file).
- Retrieval abstention when no indexed content is returned.

## Architecture

- **Frontend:** React, TypeScript, Vite
- **API:** Python, FastAPI
- **LLM:** Groq API (requires a Groq API key; free-tier availability and limits are controlled by Groq)
- **Embeddings:** FastEmbed running locally
- **Vector store:** Qdrant local mode
- **Document processing:** Docling and Marimo

No paid hosting, hosted embedding API, or managed vector database is required for local development. The first run downloads the embedding model to the local machine.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- Node.js and npm
- A Groq API key for answer generation

## Setup

### 1. Clone and configure

```powershell
git clone https://github.com/samrriddhi/AstraRAG-AI.git
cd AstraRAG-AI
Copy-Item .env.example .env
```

Open `.env` and set `GROQ_API_KEY`. The default model and local Qdrant settings are already provided in `.env.example`.

### 2. Install Python dependencies

```powershell
uv sync --all-groups
```

### 3. Start the API

```powershell
uv run uvicorn rag.entrypoints.rest:app --reload --host 127.0.0.1 --port 8000
```

API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 4. Start the React interface

In a second terminal:

```powershell
cd rag-ui
npm install
npm run dev
```

Open the local URL printed by Vite. Its development proxy forwards `/api` requests to the FastAPI server on port 8000.

## Index documents

The React chat interface includes an **Upload PDF** control. Select a PDF (up to 25 MB); the API converts it with Docling, chunks it with `HybridChunker`, generates dense and BM25 embeddings locally, and appends the chunks to the configured Qdrant collection. The upload endpoint is `POST /api/documents/upload` and returns the number of chunks indexed.

The Marimo notebook remains available for batch ingestion and alternative chunking strategies:

```powershell
# From the repository root
uv run marimo edit notebooks/create_collection.py
```

**Important:** Stop the FastAPI process before running the Marimo notebook. Both processes use Qdrant local storage, which should not be opened by two processes at the same time.

Both ingestion paths preserve an existing collection and append new points with UUIDs; they do not intentionally delete existing points. Re-uploading the same PDF currently creates additional chunks rather than replacing earlier chunks. Keep a backup of `qdrant_storage/` before experimenting with ingestion or collection settings.

## Tests

Run the unit suite:

```powershell
uv run pytest tests/unit -q
```

Some legacy integration and end-to-end tests still depend on Docker/Testcontainers and older API assumptions. They should not be treated as passing until they are updated and run in a working Docker environment.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq chat authentication | unset |
| `GROQ_MODEL` | Chat model | `openai/gpt-oss-120b` |
| `EMBEDDING_MODEL` | Local dense embedding model | `BAAI/bge-small-en-v1.5` |
| `QDRANT_COLLECTION` | Qdrant collection | `astra-rag` |
| `QDRANT_LOCAL_PATH` | Local Qdrant data directory | `./qdrant_storage` |
| `HOST` | API bind address | `127.0.0.1` |
| `PORT` | API port | `8000` |

## Known limitations / next steps

- Store page numbers and richer document metadata for citations.
- Add API and frontend tests for streaming, validation, and error states.
- Modernize legacy integration and end-to-end tests.
- Reconcile Docker configuration with the current local-first Groq/Qdrant setup.
- Add authentication, persistent chat history, and deployment only when needed.

## License

MIT
