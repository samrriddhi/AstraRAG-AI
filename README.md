# AstraRAG AI

A local-first document question-answering application built with retrieval-augmented generation (RAG). It indexes documents into Qdrant and answers questions using retrieved passages, with source and chunk citations.

> **Development status:** Phase 1 local-first RAG functionality is implemented, including PDF ingestion, hybrid retrieval, page-aware source citations, duplicate-upload prevention, and streaming error handling. Run the unit tests and frontend production build before considering the current branch release-ready.

## Current capabilities

- FastAPI chat API with streaming Server-Sent Events.
- Groq chat completions through the OpenAI-compatible API.
- Local dense embeddings using FastEmbed and `BAAI/bge-small-en-v1.5` (384 dimensions).
- Local persistent Qdrant storage.
- Dense semantic search, BM25 keyword search, and hybrid retrieval with Reciprocal Rank Fusion.
- PDF conversion and chunking with Docling through the browser upload API or Marimo notebook.
- Source filename, PDF page numbers (when available), and Qdrant chunk ID included in retrieved context.
- Idempotent PDF uploads using a SHA-256 document hash and deterministic chunk IDs.
- Re-uploading a changed PDF with the same filename indexes the new version before deleting older chunks for that filename.
- Streaming chat errors are logged by the API and surfaced to the chat UI.
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

The React chat interface includes an **Upload PDF** control. Select a PDF (up to 25 MB); the API converts it with Docling, chunks it with `HybridChunker`, generates dense and BM25 embeddings locally, and indexes the chunks in the configured Qdrant collection. The upload endpoint is `POST /api/documents/upload` and returns the number of chunks indexed.

The Marimo notebook remains available for batch ingestion and alternative chunking strategies:

```powershell
# From the repository root
uv run marimo edit notebooks/create_collection.py
```

**Important:** Stop the FastAPI process before running the Marimo notebook. Both processes use Qdrant local storage, which should not be opened by two processes at the same time.

The browser upload path is idempotent per filename and file content: re-uploading the same PDF reuses deterministic chunk IDs, while uploading changed content under the same filename indexes the new version first and then removes older chunks for that filename. Other filenames are not targeted by that cleanup. The Marimo notebook is a separate batch-ingestion path and does not currently share this replacement workflow. Keep a backup of `qdrant_storage/` before experimenting with ingestion or collection settings.

## Tests

Run the unit suite:

```powershell
uv run pytest tests/unit -q
```

Some legacy integration and end-to-end tests still depend on Docker/Testcontainers and older API assumptions. They should not be treated as passing until they are updated and run in a working Docker environment.

## Retrieval evaluation (Phase 2)

A small offline evaluation module is available at `src/rag/evaluation/metrics.py`. It provides:

- **Precision@K:** proportion of the top-K retrieved chunks that are relevant.
- **Recall@K:** proportion of labelled relevant chunks found in the top-K results.
- **Reciprocal Rank (RR):** inverse rank of the first relevant chunk.
- **Mean Reciprocal Rank (MRR):** average RR across evaluation queries.

These metrics use human-labelled relevant chunk IDs; they do not call an LLM or require an external service. Add a representative set of questions and their relevant chunk IDs before interpreting scores. The current unit tests validate metric calculations, not AstraRAG's retrieval quality on a labelled corpus.

Run the metric tests with:

```powershell
uv run pytest tests/unit/evaluation -q
```

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

- Add dedicated API and frontend automated tests for upload, streaming, and citation rendering.
- Modernize legacy integration and end-to-end tests.
- Reconcile Docker configuration with the current local-first Groq/Qdrant setup.
- Add authentication, persistent chat history, and deployment only when needed.

## License

MIT
