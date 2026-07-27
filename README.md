# 🚀 AstraRAG AI (Ongoing)

> An AI-powered Retrieval-Augmented Generation (RAG) application for document-based question answering.

> 🚧 **Project Status:** Currently under active development.

---

## Overview

AstraRAG AI is a Retrieval-Augmented Generation (RAG) application that allows users to upload documents and interact with them using natural language queries.

The project is being built with a modern backend architecture using **FastAPI**, **React**, **Qdrant**, **LiteLLM**, and **Docker**. It is designed to demonstrate scalable AI application development using vector search and Large Language Models.

---

## Current Features

- REST API built with FastAPI
- React-based frontend
- LiteLLM integration for LLM orchestration
- Qdrant vector database integration
- Docker Compose infrastructure
- Environment-based configuration
- Modular project architecture

---

## Tech Stack

### Backend
- Python
- FastAPI
- LiteLLM

### Frontend
- React

### Vector Database
- Qdrant

### DevOps
- Docker
- Docker Compose

### Tools
- UV
- Pydantic Settings

---

## Project Structure

```
AstraRAG-AI/
│
├── backend/
├── frontend/
├── infra/
│   ├── qdrant/
│   ├── litellm/
│   └── docker-compose.yml
├── notebooks/
├── README.md
└── .env.example
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- UV
- Docker Desktop
- Node.js

### Clone Repository

```bash
git clone https://github.com/samrriddhi/AstraRAG-AI.git
cd AstraRAG-AI
```

### Backend

```bash
uv sync
uv run api
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Infrastructure

```bash
docker compose --profile infra up
```

---

## Current Development

The following features are currently being integrated:

- Document upload pipeline
- Embedding generation
- Semantic search with Qdrant
- End-to-end RAG workflow
- Production deployment

---

## Planned Features

- Authentication
- Chat history
- Multiple document collections
- Streaming AI responses
- AWS Deployment
- CI/CD Pipeline

---

## Repository

GitHub:
https://github.com/samrriddhi/AstraRAG-AI

---

## License

MIT