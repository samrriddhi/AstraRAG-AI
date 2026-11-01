
"""REST API entry point for AstraRAG AI.

Exposes the Agent through FastAPI endpoints for model discovery
and document-based question answering.
"""

import json
import os
from collections.abc import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from rag.agent import Agent
from rag.config import settings

from .models import BaseModel, Messages


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
            async for chunk in response:
                yield f"data: {json.dumps(chunk)}\n\n"

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
