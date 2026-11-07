"""Smoke tests for the FastAPI REST entrypoint."""

from fastapi.testclient import TestClient

from rag.config import settings
from rag.entrypoints.rest import app


client = TestClient(app)


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_models_endpoint_returns_configured_model() -> None:
    response = client.get("/api/models")

    assert response.status_code == 200
    assert response.json() == {"data": [settings.groq_model]}
