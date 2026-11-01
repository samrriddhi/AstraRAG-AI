"""Application configuration and component factories."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .components import chat, embed, search

__all__ = ["get_openai_chat", "get_openai_embed", "get_qdrant"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    qdrant_collection: str = "astra-rag"
    qdrant_local_path: str = "./qdrant_storage"

    groq_api_key: str = Field(default="")
    groq_model: str = "openai/gpt-oss-120b"

    embedding_model: str = "BAAI/bge-small-en-v1.5"


settings = Settings()


@lru_cache(1)
def get_qdrant() -> search.QdrantSearch:
    """Create a persistent local Qdrant search component."""
    return search.QdrantSearch(
        collection=settings.qdrant_collection,
        url=settings.qdrant_local_path,
        api_key=None,
    )


@lru_cache(1)
def get_openai_chat() -> chat.OpenAIChat:
    """Create the Groq-backed chat component."""
    return chat.OpenAIChat(
        base_url="https://api.groq.com/openai/v1",
        api_key=settings.groq_api_key,
    )


@lru_cache(1)
def get_openai_embed() -> embed.OpenAIEmbed:
    """Create the local embedding component."""
    return embed.OpenAIEmbed(
        model=settings.embedding_model,
        api_key="local",
        base_url=None,
    )