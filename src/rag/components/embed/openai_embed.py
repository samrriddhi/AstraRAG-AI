"""Local embedding component using FastEmbed."""

from typing import Any

from fastembed import TextEmbedding


class OpenAIEmbed:
    """Embedding component backed by a local FastEmbed model."""

    def __init__(
        self,
        model: str,
        api_key: str = "",
        base_url: str | None = None,
        _openai_client_class: Any = None,
    ) -> None:
        self.model = model
        self.embedder = TextEmbedding(model_name=model)

    async def generate_embedding(
        self,
        text: str,
        **kwargs: Any,
    ) -> list[float]:
        """Generate a dense embedding locally."""
        embedding = next(iter(self.embedder.embed([text])))
        return embedding.tolist()