from rag import config
from rag.components import chat, search, embed


settings = config.Settings(_env_file=None)


def test_settings():
    assert settings.qdrant_collection == "astra-rag"
    assert settings.qdrant_local_path == "./qdrant_storage"

    assert settings.groq_model == "openai/gpt-oss-120b"
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"


def test_get_qdrant():
    qdrant = config.get_qdrant()

    assert isinstance(qdrant, search.QdrantSearch)

    second_qdrant = config.get_qdrant()

    assert qdrant is second_qdrant


def test_get_openai_chat(monkeypatch):
    # A placeholder key is sufficient: this test checks client construction,
    # not authentication against the Groq API.
    monkeypatch.setattr(config.settings, "groq_api_key", "test-api-key")
    config.get_openai_chat.cache_clear()
    try:
        openai = config.get_openai_chat()

        assert isinstance(openai, chat.OpenAIChat)

        second_openai = config.get_openai_chat()

        assert openai is second_openai
    finally:
        config.get_openai_chat.cache_clear()


def test_get_openai_embed():
    openai = config.get_openai_embed()

    assert isinstance(openai, embed.OpenAIEmbed)

    second_openai = config.get_openai_embed()

    assert openai is second_openai