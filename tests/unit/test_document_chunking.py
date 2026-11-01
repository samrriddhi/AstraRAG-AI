"""Tests for keeping contextualized document chunks within model limits."""

from rag.entrypoints.rest import _truncate_embedding_text


class FakeTokenizer:
    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = list(range(1, len(text.split()) + 1))
        return ([101] + tokens + [102]) if add_special_tokens else tokens

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            token_ids = [token for token in token_ids if token not in {101, 102}]
        return " ".join(f"token{token}" for token in token_ids)


def test_truncate_embedding_text_keeps_text_under_limit() -> None:
    tokenizer = FakeTokenizer()
    text = " ".join(f"word{i}" for i in range(20))

    result = _truncate_embedding_text(text, tokenizer, max_tokens=8)

    assert result
    assert len(tokenizer.encode(result, add_special_tokens=True)) <= 8


def test_truncate_embedding_text_preserves_short_text() -> None:
    tokenizer = FakeTokenizer()
    text = "a short chunk"

    result = _truncate_embedding_text(text, tokenizer, max_tokens=8)

    assert result == text
