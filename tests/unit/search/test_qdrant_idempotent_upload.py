"""Tests for idempotent, version-safe Qdrant document indexing."""

from types import SimpleNamespace

import pytest

from rag.components.search.qdrant_search import QdrantSearch


class FakeArray(list):
    def tolist(self):
        return list(self)


class FakeSparseEmbedding:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def query_embed(self, text: str):
        return iter([
            SimpleNamespace(indices=FakeArray([1, 2]), values=FakeArray([0.4, 0.6]))
        ])


class FakeQdrantClient:
    def __init__(self, *args, fail_upsert: bool = False, **kwargs):
        self.points = []
        self.deleted = []
        self.fail_upsert = fail_upsert

    async def upsert(self, **kwargs):
        if self.fail_upsert:
            raise RuntimeError("simulated upsert failure")
        self.points.extend(kwargs["points"])

    async def delete(self, **kwargs):
        self.deleted.append(kwargs)


def build_search(fail_upsert: bool = False) -> QdrantSearch:
    class Client(FakeQdrantClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, fail_upsert=fail_upsert, **kwargs)

    return QdrantSearch(
        collection="test",
        url="",
        _qdrant_client_class=Client,
        _sparse_text_embedding_class=FakeSparseEmbedding,
    )


def make_chunk(document_hash: str = "hash-v1") -> dict:
    return {
        "content": "A document passage",
        "source": "guide.pdf",
        "pages": [2],
        "document_hash": document_hash,
    }


@pytest.mark.asyncio
async def test_upsert_chunks_uses_stable_ids_and_removes_older_versions():
    search = build_search()

    count = await search.upsert_chunks(
        [make_chunk()],
        [[0.1, 0.2, 0.3]],
    )

    assert count == 1
    assert len(search.qdrant.points) == 1
    assert search.qdrant.points[0].payload["document_hash"] == "hash-v1"
    assert search.qdrant.points[0].payload["pages"] == [2]
    assert len(search.qdrant.deleted) == 1

    first_id = search.qdrant.points[0].id
    await search.upsert_chunks([make_chunk()], [[0.1, 0.2, 0.3]])
    assert search.qdrant.points[0].id == first_id


@pytest.mark.asyncio
async def test_changed_document_version_gets_different_point_ids():
    search = build_search()

    await search.upsert_chunks([make_chunk("hash-v1")], [[0.1, 0.2, 0.3]])
    first_id = search.qdrant.points[0].id

    await search.upsert_chunks([make_chunk("hash-v2")], [[0.4, 0.5, 0.6]])

    assert search.qdrant.points[1].id != first_id
    assert len(search.qdrant.deleted) == 2


@pytest.mark.asyncio
async def test_failed_upsert_does_not_delete_existing_document():
    search = build_search(fail_upsert=True)

    with pytest.raises(RuntimeError, match="simulated upsert failure"):
        await search.upsert_chunks([make_chunk()], [[0.1, 0.2, 0.3]])

    assert search.qdrant.deleted == []


@pytest.mark.asyncio
async def test_upsert_rejects_chunks_from_multiple_document_versions():
    search = build_search()

    with pytest.raises(ValueError, match="same document version"):
        await search.upsert_chunks(
            [make_chunk("hash-v1"), make_chunk("hash-v2")],
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        )

    assert search.qdrant.points == []
    assert search.qdrant.deleted == []
