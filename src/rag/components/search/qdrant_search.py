
"""
Qdrant search component.

Supports:
- Hybrid search using dense vectors and BM25 sparse vectors.
- Semantic search using dense vectors.
- Keyword search using BM25 sparse vectors.
- Retrieval results with source metadata and chunk identifiers.
"""

from fastembed import SparseTextEmbedding
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.models import QueryResponse

from rag.types import SearchResult


class QdrantSearch:
    """Component for searching documents stored in Qdrant."""

    def __init__(
        self,
        collection: str,
        url: str,
        api_key: str | None = None,
        _dense_index: str = "dense",
        _sparse_index: str = "sparse",
        _qdrant_client_class: type[AsyncQdrantClient] = AsyncQdrantClient,
        _sparse_text_embedding_class: type[SparseTextEmbedding] = SparseTextEmbedding,
    ) -> None:
        """Initialize the Qdrant search component."""

        self.collection = collection

        self.qdrant = _qdrant_client_class(
            path=url,
            api_key=api_key,
        )

        self.bm25 = _sparse_text_embedding_class(
            model_name="Qdrant/bm25"
        )

        self.dense_index = _dense_index
        self.sparse_index = _sparse_index

    def _build_result(
        self,
        response: QueryResponse,
    ) -> list[SearchResult]:
        """Convert Qdrant results into application search results.

        Preserves the original payload and adds the Qdrant point ID
        as a stable chunk identifier.
        """

        results: list[SearchResult] = []

        for point in response.points:
            if point.payload is None:
                continue

            results.append(
                {
                    "score": point.score,
                    "data": {
                        **point.payload,
                        "chunk_id": str(point.id),
                    },
                }
            )

        return results

    async def upsert_chunks(
        self,
        chunks: list[dict[str, str]],
        dense_vectors: list[list[float]],
    ) -> int:
        """Insert document chunks with dense and sparse vectors."""
        from uuid import uuid4

        if len(chunks) != len(dense_vectors):
            raise ValueError("Each chunk must have exactly one dense vector.")

        points = []
        for chunk, dense_vector in zip(chunks, dense_vectors, strict=True):
            sparse_embedding = next(
                iter(self.bm25.query_embed(chunk["content"]))
            )
            points.append(
                models.PointStruct(
                    id=str(uuid4()),
                    vector={
                        self.dense_index: dense_vector,
                        self.sparse_index: models.SparseVector(
                            indices=sparse_embedding.indices.tolist(),
                            values=sparse_embedding.values.tolist(),
                        ),
                    },
                    payload={
                        "content": chunk["content"],
                        "source": chunk["source"],
                    },
                )
            )

        if points:
            await self.qdrant.upsert(
                collection_name=self.collection,
                points=points,
                wait=True,
            )
        return len(points)

    async def hybrid_search(
        self,
        query: list[float],
        keywords: list[str],
        limit: int = 25,
    ) -> list[SearchResult]:
        """Search using dense and sparse retrieval with RRF fusion."""

        text = " ".join(keywords)

        embedding = next(
            iter(self.bm25.query_embed(text))
        )

        sparse_vector = models.SparseVector(
            indices=embedding.indices.tolist(),
            values=embedding.values.tolist(),
        )

        prefetch = [
            models.Prefetch(
                query=query,
                using=self.dense_index,
                limit=limit,
            ),
            models.Prefetch(
                query=sparse_vector,
                using=self.sparse_index,
                limit=limit,
            ),
        ]

        response = await self.qdrant.query_points(
            collection_name=self.collection,
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),
            limit=limit,
            with_payload=True,
        )

        return self._build_result(response)

    async def semantic_search(
        self,
        query: list[float],
        limit: int = 25,
    ) -> list[SearchResult]:
        """Search using dense semantic embeddings."""

        response = await self.qdrant.query_points(
            collection_name=self.collection,
            query=query,
            using=self.dense_index,
            limit=limit,
            with_payload=True,
        )

        return self._build_result(response)

    async def keyword_search(
        self,
        keywords: list[str],
        limit: int = 25,
    ) -> list[SearchResult]:
        """Search using BM25 sparse embeddings."""

        text = " ".join(keywords)

        embedding = next(
            iter(self.bm25.query_embed(text))
        )

        sparse_vector = models.SparseVector(
            indices=embedding.indices.tolist(),
            values=embedding.values.tolist(),
        )

        response = await self.qdrant.query_points(
            collection_name=self.collection,
            query=sparse_vector,
            using=self.sparse_index,
            limit=limit,
            with_payload=True,
        )

        return self._build_result(response)
