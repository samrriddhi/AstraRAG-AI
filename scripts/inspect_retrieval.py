"""Inspect the actual chunks returned for the labelled evaluation questions."""

import asyncio
import json
from pathlib import Path
from typing import Any

from rag.config import get_openai_embed, get_qdrant

DATASET_PATH = Path("data/evaluation/retrieval_dataset.json")
TOP_K = 5


async def inspect(dataset: list[dict[str, Any]]) -> None:
    """Print retrieved chunk IDs, pages, scores, and text for manual labelling."""
    embedder = get_openai_embed()
    search = get_qdrant()

    for item in dataset:
        query = item["query"]
        vector = await embedder.generate_embedding(text=query)
        results = await search.hybrid_search(
            query=vector,
            keywords=item.get("keywords") or query.split(),
            limit=TOP_K,
        )

        print("=" * 88)
        print(f"QUESTION: {query}")
        print("=" * 88)

        for rank, result in enumerate(results, start=1):
            data = result.get("data", {})
            print(f"\nRANK: {rank}")
            print(f"CHUNK_ID: {data.get('chunk_id', 'Unavailable')}")
            print(f"SOURCE: {data.get('source', 'Unavailable')}")
            print(f"PAGES: {data.get('pages', [])}")
            print(f"SCORE: {result.get('score', 'Unavailable')}")
            print("CONTENT:")
            print(data.get("content", "[No content in payload]"))
            print("-" * 88)


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not dataset:
        raise ValueError("Evaluation dataset is empty.")
    asyncio.run(inspect(dataset))


if __name__ == "__main__":
    main()
