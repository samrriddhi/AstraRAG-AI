"""Export indexed chunk text and metadata for building a human-labelled eval set.

This is read-only: it scrolls the configured Qdrant collection and writes JSON.
It does not modify or delete any points.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from rag.config import get_qdrant

OUTPUT_PATH = Path("data/evaluation/indexed_chunks.json")
PAGE_SIZE = 100


async def export_chunks() -> list[dict[str, Any]]:
    """Read every point from the configured Qdrant collection."""
    search = get_qdrant()
    offset = None
    chunks: list[dict[str, Any]] = []

    while True:
        points, next_offset = await search.qdrant.scroll(
            collection_name=search.collection,
            limit=PAGE_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}
            chunks.append(
                {
                    "chunk_id": str(point.id),
                    "source": payload.get("source"),
                    "pages": payload.get("pages", []),
                    "content": payload.get("content", ""),
                }
            )

        if next_offset is None:
            break
        offset = next_offset

    return chunks


async def main() -> None:
    chunks = await export_chunks()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Exported {len(chunks)} chunks to {OUTPUT_PATH}")
    print("Read-only export complete; the Qdrant collection was not modified.")


if __name__ == "__main__":
    asyncio.run(main())
