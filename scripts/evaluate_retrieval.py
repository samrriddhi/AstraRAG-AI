"""Run labelled retrieval evaluation against the local AstraRAG index."""

import asyncio
import json
from pathlib import Path
from typing import Any

from rag.config import get_openai_embed, get_qdrant
from rag.evaluation.metrics import mean_reciprocal_rank, precision_at_k, recall_at_k


DATASET_PATH = Path("data/evaluation/retrieval_dataset.json")
TOP_K = 5


async def evaluate(dataset: list[dict[str, Any]]) -> dict[str, float]:
    """Retrieve top-K chunks for each labelled query and calculate metrics."""
    embedder = get_openai_embed()
    search = get_qdrant()
    precisions: list[float] = []
    recalls: list[float] = []
    runs: list[list[str]] = []
    relevant_sets: list[list[str]] = []

    for item in dataset:
        query = item["query"]
        relevant_ids = item["relevant_chunk_ids"]
        if not relevant_ids:
            raise ValueError(
                f"Add at least one relevant_chunk_id for query: {query!r}"
            )

        query_vector = await embedder.generate_embedding(text=query)
        keywords = item.get("keywords") or query.split()
        results = await search.hybrid_search(
            query=query_vector,
            keywords=keywords,
            limit=TOP_K,
        )
        retrieved_ids = [
            str(result["data"]["chunk_id"])
            for result in results
            if result.get("data", {}).get("chunk_id") is not None
        ]

        precisions.append(precision_at_k(retrieved_ids, relevant_ids, TOP_K))
        recalls.append(recall_at_k(retrieved_ids, relevant_ids, TOP_K))
        runs.append(retrieved_ids)
        relevant_sets.append(relevant_ids)

        print(f"Query: {query}")
        print(f"Retrieved: {retrieved_ids}")
        print(f"Relevant:  {relevant_ids}")
        print()

    count = len(dataset)
    return {
        "queries_evaluated": float(count),
        "precision_at_5": sum(precisions) / count,
        "recall_at_5": sum(recalls) / count,
        "mrr": mean_reciprocal_rank(runs, relevant_sets),
    }


def main() -> None:
    """Load the labelled dataset and print aggregate retrieval metrics."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}. "
            "Copy the example dataset and add labelled chunk IDs."
        )

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not dataset:
        raise ValueError("The retrieval evaluation dataset is empty.")

    metrics = asyncio.run(evaluate(dataset))
    print("Retrieval evaluation")
    for name, value in metrics.items():
        if name == "queries_evaluated":
            print(f"{name}: {int(value)}")
        else:
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
