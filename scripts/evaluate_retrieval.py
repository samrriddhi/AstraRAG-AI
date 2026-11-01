"""Compare dense, sparse, and hybrid candidate-pool sizes on labelled queries."""

import asyncio
import json
from pathlib import Path
from typing import Any

from rag.config import get_openai_embed, get_qdrant
from rag.evaluation.metrics import mean_reciprocal_rank, precision_at_k, recall_at_k


DATASET_PATH = Path("data/evaluation/retrieval_dataset.json")
TOP_K = 5
HYBRID_CANDIDATE_SIZES = (5, 10, 20)


async def evaluate(dataset: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Run dense, sparse, and hybrid candidate-pool configurations."""
    embedder = get_openai_embed()
    search = get_qdrant()
    strategies = ["dense", "sparse"] + [
        f"hybrid_prefetch_{size}" for size in HYBRID_CANDIDATE_SIZES
    ]
    runs: dict[str, list[list[str]]] = {name: [] for name in strategies}
    precisions: dict[str, list[float]] = {name: [] for name in strategies}
    recalls: dict[str, list[float]] = {name: [] for name in strategies}
    relevant_sets: list[list[str]] = []

    for item in dataset:
        query = item["query"]
        relevant_ids = item["relevant_chunk_ids"]
        if not relevant_ids:
            raise ValueError(f"Add at least one relevant_chunk_id for query: {query!r}")

        query_vector = await embedder.generate_embedding(text=query)
        keywords = item.get("keywords") or query.split()
        results_by_strategy = {
            "dense": await search.semantic_search(query=query_vector, limit=TOP_K),
            "sparse": await search.keyword_search(keywords=keywords, limit=TOP_K),
        }
        for candidate_size in HYBRID_CANDIDATE_SIZES:
            results_by_strategy[f"hybrid_prefetch_{candidate_size}"] = (
                await search.hybrid_search(
                    query=query_vector,
                    keywords=keywords,
                    limit=TOP_K,
                    prefetch_limit=candidate_size,
                )
            )

        relevant_sets.append(relevant_ids)
        for strategy, results in results_by_strategy.items():
            retrieved_ids = [
                str(result["data"]["chunk_id"])
                for result in results
                if result.get("data", {}).get("chunk_id") is not None
            ]
            runs[strategy].append(retrieved_ids)
            precisions[strategy].append(
                precision_at_k(retrieved_ids, relevant_ids, TOP_K)
            )
            recalls[strategy].append(
                recall_at_k(retrieved_ids, relevant_ids, TOP_K)
            )

        print(f"Query: {query}")
        for strategy in strategies:
            print(f"  {strategy:20} {runs[strategy][-1]}")
        print(f"  {'relevant':20} {relevant_ids}")
        print()

    count = len(dataset)
    return {
        strategy: {
            "queries_evaluated": float(count),
            "precision_at_5": sum(precisions[strategy]) / count,
            "recall_at_5": sum(recalls[strategy]) / count,
            "mrr": mean_reciprocal_rank(runs[strategy], relevant_sets),
        }
        for strategy in strategies
    }


def main() -> None:
    """Load labels, compare retrieval configurations, and print metrics."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {DATASET_PATH}")
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not dataset:
        raise ValueError("The retrieval evaluation dataset is empty.")

    results = asyncio.run(evaluate(dataset))
    print("Retrieval configuration comparison (higher is better)")
    for strategy, metrics in results.items():
        print(f"\n{strategy.upper()}")
        for name, value in metrics.items():
            if name == "queries_evaluated":
                print(f"{name}: {int(value)}")
            else:
                print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
