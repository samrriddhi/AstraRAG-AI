"""Offline retrieval evaluation metrics for AstraRAG.

Expected relevant IDs are supplied by a small human-labelled evaluation set.
These metrics do not require an LLM, API key, or vector database.
"""

from collections.abc import Iterable, Sequence


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Return the fraction of the first k retrieved IDs that are relevant."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    if not retrieved:
        return 0.0
    relevant_ids = set(relevant)
    top_k = retrieved[:k]
    return sum(item in relevant_ids for item in top_k) / len(top_k)


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Return the fraction of all relevant IDs found in the first k results."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    relevant_ids = set(relevant)
    if not relevant_ids:
        return 0.0
    return len(set(retrieved[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Return the reciprocal rank of the first relevant retrieved item."""
    relevant_ids = set(relevant)
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    runs: Sequence[Sequence[str]],
    relevant_sets: Sequence[Iterable[str]],
) -> float:
    """Return mean reciprocal rank across aligned query result sets."""
    if len(runs) != len(relevant_sets):
        raise ValueError("Each retrieval run must have a relevant-ID set.")
    if not runs:
        return 0.0
    scores = [
        reciprocal_rank(run, relevant)
        for run, relevant in zip(runs, relevant_sets, strict=True)
    ]
    return sum(scores) / len(scores)
