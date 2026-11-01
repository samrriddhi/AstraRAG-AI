"""Run answer-quality checks against the configured AstraRAG pipeline.

This script calls the configured chat model once per evaluation question.
It records answers, retrieved chunk IDs, citations, and simple abstention checks.
Semantic correctness and factual grounding still require human review.
"""

import argparse
import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag.agent import Agent
from rag.config import get_openai_embed, get_qdrant, settings


DATASET_PATH = Path("data/evaluation/answer_quality_dataset.json")
REPORT_PATH = Path("data/evaluation/answer_quality_report.json")
CHUNK_ID_PATTERN = re.compile(
    r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
    re.IGNORECASE,
)
PAGE_PATTERN = re.compile(
    r"\b(?:Page|p\.)\s*[:.]?\s*([0-9]+(?:\s*,\s*[0-9]+)*)",
    re.IGNORECASE,
)
ABSTENTION_PHRASES = (
    "do not contain sufficient information",
    "does not contain sufficient information",
    "don't contain sufficient information",
    "could not find sufficient information",
    "not available in the indexed documents",
    "cannot determine from the provided documents",
    "unable to determine from the indexed documents",
)


class RecordingSearch:
    """Delegate retrieval to Qdrant while recording every returned chunk ID."""

    def __init__(self, search: Any) -> None:
        self.search = search
        self.retrieved_chunk_ids: list[str] = []

    def _record(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for result in results:
            chunk_id = result.get("data", {}).get("chunk_id")
            if chunk_id is not None:
                self.retrieved_chunk_ids.append(str(chunk_id))
        return results

    async def hybrid_search(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._record(await self.search.hybrid_search(**kwargs))

    async def semantic_search(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._record(await self.search.semantic_search(**kwargs))

    async def keyword_search(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self._record(await self.search.keyword_search(**kwargs))


def _contains_abstention(answer: str) -> bool:
    """Check for common explicit insufficient-evidence language."""
    normalized = answer.casefold()
    return any(phrase in normalized for phrase in ABSTENTION_PHRASES)


async def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Generate one answer and evaluate its citation and abstention signals."""
    recorder = RecordingSearch(get_qdrant())
    agent = Agent(
        model=settings.groq_model,
        _search=recorder,  # type: ignore[arg-type]
        _embed=get_openai_embed(),
    )
    messages = [
        {"role": "system", "content": agent.system},
        {"role": "user", "content": case["query"]},
    ]

    answer_parts: list[str] = []
    async for part in agent.generate(messages):
        answer_parts.append(part)
    answer = "".join(answer_parts)

    # Models sometimes use Unicode non-breaking/en/em dashes inside UUIDs.
    # Normalize dash variants before matching standard UUID syntax.
    normalized_answer = re.sub(r"[\u2010-\u2015\u2212]", "-", answer)
    citations = []
    for chunk_match in CHUNK_ID_PATTERN.finditer(normalized_answer):
        # Citation styles vary; identify the chunk UUID first, then inspect its
        # surrounding sentence/line for optional source and page metadata.
        context = normalized_answer[
            max(0, chunk_match.start() - 180):chunk_match.end() + 100
        ]
        source_match = re.search(
            r"([\w .()/-]+\.pdf)",
            context,
            re.IGNORECASE,
        )
        page_match = PAGE_PATTERN.search(context)
        citations.append(
            {
                "source": source_match.group(1).strip() if source_match else None,
                "pages": page_match.group(1).strip() if page_match else None,
                "chunk_id": chunk_match.group(1).strip(),
            }
        )
    retrieved_ids = sorted(set(recorder.retrieved_chunk_ids))
    cited_ids = sorted({item["chunk_id"] for item in citations})
    retrieved_set = set(retrieved_ids)
    invalid_citations = [chunk_id for chunk_id in cited_ids if chunk_id not in retrieved_set]
    abstained = _contains_abstention(answer)
    answerable = bool(case["answerable"])

    return {
        "query": case["query"],
        "answerable": answerable,
        "answer": answer,
        "expected_relevant_chunk_ids": case.get("expected_relevant_chunk_ids", []),
        "retrieved_chunk_ids": retrieved_ids,
        "citations": citations,
        "citation_present": bool(citations),
        "all_citations_retrieved": bool(citations) and not invalid_citations,
        "invalid_citation_chunk_ids": invalid_citations,
        "abstention_detected": abstained,
        "abstention_check_passed": abstained if not answerable else not abstained,
        "manual_review_required": True,
    }


def build_report(results: list[dict[str, Any]], total_cases: int) -> dict[str, Any]:
    """Build metrics from successful cases while reporting failed requests separately."""
    successful = [item for item in results if item["status"] == "completed"]
    failed = [item for item in results if item["status"] == "failed"]
    answerable = [item for item in successful if item["answerable"]]
    unanswerable = [item for item in successful if not item["answerable"]]
    cited = sum(item["citation_present"] for item in answerable)
    valid = sum(item["all_citations_retrieved"] for item in answerable)
    abstention_passes = sum(item["abstention_detected"] for item in unanswerable)
    answerable_false_abstentions = sum(item["abstention_detected"] for item in answerable)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": settings.groq_model,
        "collection": settings.qdrant_collection,
        "queries_total": total_cases,
        "queries_completed": len(successful),
        "queries_failed": len(failed),
        "summary": {
            "answerable_queries": len(answerable),
            "unanswerable_queries": len(unanswerable),
            "answerable_citation_rate": cited / len(answerable) if answerable else None,
            "valid_citation_rate": valid / len(answerable) if answerable else None,
            "unanswerable_abstention_rate": (
                abstention_passes / len(unanswerable) if unanswerable else None
            ),
            "answerable_false_abstention_rate": (
                answerable_false_abstentions / len(answerable) if answerable else None
            ),
        },
        "results": results,
        "evaluation_note": (
            "Citation and abstention checks are automatic heuristics. "
            "Failed API requests are excluded from quality metric denominators. "
            "Answer correctness, completeness, and factual grounding require human review."
        ),
    }


def save_report(report: dict[str, Any]) -> None:
    """Persist the latest report so completed cases survive an interrupted run."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


async def run(dataset: list[dict[str, Any]], limit: int | None) -> dict[str, Any]:
    """Evaluate sequentially, retry transient failures, and checkpoint after each case."""
    selected = dataset[:limit] if limit else dataset
    results: list[dict[str, Any]] = []
    for index, case in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {case['query']}", flush=True)
        result = None
        for attempt in range(3):
            try:
                result = await evaluate_case(case)
                result["status"] = "completed"
                break
            except Exception as exc:
                if attempt < 2:
                    delay = 2 ** (attempt + 1)
                    print(
                        f"  Attempt {attempt + 1} failed ({type(exc).__name__}: {exc}). "
                        f"Retrying in {delay}s...",
                        flush=True,
                    )
                    await asyncio.sleep(delay)
                else:
                    result = {
                        "query": case["query"],
                        "answerable": bool(case["answerable"]),
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "manual_review_required": True,
                    }
                    print("  Failed after 3 attempts; continuing.", flush=True)

        results.append(result)
        save_report(build_report(results, len(selected)))

    return build_report(results, len(selected))


def main() -> None:
    """Load evaluation cases, call the RAG agent, and write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N cases (useful for a small smoke run).",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {DATASET_PATH}")

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not dataset:
        raise ValueError("The answer-quality dataset is empty.")

    report = asyncio.run(run(dataset, args.limit))
    save_report(report)
    print("\nAnswer-quality evaluation")
    print(json.dumps(report["summary"], indent=2))
    print(f"Full report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
