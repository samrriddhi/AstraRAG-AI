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
SOURCE_PATTERN = re.compile(r"([\\w .()/-]+\\.pdf)\\b", re.IGNORECASE)
PAGE_PATTERN = re.compile(
    r"\\b(?:Pages?|pp?\\.)\\s*[:.]?\\s*([0-9]+(?:\\s*[-–—,]\\s*[0-9]+)*)",
    re.IGNORECASE,
)


def extract_citations(answer: str) -> list[dict[str, str | None]]:
    """Extract metadata from each citation's line, avoiding adjacent-page leakage."""
    normalized = re.sub(r"[\\u2010-\\u2015\\u2212]", "-", answer)
    lines = normalized.splitlines()
    starts: list[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line) + 1

    citations: list[dict[str, str | None]] = []
    for match in CHUNK_ID_PATTERN.finditer(normalized):
        line_index = max(
            (index for index, start in enumerate(starts) if start <= match.start()),
            default=0,
        )
        local_text = lines[line_index] if lines else ""
        source_match = SOURCE_PATTERN.search(local_text)
        page_match = PAGE_PATTERN.search(local_text)

        # Some citation renderers wrap metadata and the UUID onto adjacent lines.
        # Look back one line only when it explicitly contains source/page metadata.
        if line_index > 0 and (not source_match or not page_match):
            previous = lines[line_index - 1]
            if SOURCE_PATTERN.search(previous) or PAGE_PATTERN.search(previous):
                local_text = previous + " " + local_text
                source_match = SOURCE_PATTERN.search(local_text)
                page_match = PAGE_PATTERN.search(local_text)

        citations.append({
            "source": source_match.group(1).strip() if source_match else None,
            "pages": re.sub(r"\\s+", "", page_match.group(1)) if page_match else None,
            "chunk_id": match.group(1),
        })
    return citations


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

    citations = extract_citations(answer)
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


def _rate_limit_wait_seconds(error: Exception) -> float | None:
    """Read Groq's suggested reset delay from a rate-limit error message."""
    message = str(error)
    match = re.search(
        r"try again in\s+(?:(\d+)m)?\s*([\d.]+)s",
        message,
        re.IGNORECASE,
    )
    if not match:
        return None
    minutes = int(match.group(1) or 0)
    seconds = float(match.group(2))
    return minutes * 60 + seconds + 2


async def run(
    dataset: list[dict[str, Any]],
    limit: int | None,
    resume: bool = False,
) -> dict[str, Any]:
    """Evaluate sequentially, checkpoint results, and optionally resume completed work."""
    selected = dataset[:limit] if limit else dataset
    results_by_query: dict[str, dict[str, Any]] = {}
    if resume and REPORT_PATH.exists():
        previous = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        for item in previous.get("results", []):
            if item.get("status") == "completed":
                results_by_query[item["query"]] = item
        print(
            f"Resuming: keeping {len(results_by_query)} completed result(s).",
            flush=True,
        )

    ordered_results: list[dict[str, Any]] = []
    for index, case in enumerate(selected, start=1):
        query = case["query"]
        if query in results_by_query:
            ordered_results.append(results_by_query[query])
            print(f"[{index}/{len(selected)}] Already completed: {query}", flush=True)
            continue

        print(f"[{index}/{len(selected)}] {query}", flush=True)
        result: dict[str, Any] | None = None
        stop_after_case = False
        for attempt in range(3):
            try:
                result = await evaluate_case(case)
                result["status"] = "completed"
                break
            except Exception as exc:
                is_rate_limit = type(exc).__name__ == "RateLimitError"
                if is_rate_limit:
                    delay = _rate_limit_wait_seconds(exc)
                    if delay is not None and attempt == 0:
                        print(
                            f"  Groq rate limit reached. Waiting {delay:.0f}s "
                            "as requested by the API before one retry...",
                            flush=True,
                        )
                        await asyncio.sleep(delay)
                        continue
                    result = {
                        "query": query,
                        "answerable": bool(case["answerable"]),
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "manual_review_required": True,
                    }
                    print(
                        "  Groq rate limit persists; checkpointing and pausing "
                        "so remaining requests don't waste quota.",
                        flush=True,
                    )
                    stop_after_case = True
                    break

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
                        "query": query,
                        "answerable": bool(case["answerable"]),
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "manual_review_required": True,
                    }
                    print("  Failed after 3 attempts; continuing.", flush=True)

        if result is None:
            raise RuntimeError(f"No evaluation result produced for: {query}")
        ordered_results.append(result)
        save_report(build_report(ordered_results, len(selected)))

        if stop_after_case:
            for remaining in selected[index:]:
                ordered_results.append(
                    {
                        "query": remaining["query"],
                        "answerable": bool(remaining["answerable"]),
                        "status": "skipped",
                        "error": "Paused after Groq rate limit; rerun with --resume.",
                        "manual_review_required": True,
                    }
                )
            save_report(build_report(ordered_results, len(selected)))
            break

    return build_report(ordered_results, len(selected))


def main() -> None:
    """Load evaluation cases, call the RAG agent, and write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N cases (useful for a small smoke run).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed cases from the existing report and retry failed/skipped cases.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {DATASET_PATH}")

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if not dataset:
        raise ValueError("The answer-quality dataset is empty.")

    report = asyncio.run(run(dataset, args.limit, args.resume))
    save_report(report)
    print("\nAnswer-quality evaluation")
    print(json.dumps(report["summary"], indent=2))
    print(f"Full report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
