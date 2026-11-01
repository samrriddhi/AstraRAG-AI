"""Citation parsing helpers used by offline evaluation tests."""

import re

CHUNK_ID_PATTERN = re.compile(
    r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
    re.IGNORECASE,
)
SOURCE_PATTERN = re.compile(r"([\w .()/-]+\.pdf)\b", re.IGNORECASE)
PAGE_PATTERN = re.compile(
    r"\b(?:Pages?|pp?\.)\s*[:.]?\s*([0-9]+(?:\s*[-–—,]\s*[0-9]+)*)",
    re.IGNORECASE,
)


def extract_citations(answer: str) -> list[dict[str, str | None]]:
    """Extract metadata from each citation's line, avoiding adjacent-page leakage."""
    normalized = re.sub(r"[\u2010-\u2015\u2212]", "-", answer)
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
            if (
                not CHUNK_ID_PATTERN.search(previous)
                and (SOURCE_PATTERN.search(previous) or PAGE_PATTERN.search(previous))
            ):
                local_text = previous + " " + local_text
                source_match = SOURCE_PATTERN.search(local_text)
                page_match = PAGE_PATTERN.search(local_text)

        citations.append({
            "source": source_match.group(1).strip() if source_match else None,
            "pages": re.sub(r"\s+", "", page_match.group(1)) if page_match else None,
            "chunk_id": match.group(1),
        })
    return citations

