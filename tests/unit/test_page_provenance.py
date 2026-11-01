"""Tests for preserving PDF page provenance from Docling chunks."""

from types import SimpleNamespace

from rag.entrypoints.rest import _extract_chunk_pages


def test_extract_chunk_pages_returns_unique_sorted_page_numbers() -> None:
    chunk = SimpleNamespace(
        meta=SimpleNamespace(
            doc_items=[
                SimpleNamespace(
                    prov=[
                        SimpleNamespace(page_no=3),
                        SimpleNamespace(page_no=1),
                    ]
                ),
                SimpleNamespace(
                    prov=[
                        SimpleNamespace(page_no=3),
                        SimpleNamespace(page_no=2),
                    ]
                ),
            ]
        )
    )

    assert _extract_chunk_pages(chunk) == [1, 2, 3]


def test_extract_chunk_pages_handles_missing_provenance() -> None:
    chunk = SimpleNamespace(meta=SimpleNamespace(doc_items=[]))

    assert _extract_chunk_pages(chunk) == []


def test_extract_chunk_pages_ignores_invalid_page_numbers() -> None:
    chunk = SimpleNamespace(
        meta=SimpleNamespace(
            doc_items=[
                SimpleNamespace(
                    prov=[
                        SimpleNamespace(page_no=0),
                        SimpleNamespace(page_no=None),
                        SimpleNamespace(page_no="2"),
                    ]
                )
            ]
        )
    )

    assert _extract_chunk_pages(chunk) == []
