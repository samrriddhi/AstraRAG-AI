from rag.evaluation.citations import extract_citations


def test_adjacent_citations_keep_their_own_page_numbers():
    answer = """- Source: ocean.pdf, Page: 3, Chunk ID: d2178103-35c1-5908-823f-f4a8f9775957
- Source: ocean.pdf, Page: 2, Chunk ID: 91497edb-c2a8-5da0-9895-b6a3776ad0eb"""
    citations = extract_citations(answer)

    assert citations == [
        {
            "source": "ocean.pdf",
            "pages": "3",
            "chunk_id": "d2178103-35c1-5908-823f-f4a8f9775957",
        },
        {
            "source": "ocean.pdf",
            "pages": "2",
            "chunk_id": "91497edb-c2a8-5da0-9895-b6a3776ad0eb",
        },
    ]


def test_page_ranges_are_preserved():
    answer = "Source: ocean.pdf, pages 4-5, Chunk ID: 99999999-9999-4999-8999-999999999999"

    assert extract_citations(answer)[0]["pages"] == "4-5"


def test_metadata_on_immediately_preceding_line_is_supported():
    answer = """Source: ocean.pdf, Page 8
Chunk ID: 08bf0000-0000-4000-8000-000000000008"""

    citation = extract_citations(answer)[0]
    assert citation["source"] == "ocean.pdf"
    assert citation["pages"] == "8"


def test_missing_metadata_stays_unavailable():
    answer = "Chunk ID: 08bf0000-0000-4000-8000-000000000008"

    assert extract_citations(answer) == [
        {
            "source": None,
            "pages": None,
            "chunk_id": "08bf0000-0000-4000-8000-000000000008",
        }
    ]


def test_page_from_previous_citation_does_not_leak():
    answer = """Source: ocean.pdf, Page 3, Chunk ID: d2178103-35c1-5908-823f-f4a8f9775957
Chunk ID: 91497edb-c2a8-5da0-9895-b6a3776ad0eb"""
    citations = extract_citations(answer)

    assert citations[0]["pages"] == "3"
    assert citations[1]["pages"] is None
