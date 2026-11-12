
import re


CITATION_PATTERN = re.compile(
    r"【\s*([a-zA-Z0-9_-]+)\s*\|\s*Chunk:\s*"
    r"(\d+)(?:\s*[-–]\s*(\d+))?\s*】"
)


def validate_citations(answer, retrieved_passages):

    available_sources = {
        (item["source"], item["chunk_index"])
        for item in retrieved_passages
    }

    matches = CITATION_PATTERN.findall(answer)

    valid_citations = []
    invalid_citations = []

    for source, start_chunk, end_chunk in matches:

        first = int(start_chunk)
        last = int(end_chunk) if end_chunk else first

        for chunk_index in range(first, last + 1):

            citation = (source, chunk_index)

            if citation in available_sources:
                valid_citations.append(citation)
            else:
                invalid_citations.append(citation)

    return {
        "citations_found": len(valid_citations) + len(invalid_citations),
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "citation_status": (
            "no_citations"
            if not matches
            else "all_valid"
            if not invalid_citations
            else "invalid_citations_found"
        )
    }
