
import csv
import json
import re
import sys
import time
from pathlib import Path
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

QUESTIONS_FILE = PROJECT_ROOT / "data" / "questions" / "questions.json"
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "chunks.json"
OUTPUT_FILE = PROJECT_ROOT / "results" / "query_expansion_comparison.csv"

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")

STOPWORDS = {
    "the", "is", "a", "an", "and", "or", "of", "in", "on",
    "at", "to", "for", "how", "does", "do", "what", "why",
    "when", "where", "who", "which", "with", "by", "from",
    "this", "that", "it", "its", "are", "was", "were",
    "be", "been", "being", "as", "about"
}


def tokenize(text):
    tokens = TOKEN_PATTERN.findall(text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def expand_query(question_data):
    question_id = question_data["id"]

    # Manually expanded Moshi question (Q6).
    if question_id == 6:
        return (
            question_data["question"]
            + " speech text foundation model full duplex "
            + "spoken dialogue latency speech recognition "
            + "text to speech interruptions overlapping speech "
            + "textual information bottleneck turn based modeling"
        )

    # Other questions remain unchanged.
    return question_data["question"]


def main():
    questions = json.loads(
        QUESTIONS_FILE.read_text(encoding="utf-8")
    )

    chunks = json.loads(
        CHUNKS_FILE.read_text(encoding="utf-8")
    )

    tokenized_chunks = [
        tokenize(chunk["text"]) for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    output_rows = []

    print("\nQUERY EXPANSION EXPERIMENT")
    print("=" * 65)
    print("Questions:", len(questions))
    print("Indexed chunks:", len(chunks))

    for question_data in questions:
        expected_source = question_data["expected_source"]
        expected_concepts = question_data["expected_concepts"]

        queries = {
            "original": question_data["question"],
            "expanded": expand_query(question_data)
        }

        for query_type, query in queries.items():
            query_tokens = tokenize(query)

            start_time = time.perf_counter()
            scores = bm25.get_scores(query_tokens)

            ranked_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:5]

            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            retrieved = [
                chunks[index] for index in ranked_indices
            ]

            top1_match = (
                retrieved[0]["source"] == expected_source
            )

            top5_match = any(
                chunk["source"] == expected_source
                for chunk in retrieved
            )

            expected_tokens = set()

            for concept in expected_concepts:
                expected_tokens.update(tokenize(concept))

            retrieved_text = " ".join(
                chunk["text"] for chunk in retrieved
            )

            retrieved_tokens = set(tokenize(retrieved_text))
            matched = expected_tokens & retrieved_tokens

            coverage = (
                len(matched) / len(expected_tokens)
                if expected_tokens else 0
            )

            retrieved_chars = sum(
                len(chunk["text"]) for chunk in retrieved
            )

            output_rows.append({
                "question_id": question_data["id"],
                "query_type": query_type,
                "query": query,
                "top1_source_match": top1_match,
                "top5_source_match": top5_match,
                "concept_coverage": round(coverage, 4),
                "matched_concepts": len(matched),
                "expected_concepts": len(expected_tokens),
                "latency_ms": round(latency_ms, 3),
                "approx_retrieved_tokens": round(
                    retrieved_chars / 4
                )
            })

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    fieldnames = list(output_rows[0].keys())

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("\nSUMMARY")
    print("-" * 65)

    for query_type in ["original", "expanded"]:
        rows = [
            row for row in output_rows
            if row["query_type"] == query_type
        ]

        count = len(rows)

        top1 = sum(
            row["top1_source_match"] for row in rows
        ) / count * 100

        top5 = sum(
            row["top5_source_match"] for row in rows
        ) / count * 100

        coverage = sum(
            row["concept_coverage"] for row in rows
        ) / count * 100

        latency = sum(
            row["latency_ms"] for row in rows
        ) / count

        tokens = sum(
            row["approx_retrieved_tokens"] for row in rows
        ) / count

        print("\nQuery type:", query_type)
        print(f"Top-1 source match: {top1:.2f}%")
        print(f"Top-5 source match: {top5:.2f}%")
        print(f"Concept coverage: {coverage:.2f}%")
        print(f"Average retrieval latency: {latency:.3f} ms")
        print(f"Average retrieved tokens: {tokens:.1f}")

    print("\nQ6 COMPARISON")

    for row in output_rows:
        if row["question_id"] == 6:
            print(
                row["query_type"],
                "| Top-1:", row["top1_source_match"],
                "| Top-5:", row["top5_source_match"],
                "| Coverage:", row["concept_coverage"],
                "| Tokens:", row["approx_retrieved_tokens"],
                "| Query:", row["query"]
            )

    print("\nSaved:", OUTPUT_FILE)
    print("Total rows:", len(output_rows))


if __name__ == "__main__":
    main()
