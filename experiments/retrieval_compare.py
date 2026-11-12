
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
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted_papers"
OUTPUT_FILE = PROJECT_ROOT / "results" / "retrieval_comparison.csv"

SETTINGS = [
    {"name": "small_600", "chunk_size": 600, "overlap": 100},
    {"name": "baseline_1200", "chunk_size": 1200, "overlap": 200},
    {"name": "large_2000", "chunk_size": 2000, "overlap": 300},
]

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


def create_chunks(text, source, chunk_size, overlap):
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append({
                "source": source,
                "chunk_index": chunk_index,
                "start_char": start,
                "end_char": end,
                "text": chunk_text
            })

        chunk_index += 1

        if end == len(text):
            break

        start = end - overlap

    return chunks


def load_papers():
    papers = {}

    for file_path in EXTRACTED_DIR.glob("*.txt"):
        source = file_path.stem

        papers[source] = file_path.read_text(
            encoding="utf-8"
        )

    return papers


def build_index(papers, chunk_size, overlap):
    all_chunks = []

    for source, text in papers.items():
        all_chunks.extend(
            create_chunks(
                text,
                source,
                chunk_size,
                overlap
            )
        )

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in all_chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    return all_chunks, bm25


def evaluate():
    questions = json.loads(
        QUESTIONS_FILE.read_text(encoding="utf-8")
    )

    papers = load_papers()

    if len(papers) != 3:
        raise ValueError(
            f"Expected 3 extracted papers, found {len(papers)}. "
            f"Found: {list(papers.keys())}"
        )

    output_rows = []

    print("\nRETRIEVAL COMPARISON")
    print("=" * 75)
    print("Questions:", len(questions))
    print("Papers:", len(papers))

    for setting in SETTINGS:
        print("\nSetting:", setting["name"])
        print(
            "Chunk size:", setting["chunk_size"],
            "| Overlap:", setting["overlap"]
        )

        chunks, bm25 = build_index(
            papers,
            setting["chunk_size"],
            setting["overlap"]
        )

        print("Total generated chunks:", len(chunks))

        for question_data in questions:
            question = question_data["question"]
            expected_source = question_data["expected_source"]
            expected_concepts = question_data["expected_concepts"]

            query_tokens = tokenize(question)

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
                chunks[index]
                for index in ranked_indices
            ]

            top1_source_match = (
                retrieved[0]["source"] == expected_source
            )

            top5_source_match = any(
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

            matched_concepts = (
                expected_tokens & retrieved_tokens
            )

            concept_coverage = (
                len(matched_concepts) / len(expected_tokens)
                if expected_tokens else 0
            )

            retrieved_chars = sum(
                len(chunk["text"]) for chunk in retrieved
            )

            approximate_tokens = round(
                retrieved_chars / 4
            )

            output_rows.append({
                "question_id": question_data["id"],
                "setting": setting["name"],
                "chunk_size": setting["chunk_size"],
                "overlap": setting["overlap"],
                "total_chunks": len(chunks),
                "latency_ms": round(latency_ms, 3),
                "top1_source_match": top1_source_match,
                "top5_source_match": top5_source_match,
                "concept_coverage": round(concept_coverage, 4),
                "matched_concepts": len(matched_concepts),
                "expected_concepts": len(expected_tokens),
                "retrieved_chars": retrieved_chars,
                "approx_retrieved_tokens": approximate_tokens
            })

        setting_rows = [
            row for row in output_rows
            if row["setting"] == setting["name"]
        ]

        top1_rate = sum(
            row["top1_source_match"] for row in setting_rows
        ) / len(setting_rows) * 100

        top5_rate = sum(
            row["top5_source_match"] for row in setting_rows
        ) / len(setting_rows) * 100

        avg_coverage = sum(
            row["concept_coverage"] for row in setting_rows
        ) / len(setting_rows) * 100

        avg_tokens = sum(
            row["approx_retrieved_tokens"]
            for row in setting_rows
        ) / len(setting_rows)

        avg_latency = sum(
            row["latency_ms"] for row in setting_rows
        ) / len(setting_rows)

        print(f"Top-1 source match: {top1_rate:.2f}%")
        print(f"Top-5 source match: {top5_rate:.2f}%")
        print(f"Average concept keyword coverage: {avg_coverage:.2f}%")
        print(f"Average retrieved token estimate: {avg_tokens:.1f}")
        print(f"Average retrieval latency: {avg_latency:.3f} ms")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = list(output_rows[0].keys())

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("\nComparison saved to:", OUTPUT_FILE)
    print("Total evaluation rows:", len(output_rows))


if __name__ == "__main__":
    evaluate()
