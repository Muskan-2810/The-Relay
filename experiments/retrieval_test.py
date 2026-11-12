
import csv
import json
import sys
from pathlib import Path

# Project root ko Python path mein add karna
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.retriever import PaperRetriever

QUESTIONS_FILE = PROJECT_ROOT / "data" / "questions" / "questions.json"
OUTPUT_FILE = PROJECT_ROOT / "results" / "source_retrieval_baseline.csv"


def evaluate_retrieval():
    # Questions load karo
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as file:
        questions = json.load(file)

    # Retriever initialize karo
    retriever = PaperRetriever()

    evaluation_results = []

    top1_matches = 0
    top5_matches = 0

    print("\nRETRIEVAL EVALUATION")
    print("=" * 65)

    for question_data in questions:
        question_id = question_data["id"]
        question = question_data["question"]
        expected_source = question_data["expected_source"]

        # Top 5 passages retrieve karo
        results = retriever.search(question, top_k=5)

        retrieved_sources = [
            result["source"] for result in results
        ]

        top1_source = (
            retrieved_sources[0] if retrieved_sources else "none"
        )

        # Kya first result expected paper se hai?
        top1_match = top1_source == expected_source

        # Kya top 5 mein expected paper present hai?
        top5_match = expected_source in retrieved_sources

        if top1_match:
            top1_matches += 1

        if top5_match:
            top5_matches += 1

        evaluation_results.append({
            "question_id": question_id,
            "question": question,
            "expected_source": expected_source,
            "top1_source": top1_source,
            "top5_sources": ";".join(retrieved_sources),
            "top1_source_match": top1_match,
            "top5_source_hit": top5_match
        })

        print(f"\nQuestion {question_id}: {question}")
        print("Expected:", expected_source)
        print("Top 1:", top1_source)
        print("Top 5:", retrieved_sources)
        print("Top 1 match:", top1_match)
        print("Top 5 hit:", top5_match)

    # Metrics calculate karo
    total = len(questions)

    top1_rate = (top1_matches / total) * 100
    top5_rate = (top5_matches / total) * 100

    # CSV save karo
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        fieldnames = evaluation_results[0].keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evaluation_results)

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)
    print("Total questions:", total)
    print("Top 1 source matches:", top1_matches)
    print("Top 1 source match rate:", round(top1_rate, 2), "%")
    print("Top 5 source hits:", top5_matches)
    print("Top 5 source hit rate:", round(top5_rate, 2), "%")
    print("\nResults saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    evaluate_retrieval()
