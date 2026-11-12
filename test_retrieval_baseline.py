
import json
from app.retriever import PaperRetriever

retriever = PaperRetriever()

with open("data/questions/questions.json", encoding="utf-8") as f:
    questions = json.load(f)

correct = 0

print("\nRETRIEVAL BASELINE")
print("=" * 70)

for q in questions:
    results = retriever.search(q["question"], top_k=3)

    top1 = results[0]["source"] if results else "NONE"
    score = round(results[0]["score"], 3) if results else 0
    top3 = list(dict.fromkeys(x["source"] for x in results))

    match = top1 == q["expected_source"]

    if match:
        correct += 1

    print(f"\nQ{q['id']}: {q['question']}")
    print(f"Expected: {q['expected_source']}")
    print(f"Top-1: {top1} | Score: {score}")
    print(f"Top-3 sources: {top3}")
    print(f"MATCH: {'YES' if match else 'NO'}")
    print("-" * 70)

print("\nTOP-1 SOURCE MATCH:")
print(f"{correct}/{len(questions)} = {correct / len(questions) * 100:.2f}%")
