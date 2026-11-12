
import json
from app.scope_gate import detect_research_scope

with open("data/questions/questions.json", encoding="utf-8") as f:
    questions = json.load(f)

correct = 0
missed = []

print("\nRESEARCH SCOPE GATE VALIDATION")
print("=" * 65)

for q in questions:
    result = detect_research_scope(q["question"])

    expected = q["expected_source"]
    detected = result["matched_papers"]

    match = expected in detected

    if match:
        correct += 1
    else:
        missed.append(q["id"])

    print(f"\nQ{q['id']}: {q['question']}")
    print(f"Expected: {expected}")
    print(f"Detected: {detected}")
    print(f"MATCH: {'YES' if match else 'NO'}")

print("\n" + "=" * 65)
print(f"EXPECTED PAPER DETECTED: {correct}/{len(questions)}")
print(f"DETECTION RATE: {correct / len(questions) * 100:.2f}%")
print("MISSED QUESTION IDs:", missed)
