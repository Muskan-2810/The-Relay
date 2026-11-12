
from app.scope_gate import detect_research_scope

tests = [
    {
        "question": "What is the capital of France?",
        "expected": False
    },
    {
        "question": "Explain photosynthesis in plants.",
        "expected": False
    },
    {
        "question": "How can I pay attention in class?",
        "expected": False
    },
    {
        "question": "How does attention work in Transformers?",
        "expected": True
    },
    {
        "question": "Can attention help identify meaningful chunk boundaries?",
        "expected": True
    }
]

correct = 0

print("\nNEGATIVE AND PARTIAL SCOPE TEST")
print("=" * 65)

for test in tests:
    result = detect_research_scope(test["question"])

    actual = result["in_scope"]
    expected = test["expected"]

    match = actual == expected

    if match:
        correct += 1

    print("\nQuestion:", test["question"])
    print("Expected in_scope:", expected)
    print("Actual in_scope:", actual)
    print("Detected papers:", result["matched_papers"])
    print("MATCH:", "YES" if match else "NO")

print("\n" + "=" * 65)
print("RESULT:", correct, "/", len(tests))
print("ACCURACY:", round(correct / len(tests) * 100, 2), "%")
