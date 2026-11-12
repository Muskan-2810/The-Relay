
from app.retriever import PaperRetriever

retriever = PaperRetriever()

questions = [
    {
        "type": "UNRELATED",
        "question": "What is the capital of France?"
    },
    {
        "type": "UNRELATED",
        "question": "Explain photosynthesis in plants."
    },
    {
        "type": "PARTIALLY RELATED",
        "question": "Can attention mechanisms help identify meaningful chunk boundaries in long sequences?"
    }
]

print("\nNEGATIVE RETRIEVAL TEST")
print("=" * 70)

for item in questions:
    results = retriever.search(item["question"], top_k=3)

    print(f"\nType: {item['type']}")
    print(f"Question: {item['question']}")

    if results:
        for i, result in enumerate(results, start=1):
            print(
                f"Rank {i}: {result['source']} | "
                f"Chunk: {result['chunk_index']} | "
                f"Score: {result['score']:.3f}"
            )
            print("Text:", result["text"][:180].replace("\n", " "))
    else:
        print("No results returned.")

    print("-" * 70)
