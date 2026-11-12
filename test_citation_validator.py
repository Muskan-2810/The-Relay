from app.citation_validator import validate_citations

retrieved = [
    {"source": "moshi_v1", "chunk_index": 1, "text": "sample"},
    {"source": "attention_v7", "chunk_index": 9, "text": "sample"},
    {"source": "attention_v7", "chunk_index": 10, "text": "sample"},
    {"source": "dynamic_chunking_v2", "chunk_index": 5, "text": "sample"},
    {"source": "dynamic_chunking_v2", "chunk_index": 12, "text": "sample"},
]

answer = """
Moshi evidence 【moshi_v1 | Chunk: 1】
Attention evidence 【attention_v7 | Chunk: 9–10】
Dynamic evidence 【dynamic_chunking_v2 | Chunk: 5–12】
Fake evidence 【moshi_v1 | Chunk: 999】
"""

result = validate_citations(answer, retrieved)

print("CITATION VALIDATION TEST")
print("=" * 40)
print(result)