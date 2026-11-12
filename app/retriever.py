
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


# -----------------------------------
# 1. Project paths
# -----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "chunks.json"


# -----------------------------------
# 2. Stopwords
# -----------------------------------

STOPWORDS = {
    "the", "is", "a", "an", "and", "or",
    "of", "in", "on", "at", "to", "for",
    "how", "does", "do", "what", "why",
    "when", "where", "who", "which",
    "with", "by", "from", "this", "that",
    "it", "its", "are", "was", "were",
    "be", "been", "being", "as", "about"
}


# -----------------------------------
# 3. Text tokenizer
# -----------------------------------

def tokenize(text):

    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )

    return [
        word for word in words
        if word not in STOPWORDS
    ]


# -----------------------------------
# 4. Load extracted chunks
# -----------------------------------

def load_chunks():

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}"
        )

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("No valid chunks found.")

    required_fields = {
        "source",
        "chunk_index",
        "text"
    }

    for index, chunk in enumerate(chunks):

        missing = required_fields - set(chunk)

        if missing:
            raise ValueError(
                f"Chunk {index} is missing fields: {missing}"
            )

    return chunks


# -----------------------------------
# 5. BM25 Retriever
# -----------------------------------

class PaperRetriever:

    def __init__(self):

        self.chunks = load_chunks()

        self.tokenized_chunks = [
            tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_chunks
        )

        print("Retriever initialized!")
        print("Total indexed chunks:", len(self.chunks))


    # -----------------------------------
    # 6. Global search
    # -----------------------------------

    def search(
        self,
        query,
        top_k=5,
        min_score=0.0
    ):

        if not isinstance(query, str) or not query.strip():
            return []

        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        tokenized_query = tokenize(query)

        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        results = []
        seen_chunks = set()

        for index in ranked_indices:

            score = float(scores[index])

            if score <= min_score:
                continue

            chunk = self.chunks[index]

            identity = (
                chunk["source"],
                chunk["chunk_index"]
            )

            if identity in seen_chunks:
                continue

            seen_chunks.add(identity)

            results.append({
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "score": score,
                "text": chunk["text"]
            })

            if len(results) >= top_k:
                break

        return results


    # -----------------------------------
    # 7. Paper-wise search
    # -----------------------------------

    def search_by_source(
        self,
        query,
        source,
        top_k=2,
        min_score=0.0
    ):

        if not isinstance(query, str) or not query.strip():
            return []

        if not isinstance(source, str) or not source.strip():
            return []

        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        tokenized_query = tokenize(query)

        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        results = []

        for index in ranked_indices:

            chunk = self.chunks[index]

            # Only retrieve from the requested paper
            if chunk["source"] != source:
                continue

            score = float(scores[index])

            if score <= min_score:
                continue

            results.append({
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "score": score,
                "text": chunk["text"]
            })

            if len(results) >= top_k:
                break

        return results


# -----------------------------------
# 8. Run standalone retrieval test
# -----------------------------------

if __name__ == "__main__":

    retriever = PaperRetriever()

    question = (
        "Moshi speech text foundation model full duplex "
        "spoken dialogue latency speech recognition "
        "text to speech interruptions overlapping speech"
    )

    results = retriever.search(
        question,
        top_k=5
    )

    print("\nQuestion:", question)

    print("\nRetrieved passages:", len(results))

    for rank, result in enumerate(results, start=1):

        print("\nRank:", rank)
        print("Source:", result["source"])
        print("Chunk:", result["chunk_index"])
        print("Score:", round(result["score"], 4))
        print("Text:", result["text"][:300])
        print("--------------------------------")
