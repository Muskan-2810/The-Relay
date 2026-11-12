

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from app.retriever import PaperRetriever
from app.scope_gate import detect_research_scope
from app.citation_validator import validate_citations


# -----------------------------------
# 1. Configuration
# -----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found. Check your .env file."
    )

MODEL = "openai/gpt-oss-20b"

client = Groq(api_key=API_KEY)

retriever = PaperRetriever()


# -----------------------------------
# 2. Focused paper search queries
# -----------------------------------

PAPER_SEARCH_QUERIES = {
    "moshi_v1": (
        "Moshi full duplex speech dialogue "
        "speech to speech generation audio tokens "
        "latency interruptions overlapping speech"
    ),

    "attention_v7": (
        "Transformer self attention queries keys values "
        "scaled dot product attention multi head attention"
    ),

    "dynamic_chunking_v2": (
        "H-Net recursive data-dependent dynamic chunking "
        "content-aware context-dependent segmentation "
        "hierarchical U-Net learned chunk boundaries "
        "semantic importance compression ratio"
    )
}


# -----------------------------------
# 3. Empty citation result
# -----------------------------------

def empty_citation_result():

    return {
        "citations_found": 0,
        "valid_citations": [],
        "invalid_citations": [],
        "citation_status": "not_applicable"
    }


# -----------------------------------
# 4. Research scope + BM25 retrieval
# -----------------------------------

def retrieve_paper_passages(query, matched_papers, top_k=5):

    if len(matched_papers) >= 2:

        results = []

        for source in matched_papers:

            focused_query = PAPER_SEARCH_QUERIES[source]

            paper_results = retriever.search_by_source(
                query=focused_query,
                source=source,
                top_k=2
            )

            results.extend(paper_results)

    elif len(matched_papers) == 1:

        source = matched_papers[0]

        results = retriever.search_by_source(
            query=query,
            source=source,
            top_k=top_k
        )

    else:
        return []

    return [
        {
            "source": item["source"],
            "chunk_index": item["chunk_index"],
            "score": round(item["score"], 4),
            "text": item["text"]
        }
        for item in results
    ]


# -----------------------------------
# 5. End-to-end RAG Pipeline
# -----------------------------------

def answer_question(question):

    start = time.perf_counter()

    scope_result = detect_research_scope(question)

    matched_papers = scope_result["matched_papers"]

    # Step 1: Reject unrelated questions

    if not scope_result["in_scope"]:

        return {
            "answer": (
                "This question appears to be outside "
                "the scope of the indexed research papers. "
                "I can currently answer questions about "
                "Moshi, Transformer Attention, and "
                "Dynamic Chunking research."
            ),
            "sources": [],
            "tokens_in": 0,
            "tokens_out": 0,
            "total_ms": round(
                (time.perf_counter() - start) * 1000, 2
            ),
            "scope_status": "out_of_scope",
            **empty_citation_result()
        }

    # Step 2: Retrieve evidence

    retrieved_passages = retrieve_paper_passages(
        query=question,
        matched_papers=matched_papers,
        top_k=5
    )

    if not retrieved_passages:

        return {
            "answer": (
                "I could not find relevant passages in "
                "the indexed research papers for this question."
            ),
            "sources": [],
            "tokens_in": 0,
            "tokens_out": 0,
            "total_ms": round(
                (time.perf_counter() - start) * 1000, 2
            ),
            "scope_status": "in_scope_no_evidence",
            **empty_citation_result()
        }

    # Step 3: Prepare evidence context

    context_parts = []

    for item in retrieved_passages:

        context_parts.append(
            f"Source: {item['source']} | "
            f"Chunk: {item['chunk_index']}\n"
            f"Content: {item['text']}"
        )

    context = "\n\n".join(context_parts)

    # Step 4: Prepare model messages

    messages = [
        {
            "role": "system",
            "content": (
                "You are a research paper assistant. "
                "Answer using only the supplied research passages. "
                "Do not invent facts or unsupported claims. "
                "If evidence for a requested paper is missing, "
                "clearly state that evidence was not retrieved. "
                "Do not substitute evidence from another paper. "
                "Cite relevant claims using exactly this format: "
                "【paper_filename | Chunk: number】. "
                "Only cite chunks explicitly present in the supplied "
                "passages. Never invent chunk numbers. "
                "Do not use a range unless every chunk in that "
                "range was supplied. "
                "For comparison questions, discuss each paper "
                "separately before comparing their approaches. "
                "Distinguish the paper's claims from your own "
                "interpretation."
            )
        },
        {
            "role": "user",
            "content": (
                f"Research question:\n{question}\n\n"
                f"Retrieved research passages:\n{context}"
            )
        }
    ]

    # Step 5: Generate answer using Groq

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_completion_tokens=2500
    )

    answer = response.choices[0].message.content or ""

    # Step 6: Collect token usage

    tokens_in = 0
    tokens_out = 0

    if response.usage:

        tokens_in = response.usage.prompt_tokens or 0
        tokens_out = response.usage.completion_tokens or 0

    # Step 7: Validate generated citations

    citation_result = validate_citations(
        answer,
        retrieved_passages
    )

    # Step 8: Collect unique source references

    unique_sources = []
    seen = set()

    for item in retrieved_passages:

        key = (
            item["source"],
            item["chunk_index"]
        )

        if key not in seen:

            seen.add(key)
            unique_sources.append(key)

    total_ms = (
        time.perf_counter() - start
    ) * 1000

    return {
        "answer": answer,
        "sources": unique_sources,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "total_ms": round(total_ms, 2),
        "scope_status": "in_scope",
        **citation_result
    }


# -----------------------------------
# 6. Interactive CLI
# -----------------------------------

def main():

    print("\n===================================")
    print("       THE RELAY - PAPER QA")
    print("===================================")

    print("Model:", MODEL)
    print("Type exit to quit.\n")

    while True:

        question = input(
            "\nEnter your research question: "
        ).strip()

        if question.lower() in ["exit", "quit"]:

            print("Closing The Relay.")
            break

        if not question:

            print("Please enter a valid question.")
            continue

        print(
            "\nChecking research scope and searching papers...\n"
        )

        try:

            result = answer_question(question)

            print("\n========== SCOPE STATUS ==========")
            print(result["scope_status"])

            print("\n========== ANSWER ==========\n")
            print(result["answer"])

            print("\n========== SOURCES ==========")

            for source, chunk_index in result["sources"]:

                print(
                    f"{source} | chunk {chunk_index}"
                )

            print("\n========== CITATION VALIDATION ==========")

            print(
                "Status:",
                result["citation_status"]
            )

            print(
                "Citations checked:",
                result["citations_found"]
            )

            print(
                "Valid citations:",
                len(result["valid_citations"])
            )

            print(
                "Invalid citations:",
                len(result["invalid_citations"])
            )

            if result["valid_citations"]:

                print("\nValid references:")

                for source, chunk in result["valid_citations"]:

                    print(f"  VALID: {source} | Chunk: {chunk}")

            if result["invalid_citations"]:

                print("\nInvalid references:")

                for source, chunk in result["invalid_citations"]:

                    print(f"  INVALID: {source} | Chunk: {chunk}")

            print("\n========== METRICS ==========")

            print(
                "Total latency:",
                result["total_ms"],
                "ms"
            )

            print(
                "Input tokens:",
                result["tokens_in"]
            )

            print(
                "Output tokens:",
                result["tokens_out"]
            )

        except Exception as error:

            print(
                "\nRequest failed:",
                type(error).__name__
            )

            print("Details:", str(error))

            print(
                "Please check the service and try again."
            )


if __name__ == "__main__":
    main()
