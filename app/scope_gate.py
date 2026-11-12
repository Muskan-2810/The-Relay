
import re


PAPER_SCOPE_TERMS = {
    "moshi_v1": [
        "moshi",
        "mimi",
        "helium",
        "speech",
        "spoken",
        "audio",
        "dialogue",
        "full duplex",
        "codec",
        "audio tokens",
    ],

    "attention_v7": [
    "self attention", "self-attention",
    "attention mechanism", "attention mechanisms",
    "transformer", "transformers",
    "multi head", "queries", "keys", "values",
    "positional encoding", "encoder", "decoder",
    "recurrent layers",
],
    "dynamic_chunking_v2": [
        "dynamic chunking",
        "chunk boundaries",
        "fixed size chunk",
        "hierarchical sequence",
        "sequence modeling",
        "sequence segmentation",
        "h-net",
        "variable chunks",
        "tokenizer free",
    ],
}


def detect_research_scope(question):
    if not isinstance(question, str) or not question.strip():
        return {
            "in_scope": False,
            "matched_papers": [],
            "matched_terms": {},
        }

    normalized = re.sub(r"[^a-z0-9]+", " ", question.lower())

    matched_terms = {}

    for paper, terms in PAPER_SCOPE_TERMS.items():
        matches = []

        for term in terms:
            normalized_term = re.sub(
                r"[^a-z0-9]+", " ", term.lower()
            )

            pattern = r"\b" + re.escape(normalized_term) + r"\b"

            if re.search(pattern, normalized):
                matches.append(term)

        if matches:
            matched_terms[paper] = matches

    return {
        "in_scope": bool(matched_terms),
        "matched_papers": list(matched_terms.keys()),
        "matched_terms": matched_terms,
    }
