
from pathlib import Path
import json

# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FOLDER = PROJECT_ROOT / "data" / "extracted_papers"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "chunks"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Chunking configuration
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def create_chunks(text, chunk_size, overlap):
    chunks = []

    start = 0
    chunk_index = 0

    while start < len(text):

        end = start + chunk_size

        chunk_text = text[start:end]

        chunks.append({
            "chunk_index": chunk_index,
            "start_char": start,
            "end_char": min(end, len(text)),
            "text": chunk_text
        })

        chunk_index += 1

        # Move forward, keeping some text from previous chunk
        start = end - overlap

    return chunks


def process_all_papers():

    all_chunks = []

    text_files = sorted(INPUT_FOLDER.glob("*.txt"))

    if not text_files:
        print("No extracted text files found!")
        return

    for file_path in text_files:

        print("Chunking:", file_path.name)

        text = file_path.read_text(encoding="utf-8")

        paper_chunks = create_chunks(
            text,
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )

        for chunk in paper_chunks:
            chunk["source"] = file_path.stem

        all_chunks.extend(paper_chunks)

        print("Chunks created:", len(paper_chunks))
        print("--------------------------------")

    output_path = OUTPUT_FOLDER / "chunks.json"

    output_path.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("Total chunks:", len(all_chunks))
    print("Saved to:", output_path)


if __name__ == "__main__":
    process_all_papers()
