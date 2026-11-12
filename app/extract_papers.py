
import pymupdf
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PAPERS_FOLDER = PROJECT_ROOT / "papers"

OUTPUT_FOLDER = PROJECT_ROOT / "data" / "extracted_papers"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


def extract_text_from_pdf(pdf_path):

    document = pymupdf.open(pdf_path)

    extracted_text = ""

    for page_number, page in enumerate(document):

        page_text = page.get_text("text")

        extracted_text += page_text

    document.close()

    return extracted_text


def process_all_papers():

    pdf_files = sorted(PAPERS_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found!")
        return

    for pdf_path in pdf_files:

        print("Processing:", pdf_path.name)

        extracted_text = extract_text_from_pdf(pdf_path)

        output_path = OUTPUT_FOLDER / f"{pdf_path.stem}.txt"

        output_path.write_text(extracted_text, encoding="utf-8")

        print("Saved:", output_path.name)
        print("Characters extracted:", len(extracted_text))
        print("--------------------------------")


if __name__ == "__main__":
    process_all_papers()
