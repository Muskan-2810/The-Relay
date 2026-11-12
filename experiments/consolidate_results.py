
import csv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

INPUT_FILES = [
    "real_results.csv",
    "standin_results.csv",
    "retry_results.csv"
]

OUTPUT_FILE = RESULTS / "final_results.csv"

COLUMNS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade"
]


def normalize_source(source):
    value = source.strip().lower()

    if value in ("standin", "stand-in"):
        return "stand-in"

    if value == "real":
        return "real"

    return value


def main():
    all_rows = []
    seen_ids = set()

    for filename in INPUT_FILES:
        path = RESULTS / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing input file: {path}"
            )

        with open(
            path,
            encoding="utf-8-sig",
            newline=""
        ) as file:
            reader = csv.DictReader(file)

            if reader.fieldnames != COLUMNS:
                raise ValueError(
                    f"Unexpected columns in {filename}: "
                    f"{reader.fieldnames}"
                )

            count = 0

            for row in reader:
                request_id = row["request_id"].strip()

                if not request_id:
                    raise ValueError(
                        f"Empty request_id in {filename}"
                    )

                if request_id in seen_ids:
                    raise ValueError(
                        f"Duplicate request_id: {request_id}"
                    )

                seen_ids.add(request_id)

                row["source"] = normalize_source(
                    row["source"]
                )

                all_rows.append(row)
                count += 1

            print(f"{filename}: {count} records")

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=COLUMNS
        )

        writer.writeheader()
        writer.writerows(all_rows)

    print("\nCONSOLIDATION COMPLETE")
    print("Output:", OUTPUT_FILE)
    print("Total records:", len(all_rows))
    print("Sources:", dict(Counter(
        row["source"] for row in all_rows
    )))
    print("Statuses:", dict(Counter(
        row["status"] for row in all_rows
    )))
    print("Unique request IDs:", len(seen_ids))
    print("Original CSV files were not modified.")


if __name__ == "__main__":
    main()
