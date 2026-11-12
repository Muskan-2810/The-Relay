
import csv
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx


# -----------------------------------
# 1. Configuration
# -----------------------------------

URL = "http://127.0.0.1:8001/stream"

PROMPT = "Explain BM25 retrieval in simple terms"

MODES = ["normal", "slow", "fail", "rate_limit"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_FILE = PROJECT_ROOT / "results" / "standin_results.csv"

CSV_COLUMNS = [
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


# -----------------------------------
# 2. Run one streaming experiment
# -----------------------------------

def run_experiment(client, mode):

    request_id = f"standin-{uuid.uuid4().hex[:8]}"
    start_time = datetime.now().astimezone().isoformat()

    payload = {
        "prompt": PROMPT,
        "mode": mode
    }

    start = time.perf_counter()
    first_token_ms = ""

    status = "error"
    http_status = "N/A"

    try:

        with client.stream("POST", URL, json=payload) as response:

            http_status = response.status_code

            if response.status_code == 200:

                status = "ok"

                for chunk in response.iter_text():

                    if chunk and first_token_ms == "":
                        first_token_ms = round(
                            (time.perf_counter() - start) * 1000,
                            2
                        )

            elif response.status_code == 429:
                status = "rate_limited"
                response.read()

            elif response.status_code == 408:
                status = "timeout"
                response.read()

            else:
                status = "error"
                response.read()

    except httpx.TimeoutException:
        status = "timeout"

    except httpx.RequestError as error:
        print("Request error:", error)
        status = "error"

    total_ms = round(
        (time.perf_counter() - start) * 1000,
        2
    )

    row = {
        "request_id": request_id,
        "scenario": f"standin_stream_{mode}",
        "source": "stand-in",
        "start_time": start_time,
        "first_token_ms": first_token_ms,
        "total_ms": total_ms,
        "status": status,
        "retries": 0,
        "tokens_in": "",
        "tokens_out": "",
        "retrieval_setting": "none",
        "answer_grade": ""
    }

    print(
        f"Mode: {mode} | "
        f"HTTP: {http_status} | "
        f"TTFT: {first_token_ms} ms | "
        f"Total: {total_ms} ms | "
        f"Status: {status}"
    )

    return row


# -----------------------------------
# 3. Save CSV
# -----------------------------------

def save_results(rows):

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    file_exists = (
        RESULTS_FILE.exists()
        and RESULTS_FILE.stat().st_size > 0
    )

    with open(
        RESULTS_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)

    print("\nResults saved to:", RESULTS_FILE)


# -----------------------------------
# 4. Main
# -----------------------------------

def main():

    print("Starting streaming experiments...\n")

    results = []

    with httpx.Client(timeout=30) as client:

        for mode in MODES:

            row = run_experiment(client, mode)
            results.append(row)

    print("\n--- STREAMING SUMMARY ---")

    for row in results:

        print(
            row["scenario"],
            "| TTFT:",
            row["first_token_ms"],
            "| Total:",
            row["total_ms"],
            "| Status:",
            row["status"]
        )

    save_results(results)


if __name__ == "__main__":
    main()
