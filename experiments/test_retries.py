
import csv
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx

URL = "http://127.0.0.1:8001/generate"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_ROOT / "results" / "retry_results.csv"

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

TEST_MODES = ["rate_limit", "fail"]
MAX_RETRIES = 2


def run_strategy(mode, strategy):
    request_id = f"retry-{uuid.uuid4().hex[:8]}"
    start_time = datetime.now().astimezone().isoformat()
    start = time.perf_counter()

    retries = 0
    final_status = "error"

    with httpx.Client(timeout=10) as client:
        while True:
            try:
                response = client.post(
                    URL,
                    json={
                        "prompt": "Explain BM25 retrieval",
                        "mode": mode
                    }
                )

                if response.status_code == 200:
                    final_status = "ok"
                    break

                if response.status_code == 429:
                    final_status = "rate_limited"
                elif response.status_code == 503:
                    final_status = "error"
                else:
                    final_status = "error"
                    break

            except httpx.TimeoutException:
                final_status = "timeout"
            except httpx.RequestError:
                final_status = "error"

            if strategy == "no_retry":
                break

            if retries >= MAX_RETRIES:
                break

            retries += 1

            if strategy == "immediate":
                wait_seconds = 0
            else:
                wait_seconds = 0.5 * (2 ** (retries - 1))

            if wait_seconds:
                time.sleep(wait_seconds)

    total_ms = round((time.perf_counter() - start) * 1000, 2)

    row = {
        "request_id": request_id,
        "scenario": f"retry_{mode}_{strategy}",
        "source": "stand-in",
        "start_time": start_time,
        "first_token_ms": "",
        "total_ms": total_ms,
        "status": final_status,
        "retries": retries,
        "tokens_in": "",
        "tokens_out": "",
        "retrieval_setting": "none",
        "answer_grade": ""
    }

    print(
        f"{mode} | {strategy} | "
        f"{final_status} | retries={retries} | {total_ms} ms"
    )

    return row


def main():
    rows = []

    print("\nStarting retry comparison...")

    for mode in TEST_MODES:
        for strategy in ["no_retry", "immediate", "backoff"]:
            rows.append(run_strategy(mode, strategy))

    file_exists = (
        RESULTS_FILE.exists()
        and RESULTS_FILE.stat().st_size > 0
    )

    with open(
        RESULTS_FILE,
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)

    print("\nResults saved to:", RESULTS_FILE)
    print("Total experiments:", len(rows))


if __name__ == "__main__":
    main()
