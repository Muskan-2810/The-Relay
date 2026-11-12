
import csv
import time
import uuid
import threading
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx


# -----------------------------------
# 1. Configuration
# -----------------------------------

URL = "http://127.0.0.1:8001/generate"

PROMPT = "Explain BM25 retrieval in simple terms"

CONCURRENT_REQUESTS = 4

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
# 2. Concurrency tracking
# -----------------------------------

active_requests = 0
peak_in_flight = 0

counter_lock = threading.Lock()


def request_started():

    global active_requests, peak_in_flight

    with counter_lock:

        active_requests += 1

        peak_in_flight = max(
            peak_in_flight,
            active_requests
        )


def request_finished():

    global active_requests

    with counter_lock:
        active_requests -= 1


# -----------------------------------
# 3. Run one request
# -----------------------------------

def run_one_request():

    request_id = f"standin-{uuid.uuid4().hex[:8]}"

    start_time = datetime.now().astimezone().isoformat()

    start = time.perf_counter()

    status = "error"
    http_status = "N/A"

    payload = {
        "prompt": PROMPT,
        "mode": "normal"
    }

    request_started()

    try:

        with httpx.Client(timeout=15) as client:

            response = client.post(
                URL,
                json=payload
            )

            http_status = response.status_code

            if response.status_code == 200:
                status = "ok"

            elif response.status_code == 429:
                status = "rate_limited"

            elif response.status_code == 408:
                status = "timeout"

            else:
                status = "error"

    except httpx.TimeoutException:

        status = "timeout"

    except httpx.RequestError as error:

        print("Request error:", error)

        status = "error"

    finally:

        request_finished()

    total_ms = round(
        (time.perf_counter() - start) * 1000,
        2
    )

    row = {

        "request_id": request_id,

        "scenario": "standin_concurrency_v2",

        "source": "stand-in",

        "start_time": start_time,

        "first_token_ms": "",

        "total_ms": total_ms,

        "status": status,

        "retries": 0,

        "tokens_in": "",

        "tokens_out": "",

        "retrieval_setting": "none",

        "answer_grade": ""

    }

    print(
        f"{request_id} | "
        f"HTTP {http_status} | "
        f"{status} | "
        f"{total_ms} ms"
    )

    return row


# -----------------------------------
# 4. Save results without overwriting
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
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)

    print("\nResults appended to:", RESULTS_FILE)


# -----------------------------------
# 5. Concurrent experiment
# -----------------------------------

def main():

    global peak_in_flight

    print("\nStarting concurrent stand-in experiment...")

    print(
        "Number of simultaneous workers:",
        CONCURRENT_REQUESTS
    )

    results = []

    experiment_start = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=CONCURRENT_REQUESTS
    ) as executor:

        futures = [
            executor.submit(run_one_request)
            for _ in range(CONCURRENT_REQUESTS)
        ]

        for future in as_completed(futures):

            results.append(future.result())

    wall_time_ms = round(
        (time.perf_counter() - experiment_start) * 1000,
        2
    )

    save_results(results)

    successful = sum(
        row["status"] == "ok"
        for row in results
    )

    failed = len(results) - successful

    print("\n========== EXPERIMENT SUMMARY ==========")

    print("Total requests:", len(results))

    print("Successful:", successful)

    print("Failed:", failed)

    print("Peak client-side in-flight:", peak_in_flight)

    print("Total wall-clock time:", wall_time_ms, "ms")

    print("\nIndividual request latencies:")

    for row in sorted(results, key=lambda r: float(r["total_ms"])):

        print(
            row["request_id"],
            "|",
            row["total_ms"],
            "ms |",
            row["status"]
        )

    print("\nExperiment completed.")


if __name__ == "__main__":

    main()
