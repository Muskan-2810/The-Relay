
import os
import csv
import time
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from groq import Groq


# -----------------------------------
# 1. Configuration
# -----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_FILE = PROJECT_ROOT / "results" / "real_results.csv"

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env")

client = Groq(api_key=API_KEY)

MODEL = "openai/gpt-oss-20b"

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


# -----------------------------------
# 2. Unique request ID
# -----------------------------------

def get_next_request_id():

    if not OUTPUT_FILE.exists():
        return "real-001"

    with open(
        OUTPUT_FILE,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        rows = list(csv.DictReader(file))

    numbers = []

    for row in rows:

        request_id = row.get("request_id", "")

        if request_id.startswith("real-"):

            try:
                numbers.append(
                    int(request_id.split("-")[1])
                )
            except (ValueError, IndexError):
                pass

    return f"real-{max(numbers, default=0) + 1:03d}"


# -----------------------------------
# 3. Streaming experiment
# -----------------------------------

def run_streaming_experiment():

    request_id = get_next_request_id()

    start_time = datetime.now(timezone.utc).isoformat()

    first_token_ms = ""
    total_ms = 0

    response_parts = []

    status = "error"
    finish_reason = ""

    start = time.perf_counter()

    print("Starting streaming request...\n")

    try:

        stream = client.chat.completions.create(

            model=MODEL,

            messages=[
                {
                    "role": "user",
                    "content": (
                        "Explain BM25 retrieval in simple terms. "
                        "Describe term frequency and inverse "
                        "document frequency, and explain how "
                        "they help rank search results. "
                        "Use approximately three sentences."
                    )
                }
            ],

            temperature=0,

            max_completion_tokens=256,

            stream=True
        )

        for chunk in stream:

            elapsed = (
                time.perf_counter() - start
            ) * 1000

            if chunk.choices:

                choice = chunk.choices[0]

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                delta = choice.delta

                content = delta.content

                if content:

                    if first_token_ms == "":
                        first_token_ms = round(elapsed, 2)

                    response_parts.append(content)

                    print(content, end="", flush=True)

        total_ms = (
            time.perf_counter() - start
        ) * 1000

        response_text = "".join(response_parts)

        if response_text.strip() and finish_reason == "stop":

            status = "ok"

        else:

            status = "error"

        print("\n\n--- STREAMING COMPLETE ---")

        print("Finish reason:", finish_reason)

        print("Visible response received:", bool(response_text.strip()))

    except Exception as error:

        total_ms = (
            time.perf_counter() - start
        ) * 1000

        print("\nStreaming request failed:", type(error).__name__)

        print("Error details:", str(error))

    # -----------------------------------
    # 4. Save measurements
    # -----------------------------------

    row = {

        "request_id": request_id,

        "scenario": "real_smoke_streaming",

        "source": "real",

        "start_time": start_time,

        "first_token_ms": first_token_ms,

        "total_ms": round(total_ms, 2),

        "status": status,

        "retries": 0,

        # Streaming usage is not assumed.
        # Leave blank because stream_options
        # is unsupported by our installed SDK.
        "tokens_in": "",

        "tokens_out": "",

        "retrieval_setting": "none",

        "answer_grade": ""
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    file_exists = OUTPUT_FILE.exists()

    with open(
        OUTPUT_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=COLUMNS
        )

        if not file_exists or OUTPUT_FILE.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(row)

    # -----------------------------------
    # 5. Display measurements
    # -----------------------------------

    print("\n--- SAVED STREAMING MEASUREMENT ---")

    for key, value in row.items():

        print(f"{key}: {value}")

    if first_token_ms != "":
        print(
            "\nTime from first visible text to completion:",
            round(total_ms - first_token_ms, 2),
            "ms"
        )

    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":

    run_streaming_experiment()
