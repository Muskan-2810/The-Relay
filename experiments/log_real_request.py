
import os
import csv
import time
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from groq import Groq


# -----------------------------------
# 1. Project paths and configuration
# -----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_FILE = PROJECT_ROOT / ".env"
OUTPUT_FILE = PROJECT_ROOT / "results" / "real_results.csv"

load_dotenv(ENV_FILE)

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found. Check your .env file."
    )

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
# 2. Generate unique request ID
# -----------------------------------

def get_next_request_id():

    existing_numbers = []

    if OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:

        with open(
            OUTPUT_FILE,
            "r",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                request_id = (row.get("request_id") or "").strip()

                if request_id.startswith("real-"):

                    try:
                        number = int(request_id.split("-")[1])
                        existing_numbers.append(number)

                    except (ValueError, IndexError):
                        continue

    next_number = max(existing_numbers, default=0) + 1

    return f"real-{next_number:03d}"


# -----------------------------------
# 3. Run one real API request
# -----------------------------------

def run_experiment():

    request_id = get_next_request_id()

    scenario = "real_smoke_non_streaming"

    start_time = datetime.now(timezone.utc).isoformat()

    status = "error"

    elapsed = 0
    tokens_in = 0
    tokens_out = 0
    first_token_ms = ""

    response_text = ""
    finish_reason = ""

    start = time.perf_counter()

    try:

        response = client.chat.completions.create(

            model=MODEL,

            messages=[
                {
                    "role": "user",
                    "content": (
                        "Explain BM25 retrieval in simple terms. "
                        "Describe how term frequency and inverse "
                        "document frequency help rank documents. "
                        "Use approximately three sentences."
                    )
                }
            ],

            temperature=0,

            max_completion_tokens=256
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        response_text = (
            response.choices[0].message.content or ""
        )

        finish_reason = response.choices[0].finish_reason

        if response.usage:

            tokens_in = response.usage.prompt_tokens or 0

            tokens_out = response.usage.completion_tokens or 0

        # Do not mark incomplete responses as successful.

        if finish_reason == "stop" and response_text.strip():

            status = "ok"

        else:

            status = "error"

        print("\nModel response:")
        print(response_text)

        print("\nFinish reason:", finish_reason)

        if finish_reason == "length":

            print(
                "WARNING: Response reached the token limit "
                "and may be incomplete."
            )

        elif not response_text.strip():

            print("WARNING: Model returned empty text.")

    except Exception as error:

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        status = "error"

        # Avoid printing credentials or request headers.
        print("\nRequest failed:", type(error).__name__)
        print("Error details:", str(error))

    # -----------------------------------
    # 4. Prepare CSV row
    # -----------------------------------

    row = {

        "request_id": request_id,

        "scenario": scenario,

        "source": "real",

        "start_time": start_time,

        # Non-streaming request:
        # First-token latency cannot be measured separately.
        "first_token_ms": first_token_ms,

        "total_ms": round(elapsed, 2),

        "status": status,

        "retries": 0,

        "tokens_in": tokens_in,

        "tokens_out": tokens_out,

        "retrieval_setting": "none",

        "answer_grade": ""
    }

    # -----------------------------------
    # 5. Save raw measurement
    # -----------------------------------

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
    # 6. Display saved measurement
    # -----------------------------------

    print("\n--- SAVED MEASUREMENT ---")

    for key, value in row.items():

        print(f"{key}: {value}")

    print("\nFinish reason:", finish_reason)

    print("\nSaved to:", OUTPUT_FILE)


# -----------------------------------
# 7. Main
# -----------------------------------

if __name__ == "__main__":

    run_experiment()
