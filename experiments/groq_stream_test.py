
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

start = time.perf_counter()
first_event_ms = None
first_text_ms = None
response_parts = []

print("Starting streaming test...")

try:
    stream = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": "Explain BM25 retrieval in two simple sentences."
            }
        ],
        temperature=0,
        max_completion_tokens=256,
        stream=True
    )

    for chunk in stream:

        elapsed = (time.perf_counter() - start) * 1000

        if first_event_ms is None:
            first_event_ms = elapsed

        if chunk.choices:
            delta = chunk.choices[0].delta

            content = delta.content

            if content:
                if first_text_ms is None:
                    first_text_ms = elapsed

                response_parts.append(content)
                print(content, end="", flush=True)

    total_ms = (time.perf_counter() - start) * 1000

    print("\n\n--- STREAMING MEASUREMENTS ---")
    print("First event (ms):", round(first_event_ms, 2) if first_event_ms is not None else "Not received")
    print("First visible text (ms):", round(first_text_ms, 2) if first_text_ms is not None else "Not received")
    print("Total latency (ms):", round(total_ms, 2))
    print("Visible response received:", bool(response_parts))
    print("Request status:", "OK" if response_parts else "EMPTY_RESPONSE")

except Exception as error:
    print("\nRequest failed:", type(error).__name__)
    print("Error:", str(error))
