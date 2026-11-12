
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL = "openai/gpt-oss-20b"

print("Sending non-streaming request...")
start = time.perf_counter()

try:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "Explain BM25 retrieval in exactly two simple sentences."
            }
        ],
        temperature=0,
        max_completion_tokens=256
    )

    elapsed = (time.perf_counter() - start) * 1000

    message = response.choices[0].message

    print("\n--- MODEL RESPONSE ---")
    print("Content:", repr(message.content))
    print("Finish reason:", response.choices[0].finish_reason)

    print("\n--- TOKEN USAGE ---")
    print("Input tokens:", response.usage.prompt_tokens)
    print("Output tokens:", response.usage.completion_tokens)
    print("Total tokens:", response.usage.total_tokens)

    print("\n--- LATENCY ---")
    print("Total latency (ms):", round(elapsed, 2))

except Exception as error:
    print("\nRequest failed:", type(error).__name__)
    print("Error:", str(error))
