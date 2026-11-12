
import asyncio
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse


# -----------------------------------
# 1. Application
# -----------------------------------

app = FastAPI(
    title="The Relay - Stand-in Model Service",
    description="A local model service for latency and failure experiments",
    version="1.1"
)


# -----------------------------------
# 2. Request model
# -----------------------------------

class ModelRequest(BaseModel):
    prompt: str
    mode: str = "normal"


# -----------------------------------
# 3. Service configuration
# -----------------------------------

in_flight = 0

MAX_CONCURRENT_REQUESTS = 2

BASE_DELAY = 0.25

SLOW_DELAY = 2.0

CHUNK_DELAY = 0.1

VALID_MODES = ["normal", "slow", "fail", "rate_limit"]


# -----------------------------------
# 4. Request validation
# -----------------------------------

def validate_request(request):

    if not request.prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Prompt cannot be empty"
        )

    if request.mode not in VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail="Unknown simulation mode"
        )

    if request.mode == "rate_limit":
        raise HTTPException(
            status_code=429,
            detail="Simulated rate limit exceeded"
        )

    if request.mode == "fail":
        raise HTTPException(
            status_code=503,
            detail="Simulated model service unavailable"
        )


# -----------------------------------
# 5. Calculate simulated delay
# -----------------------------------

def calculate_delay(mode):

    overload = max(
        0,
        in_flight - MAX_CONCURRENT_REQUESTS
    )

    delay = BASE_DELAY + (overload * 0.5)

    if mode == "slow":
        delay += SLOW_DELAY

    return delay


# -----------------------------------
# 6. Health endpoint
# -----------------------------------

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "service": "standin-model",
        "in_flight": in_flight
    }


# -----------------------------------
# 7. Generate endpoint
# -----------------------------------

@app.post("/generate")
async def generate(request: ModelRequest):

    global in_flight

    validate_request(request)

    in_flight += 1

    start = time.perf_counter()

    try:

        delay = calculate_delay(request.mode)

        await asyncio.sleep(delay)

        answer = (
            "This is a simulated response from The Relay "
            "stand-in model service. "
            "The service is used to study latency, "
            "concurrency and failure behaviour."
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        return {
            "answer": answer,
            "mode": request.mode,
            "simulated_delay_ms": round(delay * 1000, 2),
            "total_ms": round(elapsed, 2),
            "in_flight": in_flight,
            "status": "ok"
        }

    finally:

        in_flight -= 1


# -----------------------------------
# 8. Streaming endpoint
# -----------------------------------

@app.post("/stream")
async def stream(request: ModelRequest):

    global in_flight

    validate_request(request)

    answer = (
        "The Relay is a local stand-in model service. "
        "It helps us measure response latency, "
        "streaming behaviour and simulated failures."
    )

    async def generate_chunks():

        global in_flight

        in_flight += 1

        try:

            # Apply initial service delay

            delay = calculate_delay(request.mode)

            await asyncio.sleep(delay)

            words = answer.split()

            for word in words:

                yield word + " "

                await asyncio.sleep(CHUNK_DELAY)

        finally:

            in_flight -= 1

    return StreamingResponse(
        generate_chunks(),
        media_type="text/plain"
    )
