"""
app/main.py
===========
FastAPI application with two endpoints:
  GET  /health  → readiness probe
  POST /chat    → conversational agent

Stateless: no session state stored server-side.
Every /chat call receives and returns the full conversation.
"""

import logging
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent import run_agent
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.retriever import retriever

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── lifespan: load index once at startup ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load FAISS index + embedding model before serving requests."""
    log.info("Starting up — loading retriever...")
    retriever.load()
    log.info("Startup complete ✅")
    yield
    log.info("Shutting down.")


# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "SHL Assessment Recommender",
    description = "Conversational agent for selecting SHL Individual Test Solutions",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request timing middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def log_timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - t0
    log.info("%s %s  %.2fs  %d", request.method, request.url.path, elapsed, response.status_code)
    return response


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """
    Readiness probe.
    Returns 200 {"status": "ok"} when the service is ready to serve.
    The evaluator allows up to 2 minutes on cold start before checking this.
    """
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Stateless conversational endpoint.

    Accepts the full conversation history on every call.
    Returns the agent's next reply plus optional assessment recommendations.

    - recommendations is [] when clarifying or refusing
    - recommendations has 1-10 items when the agent commits to a shortlist
    - end_of_conversation is true only when the task is complete
    """
    if not request.messages:
        raise HTTPException(status_code=422, detail="messages array cannot be empty")

    # validate last message is from user
    if request.messages[-1].role != "user":
        raise HTTPException(
            status_code=422,
            detail="Last message must be from 'user'",
        )

    try:
        response = run_agent(request)
        return response
    except Exception as e:
        log.exception("Unhandled error in /chat: %s", e)
        # return a graceful fallback instead of 500
        return ChatResponse(
            reply               = "I encountered an unexpected error. Please try again.",
            recommendations     = [],
            end_of_conversation = False,
        )


# ── dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host    = "0.0.0.0",
        port    = int(os.getenv("PORT", 8000)),
        reload  = True,
        workers = 1,  # keep 1 worker so FAISS index isn't duplicated
    )