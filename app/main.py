"""REST API for the support assistant.

Endpoints:
  GET  /          the chat page
  GET  /health    is the service up, and how many cards are loaded
  POST /chat      send a customer message, get the agent's decision and answer
  POST /feedback  thumbs up / down on a reply, saved with the decision behind it (JSONL file)
  GET  /feedback/stats  thumbs up / down per decision type

Interactive docs (try the API in the browser): http://localhost:8000/docs
"""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import AgentResponse, SupportAgent
from app.config import settings
from app.embeddings import FastEmbedEmbedder
from app.knowledge_base import load_knowledge_base
from app.llm import OllamaClient
from app.reranker import CrossEncoderReranker
from app.vector_store import VectorStore, make_client

logger = logging.getLogger("support-assistant")


# ---- request and response shapes -------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000, examples=["I forgot my password"])
    previous_message: str | None = Field(
        default=None,
        max_length=1000,
        description="The customer's earlier message, when they reply to a clarifying question.",
    )
    mode: Literal["llm", "rules"] | None = Field(
        default=None, description="Override the decision mode for this message (for comparison)."
    )


class SourceCard(BaseModel):
    id: str
    category: str
    question: str
    answer: str


class OptionOut(BaseModel):
    card_id: str
    label: str


class MatchOut(BaseModel):
    card_id: str
    question: str
    score: float
    rerank_score: float | None = None  # only when a re-ranking model is used
    out_of_scope: bool = False


class ChatResponse(BaseModel):
    id: str  # used to send feedback about this answer
    action: Literal["answer", "clarify", "decline"]
    message: str
    reason: str
    decided_by: Literal["llm", "guardrail", "rules"]
    confidence: float
    source: SourceCard | None = None
    options: list[OptionOut] = []
    matches: list[MatchOut] = []
    search_query: str | None = None  # the rewritten search, if the first search was weak


class MatchIn(BaseModel):
    card_id: str = Field(max_length=20)
    score: float


class FeedbackRequest(BaseModel):
    response_id: str = Field(max_length=64)
    rating: Literal["up", "down"]
    question: str = Field(max_length=1000)
    answer: str = Field(max_length=5000)
    comment: str | None = Field(default=None, max_length=1000)
    # The decision behind the reply. With these fields, a feedback record can later become
    # a test case or a training example (see scripts/export_feedback.py).
    previous_message: str | None = Field(default=None, max_length=1000)
    action: Literal["answer", "clarify", "decline"] | None = None
    source_card: str | None = Field(default=None, max_length=20)
    options: list[str] = Field(default=[], max_length=10)  # cards offered in a clarifying question
    decided_by: Literal["llm", "guardrail", "rules"] | None = None
    reason: str | None = Field(default=None, max_length=2000)
    search_query: str | None = Field(default=None, max_length=1000)
    matches: list[MatchIn] = Field(default=[], max_length=10)


# ---- building the agent once at startup --------------------------------------------

def index_with_retry(store: VectorStore, entries, attempts: int = 10, wait_seconds: float = 3.0) -> None:
    """In Docker, Qdrant may still be starting when the API starts. Wait instead of crashing."""
    for attempt in range(1, attempts + 1):
        try:
            store.index(entries)
            return
        except Exception as error:  # connection refused, timeout, ...
            if attempt == attempts:
                raise
            logger.warning("Qdrant not ready (attempt %d/%d): %s", attempt, attempts, error)
            time.sleep(wait_seconds)


def build_agent() -> SupportAgent:
    entries = load_knowledge_base(settings.knowledge_base_path)
    store = VectorStore(
        make_client(settings.qdrant_url),
        FastEmbedEmbedder(settings.embedding_model),
        settings.qdrant_collection,
    )
    index_with_retry(store, entries)
    llm = OllamaClient(settings.ollama_url, settings.ollama_model, settings.llm_timeout_seconds)
    reranker = CrossEncoderReranker(settings.reranker_model) if settings.reranker_model else None
    return SupportAgent(
        store,
        llm,
        min_relevance=settings.min_relevance,
        ambiguity_gap=settings.ambiguity_gap,
        second_check_gap=settings.second_check_gap,
        rewrite_weak_queries=settings.rewrite_weak_queries,
        reranker=reranker,
        min_rerank=settings.min_rerank,
        mode=settings.agent_mode,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model and index the knowledge base once, not on every request.
    if not hasattr(app.state, "agent"):
        app.state.agent = build_agent()
    yield


app = FastAPI(title="Agentic Customer Support Assistant", version="1.0", lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def get_agent(request: Request) -> SupportAgent:
    return request.app.state.agent


# ---- endpoints ------------------------------------------------------------------

def to_chat_response(result: AgentResponse) -> ChatResponse:
    source = None
    if result.source:
        entry = next(m.entry for m in result.matches if m.entry.id == result.source)
        source = SourceCard(id=entry.id, category=entry.category, question=entry.question, answer=entry.answer)
    return ChatResponse(
        id=uuid.uuid4().hex,
        action=result.action.value,
        message=result.message,
        reason=result.reason,
        decided_by=result.decided_by,
        confidence=round(result.confidence, 3),
        source=source,
        options=[OptionOut(card_id=o.card_id, label=o.label) for o in result.options],
        search_query=result.search_query,
        matches=[
            MatchOut(
                card_id=m.entry.id,
                question=m.entry.question,
                score=round(m.score, 3),
                rerank_score=round(result.rerank[m.entry.id], 3) if m.entry.id in result.rerank else None,
                out_of_scope=m.entry.out_of_scope,
            )
            for m in result.matches
        ],
    )


@app.get("/", include_in_schema=False)
def chat_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(agent: SupportAgent = Depends(get_agent)) -> dict:
    cards = agent.store.client.count(agent.store.collection).count
    llm_status = agent.llm.status() if hasattr(agent.llm, "status") else "unknown"
    return {
        "status": "ok",
        "cards": cards,
        "llm": llm_status,  # "ok", "model_missing" or "unreachable"
        "model": settings.ollama_model,
        "reranker": settings.reranker_model or None,
        "mode": agent.mode,
        "min_relevance": agent.min_relevance,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, agent: SupportAgent = Depends(get_agent)) -> ChatResponse:
    result = agent.respond(request.message, previous_message=request.previous_message, mode=request.mode)
    response = to_chat_response(result)
    logger.info(
        "chat id=%s action=%s decided_by=%s source=%s confidence=%.3f",
        response.id, response.action, response.decided_by, response.source and response.source.id,
        response.confidence,
    )
    return response


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> dict:
    """Save thumbs up/down together with the decision behind the reply.

    Like the clinician accept/reject step in ACIE, these judgments first measure quality
    on real questions. scripts/export_feedback.py turns thumbs-down replies into cases to
    review and thumbs-up decisions into training examples for fine-tuning later.
    The file holds customer messages, so it stays out of git (see .gitignore).
    """
    path = Path(settings.feedback_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "model": settings.ollama_model,
        "agent_mode": settings.agent_mode,
        **request.model_dump(),
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "saved"}


@app.get("/feedback/stats")
def feedback_stats() -> dict:
    """Thumbs up and down per decision type: a first, simple quality monitor."""
    ratings = {"up": 0, "down": 0}
    by_action: dict[str, dict[str, int]] = {}
    path = Path(settings.feedback_path)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                rating = record["rating"]
                ratings[rating] += 1
            except (ValueError, KeyError):
                continue  # skip a damaged line instead of failing
            action = record.get("action") or "unknown"
            by_action.setdefault(action, {"up": 0, "down": 0})[rating] += 1
    return {"total": sum(ratings.values()), "ratings": ratings, "by_action": by_action}
