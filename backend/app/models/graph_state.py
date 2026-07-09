# models/graph_state.py
#
# The state object that flows through every node in the LangGraph RAG graph.
#
# WHY TypedDict and not a Pydantic model or dataclass:
# LangGraph requires state to be a TypedDict. This is a LangGraph convention —
# each node receives the full state dict and returns a PARTIAL update dict.
# LangGraph merges the partial update into the full state automatically.
#
# HOW IT WORKS:
#   Node 1 (retrieve) returns:  {"retrieved_chunks": [...]}
#   Node 2 (generate) returns:  {"answer_draft": "...", "citations": [...]}
#   Node 3 (critic)   returns:  {"staleness_flags": [...], "confidence": 0.8}
#   LangGraph merges all of these into one running state object.
#
# Every node receives the FULL state (all fields) and only writes
# back the fields it's responsible for. Clean separation of concerns.

from typing import TypedDict, Annotated
from app.models.chunk import ChunkResult, Citation, StalenessFlag
import operator


class GraphState(TypedDict):
    """
    Shared state passed between all nodes in the RAG graph.
    Each field is written by exactly one node — documented below.
    """

    # ── Set by the API route BEFORE the graph starts ──────────────────────
    query:    str    # the original user question — never modified by nodes
    repo_id:  str    # scopes all pgvector searches to this repo

    # ── Written by: retrieve node ─────────────────────────────────────────
    # Top-50 chunks from pgvector similarity search.
    # On retry, this is overwritten with a new search using refined_query.
    retrieved_chunks: list[ChunkResult]

    # ── Written by: generate node ─────────────────────────────────────────
    # The LLM's answer before the critic reviews it.
    answer_draft: str

    # Citations built from the top-8 chunk metadata after reranking.
    # Passed to the critic so it knows which sources were used.
    citations: list[Citation]

    # ── Written by: critic node ───────────────────────────────────────────
    # List of staleness issues found. Empty list = answer is clean.
    staleness_flags: list[StalenessFlag]

    # 0.0 to 1.0. Drives the conditional edge:
    #   >= 0.7 → finalize (answer is good enough)
    #   <  0.7 → retry retrieve with refined_query
    confidence: float

    # Rewritten query used on retry.
    # The critic sets this when it knows WHY the retrieval was bad.
    # e.g. "closed issue detected — add 'current behavior' to query"
    refined_query: str | None

    # ── Written by: conditional edge logic ────────────────────────────────
    # Incremented each time we loop back to retrieve.
    # Max value is 2 — after that we finalize regardless of confidence.
    # Prevents infinite retry loops.
    retry_count: int

    # ── Written by: finalize node ─────────────────────────────────────────
    # The final answer sent to the user. Includes inline staleness warnings
    # if the critic fired but we hit max retries.
    final_answer: str | None

    # Total tokens used by the LLM call — logged for billing.
    tokens_used: int


def initial_state(query: str, repo_id: str) -> GraphState:
    """
    Creates a fresh GraphState for a new query.
    Call this in the chat route before invoking the graph.

    Usage:
        state = initial_state(question, repo_id)
        result = await graph.ainvoke(state)
    """
    return GraphState(
        query=query,
        repo_id=repo_id,
        retrieved_chunks=[],
        answer_draft="",
        citations=[],
        staleness_flags=[],
        confidence=0.0,
        refined_query=None,
        retry_count=0,
        final_answer=None,
        tokens_used=0,
    )
