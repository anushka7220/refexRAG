# api/routes/chat.py
#
# Chat endpoints. The most complex routes in the entire backend.
#
# WHAT MAKES THIS COMPLEX:
# Every other route returns JSON. This one opens a long-lived HTTP
# connection and streams tokens to the frontend as the LLM generates them.
# It also invokes the full LangGraph pipeline and persists the result
# to the database after streaming completes, all within one request.
#
# SSE STREAMING PATTERN:
#   1. Frontend calls POST /sessions/{id}/chat
#   2. We return an EventSourceResponse immediately (opens the stream)
#   3. Inside the async generator we invoke the LangGraph graph
#   4. As LangGraph completes each node, we yield SSE events
#   5. After all events are yielded, we save the message to the DB
#   6. We yield the done event and the generator exits
#   7. The HTTP connection closes

import uuid
import structlog
from fastapi import APIRouter, HTTPException, Depends, status
from sse_starlette.sse import EventSourceResponse
from typing import AsyncGenerator

from app.core.dependencies import get_current_user
from app.core.supabase import supabase_admin, execute
from app.models.user import UserProfile
from app.models.message import ChatRequest, MessageResponse, ChatSessionResponse, MessageRecord
from app.models.graph_state import initial_state
from app.services.rag.graph import rag_graph
from app.services.rag.cache import semantic_cache
from app.utils.sse import stream_rag_response

log = structlog.get_logger(__name__)

router = APIRouter()


# POST /repos/{repo_id}/sessions — create a chat session

@router.post(
    "/repos/{repo_id}/sessions",
    response_model=ChatSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    repo_id: str,
    current_user: UserProfile = Depends(get_current_user),
):
    """
    Creates a new chat session for a repo.
    A session groups related messages together under one conversation.
    The frontend creates a session before sending the first message.
    """
    _assert_user_has_repo_access(current_user.id, repo_id)

    session_id = str(uuid.uuid4())
    rows = execute(
        supabase_admin.table("chat_sessions").insert({
            "id":      session_id,
            "user_id": current_user.id,
            "repo_id": repo_id,
            "title":   None,
        }).execute()
    )

    log.info("session_created", session_id=session_id, repo_id=repo_id)
    return ChatSessionResponse(**rows[0])


# GET /repos/{repo_id}/sessions — list sessions

@router.get("/repos/{repo_id}/sessions", response_model=list[ChatSessionResponse])
async def list_sessions(
    repo_id: str,
    current_user: UserProfile = Depends(get_current_user),
):
    """Returns all chat sessions the current user has for a repo."""
    _assert_user_has_repo_access(current_user.id, repo_id)

    rows = execute(
        supabase_admin.table("chat_sessions")
        .select("*")
        .eq("user_id", current_user.id)
        .eq("repo_id", repo_id)
        .order("created_at", desc=True)
        .execute()
    )

    return [ChatSessionResponse(**r) for r in rows]


# GET /sessions/{session_id}/messages — message history

@router.get("/sessions/{session_id}/messages", response_model=list[MessageResponse])
async def get_messages(
    session_id: str,
    current_user: UserProfile = Depends(get_current_user),
):
    """Returns the full message history for a session."""
    _assert_user_owns_session(current_user.id, session_id)

    rows = execute(
        supabase_admin.table("messages")
        .select("*")
        .eq("session_id", session_id)
        .order("created_at", desc=False)
        .execute()
    )

    return [MessageResponse(**r) for r in rows]


# POST /sessions/{session_id}/chat — the core streaming endpoint

@router.post("/sessions/{session_id}/chat")
async def chat(
    session_id: str,
    body: ChatRequest,
    current_user: UserProfile = Depends(get_current_user),
):
    """
    The core chat endpoint. Streams the RAG response via SSE.

    Flow:
        1. Validate the session belongs to this user.
        2. Save the user's message to the DB.
        3. Check the semantic cache. On hit, stream the cached answer.
        4. On cache miss, invoke the LangGraph pipeline.
        5. Stream tokens, citations, and staleness flags as SSE events.
        6. Save the assistant's response to the DB.
        7. Update the cache with the new answer.
        8. Yield the done event.

    Returns an EventSourceResponse which keeps the HTTP connection open
    and streams events until the generator exits.
    """
    session = _assert_user_owns_session(current_user.id, session_id)
    repo_id = session["repo_id"]

    question = body.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be empty.",
        )

    # Save user message to DB before generating the response.
    # This way the message exists even if generation fails.
    user_message_id = _save_message(
        session_id=session_id,
        role="user",
        content=question,
    )

    return EventSourceResponse(
        _stream_response(
            question=question,
            session_id=session_id,
            repo_id=repo_id,
        ),
        media_type="text/event-stream",
    )


async def _stream_response(
    question: str,
    session_id: str,
    repo_id: str,
) -> AsyncGenerator[str, None]:
    """
    Async generator that runs the RAG pipeline and yields SSE events.

    This is the function that does all the real work. Separated from
    the route handler so it can be an async generator while the route
    handler returns a synchronous EventSourceResponse wrapping it.
    """
    try:
        # Check semantic cache first.
        # A cache hit skips the entire LangGraph pipeline.
        cached = await semantic_cache.get(question, repo_id)

        if cached:
            log.info("cache_hit_serving", repo_id=repo_id)
            async for event in stream_rag_response(
                final_answer=cached["final_answer"],
                citations=cached["citations"],
                staleness_flags=cached["staleness_flags"],
                message_id=cached.get("message_id", str(uuid.uuid4())),
                tokens_used=cached.get("tokens_used", 0),
            ):
                yield event
            return

        # Cache miss — run the full LangGraph pipeline.
        state = initial_state(query=question, repo_id=repo_id)

        log.info("rag_graph_invoke_start", repo_id=repo_id, question=question[:100])

        result = await rag_graph.ainvoke(state)

        final_answer    = result.get("final_answer", "")
        citations       = result.get("citations", [])
        staleness_flags = result.get("staleness_flags", [])
        tokens_used     = result.get("tokens_used", 0)

        if not final_answer:
            final_answer = (
                "I could not find relevant information in this repository "
                "to answer your question."
            )

        # Save assistant message to DB.
        message_id = _save_message(
            session_id=session_id,
            role="assistant",
            content=final_answer,
            citations=[c.model_dump() for c in citations],
            staleness_flags=[f.model_dump() for f in staleness_flags],
            tokens_used=tokens_used,
        )

        # Store in semantic cache so similar future questions are served instantly.
        await semantic_cache.set(question, repo_id, {
            "final_answer":    final_answer,
            "citations":       [c.model_dump() for c in citations],
            "staleness_flags": [f.model_dump() for f in staleness_flags],
            "message_id":      message_id,
            "tokens_used":     tokens_used,
        })

        # Set session title from first question if not already set.
        _set_session_title_if_empty(session_id, question)

        # Stream everything to the frontend.
        async for event in stream_rag_response(
            final_answer=final_answer,
            citations=citations,
            staleness_flags=staleness_flags,
            message_id=message_id,
            tokens_used=tokens_used,
        ):
            yield event

        log.info(
            "rag_graph_invoke_done",
            repo_id=repo_id,
            tokens_used=tokens_used,
            citations=len(citations),
            staleness_flags=len(staleness_flags),
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        log.error("chat_stream_error", error=str(e))
        import json
        yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"


# DELETE /sessions/{session_id} — delete a session

@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    current_user: UserProfile = Depends(get_current_user),
):
    """Deletes a session and all its messages."""
    _assert_user_owns_session(current_user.id, session_id)

    # Messages are deleted by cascade from the DB schema.
    supabase_admin.table("chat_sessions").delete().eq(
        "id", session_id
    ).execute()


# Internal helpers

def _assert_user_has_repo_access(user_id: str, repo_id: str) -> None:
    """Raises 403 if user does not have this repo linked."""
    links = execute(
        supabase_admin.table("user_repos")
        .select("repo_id")
        .eq("user_id", user_id)
        .eq("repo_id", repo_id)
        .execute()
    )
    if not links:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this repo.",
        )


def _assert_user_owns_session(user_id: str, session_id: str) -> dict:
    """
    Raises 403 if user does not own this session.
    Returns the session row so callers can read repo_id from it
    without a second database query.
    """
    rows = execute(
        supabase_admin.table("chat_sessions")
        .select("*")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session not found or access denied.",
        )
    return rows[0]


def _save_message(
    session_id: str,
    role: str,
    content: str,
    citations: list = None,
    staleness_flags: list = None,
    tokens_used: int = None,
) -> str:
    """
    Saves a message to the DB and returns its UUID.
    Used for both user messages and assistant responses.
    """
    message_id = str(uuid.uuid4())
    supabase_admin.table("messages").insert({
        "id":              message_id,
        "session_id":      session_id,
        "role":            role,
        "content":         content,
        "citations":       citations or [],
        "staleness_flags": staleness_flags or [],
        "tokens_used":     tokens_used,
    }).execute()
    return message_id


def _set_session_title_if_empty(session_id: str, question: str) -> None:
    """
    Sets the session title to the first question asked, truncated to 60 chars.
    Only sets it once — if a title already exists, does nothing.
    This gives sessions a human readable name in the sidebar.
    """
    rows = execute(
        supabase_admin.table("chat_sessions")
        .select("title")
        .eq("id", session_id)
        .execute()
    )
    if rows and rows[0].get("title"):
        return

    title = question[:60] + ("..." if len(question) > 60 else "")
    supabase_admin.table("chat_sessions").update({
        "title": title
    }).eq("id", session_id).execute()

