# models/message.py
#
# Shapes for chat sessions and messages.
# Messages are the most complex response object — they contain
# the answer text, citations, and staleness flags all in one.

from pydantic import BaseModel
from datetime import datetime
from typing import Literal
from app.models.chunk import Citation, StalenessFlag


# ── Request: user sends a question ────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str

    def validate_question(self) -> str:
        q = self.question.strip()
        if not q:
            raise ValueError("Question cannot be empty")
        if len(q) > 2000:
            raise ValueError("Question too long — max 2000 characters")
        return q


# ── Response: a single message in a chat session ──────────────────────────
class MessageResponse(BaseModel):
    id:              str
    role:            Literal["user", "assistant"]
    content:         str
    citations:       list[Citation]      = []
    staleness_flags: list[StalenessFlag] = []
    model_used:      str | None
    tokens_used:     int | None
    created_at:      datetime

    class Config:
        extra = "ignore"


# ── Response: a chat session summary ──────────────────────────────────────
class ChatSessionResponse(BaseModel):
    id:            str
    repo_id:       str
    title:         str | None
    message_count: int = 0
    created_at:    datetime

    class Config:
        extra = "ignore"


# ── Internal: what gets saved to the DB after a full RAG response ─────────
class MessageRecord(BaseModel):
    session_id:      str
    role:            Literal["user", "assistant"]
    content:         str
    citations:       list[dict] = []       # serialised Citation dicts
    staleness_flags: list[dict] = []       # serialised StalenessFlag dicts
    model_used:      str | None = None
    tokens_used:     int | None = None
