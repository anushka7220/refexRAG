# models/decision.py
#
# Decision archaeology feature — the structured object we extract
# from PRs and issues that captures WHY a decision was made,
# not just what was decided.

from dataclasses import dataclass, field
from pydantic import BaseModel
from datetime import datetime


# ── Internal: what DecisionExtractor produces ─────────────────────────────
@dataclass
class DecisionNode:
    """
    A structured decision extracted from a PR or issue during ingestion.

    Fields:
        repo_id:              Which repo this belongs to.
        decision:             One-sentence summary of what was decided.
                              e.g. "Use JWT over session-based auth"
        alternatives_rejected: List of dicts, each with "option" and "reason".
                              e.g. [{"option": "sessions", "reason": "stateful, doesn't scale"}]
        reasoning:            Full reasoning paragraph extracted from the PR discussion.
        source_chunk_ids:     IDs of the chunks this was derived from.
                              Lets us show "see PR #341, comment by @alice" in the UI.
        embedding:            1024-dim vector of the decision text.
                              Stored separately so decisions are searchable independently.
        id:                   UUID assigned after DB insert.
    """
    repo_id:               str
    decision:              str
    alternatives_rejected: list[dict]  = field(default_factory=list)
    reasoning:             str         = ""
    source_chunk_ids:      list[str]   = field(default_factory=list)
    embedding:             list[float] = field(default_factory=list)
    id:                    str         = ""


# ── Response: what the frontend gets ──────────────────────────────────────
class RejectedAlternative(BaseModel):
    option: str
    reason: str


class DecisionEvidenceItem(BaseModel):
    chunk_id:    str
    source_type: str
    source_id:   str
    url:         str
    excerpt:     str


class DecisionNodeResponse(BaseModel):
    id:                    str
    decision:              str
    alternatives_rejected: list[RejectedAlternative]
    reasoning:             str
    evidence:              list[DecisionEvidenceItem]
    created_at:            datetime

    class Config:
        extra = "ignore"


# ── What Gemini Flash returns (structured JSON we parse) ──────────────────
# Used in decision_extractor.py to parse the LLM output.
class DecisionExtraction(BaseModel):
    decision:              str
    alternatives_rejected: list[RejectedAlternative]
    reasoning:             str
    confidence:            float   # 0.0-1.0 — skip storing if below 0.6
