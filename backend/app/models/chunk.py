# models/chunk.py
#
# The atomic unit of the entire RAG pipeline.
# A chunk is one piece of text from GitHub (one issue body, one PR comment,
# one release note) with its embedding vector and metadata attached.
#
# WHY a dataclass and not Pydantic:
# Chunks are internal — they never go directly to the frontend as HTTP responses.
# They flow between services: Chunker → EmbeddingService → VectorStore.
# Dataclasses are lighter and faster for internal data passing.

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


# ── Source type: what kind of GitHub object this chunk came from ───────────
SourceType = Literal["issue", "pr", "comment", "commit", "release", "discussion"]

# ── Status: current state of the source object on GitHub ──────────────────
SourceStatus = Literal["open", "closed", "merged", "none"]


@dataclass
class Chunk:
    """
    One piece of text from a GitHub object, ready to be embedded and stored.

    Fields:
        repo_id:          UUID of the repo this chunk belongs to.
        content:          Raw text of this chunk, max roughly 512 tokens.
        source_type:      What kind of object this came from.
        source_id:        GitHub's identifier, issue number, PR number, commit SHA.
        status:           Current state of the source object on GitHub.
        content_hash:     sha256 of content, used for deduplication.
                          If this hash already exists in the DB, we skip re embedding.
        source_created_at: When the GitHub object was created.
                           The critic uses this to detect stale sources.
        url:              Full GitHub URL to the source, built once at chunk
                          creation time when owner and repo_name are known.
                          Stored so citations can link directly without
                          reconstructing the URL later with incomplete data.
        version_tag:      Nearest release tag at time of creation, e.g. v0.3.1.
                          Extracted during ingestion. Used for version mismatch detection.
        embedding:        1024 dim float vector from bge-large-en-v1.5.
                          Empty list until EmbeddingService fills it.
        id:               UUID assigned when stored in Supabase. Empty until then.
    """
    repo_id:           str
    content:           str
    source_type:       SourceType
    source_id:         str
    status:            SourceStatus
    content_hash:      str
    source_created_at: datetime
    url:               str

    # Optional fields with defaults
    version_tag:  str | None       = None
    embedding:    list[float]      = field(default_factory=list)
    id:           str              = ""   # filled after DB insert


@dataclass
class ChunkResult:
    """
    A chunk returned from a similarity search, with relevance scores attached.

    Fields:
        chunk:        The full Chunk object.
        score:        Cosine similarity score from pgvector (0.0 to 1.0).
                      Higher = more similar to the query.
        rerank_score: Score from the cross-encoder reranker (optional).
                      Only set after the reranking step in the RAG graph.
                      Higher = more relevant. Used to sort the final top-k.
    """
    chunk:        Chunk
    score:        float
    rerank_score: float | None = None


# ── What we send to the frontend in citations ─────────────────────────────
# Pydantic model because this goes in the HTTP response (inside Message).

from pydantic import BaseModel

class Citation(BaseModel):
    chunk_id:    str
    source_type: SourceType
    source_id:   str
    status:      SourceStatus
    version_tag: str | None
    url:         str       # full GitHub URL to the source
    excerpt:     str       # first 200 chars of content


class StalenessFlag(BaseModel):
    chunk_id:  str
    reason:    Literal[
        "source_closed",
        "version_mismatch",
        "contradiction",
        "outdated_timestamp",
    ]
    severity:  Literal["warn", "error"]
    detail:    str          # human-readable explanation shown in the UI
