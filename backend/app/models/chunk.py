#this is the atomic unit of entire RAG pipeline
#chunk is one peice of text from github(one issue body, one pR comment, one release note)
#along with its embedding vector and metadata attached
#WHY A DATACLASS NOT PYDANTIC?
#chunks are internal-they never go directly to the frontend as http response
#they flow between services: chunker-> embeddings->vectorSearch
#dataclass are lighter and faster for internal data passing

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

sourceType = Literal["issue", "pr", "comment", "release", "discussion"]

#current state of source object on github
SourceStatus = Literal["open", "closed", "merged", "none"]

@dataclass
class Chunk:
    repo_id: str #UUID of the repo this chunk belongs to
    content: str #raw text of the chunk
    source_type: sourceType #what kind of object
    source_id: str #github's identifier
    status: SourceStatus #current state of source object
    content_hash: str #sha256 of content — used for deduplication. If this hash already exists in the DB, we skip re-embedding.
    source_created_at: datetime

    #optional fields with defaults
    version_tag: str | None = None
    embedding: list[float] = field(default_factory=list) #1024-dim float vector from bge-large-en-v1.5.
    id: str = "" #field after db insert, UUID assigned when stored in Supabase. Empty until then.


@dataclass 
class ChunkResult:
    chunk: Chunk
    score: float
    rerank_score: float | None = None

#what we send to the frontend
#NOW PYDANTIC MODEL because this goes in the http response

from pydantic import BaseModel

class Citation(BaseModel):
    chunk_id: str
    source_type: sourceType
    source_id: str
    status: SourceStatus
    version_tag: str | None
    url: str
    excerpt: str

class StalenessFalg(BaseModel):
    chunk_id: str
    reason: Literal[
        "source_closed",
        "version_mismatch",
        "contradiction",
        "outdated timestamp",
    ]
    severity: Literal["warn", "error"]
    detail: str