# Shapes for repos and ingestion jobs.
# Two categories of models:
#   - Request models: what the frontend sends IN (validated by FastAPI)
#   - Response models: what we send OUT to the frontend

from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import Literal
import re

RepoStatus = Literal["queued", "fetching", "chunking", "embedding", "extracting", "done", "failed"]
IngestionStage = Literal["queued", "fetching", "chunking", "embedding", "extracting", "done", "failed"]

#user submits a github url
class IngestRepoRequest(BaseModel):
    github_url: str
    @field_validator("github_url")
    @classmethod
    def validate_github_url(cls, v: str) -> str:
        # Strip trailing slash and whitespace
        v = v.strip().rstrip("/")
 
        # Add scheme if missing
        if not v.startswith("http"):
            v = "https://" + v
 
        pattern = r"^https://github\.com/[\w\-\.]+/[\w\-\.]+$"
        if not re.match(pattern, v):
            raise ValueError(
                "Must be a valid GitHub repo URL. "
                "Example: https://github.com/owner/repo-name"
            )
        return v

#repo summary sent to the frontend
class RepoResponse(BaseModel):
    id: str
    github_url:       str
    owner:            str
    name:             str
    status:           RepoStatus
    chunk_count:      int
    decision_count:   int
    last_ingested_at: datetime | None
 
    class Config:
        extra = "ignore"

#ingestion job status
class IngestionStatusResponse(BaseModel):
    repo_id:      str
    stage:        IngestionStage
    progress_pct: int
    error_msg:    str | None
 
    class Config:
        extra = "ignore"
 