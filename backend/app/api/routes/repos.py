# api/routes/repos.py
#
# Repo ingestion and management endpoints.
#
# WHAT THIS FILE DOES:
# Creates repo records, queues ingestion jobs, returns status.
# The actual ingestion pipeline runs in the Celery worker process,
# not here. This file only queues the work and tracks its status.

import uuid
from fastapi import APIRouter, HTTPException, Depends, status

from app.core.dependencies import get_current_user, check_repo_limit
from app.core.supabase import supabase_admin, execute
from app.models.user import UserProfile
from app.models.repo import IngestRepoRequest, RepoResponse, IngestionStatusResponse
from app.utils.github_parser import parse_github_url
from celery_worker.tasks import ingest_repo

import structlog

log = structlog.get_logger(__name__)

router = APIRouter()


# ── POST /repos — submit a GitHub URL for ingestion ───────────────────────

@router.post("", response_model=RepoResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_repo(
    body: IngestRepoRequest,
    current_user: UserProfile = Depends(check_repo_limit),
):
    """
    Submits a GitHub URL for ingestion.

    Steps:
        1. Validate and parse the GitHub URL.
        2. Check if this repo is already ingested by any user.
           If yes, link the current user to the existing repo
           and return immediately without re-ingesting.
        3. If new, create a repo row and queue the Celery task.
        4. Increment the user's repos_used count.

    Returns 202 Accepted because the work is queued, not done yet.
    The frontend should poll GET /repos/{repo_id}/status for progress.
    """
    parsed = parse_github_url(body.github_url)
    if not parsed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid GitHub URL. Expected: https://github.com/owner/repo",
        )
    owner, repo_name = parsed
    canonical_url = f"https://github.com/{owner}/{repo_name}"

    # Check if any user has already ingested this repo.
    # If yes, link this user to the existing repo and skip re-ingestion.
    # This is the shared chunk store deduplication at the user level.
    existing = execute(
        supabase_admin.table("repos")
        .select("*")
        .eq("github_url", canonical_url)
        .execute()
    )

    if existing:
        repo_row = existing[0]
        repo_id = repo_row["id"]

        # Check if this user already has this repo linked
        already_linked = execute(
            supabase_admin.table("user_repos")
            .select("repo_id")
            .eq("user_id", current_user.id)
            .eq("repo_id", repo_id)
            .execute()
        )

        if not already_linked:
            # Link this user to the existing repo
            supabase_admin.table("user_repos").insert({
                "user_id": current_user.id,
                "repo_id": repo_id,
            }).execute()

            # Increment user's repo count
            _increment_repos_used(current_user.id)

            log.info(
                "repo_linked_existing",
                repo_id=repo_id,
                user_id=current_user.id,
            )

        return RepoResponse(**repo_row)

    # New repo — create the row and queue ingestion
    repo_id = str(uuid.uuid4())

    repo_row = execute(
        supabase_admin.table("repos").insert({
            "id":         repo_id,
            "github_url": canonical_url,
            "owner":      owner,
            "name":       repo_name,
            "status":     "queued",
        }).execute()
    )[0]

    # Link this user to the new repo
    supabase_admin.table("user_repos").insert({
        "user_id": current_user.id,
        "repo_id": repo_id,
    }).execute()

    # Increment user's repo count
    _increment_repos_used(current_user.id)

    # Queue the ingestion task in Celery.
    # .delay() returns immediately — the worker picks it up from Redis.
    ingest_repo.delay(repo_id=repo_id, github_url=canonical_url)

    log.info(
        "repo_ingestion_queued",
        repo_id=repo_id,
        github_url=canonical_url,
        user_id=current_user.id,
    )

    return RepoResponse(**repo_row)


# ── GET /repos — list all repos for current user ──────────────────────────

@router.get("", response_model=list[RepoResponse])
async def list_repos(current_user: UserProfile = Depends(get_current_user)):
    """
    Returns all repos the current user has ingested or linked to.

    Joins user_repos with repos to get only this user's repos,
    not everyone's. Each user sees only their own linked repos.
    """
    # Get repo IDs linked to this user
    links = execute(
        supabase_admin.table("user_repos")
        .select("repo_id")
        .eq("user_id", current_user.id)
        .execute()
    )

    if not links:
        return []

    repo_ids = [row["repo_id"] for row in links]

    repos = execute(
        supabase_admin.table("repos")
        .select("*")
        .in_("id", repo_ids)
        .order("last_ingested_at", desc=True)
        .execute()
    )

    return [RepoResponse(**r) for r in repos]


# ── GET /repos/{repo_id} — single repo detail ─────────────────────────────

@router.get("/{repo_id}", response_model=RepoResponse)
async def get_repo(
    repo_id: str,
    current_user: UserProfile = Depends(get_current_user),
):
    """Returns a single repo if the current user has access to it."""
    _assert_user_has_access(current_user.id, repo_id)

    rows = execute(
        supabase_admin.table("repos")
        .select("*")
        .eq("id", repo_id)
        .execute()
    )

    if not rows:
        raise HTTPException(status_code=404, detail="Repo not found")

    return RepoResponse(**rows[0])


# ── GET /repos/{repo_id}/status — ingestion progress ──────────────────────

@router.get("/{repo_id}/status", response_model=IngestionStatusResponse)
async def get_ingestion_status(
    repo_id: str,
    current_user: UserProfile = Depends(get_current_user),
):
    """
    Returns the current ingestion progress for a repo.
    The frontend polls this endpoint every 3 seconds while ingestion runs.
    Reads from ingestion_jobs table which the orchestrator updates at every stage.
    """
    _assert_user_has_access(current_user.id, repo_id)

    rows = execute(
        supabase_admin.table("ingestion_jobs")
        .select("*")
        .eq("repo_id", repo_id)
        .order("started_at", desc=True)
        .limit(1)
        .execute()
    )

    if not rows:
        raise HTTPException(status_code=404, detail="No ingestion job found for this repo")

    job = rows[0]
    return IngestionStatusResponse(
        repo_id=repo_id,
        stage=job["stage"],
        progress_pct=job["progress_pct"],
        error_msg=job.get("error_msg"),
    )


# ── DELETE /repos/{repo_id} — remove a repo ───────────────────────────────

@router.delete("/{repo_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_repo(
    repo_id: str,
    current_user: UserProfile = Depends(get_current_user),
):
    """
    Removes a repo from the current user's dashboard.

    Unlinks the user from the repo. Only deletes the actual repo row
    and its chunks if no other users have it linked. This preserves
    the shared chunk store for other users.
    """
    _assert_user_has_access(current_user.id, repo_id)

    # Remove this user's link
    supabase_admin.table("user_repos").delete().eq(
        "user_id", current_user.id
    ).eq("repo_id", repo_id).execute()

    # Decrement repos_used
    supabase_admin.table("profiles").update({
        "repos_used": max(0, current_user.repos_used - 1)
    }).eq("id", current_user.id).execute()

    # Check if any other user still has this repo linked
    remaining_links = execute(
        supabase_admin.table("user_repos")
        .select("user_id")
        .eq("repo_id", repo_id)
        .execute()
    )

    if not remaining_links:
        # No other users have this repo. Safe to delete everything.
        from app.services.ingestion.vector_store import vector_store
        vector_store.delete_chunks_for_repo(repo_id)

        supabase_admin.table("decision_nodes").delete().eq("repo_id", repo_id).execute()
        supabase_admin.table("contributors").delete().eq("repo_id", repo_id).execute()
        supabase_admin.table("file_areas").delete().eq("repo_id", repo_id).execute()
        supabase_admin.table("ingestion_jobs").delete().eq("repo_id", repo_id).execute()
        supabase_admin.table("repos").delete().eq("id", repo_id).execute()

        log.info("repo_fully_deleted", repo_id=repo_id)
    else:
        log.info(
            "repo_unlinked_only",
            repo_id=repo_id,
            remaining_users=len(remaining_links),
        )


# ── Internal helpers ───────────────────────────────────────────────────────

def _assert_user_has_access(user_id: str, repo_id: str) -> None:
    """
    Raises 403 if the user does not have this repo linked.
    Called before any operation that reads or modifies a specific repo.
    """
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


def _increment_repos_used(user_id: str) -> None:
    """Increments the repos_used counter on the user's profile."""
    rows = execute(
        supabase_admin.table("profiles")
        .select("repos_used")
        .eq("id", user_id)
        .execute()
    )
    current_count = rows[0]["repos_used"] if rows else 0
    supabase_admin.table("profiles").update({
        "repos_used": current_count + 1
    }).eq("id", user_id).execute()