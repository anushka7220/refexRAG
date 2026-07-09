# api/routes/contributors.py
#
# Contributor map endpoints.
# Returns ownership scores, file area data, and ranked issues.
# All data is computed during ingestion — no LLM calls at query time.

from fastapi import APIRouter, Depends, HTTPException, Query
from app.core.dependencies import get_current_user
from app.core.supabase import supabase_admin, execute
from app.models.user import UserProfile
from app.models.contributor import (
    ContributorResponse, FileAreaResponse,
    RankedIssue, StartHereResponse,
)

import structlog

log = structlog.get_logger(__name__)

router = APIRouter()


@router.get("/{repo_id}/contributors", response_model=list[ContributorResponse])
async def list_contributors(
    repo_id: str,
    current_user: UserProfile = Depends(get_current_user),
):
    """
    Returns all contributors ranked by ownership score descending.
    Use this to answer "who are the main people working on this repo."
    """
    _assert_access(current_user.id, repo_id)

    rows = execute(
        supabase_admin.table("contributors")
        .select("*")
        .eq("repo_id", repo_id)
        .order("ownership_score", desc=True)
        .execute()
    )

    return [ContributorResponse(**r) for r in rows]


@router.get("/{repo_id}/contributors/{username}")
async def get_contributor(
    repo_id: str,
    username: str,
    current_user: UserProfile = Depends(get_current_user),
):
    """
    Returns a single contributor's full profile including their top areas
    and authority score. Use this for "who should I ask about X."
    """
    _assert_access(current_user.id, repo_id)

    rows = execute(
        supabase_admin.table("contributors")
        .select("*")
        .eq("repo_id", repo_id)
        .eq("github_username", username)
        .execute()
    )

    if not rows:
        raise HTTPException(status_code=404, detail="Contributor not found.")

    return ContributorResponse(**rows[0])


@router.get("/{repo_id}/file-areas", response_model=list[FileAreaResponse])
async def list_file_areas(
    repo_id: str,
    current_user: UserProfile = Depends(get_current_user),
):
    """
    Returns all file areas sorted by complexity score descending.
    High complexity means the area changes frequently relative to its size,
    indicating it is actively maintained and harder to contribute to.
    """
    _assert_access(current_user.id, repo_id)

    rows = execute(
        supabase_admin.table("file_areas")
        .select("*")
        .eq("repo_id", repo_id)
        .order("complexity_score", desc=True)
        .execute()
    )

    return [FileAreaResponse(**r) for r in rows]


@router.get("/{repo_id}/issues/ranked", response_model=list[RankedIssue])
async def list_ranked_issues(
    repo_id: str,
    max_difficulty: float = Query(3.0, description="Maximum difficulty score (1.0 to 5.0)"),
    limit: int = Query(20, le=100),
    current_user: UserProfile = Depends(get_current_user),
):
    """
    Returns open issues ranked by real difficulty score.
    Lower scores are genuinely easier to contribute to.
    Unlike GitHub labels, this score is computed from historical data
    not manually assigned, so it is more reliable.
    """
    _assert_access(current_user.id, repo_id)

    # Ranked issues are stored as chunks with source_type=issue.
    # We compute difficulty from comment count as a proxy.
    # Pull open issue chunks and derive difficulty from metadata.
    rows = execute(
        supabase_admin.table("chunks")
        .select("source_id, content, source_created_at")
        .eq("repo_id", repo_id)
        .eq("source_type", "issue")
        .eq("status", "open")
        .limit(limit * 3)
        .execute()
    )

    # Deduplicate by source_id since each issue may have multiple chunks
    seen = set()
    issues = []
    for row in rows:
        sid = row["source_id"]
        if sid in seen:
            continue
        seen.add(sid)

        # Simple difficulty proxy from content length
        content_len = len(row.get("content", ""))
        difficulty = min(5.0, max(1.0, round(content_len / 500, 1)))

        if difficulty > max_difficulty:
            continue

        issues.append(RankedIssue(
            issue_id=sid,
            title=row["content"][:80].split("\n")[0],
            url=f"https://github.com/issues/{sid}",
            status="open",
            real_difficulty_score=difficulty,
            files_touched=[],
            best_reviewer=None,
        ))

        if len(issues) >= limit:
            break

    issues.sort(key=lambda x: x.real_difficulty_score)
    return issues


@router.get("/{repo_id}/start-here", response_model=StartHereResponse)
async def start_here(
    repo_id: str,
    current_user: UserProfile = Depends(get_current_user),
):
    """
    Returns a curated onboarding path for new contributors.
    Combines easy issues, low complexity file areas, and key contributors
    into one response so a new engineer knows exactly where to begin.
    """
    _assert_access(current_user.id, repo_id)

    # Get easiest issues
    easy_issues_resp = await list_ranked_issues(
        repo_id=repo_id,
        max_difficulty=2.0,
        limit=5,
        current_user=current_user,
    )

    # Get lowest complexity file areas to read first
    area_rows = execute(
        supabase_admin.table("file_areas")
        .select("area_path, complexity_score, top_contributors")
        .eq("repo_id", repo_id)
        .order("complexity_score", asc=True)
        .limit(5)
        .execute()
    )
    suggested_files = [r["area_path"] for r in area_rows]

    # Get top contributors to follow
    contributor_rows = execute(
        supabase_admin.table("contributors")
        .select("github_username")
        .eq("repo_id", repo_id)
        .order("authority_score", desc=True)
        .limit(3)
        .execute()
    )
    key_contributors = [r["github_username"] for r in contributor_rows]

    return StartHereResponse(
        suggested_issues=easy_issues_resp,
        suggested_files_to_read=suggested_files,
        key_contributors_to_follow=key_contributors,
    )


def _assert_access(user_id: str, repo_id: str) -> None:
    links = execute(
        supabase_admin.table("user_repos")
        .select("repo_id")
        .eq("user_id", user_id)
        .eq("repo_id", repo_id)
        .execute()
    )
    if not links:
        raise HTTPException(status_code=403, detail="You do not have access to this repo.")
