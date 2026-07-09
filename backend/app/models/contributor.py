# models/contributor.py
#
# Contributor map feature — ownership scores, authority maps,
# file areas, and real issue difficulty ratings.

from dataclasses import dataclass, field
from pydantic import BaseModel
from datetime import datetime


# ── Internal: built by ContributorBuilder during ingestion ────────────────
@dataclass
class ContributorScore:
    """
    Computed ownership and authority scores for one contributor in one repo.

    ownership_score: How much of the codebase this person has touched.
                     Computed as: (commits_by_this_person / total_commits) weighted by recency.
                     Range: 0.0 to 1.0.

    authority_score: How often this person's reviews are the deciding factor for merges.
                     Computed as: (prs_they_approved_that_merged / total_prs_they_reviewed).
                     Range: 0.0 to 1.0.

    top_areas:       File path prefixes where they have the highest ownership.
                     e.g. ["src/auth", "api/routes"]
    """
    repo_id:         str
    github_username: str
    ownership_score: float        = 0.0
    authority_score: float        = 0.0
    top_areas:       list[str]    = field(default_factory=list)
    id:              str          = ""


@dataclass
class FileAreaScore:
    """
    Ownership and complexity data for one logical file area.

    area_path:       Path prefix. e.g. "src/auth", "api", "tests"
    complexity_score: How often this area changes relative to its size.
                      High complexity = changes frequently = harder to contribute to.
    co_changes_with: Other areas that tend to change in the same commits.
                     Useful for "if I change X, I probably also need to change Y".
    top_contributors: Usernames ranked by ownership_score for this area.
    """
    repo_id:          str
    area_path:        str
    complexity_score: float     = 0.0
    co_changes_with:  list[str] = field(default_factory=list)
    top_contributors: list[str] = field(default_factory=list)
    id:               str       = ""


# ── Response models: what the frontend gets ───────────────────────────────
class ContributorResponse(BaseModel):
    github_username: str
    avatar_url:      str | None = None
    ownership_score: float
    authority_score: float
    top_areas:       list[str]

    class Config:
        extra = "ignore"


class FileAreaResponse(BaseModel):
    area_path:        str
    complexity_score: float
    co_changes_with:  list[str]
    top_contributors: list[str]

    class Config:
        extra = "ignore"


class RankedIssue(BaseModel):
    """
    A GitHub issue with a computed real difficulty score.

    real_difficulty_score: 1.0 (trivial) to 5.0 (very hard).
                           Computed from historical data:
                           - Median time-to-merge for similar past issues
                           - Number of files touched in the fixing PR
                           - Number of review rounds needed before merge
                           Not from the label — labels lie.
    """
    issue_id:             str
    title:                str
    url:                  str
    status:               str
    real_difficulty_score: float
    files_touched:        list[str]
    best_reviewer:        str | None  # contributor with highest authority in touched areas


class StartHereResponse(BaseModel):
    """
    Curated onboarding path for a new contributor.
    Returned by GET /repos/{id}/start-here.
    """
    suggested_issues:           list[RankedIssue]
    suggested_files_to_read:    list[str]
    key_contributors_to_follow: list[str]
