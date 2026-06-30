# contributor_builder.py
#
# Computes ownership scores, authority scores, and file area data from
# commit history and PR review patterns. No LLM calls. Pure computation.
#
# WHY NO LLM HERE:
# Ownership and authority are derivable facts from git history.
# Who committed to a file, how often, how recently. Who reviewed PRs
# and whose approval correlated with the PR getting merged.
# An LLM would only add latency and cost without improving accuracy.
#
# OWNERSHIP SCORE:
# Weighted by recency. A commit from last month counts more than a
# commit from three years ago. Formula:
#   score = sum(recency_weight(commit) for commit in their commits to this file)
#           / sum(recency_weight(commit) for all commits to this file)
#
# AUTHORITY SCORE:
# Measures how often a reviewer's approval correlates with a merge.
#   score = prs_they_approved_that_merged / total_prs_they_reviewed
#
# FILE AREAS:
# Files are grouped by their top level directory as a simple area boundary.
# Co-change tracking counts how often two areas appear in the same commit.

import structlog
from collections import defaultdict
from datetime import datetime, timezone
from dataclasses import dataclass

from app.models.contributor import ContributorScore, FileAreaScore

log = structlog.get_logger(__name__)

# Half life for recency weighting, in days.
# A commit from 180 days ago counts half as much as one from today.
RECENCY_HALF_LIFE_DAYS = 180


def _recency_weight(commit_date: datetime) -> float:
    """
    Returns a weight between 0 and 1 based on how recent a commit is.
    Uses exponential decay so very old commits contribute little
    without dropping to exactly zero.
    """
    now = datetime.now(timezone.utc)
    if commit_date.tzinfo is None:
        commit_date = commit_date.replace(tzinfo=timezone.utc)

    days_ago = (now - commit_date).days
    days_ago = max(0, days_ago)

    return 0.5 ** (days_ago / RECENCY_HALF_LIFE_DAYS)


def _area_from_path(file_path: str) -> str:
    """
    Reduces a full file path to its area boundary.
    Uses the top two directory levels as a reasonable granularity.

    Example:
        src/auth/jwt_handler.py becomes src/auth
        README.md becomes root
    """
    parts = file_path.split("/")
    if len(parts) <= 1:
        return "root"
    return "/".join(parts[:2])


class ContributorBuilder:
    """
    Builds contributor and file area data from raw commit and PR objects.

    Usage (called by IngestionOrchestrator):
        builder = ContributorBuilder()
        ownership = builder.build_ownership_scores(commits)
        authority = builder.build_authority_scores(prs)
        file_areas = builder.build_file_areas(commits)
    """

    def build_ownership_scores(self, commits: list) -> dict[str, float]:
        """
        Computes a recency weighted ownership score per contributor.
        The score reflects overall presence across the repo, not per file.
        Per file ownership is computed separately in build_file_areas.

        Args:
            commits: List of RawCommit objects from GitHubFetcher.

        Returns:
            Dict of github_username to normalized score between 0 and 1.
        """
        if not commits:
            return {}

        raw_scores: dict[str, float] = defaultdict(float)

        for commit in commits:
            weight = _recency_weight(commit.created_at)
            raw_scores[commit.author] += weight

        # Normalize so the highest scoring contributor lands at 1.0
        max_score = max(raw_scores.values()) if raw_scores else 1.0
        normalized = {
            author: round(score / max_score, 4)
            for author, score in raw_scores.items()
        }

        log.info("ownership_scores_built", contributors=len(normalized))
        return normalized

    def build_authority_scores(self, prs: list) -> dict[str, float]:
        """
        Computes authority score per reviewer based on review to merge correlation.

        For every PR a person reviewed, we check whether their review state
        was APPROVED and whether the PR ultimately merged. A reviewer whose
        approvals consistently precede a merge has high authority in that
        sense, their sign off tends to be the deciding factor.

        Formula:
            score = prs_they_approved_that_merged / total_prs_they_reviewed

        Args:
            prs: List of RawPR objects, each carrying a list of RawReview
                 objects with reviewer login attached.

        Returns:
            Dict of github_username to authority score between 0 and 1.
        """
        if not prs:
            return {}

        reviewed_count: dict[str, int] = defaultdict(int)
        approved_and_merged: dict[str, int] = defaultdict(int)

        for pr in prs:
            # A reviewer may appear more than once on the same PR across
            # multiple review rounds. We count distinct reviewers per PR
            # once, so a single PR cannot inflate one person's denominator.
            reviewers_on_this_pr = set()

            for review in pr.reviews:
                if review.reviewer == "unknown":
                    continue
                reviewers_on_this_pr.add((review.reviewer, review.state))

            seen_reviewers = set()
            for reviewer, state in reviewers_on_this_pr:
                if reviewer in seen_reviewers:
                    continue
                seen_reviewers.add(reviewer)

                reviewed_count[reviewer] += 1
                if state == "APPROVED" and pr.merged:
                    approved_and_merged[reviewer] += 1

        scores = {
            reviewer: round(approved_and_merged[reviewer] / reviewed_count[reviewer], 4)
            for reviewer in reviewed_count
            if reviewed_count[reviewer] > 0
        }

        log.info("authority_scores_built", reviewers=len(scores))
        return scores

    def build_file_areas(self, commits: list) -> list[FileAreaScore]:
        """
        Groups files into areas and computes complexity and co-change data.

        complexity_score: how often an area changes relative to its file count.
        Higher churn relative to size suggests a harder, more actively
        maintained part of the codebase.

        co_changes_with: other areas that frequently appear in the same
        commit as this one. Useful for "if you touch X you likely touch Y".

        Args:
            commits: List of RawCommit objects.

        Returns:
            List of FileAreaScore objects, one per detected area.
        """
        if not commits:
            return []

        area_change_count: dict[str, int] = defaultdict(int)
        area_files: dict[str, set] = defaultdict(set)
        co_change_pairs: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        area_contributors: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for commit in commits:
            areas_in_commit = set()
            weight = _recency_weight(commit.created_at)

            for file_path in commit.files:
                area = _area_from_path(file_path)
                areas_in_commit.add(area)
                area_change_count[area] += 1
                area_files[area].add(file_path)
                area_contributors[area][commit.author] += weight

            # Record co-changes for every pair of areas touched together
            areas_list = list(areas_in_commit)
            for i in range(len(areas_list)):
                for j in range(i + 1, len(areas_list)):
                    co_change_pairs[areas_list[i]][areas_list[j]] += 1
                    co_change_pairs[areas_list[j]][areas_list[i]] += 1

        max_change_count = max(area_change_count.values()) if area_change_count else 1

        results = []
        for area, change_count in area_change_count.items():
            file_count = max(1, len(area_files[area]))
            complexity = round((change_count / file_count) / max_change_count, 4)

            top_co_changes = sorted(
                co_change_pairs[area].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]

            top_contributors = sorted(
                area_contributors[area].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]

            results.append(FileAreaScore(
                repo_id="",
                area_path=area,
                complexity_score=complexity,
                co_changes_with=[name for name, _ in top_co_changes],
                top_contributors=[name for name, _ in top_contributors],
            ))

        log.info("file_areas_built", count=len(results))
        return results

    def compute_difficulty(self, issue, similar_past_issues: list) -> float:
        """
        Computes a real difficulty score for an issue, replacing label based
        difficulty which is frequently wrong or stale.

        Currently a placeholder using a simple heuristic based on comment
        count as a proxy for discussion complexity. A future improvement
        is to correlate with actual time to merge once that data is tracked
        per issue rather than per PR.

        Args:
            issue: A RawIssue object.
            similar_past_issues: Historical issues for comparison, unused
                                  in this initial version.

        Returns:
            Float score from 1.0, trivial, to 5.0, very hard.
        """
        comment_count = len(issue.comments)

        if comment_count <= 2:
            return 1.0
        elif comment_count <= 5:
            return 2.0
        elif comment_count <= 10:
            return 3.0
        elif comment_count <= 20:
            return 4.0
        else:
            return 5.0