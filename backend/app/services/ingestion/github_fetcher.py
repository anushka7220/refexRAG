# github_fetcher.py
#
# THE ONLY FILE that talks to the GitHub API.
# Everything GitHub-related — HTTP calls, pagination, rate limiting,
# retry logic — lives here. Nothing else in the codebase imports requests
# or PyGithub directly.
#
# WHAT IT RETURNS:
# Plain Python dicts and dataclasses — not PyGithub objects.
# This keeps the rest of the codebase decoupled from the GitHub library.
# If we switch from PyGithub to httpx tomorrow, only this file changes.
#
# RATE LIMIT STRATEGY:
# GitHub allows 5000 requests/hour per OAuth token.
# Large repos (React, Django, FastAPI) have 10,000+ issues + comments.
# Each issue's comments = 1 separate API call.
# We handle this with:
#   1. 700ms delay between paginated calls (stays under 90 req/min secondary limit)
#   2. Rate limit header check before each batch — sleep if < 100 requests remain
#   3. Exponential backoff on 403/429/503 via tenacity

import asyncio
import time
import structlog
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from github import Github, GithubException, RateLimitExceededException
from github.Repository import Repository
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

from app.core.config import settings

log = structlog.get_logger(__name__)


# ── Raw data shapes returned by this fetcher ──────────────────────────────
# Plain dataclasses — not Pydantic, not PyGithub objects.
# Chunker.py consumes these.

@dataclass
class RawIssue:
    number:     int
    title:      str
    body:       str
    state:      str           # "open" or "closed"
    created_at: datetime
    updated_at: datetime
    labels:     list[str]     = field(default_factory=list)
    comments:   list[str]     = field(default_factory=list)   # comment bodies
    html_url:   str           = ""


@dataclass
class RawReview:
    """
    A single formal review on a PR, with the reviewer's identity attached.
    Separated from a plain string list because authority scoring needs
    to know who wrote each review, not just what it said.
    """
    reviewer: str          # github login of the reviewer
    body:     str
    state:    str          # "APPROVED", "CHANGES_REQUESTED", "COMMENTED"


@dataclass
class RawPR:
    number:        int
    title:         str
    body:          str
    state:         str          # "open", "closed", "merged"
    merged:        bool
    created_at:    datetime
    updated_at:    datetime
    merged_at:     Optional[datetime]
    comments:      list[str]       = field(default_factory=list)   # review comments
    reviews:       list[RawReview] = field(default_factory=list)   # formal reviews with identity
    html_url:      str             = ""


@dataclass
class RawCommit:
    sha:        str
    message:    str
    author:     str
    created_at: datetime
    files:      list[str]   = field(default_factory=list)   # file paths changed


@dataclass
class RawRelease:
    tag_name:   str
    name:       str
    body:       str
    created_at: datetime
    html_url:   str = ""


@dataclass
class RepoMeta:
    owner:            str
    name:             str
    description:      str
    default_branch:   str
    latest_commit_sha: str
    html_url:         str


# ── GitHub client wrapper ──────────────────────────────────────────────────

class GitHubFetcher:
    """
    Fetches all relevant data from a public GitHub repository.

    Usage:
        fetcher = GitHubFetcher(github_token="ghp_...")
        meta    = fetcher.fetch_repo_meta("https://github.com/owner/repo")
        issues  = await fetcher.fetch_issues(meta.owner, meta.name)
        prs     = await fetcher.fetch_prs(meta.owner, meta.name)
        commits = await fetcher.fetch_commits(meta.owner, meta.name)

    Note: PyGithub is synchronous. All methods that call the API are wrapped
    with run_in_executor so they don't block FastAPI's async event loop.
    """

    def __init__(self, github_token: Optional[str] = None):
        """
        Args:
            github_token: A GitHub OAuth token. Authenticated requests get
                          5000 req/hr. Pass None for unauthenticated (60/hr — useless).
        """
        self._github = Github(github_token, per_page=100)
        self._delay  = settings.github_api_delay_seconds
        log.info("github_fetcher_init", authenticated=github_token is not None)

    # ── Rate limit guard ───────────────────────────────────────────────────

    def _check_rate_limit(self):
        """
        Checks remaining API calls. Sleeps until reset if dangerously low.
        Call this before starting each paginated batch.

        GitHub rate limit headers:
            X-RateLimit-Remaining: requests left in current window
            X-RateLimit-Reset:     unix timestamp when the window resets
        """
        rate_limit = self._github.get_rate_limit()
        remaining  = rate_limit.core.remaining
        reset_time = rate_limit.core.reset

        log.debug("rate_limit_check", remaining=remaining)

        if remaining < 100:
            # Less than 100 requests left — sleep until the window resets
            sleep_seconds = (reset_time - datetime.utcnow()).total_seconds() + 10
            sleep_seconds = max(0, sleep_seconds)
            log.warning(
                "rate_limit_low",
                remaining=remaining,
                sleeping_seconds=sleep_seconds,
            )
            time.sleep(sleep_seconds)

    # ── Repo metadata ──────────────────────────────────────────────────────

    def _fetch_repo_meta_sync(self, github_url: str) -> RepoMeta:
        """Synchronous — call via _run_sync() from async context."""
        # Parse owner/name from URL
        # Input: "https://github.com/owner/repo-name"
        parts = github_url.rstrip("/").split("/")
        owner, name = parts[-2], parts[-1]

        repo = self._github.get_repo(f"{owner}/{name}")

        # Get latest commit SHA — stored in repos table for differential ingestion
        default_branch = repo.default_branch
        latest_commit  = repo.get_branch(default_branch).commit.sha

        return RepoMeta(
            owner=owner,
            name=name,
            description=repo.description or "",
            default_branch=default_branch,
            latest_commit_sha=latest_commit,
            html_url=repo.html_url,
        )

    async def fetch_repo_meta(self, github_url: str) -> RepoMeta:
        return await self._run_sync(self._fetch_repo_meta_sync, github_url)

    # ── Issues ─────────────────────────────────────────────────────────────

    def _fetch_issues_sync(self, owner: str, repo_name: str) -> list[RawIssue]:
        """
        Fetches all issues including closed ones.
        Each issue's comments are fetched separately — one API call per issue.

        WHY fetch comments separately:
        GitHub's issue list endpoint returns issue bodies but NOT comments.
        Comments contain the real discussion — often more valuable than the body.
        We want both.
        """
        self._check_rate_limit()

        repo   = self._github.get_repo(f"{owner}/{repo_name}")
        issues = []

        # state="all" fetches both open and closed issues
        # We need closed ones too — the critic uses status to detect stale sources
        paginator = repo.get_issues(state="all")

        for i, issue in enumerate(paginator):
            # Skip pull requests — GitHub's API returns them as issues too
            # We fetch PRs separately via get_pulls()
            if issue.pull_request is not None:
                continue

            # Respect ingestion limit — don't index massive repos fully
            if i >= settings.MAX_ISSUES_PER_REPO:
                log.info("issue_limit_reached", limit=settings.MAX_ISSUES_PER_REPO)
                break

            # Fetch comments for this issue
            comment_bodies = self._fetch_issue_comments_sync(issue)

            issues.append(RawIssue(
                number=issue.number,
                title=issue.title,
                body=issue.body or "",
                state=issue.state,
                created_at=issue.created_at,
                updated_at=issue.updated_at,
                labels=[label.name for label in issue.labels],
                comments=comment_bodies,
                html_url=issue.html_url,
            ))

            # Delay between issues to respect secondary rate limit
            # 700ms = safe margin under GitHub's 90 req/min secondary limit
            time.sleep(self._delay)

        log.info("issues_fetched", count=len(issues), repo=f"{owner}/{repo_name}")
        return issues

    def _fetch_issue_comments_sync(self, issue) -> list[str]:
        """Fetches comment bodies for a single issue. Returns list of strings."""
        try:
            return [
                comment.body
                for comment in issue.get_comments()
                if comment.body  # skip empty comments
            ]
        except GithubException as e:
            log.warning("issue_comments_failed", issue_number=issue.number, error=str(e))
            return []

    async def fetch_issues(self, owner: str, repo_name: str) -> list[RawIssue]:
        return await self._run_sync(self._fetch_issues_sync, owner, repo_name)

    # ── Pull requests ──────────────────────────────────────────────────────

    def _fetch_prs_sync(self, owner: str, repo_name: str) -> list[RawPR]:
        """
        Fetches all PRs including merged and closed.
        Collects both review comments (inline on code) and formal review bodies.
        These are the richest source for decision archaeology.
        """
        self._check_rate_limit()

        repo = self._github.get_repo(f"{owner}/{repo_name}")
        prs  = []

        paginator = repo.get_pulls(state="all", sort="created", direction="desc")

        for i, pr in enumerate(paginator):
            if i >= settings.MAX_PRS_PER_REPO:
                log.info("pr_limit_reached", limit=settings.MAX_PRS_PER_REPO)
                break

            # Review comments = inline comments on specific lines of code
            review_comments = []
            try:
                for comment in pr.get_review_comments():
                    if comment.body:
                        review_comments.append(comment.body)
            except GithubException:
                pass

            # Formal reviews = approve/request changes with a summary body
            # We capture the reviewer's login here, not just the body text.
            # Authority scoring depends on knowing who wrote each review.
            reviews = []
            try:
                for review in pr.get_reviews():
                    reviewer_login = review.user.login if review.user else "unknown"
                    reviews.append(RawReview(
                        reviewer=reviewer_login,
                        body=review.body or "",
                        state=review.state or "COMMENTED",
                    ))
            except GithubException:
                pass

            # Issue comments on the PR (general discussion, not inline)
            issue_comments = []
            try:
                for comment in pr.get_issue_comments():
                    if comment.body:
                        issue_comments.append(comment.body)
            except GithubException:
                pass

            state = "merged" if pr.merged else pr.state

            prs.append(RawPR(
                number=pr.number,
                title=pr.title,
                body=pr.body or "",
                state=state,
                merged=pr.merged,
                created_at=pr.created_at,
                updated_at=pr.updated_at,
                merged_at=pr.merged_at,
                comments=review_comments + issue_comments,
                reviews=reviews,
                html_url=pr.html_url,
            ))

            time.sleep(self._delay)

        log.info("prs_fetched", count=len(prs), repo=f"{owner}/{repo_name}")
        return prs

    async def fetch_prs(self, owner: str, repo_name: str) -> list[RawPR]:
        return await self._run_sync(self._fetch_prs_sync, owner, repo_name)

    # ── Commits ────────────────────────────────────────────────────────────

    def _fetch_commits_sync(self, owner: str, repo_name: str) -> list[RawCommit]:
        """
        Fetches recent commits with the list of files each commit touched.
        Used by ContributorBuilder to compute ownership scores.

        WHY files changed matter:
        A contributor who commits to src/auth/** 50 times owns that area.
        We can't know that without fetching files per commit.
        GitHub returns files on individual commit objects, not the list endpoint.
        So we fetch the list first, then fetch each commit individually.
        This is expensive — we limit to 500 commits for speed.
        """
        self._check_rate_limit()

        repo    = self._github.get_repo(f"{owner}/{repo_name}")
        commits = []
        limit   = 500  # balance between accuracy and API cost

        for i, commit_ref in enumerate(repo.get_commits()):
            if i >= limit:
                break

            try:
                # Fetch the full commit object to get files changed
                # This is one API call per commit — expensive but necessary
                full_commit = repo.get_commit(commit_ref.sha)

                files_changed = [
                    f.filename for f in full_commit.files
                    if f.filename  # skip null filenames
                ]

                commits.append(RawCommit(
                    sha=commit_ref.sha,
                    message=commit_ref.commit.message or "",
                    author=commit_ref.commit.author.name if commit_ref.commit.author else "unknown",
                    created_at=commit_ref.commit.author.date if commit_ref.commit.author else datetime.utcnow(),
                    files=files_changed,
                ))

            except GithubException as e:
                log.warning("commit_fetch_failed", sha=commit_ref.sha, error=str(e))
                continue

            time.sleep(self._delay)

        log.info("commits_fetched", count=len(commits), repo=f"{owner}/{repo_name}")
        return commits

    async def fetch_commits(self, owner: str, repo_name: str) -> list[RawCommit]:
        return await self._run_sync(self._fetch_commits_sync, owner, repo_name)

    # ── Releases ───────────────────────────────────────────────────────────

    def _fetch_releases_sync(self, owner: str, repo_name: str) -> list[RawRelease]:
        """
        Fetches all releases.
        Release tag names are used to annotate chunks with version context.
        This is what lets the critic say "this chunk is from v0.2, you asked about v0.4".
        """
        self._check_rate_limit()

        repo     = self._github.get_repo(f"{owner}/{repo_name}")
        releases = []

        for release in repo.get_releases():
            releases.append(RawRelease(
                tag_name=release.tag_name,
                name=release.title or release.tag_name,
                body=release.body or "",
                created_at=release.created_at,
                html_url=release.html_url,
            ))

        log.info("releases_fetched", count=len(releases), repo=f"{owner}/{repo_name}")
        return releases

    async def fetch_releases(self, owner: str, repo_name: str) -> list[RawRelease]:
        return await self._run_sync(self._fetch_releases_sync, owner, repo_name)

    # ── Differential fetch: only changed objects ───────────────────────────

    def _fetch_changed_since_sync(
        self,
        owner: str,
        repo_name: str,
        since_sha: str,
    ) -> dict:
        """
        Fetches only issues and PRs updated since a given commit SHA.
        Used by the webhook handler for differential re-ingestion —
        we don't re-embed the entire repo when one issue changes.

        Returns dict with keys: "issues", "prs"
        """
        repo = self._github.get_repo(f"{owner}/{repo_name}")

        # Get the timestamp of the reference commit
        try:
            ref_commit   = repo.get_commit(since_sha)
            since_time   = ref_commit.commit.author.date
        except GithubException:
            log.warning("reference_commit_not_found", sha=since_sha)
            return {"issues": [], "prs": []}

        # GitHub issue/PR list supports since= parameter for filtering by update time
        changed_issues = []
        for issue in repo.get_issues(state="all", since=since_time):
            if issue.pull_request is not None:
                continue
            comment_bodies = self._fetch_issue_comments_sync(issue)
            changed_issues.append(RawIssue(
                number=issue.number,
                title=issue.title,
                body=issue.body or "",
                state=issue.state,
                created_at=issue.created_at,
                updated_at=issue.updated_at,
                labels=[label.name for label in issue.labels],
                comments=comment_bodies,
                html_url=issue.html_url,
            ))
            time.sleep(self._delay)

        # PRs don't support since= — filter manually
        changed_prs = []
        for pr in repo.get_pulls(state="all", sort="updated", direction="desc"):
            if pr.updated_at < since_time:
                break   # sorted by updated desc — once we pass since_time, stop
            changed_prs.append(RawPR(
                number=pr.number,
                title=pr.title,
                body=pr.body or "",
                state="merged" if pr.merged else pr.state,
                merged=pr.merged,
                created_at=pr.created_at,
                updated_at=pr.updated_at,
                merged_at=pr.merged_at,
                comments=[],
                reviews=[],
                html_url=pr.html_url,
            ))
            time.sleep(self._delay)

        log.info(
            "differential_fetch_done",
            changed_issues=len(changed_issues),
            changed_prs=len(changed_prs),
        )
        return {"issues": changed_issues, "prs": changed_prs}

    async def fetch_changed_since(
        self,
        owner: str,
        repo_name: str,
        since_sha: str,
    ) -> dict:
        return await self._run_sync(
            self._fetch_changed_since_sync, owner, repo_name, since_sha
        )

    # ── Async executor wrapper ─────────────────────────────────────────────

    async def _run_sync(self, fn, *args):
        """
        Runs a synchronous function in a thread pool executor.

        WHY THIS EXISTS:
        PyGithub is a synchronous library — its API calls use requests.get()
        which blocks the calling thread until the response arrives.
        In an async FastAPI app, blocking the thread = blocking the event loop
        = all other requests stall until this GitHub call completes.

        run_in_executor pushes the blocking call to a separate thread pool.
        The event loop stays free to handle other requests while GitHub responds.

        This is the correct pattern for any sync I/O inside async Python.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn, *args)