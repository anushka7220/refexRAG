# github_fetcher.py
#
# THE ONLY FILE that talks to the GitHub API.
# Everything GitHub-related — HTTP calls, pagination, timeout, retry,
# and rate limiting — lives here. Nothing else in the codebase imports
# requests or PyGithub directly.
#
# ── RESILIENCE STRATEGY (three independent layers) ────────────────────────
#
# Layer 1 — Client-level (PyGithub GithubRetry):
#   The Github() client is configured with a long timeout and a GithubRetry
#   object. GithubRetry is PyGithub's own urllib3.Retry subclass that ALSO
#   understands GitHub's rate-limit headers. It transparently retries 403/429
#   (rate limit) and 5xx (server) responses with exponential backoff, before
#   the exception ever reaches our code.
#
# Layer 2 — Call-level (tenacity):
#   Network read timeouts and dropped connections are NOT HTTP status codes,
#   so GithubRetry does not catch them. We wrap the small, risky, per-item
#   calls (one issue's comments, one commit's files) with a tenacity @retry
#   that specifically catches timeout / connection errors and retries THAT
#   one call a few times with backoff — instead of aborting the whole repo.
#
# Layer 3 — Loop-level (skip-and-continue):
#   If an individual item still fails after its retries, we log it and skip
#   that ONE item. One unreachable issue must never fail a 3000-issue repo.
#   Only a failure fetching the repo object itself is fatal.
#
# WHY THIS MATTERS:
#   The old version caught only GithubException. A ReadTimeoutError is a
#   urllib3/requests exception, so it slipped past every try/except and
#   killed ingestion. That is the bug these three layers fix.

import asyncio
import io
import os
import tarfile
import tempfile
import time
import logging
import requests as _requests
import structlog
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from github import Github, GithubException, GithubRetry
from requests.exceptions import (
    ReadTimeout,
    ConnectTimeout,
    Timeout,
    ConnectionError as RequestsConnectionError,
    ChunkedEncodingError,
)
from urllib3.exceptions import ReadTimeoutError, ProtocolError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from app.core.config import settings

log = structlog.get_logger(__name__)
_stdlib_log = logging.getLogger(__name__)

# Network-level failures that mean "try again", not "give up".
# These are raised by requests/urllib3, NOT by PyGithub, which is exactly
# why the old GithubException-only handling missed them.
TRANSIENT_NETWORK_ERRORS = (
    ReadTimeout,
    ConnectTimeout,
    Timeout,
    RequestsConnectionError,
    ChunkedEncodingError,
    ReadTimeoutError,
    ProtocolError,
)

# Reusable tenacity policy for a single risky call.
# 4 attempts, exponential backoff 2s → 4s → 8s (capped at 20s), then give up
# and let the caller's loop-level handler skip the item.
_retry_transient = retry(
    retry=retry_if_exception_type(TRANSIENT_NETWORK_ERRORS),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    before_sleep=before_sleep_log(_stdlib_log, logging.WARNING),
    reraise=True,
)


# ── Raw data shapes returned by this fetcher ──────────────────────────────

@dataclass
class RawIssue:
    number:     int
    title:      str
    body:       str
    state:      str
    created_at: datetime
    updated_at: datetime
    labels:     list[str] = field(default_factory=list)
    comments:   list[str] = field(default_factory=list)
    html_url:   str       = ""


@dataclass
class RawReview:
    reviewer: str
    body:     str
    state:    str


@dataclass
class RawPR:
    number:        int
    title:         str
    body:          str
    state:         str
    merged:        bool
    created_at:    datetime
    updated_at:    datetime
    merged_at:     Optional[datetime]
    comments:      list[str]       = field(default_factory=list)
    reviews:       list[RawReview] = field(default_factory=list)
    files_changed: list[str]       = field(default_factory=list)   # join key to code chunks
    html_url:      str             = ""


@dataclass
class RawCommit:
    sha:        str
    message:    str
    author:     str
    created_at: datetime
    files:      list[str] = field(default_factory=list)


@dataclass
class RawRelease:
    tag_name:   str
    name:       str
    body:       str
    created_at: datetime
    html_url:   str = ""


@dataclass
class SourceFileRaw:
    """One source file extracted from the repo tarball."""
    path:    str
    content: str


@dataclass
class RepoMeta:
    owner:             str
    name:              str
    description:       str
    default_branch:    str
    latest_commit_sha: str
    html_url:          str


# ── GitHub client wrapper ──────────────────────────────────────────────────

class GitHubFetcher:
    """
    Fetches all relevant data from a public GitHub repository, resiliently.

    Usage:
        fetcher = GitHubFetcher(github_token="ghp_...")
        meta    = await fetcher.fetch_repo_meta("https://github.com/owner/repo")
        issues  = await fetcher.fetch_issues(meta.owner, meta.name)
    """

    # How long (seconds) to wait on a single socket read before giving up.
    # The old default was 15s, which pypdf's larger comment threads exceeded.
    HTTP_TIMEOUT = 30

    # Commits are the most expensive fetch: one extra API call PER commit to
    # get its changed files. On a huge repo that is thousands of slow calls.
    # We cap how many commits we enrich with file lists. Beyond this cap we
    # still record the commit (sha, message, author) but skip the files call.
    MAX_COMMITS_WITH_FILES = 300

    # Absolute cap on commits scanned at all.
    MAX_COMMITS = 500

    def __init__(self, github_token: Optional[str] = None):
        # GithubRetry handles 403/429/5xx at the HTTP layer with backoff and
        # respects GitHub's rate-limit reset headers automatically.
        gh_retry = GithubRetry(
            total=6,
            backoff_factor=2.0,           # 2s, 4s, 8s, 16s ...
            status_forcelist=[403, 429, 500, 502, 503, 504],
        )

        self._github = Github(
            login_or_token=github_token,
            per_page=100,
            timeout=self.HTTP_TIMEOUT,
            retry=gh_retry,
        )
        self._delay = settings.github_api_delay_seconds
        log.info(
            "github_fetcher_init",
            authenticated=github_token is not None,
            timeout=self.HTTP_TIMEOUT,
        )

    # ── Rate limit guard ───────────────────────────────────────────────────

    def _check_rate_limit(self):
        """
        Defensive pre-check. Sleeps if quota is nearly exhausted.
        GithubRetry already handles hitting the limit mid-flight; this just
        avoids starting a big batch when we're about to run dry.

        Note: reset_time from GitHub is timezone-aware (UTC). We compare it
        against a timezone-aware now() to avoid the "can't subtract
        offset-naive and offset-aware datetimes" error.
        """
        try:
            rate_limit = self._github.get_rate_limit()
            core = getattr(rate_limit, "core", None) or getattr(rate_limit, "resources", None)
            if core is not None and hasattr(core, "core"):
                core = core.core

            remaining = getattr(core, "remaining", None) if core else getattr(rate_limit, "remaining", None)
            reset_time = getattr(core, "reset", None) if core else getattr(rate_limit, "reset", None)

            if remaining is None:
                log.debug("rate_limit_check_skipped", reason="unrecognized object shape")
                return

            log.debug("rate_limit_check", remaining=remaining)

            if remaining < 100 and reset_time is not None:
                if reset_time.tzinfo is None:
                    reset_time = reset_time.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                sleep_seconds = max(0, (reset_time - now).total_seconds() + 10)
                log.warning("rate_limit_low", remaining=remaining, sleeping_seconds=sleep_seconds)
                time.sleep(sleep_seconds)

        except Exception as e:
            # Never let the pre-check itself abort ingestion.
            log.debug("rate_limit_check_failed", error=str(e))

    # ── Repo metadata ──────────────────────────────────────────────────────

    def _fetch_repo_meta_sync(self, github_url: str) -> RepoMeta:
        parts = github_url.rstrip("/").split("/")
        owner, name = parts[-2], parts[-1]

        repo = self._github.get_repo(f"{owner}/{name}")
        default_branch = repo.default_branch
        latest_commit = repo.get_branch(default_branch).commit.sha

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

    @_retry_transient
    def _fetch_issue_comments_sync(self, issue) -> list[str]:
        """
        Fetches comment bodies for ONE issue.
        Wrapped with tenacity: a transient timeout here retries just this
        issue's comments, not the whole repo. Catches both GithubException
        (skips, returns empty) and — via the decorator — network timeouts
        (retries up to 4x, then raises to the caller's skip handler).
        """
        try:
            return [c.body for c in issue.get_comments() if c.body]
        except GithubException as e:
            log.warning("issue_comments_github_error", issue_number=issue.number, error=str(e))
            return []

    def _fetch_issues_sync(self, owner: str, repo_name: str) -> list[RawIssue]:
        self._check_rate_limit()
        repo = self._github.get_repo(f"{owner}/{repo_name}")
        issues = []
        skipped = 0

        paginator = repo.get_issues(state="all")
        count = 0
        for issue in paginator:
            if issue.pull_request is not None:
                continue
            if count >= settings.MAX_ISSUES_PER_REPO:
                log.info("issue_limit_reached", limit=settings.MAX_ISSUES_PER_REPO)
                break
            count += 1

            # Loop-level resilience: one bad issue must not kill the batch.
            try:
                comment_bodies = self._fetch_issue_comments_sync(issue)
            except TRANSIENT_NETWORK_ERRORS as e:
                log.warning("issue_comments_timeout_skip", issue_number=issue.number, error=str(e))
                comment_bodies = []
                skipped += 1

            try:
                issues.append(RawIssue(
                    number=issue.number,
                    title=issue.title,
                    body=issue.body or "",
                    state=issue.state,
                    created_at=issue.created_at,
                    updated_at=issue.updated_at,
                    labels=[l.name for l in issue.labels],
                    comments=comment_bodies,
                    html_url=issue.html_url,
                ))
            except TRANSIENT_NETWORK_ERRORS as e:
                log.warning("issue_body_timeout_skip", issue_number=getattr(issue, "number", "?"), error=str(e))
                skipped += 1
                continue

            time.sleep(self._delay)

        log.info("issues_fetched", count=len(issues), skipped=skipped, repo=f"{owner}/{repo_name}")
        return issues

    async def fetch_issues(self, owner: str, repo_name: str) -> list[RawIssue]:
        return await self._run_sync(self._fetch_issues_sync, owner, repo_name)

    # ── Pull requests ──────────────────────────────────────────────────────

    @_retry_transient
    def _fetch_pr_details_sync(self, pr) -> tuple[list[str], list[RawReview], list[str]]:
        """
        Fetches review comments + formal reviews + changed file paths for ONE pr.
        files_changed is the join key linking this PR's discussion to code
        chunks: any code chunk whose file_path appears here gets this PR as
        a linked discussion at retrieval time.
        tenacity-wrapped so a timeout retries just this PR's sub-fetches.
        """
        review_comments = []
        reviews = []
        issue_comments = []
        files_changed = []

        try:
            review_comments = [c.body for c in pr.get_review_comments() if c.body]
        except GithubException:
            pass

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

        try:
            issue_comments = [c.body for c in pr.get_issue_comments() if c.body]
        except GithubException:
            pass

        try:
            # Capped at 100 files per PR: giant refactor PRs would otherwise
            # link to half the repo, which dilutes the join into noise.
            files_changed = [f.filename for f in pr.get_files()[:100] if f.filename]
        except GithubException:
            pass

        return review_comments + issue_comments, reviews, files_changed

    def _fetch_prs_sync(self, owner: str, repo_name: str) -> list[RawPR]:
        self._check_rate_limit()
        repo = self._github.get_repo(f"{owner}/{repo_name}")
        prs = []
        skipped = 0

        paginator = repo.get_pulls(state="all", sort="created", direction="desc")
        for i, pr in enumerate(paginator):
            if i >= settings.MAX_PRS_PER_REPO:
                log.info("pr_limit_reached", limit=settings.MAX_PRS_PER_REPO)
                break

            try:
                comments, reviews, files_changed = self._fetch_pr_details_sync(pr)
            except TRANSIENT_NETWORK_ERRORS as e:
                log.warning("pr_details_timeout_skip", pr_number=pr.number, error=str(e))
                comments, reviews, files_changed = [], [], []
                skipped += 1

            try:
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
                    comments=comments,
                    reviews=reviews,
                    files_changed=files_changed,
                    html_url=pr.html_url,
                ))
            except TRANSIENT_NETWORK_ERRORS as e:
                log.warning("pr_body_timeout_skip", pr_number=getattr(pr, "number", "?"), error=str(e))
                skipped += 1
                continue

            time.sleep(self._delay)

        log.info("prs_fetched", count=len(prs), skipped=skipped, repo=f"{owner}/{repo_name}")
        return prs

    async def fetch_prs(self, owner: str, repo_name: str) -> list[RawPR]:
        return await self._run_sync(self._fetch_prs_sync, owner, repo_name)

    # ── Commits ────────────────────────────────────────────────────────────

    @_retry_transient
    def _fetch_single_commit_files_sync(self, repo, sha: str) -> list[str]:
        """
        Fetches the changed-file list for ONE commit.
        This is the single most expensive call in ingestion (one API round
        trip per commit), so it gets its own tenacity retry and is capped
        by MAX_COMMITS_WITH_FILES in the caller.
        """
        full_commit = repo.get_commit(sha)
        return [f.filename for f in full_commit.files if f.filename]

    def _fetch_commits_sync(self, owner: str, repo_name: str) -> list[RawCommit]:
        self._check_rate_limit()
        repo = self._github.get_repo(f"{owner}/{repo_name}")
        commits = []
        skipped = 0

        for i, commit_ref in enumerate(repo.get_commits()):
            if i >= self.MAX_COMMITS:
                break

            files_changed = []
            # Only enrich the first N commits with their file lists — the
            # expensive per-commit call. Older commits still get recorded,
            # just without files (which only affects ownership granularity).
            if i < self.MAX_COMMITS_WITH_FILES:
                try:
                    files_changed = self._fetch_single_commit_files_sync(repo, commit_ref.sha)
                except TRANSIENT_NETWORK_ERRORS as e:
                    log.warning("commit_files_timeout_skip", sha=commit_ref.sha[:8], error=str(e))
                    skipped += 1
                except GithubException as e:
                    log.warning("commit_files_github_error", sha=commit_ref.sha[:8], error=str(e))
                    skipped += 1

            try:
                author = commit_ref.commit.author
                commits.append(RawCommit(
                    sha=commit_ref.sha,
                    message=commit_ref.commit.message or "",
                    author=author.name if author else "unknown",
                    created_at=author.date if author else datetime.now(timezone.utc),
                    files=files_changed,
                ))
            except TRANSIENT_NETWORK_ERRORS as e:
                log.warning("commit_meta_timeout_skip", sha=commit_ref.sha[:8], error=str(e))
                skipped += 1
                continue

            # Only sleep when we actually made the expensive files call.
            if i < self.MAX_COMMITS_WITH_FILES:
                time.sleep(self._delay)

        log.info("commits_fetched", count=len(commits), skipped=skipped, repo=f"{owner}/{repo_name}")
        return commits

    async def fetch_commits(self, owner: str, repo_name: str) -> list[RawCommit]:
        return await self._run_sync(self._fetch_commits_sync, owner, repo_name)

    # ── Releases ───────────────────────────────────────────────────────────

    def _fetch_releases_sync(self, owner: str, repo_name: str) -> list[RawRelease]:
        self._check_rate_limit()
        repo = self._github.get_repo(f"{owner}/{repo_name}")
        releases = []

        try:
            for release in repo.get_releases():
                releases.append(RawRelease(
                    tag_name=release.tag_name,
                    name=release.title or release.tag_name,
                    body=release.body or "",
                    created_at=release.created_at,
                    html_url=release.html_url,
                ))
        except TRANSIENT_NETWORK_ERRORS as e:
            log.warning("releases_timeout_partial", got=len(releases), error=str(e))
        except GithubException as e:
            log.warning("releases_github_error", error=str(e))

        log.info("releases_fetched", count=len(releases), repo=f"{owner}/{repo_name}")
        return releases

    async def fetch_releases(self, owner: str, repo_name: str) -> list[RawRelease]:
        return await self._run_sync(self._fetch_releases_sync, owner, repo_name)

    # ── Differential fetch ─────────────────────────────────────────────────

    def _fetch_changed_since_sync(self, owner: str, repo_name: str, since_sha: str) -> dict:
        repo = self._github.get_repo(f"{owner}/{repo_name}")

        try:
            ref_commit = repo.get_commit(since_sha)
            since_time = ref_commit.commit.author.date
        except GithubException:
            log.warning("reference_commit_not_found", sha=since_sha)
            return {"issues": [], "prs": []}

        changed_issues = []
        for issue in repo.get_issues(state="all", since=since_time):
            if issue.pull_request is not None:
                continue
            try:
                comment_bodies = self._fetch_issue_comments_sync(issue)
            except TRANSIENT_NETWORK_ERRORS:
                comment_bodies = []
            changed_issues.append(RawIssue(
                number=issue.number,
                title=issue.title,
                body=issue.body or "",
                state=issue.state,
                created_at=issue.created_at,
                updated_at=issue.updated_at,
                labels=[l.name for l in issue.labels],
                comments=comment_bodies,
                html_url=issue.html_url,
            ))
            time.sleep(self._delay)

        changed_prs = []
        for pr in repo.get_pulls(state="all", sort="updated", direction="desc"):
            if pr.updated_at < since_time:
                break
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

        log.info("differential_fetch_done", changed_issues=len(changed_issues), changed_prs=len(changed_prs))
        return {"issues": changed_issues, "prs": changed_prs}

    async def fetch_changed_since(self, owner: str, repo_name: str, since_sha: str) -> dict:
        return await self._run_sync(self._fetch_changed_since_sync, owner, repo_name, since_sha)

    # ── Source code via tarball: ONE api call for the whole codebase ──────

    # Directories that are dependency or build output, never source.
    SKIP_DIRS = {
        "node_modules", "venv", ".venv", "env", ".git", "dist", "build",
        "__pycache__", "vendor", ".next", "target", ".tox", "site-packages",
        "coverage", ".pytest_cache", "migrations",
    }

    # Per-file and total caps. Embedding is the MacBook's bottleneck, so the
    # fetcher enforces limits before chunks ever exist.
    MAX_FILE_BYTES = 100_000
    MAX_SOURCE_FILES = 250

    def _fetch_source_files_sync(
        self,
        owner: str,
        repo_name: str,
        priority_paths: Optional[set] = None,
    ) -> list[SourceFileRaw]:
        """
        Downloads the ENTIRE repository as one tarball (a single API call),
        extracts it to a temp dir, and returns filtered source files.

        WHY TARBALL, NOT THE CONTENTS API:
        The Contents API costs one call per file. A 400-file repo would cost
        400 rate-limited calls at 700ms delay each. The tarball endpoint
        returns the whole tree in one call, then everything else is local
        disk I/O at zero API cost. This makes code the CHEAPEST thing we
        ingest instead of the most expensive.

        priority_paths: file paths known to appear in commit/PR history.
        When the MAX_SOURCE_FILES cap bites, discussion-linked files are kept
        FIRST, because those are the files the product's code-to-discussion
        join can actually enrich. The cap serves the thesis.
        """
        self._check_rate_limit()
        repo = self._github.get_repo(f"{owner}/{repo_name}")
        tarball_url = repo.get_archive_link("tarball")

        headers = {}
        # Reuse the client's token for the archive download if present.
        auth = getattr(self._github, "_Github__requester", None)
        token = getattr(auth, "_Requester__authorizationHeader", None) if auth else None
        if token:
            headers["Authorization"] = token

        resp = _requests.get(tarball_url, headers=headers, timeout=120, stream=True)
        resp.raise_for_status()

        results: list[SourceFileRaw] = []
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tf:
                tf.extractall(tmp, filter="data")

            # The tarball wraps everything in one top-level dir like
            # "owner-repo-sha/". Strip it so paths are repo-relative.
            roots = os.listdir(tmp)
            if not roots:
                return []
            base = os.path.join(tmp, roots[0])

            candidates: list[tuple[str, str]] = []
            for dirpath, dirnames, filenames in os.walk(base):
                # Prune skip dirs in place so os.walk never descends into them
                dirnames[:] = [d for d in dirnames if d not in self.SKIP_DIRS and not d.startswith(".")]
                for fname in filenames:
                    full = os.path.join(dirpath, fname)
                    rel = os.path.relpath(full, base)
                    candidates.append((rel, full))

            # Priority ordering: README first, then discussion-linked files,
            # then the rest. The cap trims from the tail.
            def sort_key(item):
                rel, _ = item
                is_readme = os.path.basename(rel).lower().startswith("readme")
                in_priority = priority_paths is not None and rel in priority_paths
                return (0 if is_readme else 1 if in_priority else 2, rel)

            candidates.sort(key=sort_key)

            from app.services.ingestion.code_chunker import language_for
            for rel, full in candidates:
                if len(results) >= self.MAX_SOURCE_FILES:
                    log.info("source_file_cap_reached", cap=self.MAX_SOURCE_FILES)
                    break
                if language_for(rel) is None:
                    continue
                try:
                    size = os.path.getsize(full)
                    if size == 0 or size > self.MAX_FILE_BYTES:
                        continue
                    with open(full, "r", encoding="utf-8", errors="ignore") as fh:
                        content = fh.read()
                    # Skip minified or generated files: one enormous line.
                    if content and max((len(l) for l in content.splitlines()), default=0) > 2000:
                        continue
                    results.append(SourceFileRaw(path=rel, content=content))
                except OSError:
                    continue

        log.info("source_files_fetched", count=len(results), repo=f"{owner}/{repo_name}")
        return results

    async def fetch_source_files(
        self,
        owner: str,
        repo_name: str,
        priority_paths: Optional[set] = None,
    ) -> list[SourceFileRaw]:
        return await self._run_sync(
            self._fetch_source_files_sync, owner, repo_name, priority_paths
        )

    # ── Async executor wrapper ─────────────────────────────────────────────

    async def _run_sync(self, fn, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn, *args)