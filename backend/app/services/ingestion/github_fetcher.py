# github_fetcher.py
#
# THE ONLY FILE that talks to the GitHub API.
# Everything GitHub-related — HTTP calls, pagination, timeout, retry,
# and rate limiting — lives here. Nothing else in the codebase imports
# requests or PyGithub directly.
#
# ── RESILIENCE STRATEGY (four independent layers) ─────────────────────────
#
# Layer 1 — Client-level (PyGithub GithubRetry):
#   Configured with a long timeout and a GithubRetry that understands
#   GitHub's rate-limit headers. Transparently retries 403/429/5xx with
#   backoff before the exception reaches our code.
#
# Layer 2 — Call-level (tenacity):
#   Network read timeouts and dropped connections are NOT HTTP status codes,
#   so GithubRetry does not catch them. The small per-item calls are wrapped
#   with tenacity that retries just that one call, not the whole repo.
#
# Layer 3 — Loop-level (skip-and-continue):
#   If an item still fails after retries, log it and skip that ONE item.
#
# Layer 4 — List-endpoint level (degrade-to-empty):
#   get_pulls / get_issues / get_commits fetch pages LAZILY during iteration,
#   and that page fetch can raise GithubException — a 404 if the repo was
#   renamed/moved or has PRs disabled. Wrap each iteration so a list-endpoint
#   failure degrades to "none of that kind" and lets the others still index.
#   (This is what killed the sindresorhus/is-online ingestion.)

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

TRANSIENT_NETWORK_ERRORS = (
    ReadTimeout,
    ConnectTimeout,
    Timeout,
    RequestsConnectionError,
    ChunkedEncodingError,
    ReadTimeoutError,
    ProtocolError,
)

_retry_transient = retry(
    retry=retry_if_exception_type(TRANSIENT_NETWORK_ERRORS),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    before_sleep=before_sleep_log(_stdlib_log, logging.WARNING),
    reraise=True,
)


# ── Raw data shapes ────────────────────────────────────────────────────────

@dataclass
class RawIssue:
    number:     int
    title:      str
    body:       str
    state:      str
    created_at: datetime
    updated_at: datetime
    labels:     list = field(default_factory=list)
    comments:   list = field(default_factory=list)
    html_url:   str  = ""


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
    comments:      list = field(default_factory=list)
    reviews:       list = field(default_factory=list)
    files_changed: list = field(default_factory=list)   # join key to code chunks
    html_url:      str  = ""


@dataclass
class RawCommit:
    sha:        str
    message:    str
    author:     str
    created_at: datetime
    files:      list = field(default_factory=list)


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
    HTTP_TIMEOUT = 30
    MAX_COMMITS_WITH_FILES = 60
    MAX_COMMITS = 200
    MAX_COMMENTS_PER_ISSUE = 20
    MAX_COMMENTS_PER_PR = 20

    def __init__(self, github_token: Optional[str] = None):
        gh_retry = GithubRetry(
            total=6,
            backoff_factor=2.0,
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
        try:
            rate_limit = self._github.get_rate_limit()
            core = getattr(rate_limit, "core", None) or getattr(rate_limit, "resources", None)
            if core is not None and hasattr(core, "core"):
                core = core.core
            remaining = getattr(core, "remaining", None) if core else getattr(rate_limit, "remaining", None)
            reset_time = getattr(core, "reset", None) if core else getattr(rate_limit, "reset", None)
            if remaining is None:
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
    def _fetch_issue_comments_sync(self, issue) -> list:
        try:
            if getattr(issue, "comments", 0) == 0:
                return []
            bodies = []
            for c in issue.get_comments():
                if c.body:
                    bodies.append(c.body)
                if len(bodies) >= self.MAX_COMMENTS_PER_ISSUE:
                    break
            return bodies
        except GithubException as e:
            log.warning("issue_comments_github_error", issue_number=issue.number, error=str(e))
            return []

    def _fetch_issues_sync(self, owner: str, repo_name: str) -> list:
        self._check_rate_limit()
        repo = self._github.get_repo(f"{owner}/{repo_name}")
        issues = []
        skipped = 0
        paginator = repo.get_issues(state="all")
        count = 0

        # Layer 4: paginator fetches pages lazily; a 404 here (issues disabled
        # or repo moved) degrades to what we gathered rather than failing all.
        try:
            for issue in paginator:
                if issue.pull_request is not None:
                    continue
                if count >= settings.MAX_ISSUES_PER_REPO:
                    log.info("issue_limit_reached", limit=settings.MAX_ISSUES_PER_REPO)
                    break
                count += 1

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
        except GithubException as e:
            log.warning("issue_list_unavailable", repo=f"{owner}/{repo_name}", error=str(e))
            return issues

        log.info("issues_fetched", count=len(issues), skipped=skipped, repo=f"{owner}/{repo_name}")
        return issues

    async def fetch_issues(self, owner: str, repo_name: str) -> list:
        return await self._run_sync(self._fetch_issues_sync, owner, repo_name)

    # ── Pull requests ──────────────────────────────────────────────────────

    @_retry_transient
    def _fetch_pr_details_sync(self, pr) -> tuple:
        review_comments = []
        reviews = []
        issue_comments = []
        files_changed = []

        try:
            for c in pr.get_review_comments():
                if c.body:
                    review_comments.append(c.body)
                if len(review_comments) >= self.MAX_COMMENTS_PER_PR:
                    break
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
                if len(reviews) >= self.MAX_COMMENTS_PER_PR:
                    break
        except GithubException:
            pass

        try:
            if getattr(pr, "comments", 0):
                for c in pr.get_issue_comments():
                    if c.body:
                        issue_comments.append(c.body)
                    if len(issue_comments) >= self.MAX_COMMENTS_PER_PR:
                        break
        except GithubException:
            pass

        try:
            files_changed = [f.filename for f in pr.get_files()[:100] if f.filename]
        except GithubException:
            pass

        return review_comments + issue_comments, reviews, files_changed

    def _fetch_prs_sync(self, owner: str, repo_name: str) -> list:
        self._check_rate_limit()
        repo = self._github.get_repo(f"{owner}/{repo_name}")
        prs = []
        skipped = 0
        paginator = repo.get_pulls(state="all", sort="created", direction="desc")

        # Layer 4: iterating the paginator lazily fetches pages, and THAT call
        # can raise GithubException — a 404 when the repo was renamed/moved or
        # has PRs disabled (exactly what killed sindresorhus/is-online). Wrap
        # the whole loop so a PR-list failure degrades to the PRs gathered so
        # far instead of aborting the entire ingestion task.
        try:
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
        except GithubException as e:
            log.warning("pr_list_unavailable", repo=f"{owner}/{repo_name}", error=str(e))
            return prs

        log.info("prs_fetched", count=len(prs), skipped=skipped, repo=f"{owner}/{repo_name}")
        return prs

    async def fetch_prs(self, owner: str, repo_name: str) -> list:
        return await self._run_sync(self._fetch_prs_sync, owner, repo_name)

    # ── Commits ────────────────────────────────────────────────────────────

    @_retry_transient
    def _fetch_single_commit_files_sync(self, repo, sha: str) -> list:
        full_commit = repo.get_commit(sha)
        return [f.filename for f in full_commit.files if f.filename]

    def _fetch_commits_sync(self, owner: str, repo_name: str) -> list:
        self._check_rate_limit()
        repo = self._github.get_repo(f"{owner}/{repo_name}")
        commits = []
        skipped = 0

        # Layer 4: commit list is paginated lazily too, can 404 on empty/moved.
        try:
            for i, commit_ref in enumerate(repo.get_commits()):
                if i >= self.MAX_COMMITS:
                    break

                files_changed = []
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

                if i < self.MAX_COMMITS_WITH_FILES:
                    time.sleep(self._delay)
        except GithubException as e:
            log.warning("commit_list_unavailable", repo=f"{owner}/{repo_name}", error=str(e))
            return commits

        log.info("commits_fetched", count=len(commits), skipped=skipped, repo=f"{owner}/{repo_name}")
        return commits

    async def fetch_commits(self, owner: str, repo_name: str) -> list:
        return await self._run_sync(self._fetch_commits_sync, owner, repo_name)

    # ── Releases ───────────────────────────────────────────────────────────

    def _fetch_releases_sync(self, owner: str, repo_name: str) -> list:
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

    async def fetch_releases(self, owner: str, repo_name: str) -> list:
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
        try:
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
        except GithubException as e:
            log.warning("differential_issue_list_unavailable", repo=f"{owner}/{repo_name}", error=str(e))

        changed_prs = []
        try:
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
        except GithubException as e:
            log.warning("differential_pr_list_unavailable", repo=f"{owner}/{repo_name}", error=str(e))

        log.info("differential_fetch_done", changed_issues=len(changed_issues), changed_prs=len(changed_prs))
        return {"issues": changed_issues, "prs": changed_prs}

    async def fetch_changed_since(self, owner: str, repo_name: str, since_sha: str) -> dict:
        return await self._run_sync(self._fetch_changed_since_sync, owner, repo_name, since_sha)

    # ── Source code via tarball ────────────────────────────────────────────

    SKIP_DIRS = {
        "node_modules", "venv", ".venv", "env", ".git", "dist", "build",
        "__pycache__", "vendor", ".next", "target", ".tox", "site-packages",
        "coverage", ".pytest_cache", "migrations",
    }
    MAX_FILE_BYTES = 100_000
    MAX_SOURCE_FILES = 250

    def _fetch_source_files_sync(self, owner: str, repo_name: str, priority_paths: Optional[set] = None) -> list:
        self._check_rate_limit()
        repo = self._github.get_repo(f"{owner}/{repo_name}")
        tarball_url = repo.get_archive_link("tarball")

        headers = {}
        auth = getattr(self._github, "_Github__requester", None)
        token = getattr(auth, "_Requester__authorizationHeader", None) if auth else None
        if token:
            headers["Authorization"] = token

        resp = _requests.get(tarball_url, headers=headers, timeout=120, stream=True)
        resp.raise_for_status()

        results = []
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tf:
                # filter="data" is Python 3.12+ only; fall back for 3.10/3.11.
                try:
                    tf.extractall(tmp, filter="data")
                except TypeError:
                    tmp_real = os.path.realpath(tmp)
                    for member in tf.getmembers():
                        dest = os.path.realpath(os.path.join(tmp, member.name))
                        if not dest.startswith(tmp_real + os.sep):
                            log.warning("tarball_member_skipped_unsafe", name=member.name)
                            continue
                        tf.extract(member, tmp)

            roots = os.listdir(tmp)
            if not roots:
                return []
            base = os.path.join(tmp, roots[0])

            candidates = []
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [d for d in dirnames if d not in self.SKIP_DIRS and not d.startswith(".")]
                for fname in filenames:
                    full = os.path.join(dirpath, fname)
                    rel = os.path.relpath(full, base)
                    candidates.append((rel, full))

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
                    if content and max((len(l) for l in content.splitlines()), default=0) > 2000:
                        continue
                    results.append(SourceFileRaw(path=rel, content=content))
                except OSError:
                    continue

        log.info("source_files_fetched", count=len(results), repo=f"{owner}/{repo_name}")
        return results

    async def fetch_source_files(self, owner: str, repo_name: str, priority_paths: Optional[set] = None) -> list:
        return await self._run_sync(self._fetch_source_files_sync, owner, repo_name, priority_paths)

    # ── Async executor wrapper ─────────────────────────────────────────────

    async def _run_sync(self, fn, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn, *args)