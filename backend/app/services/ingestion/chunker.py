# chunker.py
#
# Splits raw GitHub objects into embedding-ready chunks with metadata attached.
#
# WHY CHUNKING EXISTS:
# bge-large-en-v1.5 has a 512-token input limit. GitHub issue bodies can be
# 10,000+ tokens. Without chunking, the model silently truncates everything
# past token 512. You'd embed only the first ~400 words of every long issue.
#
# WHY OVERLAP:
# If we split at hard boundaries, a sentence can be cut in half:
#   Chunk 1: "...the root cause was identified as an off-by-one error in the"
#   Chunk 2: "pagination logic, which was fixed in commit abc123."
# Neither chunk alone tells the full story. With 50-token overlap, the end
# of chunk 1 appears at the start of chunk 2 — context is preserved.
#
# WHAT GETS ATTACHED TO EVERY CHUNK:
# source_type, source_id, status, created_at, version_tag, content_hash
# These fields are what make the critic possible — it reads metadata, not text.

import hashlib
import re
from datetime import datetime
from typing import Optional

import tiktoken
import structlog

from app.core.config import settings
from app.models.chunk import Chunk, SourceType, SourceStatus
from app.services.ingestion.github_fetcher import (
    RawIssue, RawPR, RawCommit, RawRelease
)
from app.utils.version_extractor import extract_version_tag

log = structlog.get_logger(__name__)

# Tokenizer — same one used by most embedding models for token counting.
# "cl100k_base" is the GPT-4 tokenizer, a good approximation for bge-large.
# We use it only for counting — bge-large does its own tokenization internally.
_tokenizer = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_tokenizer.encode(text))


def _compute_hash(text: str) -> str:
    """
    SHA-256 hash of the chunk content.
    Used as the deduplication key in pgvector.
    If this hash already exists in the DB, we skip re-embedding.
    Two identical chunks (same repo indexed by two users) share one DB row.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_github_url(
    owner: str,
    repo_name: str,
    source_type: SourceType,
    source_id: str,
) -> str:
    """Builds the full GitHub URL for a chunk's source — shown in citation cards."""
    base = f"https://github.com/{owner}/{repo_name}"
    if source_type == "issue":
        return f"{base}/issues/{source_id}"
    elif source_type in ("pr", "comment"):
        return f"{base}/pull/{source_id}"
    elif source_type == "commit":
        return f"{base}/commit/{source_id}"
    elif source_type == "release":
        return f"{base}/releases/tag/{source_id}"
    return base


def _split_text(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """
    Splits text into overlapping chunks that fit within max_tokens.

    Strategy:
    1. Split into sentences (rough — period/newline boundaries).
    2. Greedily accumulate sentences until we hit max_tokens.
    3. Start the next chunk overlap_tokens back from where we stopped.

    This preserves sentence boundaries — we never cut mid-sentence.
    A hard character split would be faster but produces incoherent chunk edges.

    Args:
        text:           The text to split.
        max_tokens:     Maximum tokens per chunk (default 512 from config).
        overlap_tokens: How many tokens to repeat between adjacent chunks (default 50).

    Returns:
        List of text strings, each under max_tokens.
    """
    if not text or not text.strip():
        return []

    # If the whole text fits in one chunk, return it as-is
    if _count_tokens(text) <= max_tokens:
        return [text.strip()]

    # Split into sentences on period, newline, or double newline
    # This is intentionally simple — perfect sentence splitting isn't worth
    # the complexity here since we have overlap to handle edge cases
    sentences = re.split(r'(?<=[.!?])\s+|\n\n+|\n(?=[A-Z#*-])', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks     = []
    current    = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _count_tokens(sentence)

        # Single sentence exceeds max — split it by words as fallback
        if sentence_tokens > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            # Hard split by words
            words  = sentence.split()
            buffer = []
            buf_tokens = 0
            for word in words:
                wt = _count_tokens(word + " ")
                if buf_tokens + wt > max_tokens and buffer:
                    chunks.append(" ".join(buffer))
                    # Overlap: keep last overlap_tokens worth of words
                    buffer     = buffer[max(0, len(buffer) - overlap_tokens // 4):]
                    buf_tokens = _count_tokens(" ".join(buffer))
                buffer.append(word)
                buf_tokens += wt
            if buffer:
                current     = buffer
                current_tokens = buf_tokens
            continue

        # Adding this sentence would exceed the limit — flush current chunk
        if current_tokens + sentence_tokens > max_tokens and current:
            chunks.append(" ".join(current))

            # Build overlap: walk backwards through current sentences
            # until we've collected overlap_tokens worth
            overlap    = []
            overlap_t  = 0
            for s in reversed(current):
                st = _count_tokens(s)
                if overlap_t + st > overlap_tokens:
                    break
                overlap.insert(0, s)
                overlap_t += st

            current        = overlap
            current_tokens = overlap_t

        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


class Chunker:
    """
    Converts raw GitHub objects into embedding-ready Chunk objects.

    Usage (called by IngestionOrchestrator):
        chunker = Chunker(repo_id="uuid", owner="facebook", repo_name="react")
        chunks  = chunker.chunk_issue(raw_issue, version_map)
        chunks += chunker.chunk_pr(raw_pr, version_map)
        ...
    """

    def __init__(self, repo_id: str, owner: str, repo_name: str):
        self.repo_id   = repo_id
        self.owner     = owner
        self.repo_name = repo_name
        self._max      = settings.MAX_CHUNK_SIZE_TOKENS
        self._overlap  = settings.CHUNK_OVERLAP_TOKENS

    # ── Issue chunking ─────────────────────────────────────────────────────

    def chunk_issue(
        self,
        issue: RawIssue,
        version_map: dict[datetime, str],
    ) -> list[Chunk]:
        """
        Produces chunks from an issue body + its comments.

        The issue body and each comment are chunked independently —
        a single comment can itself be multiple chunks if it's long.
        We prepend the issue title to every chunk so that even a small
        chunk carries enough context to be retrievable on its own.

        Args:
            issue:       Raw issue from GitHubFetcher.
            version_map: {release_created_at: tag_name} — used to find the
                         nearest release version at the time this issue was created.
                         Built from releases by IngestionOrchestrator before chunking.
        """
        chunks = []
        status: SourceStatus = "open" if issue.state == "open" else "closed"
        version = extract_version_tag(issue.created_at, version_map)
        url     = _build_github_url(self.owner, self.repo_name, "issue", str(issue.number))

        # ── Body chunks ───────────────────────────────────────────────────
        # Prepend title so every chunk knows what issue it belongs to
        body_text = f"Issue #{issue.number}: {issue.title}\n\n{issue.body}"
        for piece in _split_text(body_text, self._max, self._overlap):
            chunks.append(self._make_chunk(
                content=piece,
                source_type="issue",
                source_id=str(issue.number),
                status=status,
                created_at=issue.created_at,
                version_tag=version,
                url=url,
            ))

        # ── Comment chunks ────────────────────────────────────────────────
        # Each comment is a separate chunk — comments often contain the real fix
        for comment_body in issue.comments:
            comment_text = f"Comment on Issue #{issue.number}: {issue.title}\n\n{comment_body}"
            for piece in _split_text(comment_text, self._max, self._overlap):
                chunks.append(self._make_chunk(
                    content=piece,
                    source_type="comment",
                    source_id=str(issue.number),
                    status=status,
                    created_at=issue.created_at,
                    version_tag=version,
                    url=url,
                ))

        log.debug("issue_chunked", issue_number=issue.number, chunks=len(chunks))
        return chunks

    # ── PR chunking ────────────────────────────────────────────────────────

    def chunk_pr(
        self,
        pr: RawPR,
        version_map: dict[datetime, str],
    ) -> list[Chunk]:
        """
        Produces chunks from a PR body + review comments + formal reviews.

        PR chunks are the primary source for decision archaeology.
        We chunk the body, all review comments, and formal review bodies separately
        so the decision extractor can see the full discussion thread.

        Status mapping:
            merged → "merged" (most valuable — it was accepted)
            closed (not merged) → "closed" (rejected PR — also valuable for archaeology)
            open → "open"
        """
        chunks = []

        if pr.merged:
            status: SourceStatus = "merged"
        elif pr.state == "closed":
            status = "closed"
        else:
            status = "open"

        version  = extract_version_tag(pr.created_at, version_map)
        url      = _build_github_url(self.owner, self.repo_name, "pr", str(pr.number))

        # ── PR body ───────────────────────────────────────────────────────
        body_text = f"PR #{pr.number}: {pr.title}\n\n{pr.body}"
        for piece in _split_text(body_text, self._max, self._overlap):
            chunks.append(self._make_chunk(
                content=piece,
                source_type="pr",
                source_id=str(pr.number),
                status=status,
                created_at=pr.created_at,
                version_tag=version,
                url=url,
                files_touched=getattr(pr, "files_changed", []) or [],
            ))

        # ── Review comments (inline code comments) ────────────────────────
        for comment_body in pr.comments:
            text = f"Review comment on PR #{pr.number}: {pr.title}\n\n{comment_body}"
            for piece in _split_text(text, self._max, self._overlap):
                chunks.append(self._make_chunk(
                    content=piece,
                    source_type="comment",
                    source_id=str(pr.number),
                    status=status,
                    created_at=pr.created_at,
                    version_tag=version,
                    url=url,
                ))

        # Formal review bodies, now carrying reviewer identity
        for review in pr.reviews:
            if not review.body:
                continue
            text = f"Formal review by {review.reviewer} on PR #{pr.number}: {pr.title}\n\n{review.body}"
            for piece in _split_text(text, self._max, self._overlap):
                chunks.append(self._make_chunk(
                    content=piece,
                    source_type="comment",
                    source_id=str(pr.number),
                    status=status,
                    created_at=pr.created_at,
                    version_tag=version,
                    url=url,
                ))

        log.debug("pr_chunked", pr_number=pr.number, chunks=len(chunks))
        return chunks

    # ── Commit chunking ────────────────────────────────────────────────────

    def chunk_commit(self, commit: RawCommit) -> list[Chunk]:
        """
        Produces one chunk per commit message.

        Commit messages are usually short — rarely need splitting.
        We include the list of changed files in the chunk text so that
        queries like "what changed in src/auth?" can retrieve relevant commits.
        """
        if not commit.message.strip():
            return []

        # Include top 10 changed files — enough context without being noisy
        files_text = "\n".join(commit.files[:10])
        text = (
            f"Commit {commit.sha[:8]} by {commit.author}:\n"
            f"{commit.message}\n\n"
            f"Files changed:\n{files_text}"
        )

        chunks = []
        url = _build_github_url(self.owner, self.repo_name, "commit", commit.sha)
        for piece in _split_text(text, self._max, self._overlap):
            chunks.append(self._make_chunk(
                content=piece,
                source_type="commit",
                source_id=commit.sha,
                status="none",
                created_at=commit.created_at,
                version_tag=None,
                url=url,
                files_touched=commit.files,
            ))

        return chunks

    # ── Release chunking ───────────────────────────────────────────────────

    def chunk_release(self, release: RawRelease) -> list[Chunk]:
        """
        Produces chunks from a release's changelog body.

        Release notes are the authoritative source for version context.
        The tag_name (e.g. "v0.3.1") is stored as version_tag on these chunks
        and also used to build the version_map for issue/PR chunks.
        """
        if not release.body.strip():
            return []

        text = f"Release {release.tag_name}: {release.name}\n\n{release.body}"
        chunks = []
        for piece in _split_text(text, self._max, self._overlap):
            chunks.append(self._make_chunk(
                content=piece,
                source_type="release",
                source_id=release.tag_name,
                status="none",
                created_at=release.created_at,
                version_tag=release.tag_name,
                url=release.html_url,
            ))

        return chunks

    # ── Internal factory ───────────────────────────────────────────────────

    def _make_chunk(
        self,
        content:     str,
        source_type: SourceType,
        source_id:   str,
        status:      SourceStatus,
        created_at:  datetime,
        version_tag: Optional[str],
        url:         str,
        files_touched: Optional[list[str]] = None,
    ) -> Chunk:
        """
        Creates a Chunk with all metadata attached.
        The embedding field is left empty — EmbeddingService fills it next.
        url is built once at chunk creation time, when owner and repo_name
        are known, and travels with the chunk through storage so citations
        never need to reconstruct it later with incomplete information.
        """
        return Chunk(
            repo_id=self.repo_id,
            content=content,
            source_type=source_type,
            source_id=source_id,
            status=status,
            content_hash=_compute_hash(content),
            source_created_at=created_at,
            url=url,
            version_tag=version_tag,
            embedding=[],   # filled by EmbeddingService
            id="",          # filled after DB insert
            files_touched=files_touched or [],
        )