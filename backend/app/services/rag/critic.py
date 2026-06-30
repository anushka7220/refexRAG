# critic.py
#
# The self healing layer of the RAG pipeline. Runs three independent
# checks against the chunks used to generate an answer and produces a
# confidence score plus specific staleness flags.
#
# THE THREE CHECKS:
#   1. Staleness:     is the source old and closed, likely outdated.
#   2. Version match: does the chunk's version tag match what the user asked about.
#   3. Contradiction: do two similar chunks actually disagree in content.
#
# Checks 1 and 2 are pure logic, no LLM call, fast and deterministic.
# Check 3 only calls an LLM on pairs of chunks that are suspiciously
# similar in embedding space, since running an LLM on every pair would
# be slow and unnecessary for chunks that are obviously unrelated.
#
# CONFIDENCE SCORING:
# Starts at 1.0. Each flag found reduces it. The generate node's own
# self reported confidence, from the LLM's JSON response, is averaged
# in as well, so the critic's score reflects both structural problems
# in the sources and the model's own uncertainty about its answer.

import re
import asyncio
import structlog
from datetime import datetime, timezone

from app.models.chunk import ChunkResult, StalenessFlag
from app.services.rag.prompts import build_contradiction_check_prompt
import google.generativeai as genai
from app.core.config import settings

log = structlog.get_logger(__name__)

genai.configure(api_key=settings.GEMINI_API_KEY)

# A closed source older than this is flagged as potentially stale.
# 18 months is a deliberately conservative threshold, software changes
# fast enough that an 18 month old closed issue may no longer reflect
# current behavior, but is recent enough to not flag everything as stale.
STALENESS_THRESHOLD_DAYS = 18 * 30

# Chunks above this cosine similarity are checked for contradiction.
# Below this threshold, chunks are about different enough topics that
# a contradiction check would mostly be wasted LLM calls.
CONTRADICTION_CHECK_THRESHOLD = 0.85

# Pattern for extracting a version string like v0.3 or v1.2.1 from a query.
VERSION_PATTERN = re.compile(r"v?\d+\.\d+(\.\d+)?")


class Critic:
    """
    Runs staleness, version mismatch, and contradiction checks on the
    chunks used to generate an answer.

    Usage:
        critic = Critic()
        flags, confidence = await critic.review(query, top_chunks, llm_confidence)
    """

    async def review(
        self,
        query: str,
        top_chunks: list[ChunkResult],
        llm_confidence: float,
    ) -> tuple[list[StalenessFlag], float]:
        """
        Runs all three checks and produces a combined confidence score.

        Args:
            query:          The user's original question.
            top_chunks:     The chunks actually used to generate the answer,
                            already reranked and trimmed to top 8.
            llm_confidence: The generate node's self reported confidence
                            from its own JSON response.

        Returns:
            Tuple of (list of StalenessFlag, final confidence score 0 to 1).
        """
        flags: list[StalenessFlag] = []

        staleness_flags = self._check_staleness(top_chunks)
        flags.extend(staleness_flags)

        version_flags = self._check_version_match(query, top_chunks)
        flags.extend(version_flags)

        contradiction_flags = await self._check_contradictions(top_chunks)
        flags.extend(contradiction_flags)

        structural_confidence = self._compute_structural_confidence(flags, len(top_chunks))

        # Final confidence blends the critic's structural assessment with
        # the LLM's own self reported confidence. Weighted toward the
        # structural score since it is grounded in verifiable metadata
        # rather than the model's potentially overconfident self assessment.
        final_confidence = round(
            (structural_confidence * 0.7) + (llm_confidence * 0.3), 4
        )

        log.info(
            "critic_review_complete",
            flags_found=len(flags),
            structural_confidence=structural_confidence,
            llm_confidence=llm_confidence,
            final_confidence=final_confidence,
        )

        return flags, final_confidence

    def _check_staleness(self, top_chunks: list[ChunkResult]) -> list[StalenessFlag]:
        """
        Flags chunks that are closed and older than STALENESS_THRESHOLD_DAYS.

        A closed issue or rejected PR from years ago may no longer reflect
        the current state of the codebase. An open issue, regardless of
        age, is not flagged here since it may still be an active,
        unresolved discussion that is genuinely still relevant.
        """
        flags = []
        now = datetime.now(timezone.utc)

        for result in top_chunks:
            chunk = result.chunk

            if chunk.status not in ("closed",):
                continue

            created_at = chunk.source_created_at
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)

            age_days = (now - created_at).days

            if age_days > STALENESS_THRESHOLD_DAYS:
                flags.append(StalenessFlag(
                    chunk_id=chunk.id,
                    reason="source_closed",
                    severity="warn",
                    detail=(
                        f"This answer is based on a closed {chunk.source_type} "
                        f"from {age_days // 30} months ago. Behavior may have "
                        f"changed since then."
                    ),
                ))

        return flags

    def _check_version_match(
        self,
        query: str,
        top_chunks: list[ChunkResult],
    ) -> list[StalenessFlag]:
        """
        Flags chunks whose version tag does not match a version mentioned
        in the user's query.

        Only fires when the query explicitly mentions a version, for
        example "does this work in v0.4". If the query has no version
        reference, this check is skipped entirely since there is nothing
        to compare against.
        """
        flags = []

        query_version_match = VERSION_PATTERN.search(query)
        if not query_version_match:
            return flags

        query_version = query_version_match.group(0)

        for result in top_chunks:
            chunk = result.chunk

            if not chunk.version_tag:
                continue

            if not self._versions_match(query_version, chunk.version_tag):
                flags.append(StalenessFlag(
                    chunk_id=chunk.id,
                    reason="version_mismatch",
                    severity="warn",
                    detail=(
                        f"This source is from {chunk.version_tag}, but your "
                        f"question references {query_version}. Behavior may "
                        f"differ between versions."
                    ),
                ))

        return flags

    def _versions_match(self, query_version: str, chunk_version: str) -> bool:
        """
        Compares two version strings loosely, by major and minor number.
        v0.3 matches v0.3.1, since patch level differences rarely
        represent the kind of behavior change the version check cares about.
        """
        q_clean = query_version.lstrip("v")
        c_clean = chunk_version.lstrip("v")

        q_parts = q_clean.split(".")[:2]
        c_parts = c_clean.split(".")[:2]

        return q_parts == c_parts

    async def _check_contradictions(
        self,
        top_chunks: list[ChunkResult],
    ) -> list[StalenessFlag]:
        """
        Checks pairs of chunks with high embedding similarity for actual
        content contradiction.

        High cosine similarity means two chunks discuss similar topics.
        It does not mean they agree. This check catches the case where
        one chunk says a bug was fixed and another, similarly worded,
        chunk says it persists.

        Only pairs above CONTRADICTION_CHECK_THRESHOLD are checked with
        an LLM call, to avoid burning calls on obviously unrelated chunks.
        """
        flags = []
        pairs_to_check = []

        for i in range(len(top_chunks)):
            for j in range(i + 1, len(top_chunks)):
                sim = self._cosine_similarity(
                    top_chunks[i].chunk.embedding,
                    top_chunks[j].chunk.embedding,
                )
                if sim >= CONTRADICTION_CHECK_THRESHOLD:
                    pairs_to_check.append((top_chunks[i], top_chunks[j]))

        if not pairs_to_check:
            return flags

        log.info("contradiction_check_pairs", count=len(pairs_to_check))

        results = await asyncio.gather(
            *[self._check_pair(a, b) for a, b in pairs_to_check]
        )

        for (chunk_a, chunk_b), contradicts in zip(pairs_to_check, results):
            if contradicts:
                flags.append(StalenessFlag(
                    chunk_id=chunk_a.chunk.id,
                    reason="contradiction",
                    severity="error",
                    detail=(
                        f"This source appears to contradict another source "
                        f"used in this answer, {chunk_b.chunk.source_type} "
                        f"#{chunk_b.chunk.source_id}. The answer may be unreliable."
                    ),
                ))

        return flags

    async def _check_pair(self, result_a: ChunkResult, result_b: ChunkResult) -> bool:
        """
        Runs a single contradiction check between two chunks via Gemini Flash.
        Returns True if the model judges the passages to contradict each other.
        """
        prompt = build_contradiction_check_prompt(
            result_a.chunk.content,
            result_b.chunk.content,
        )

        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.0,
                        response_mime_type="application/json",
                    ),
                ),
            )
            import json
            parsed = json.loads(response.text.strip())
            return bool(parsed.get("contradicts", False))

        except Exception as e:
            log.error("contradiction_check_failed", error=str(e))
            # On failure, default to not flagging, a missed contradiction
            # is preferable to a false positive that erodes user trust
            # in the warning system through noise.
            return False

    def _compute_structural_confidence(
        self,
        flags: list[StalenessFlag],
        chunk_count: int,
    ) -> float:
        """
        Computes a confidence score purely from the flags found, separate
        from the LLM's own self reported confidence.

        Starts at 1.0. Each warn severity flag reduces confidence by 0.15.
        Each error severity flag, contradictions, reduces confidence by 0.35,
        since a genuine contradiction is a more serious problem than a
        single stale source.
        """
        if chunk_count == 0:
            return 0.0

        confidence = 1.0
        for flag in flags:
            if flag.severity == "warn":
                confidence -= 0.15
            elif flag.severity == "error":
                confidence -= 0.35

        return max(0.0, round(confidence, 4))

    def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """
        Computes cosine similarity. Vectors are already unit length from
        EmbeddingService, so this reduces to a dot product.
        """
        if not vec_a or not vec_b:
            return 0.0
        return sum(a * b for a, b in zip(vec_a, vec_b))


critic = Critic()