# decision_extractor.py — v2
#
# ARCHITECTURE: Parallel multi-agent decision extraction with reconciliation.
#
# WHY THIS DESIGN:
# A single LLM call per PR has a fundamental problem — the model conflates
# "what was decided" with "what was discussed" and "what was rejected" with
# "what was deferred." These are different things and a single prompt trying
# to extract all three at once produces mediocre results for all three.
#
# Solution: three specialized agents, each with a narrow job, running in
# parallel on the same PR text. Then a reconciliation step that merges
# their outputs and flags disagreements rather than silently picking one.
#
# AGENT ROLES:
#   Agent 1 — Decider:  "What concrete decision was made in this thread?"
#   Agent 2 — Rejector: "What alternatives were explicitly rejected, and why?"
#   Agent 3 — Skeptic:  "How confident are we in the above? Flag conflicts."
#
# RECONCILIATION:
#   If agents agree → merge into a DecisionNode with high confidence.
#   If agents disagree → either re-run with a tighter prompt (max 1 retry)
#                         or store the conflict explicitly for the UI to surface.
#
# ORCHESTRATION: asyncio.gather() runs all three agents truly in parallel.
# Three Gemini Flash calls fire simultaneously — total latency = slowest agent,
# not sum of all three.

import asyncio
import json
import structlog
from datetime import datetime, timezone
from typing import Optional

import google.generativeai as genai

from app.core.config import settings
from app.models.decision import DecisionNode, DecisionExtraction, RejectedAlternative
from app.models.chunk import Chunk

log = structlog.get_logger(__name__)

# Configure Gemini once at module load
genai.configure(api_key=settings.GEMINI_API_KEY)


# ── Agent prompts ──────────────────────────────────────────────────────────
# Each agent has a narrow, specific job.
# Narrow prompts outperform broad prompts for structured extraction.

DECIDER_PROMPT = """You are a precise technical analyst. Your ONLY job is to identify the single concrete decision made in this GitHub PR or issue thread.

A "decision" means: something was chosen, implemented, or agreed upon. Not discussed. Not proposed. Actually decided.

If no concrete decision was made (e.g. it's just a bug report, a question, or an unresolved discussion), respond with:
{{"decision": null, "confidence": 0.0}}

If a decision was made, respond with:
{{"decision": "one sentence describing what was decided", "confidence": 0.0-1.0}}

Confidence guide:
- 1.0: Explicitly stated ("we decided to...", "going with X", PR merged with clear purpose)
- 0.7: Clearly implied by the merge and context
- 0.4: Inferred — not explicitly stated
- 0.0: No decision found

Respond ONLY with valid JSON. No explanation, no markdown.

PR/Issue thread:
{content}"""


REJECTOR_PROMPT = """You are a precise technical analyst. Your ONLY job is to identify alternatives that were EXPLICITLY rejected in this GitHub PR or issue thread.

Rules:
- "Rejected" means: considered and ruled out, not just unmentioned.
- "Deferred" is NOT rejected. If something was deferred, say so.
- Only include alternatives with explicit reasons given.
- If nothing was explicitly rejected, return an empty list.

Respond ONLY with valid JSON in this exact format:
{{
  "alternatives": [
    {{
      "option": "the alternative that was rejected",
      "reason": "why it was rejected",
      "disposition": "rejected" | "deferred" | "superseded"
    }}
  ]
}}

No explanation, no markdown.

PR/Issue thread:
{content}"""


SKEPTIC_PROMPT = """You are a critical reviewer of AI-extracted information. You will receive two extraction attempts from the same PR/issue thread and must assess their quality.

Extraction 1 (Decider agent):
{decider_output}

Extraction 2 (Rejector agent):
{rejector_output}

Your job:
1. Does the decision in Extraction 1 actually look like a decision, or is it vague/wrong?
2. Are the rejections in Extraction 2 genuinely explicit, or inferred/hallucinated?
3. Do Extractions 1 and 2 contradict each other?

Respond ONLY with valid JSON:
{{
  "decision_valid": true | false,
  "decision_critique": "one sentence or null",
  "rejections_valid": true | false,
  "rejections_critique": "one sentence or null",
  "contradiction_detected": true | false,
  "contradiction_detail": "what specifically conflicts, or null",
  "overall_confidence": 0.0-1.0,
  "recommendation": "accept" | "retry" | "discard"
}}"""


RECONCILER_PROMPT = """You are reconciling conflicting extractions from the same PR/issue thread.

The specialized agents disagreed or produced low-confidence results. Your job is one final extraction attempt using the original thread, knowing what went wrong.

Known issues: {conflict_summary}

Produce a single, conservative extraction. When uncertain, prefer null over a guess.

Respond ONLY with valid JSON:
{{
  "decision": "one sentence or null",
  "alternatives_rejected": [
    {{"option": "...", "reason": "...", "disposition": "rejected|deferred|superseded"}}
  ],
  "reasoning": "one paragraph explaining the decision context",
  "confidence": 0.0-1.0
}}

Original thread:
{content}"""


# ── Single agent call ──────────────────────────────────────────────────────

async def _call_agent(prompt: str, agent_name: str) -> dict:
    """
    Makes one Gemini Flash call and parses the JSON response.
    Runs in an executor because google-generativeai is synchronous.

    Returns parsed dict on success, empty dict on failure.
    """
    model = genai.GenerativeModel("gemini-flash-latest")

    try:
        # Run sync Gemini call in a thread pool so it doesn't block the event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,          # low temp for structured extraction
                    response_mime_type="application/json",
                ),
            ),
        )

        raw = response.text.strip()
        parsed = json.loads(raw)
        log.info("agent_success", agent=agent_name, keys=list(parsed.keys()))
        return parsed

    except json.JSONDecodeError as e:
        log.error("agent_json_parse_failed", agent=agent_name, error=str(e))
        return {}
    except Exception as e:
        log.error("agent_call_failed", agent=agent_name, error=str(e))
        return {}


# ── Reconciliation logic ───────────────────────────────────────────────────

def _build_conflict_summary(skeptic_output: dict) -> str:
    """Builds a human-readable conflict summary from the skeptic's output."""
    parts = []
    if not skeptic_output.get("decision_valid"):
        parts.append(f"Decision invalid: {skeptic_output.get('decision_critique', 'unclear')}")
    if not skeptic_output.get("rejections_valid"):
        parts.append(f"Rejections invalid: {skeptic_output.get('rejections_critique', 'unclear')}")
    if skeptic_output.get("contradiction_detected"):
        parts.append(f"Contradiction: {skeptic_output.get('contradiction_detail', 'agents disagreed')}")
    return ". ".join(parts) if parts else "Low confidence across agents."


def _merge_outputs(
    decider: dict,
    rejector: dict,
    skeptic: dict,
) -> Optional[DecisionExtraction]:
    """
    Merges three agent outputs into one DecisionExtraction.
    Only called when skeptic recommends 'accept'.
    Returns None if the merged result doesn't meet the confidence threshold.
    """
    decision = decider.get("decision")
    if not decision:
        return None

    alternatives = []
    for alt in rejector.get("alternatives", []):
        if alt.get("option") and alt.get("reason"):
            alternatives.append(RejectedAlternative(
                option=alt["option"],
                reason=f"{alt['reason']} [{alt.get('disposition', 'rejected')}]",
            ))

    confidence = skeptic.get("overall_confidence", 0.5)

    # Don't store low-confidence extractions — they add noise
    if confidence < 0.6:
        log.info("extraction_below_threshold", confidence=confidence)
        return None

    return DecisionExtraction(
        decision=decision,
        alternatives_rejected=alternatives,
        reasoning=decider.get("reasoning", ""),
        confidence=confidence,
    )


def _parse_reconciler_output(raw: dict) -> Optional[DecisionExtraction]:
    """Parses the reconciler's output into a DecisionExtraction."""
    decision = raw.get("decision")
    if not decision or raw.get("confidence", 0) < 0.5:
        return None

    alternatives = []
    for alt in raw.get("alternatives_rejected", []):
        if alt.get("option") and alt.get("reason"):
            alternatives.append(RejectedAlternative(
                option=alt["option"],
                reason=alt["reason"],
            ))

    return DecisionExtraction(
        decision=decision,
        alternatives_rejected=alternatives,
        reasoning=raw.get("reasoning", ""),
        confidence=raw.get("confidence", 0.5),
    )


# ── Conflict log ───────────────────────────────────────────────────────────
# In-memory log of conflicts detected during extraction.
# Written to at runtime, can be queried via a debug endpoint or exported.
# In production this would go to a structured logging system.

_conflict_log: list[dict] = []

def get_conflict_log() -> list[dict]:
    """Returns all conflicts detected during this session. For debugging/MULTI_AGENT.md."""
    return _conflict_log

def _log_conflict(pr_or_issue_id: str, conflict_summary: str, resolution: str):
    _conflict_log.append({
        "source_id":       pr_or_issue_id,
        "conflict":        conflict_summary,
        "resolution":      resolution,
        "detected_at":     datetime.now(timezone.utc).isoformat(),
    })


# ── Main extractor ─────────────────────────────────────────────────────────

class DecisionExtractorV2:
    """
    Parallel multi-agent decision extractor with reconciliation.

    Usage:
        extractor = DecisionExtractorV2()
        result = await extractor.extract(pr_body, comments, source_id="pr#341")
        if result:
            # result is a DecisionExtraction — embed and store it
    """

    async def extract(
        self,
        pr_body: str,
        comments: list[str],
        source_id: str = "",
    ) -> Optional[DecisionExtraction]:
        """
        Runs three specialized agents in parallel on the PR/issue text,
        reconciles their outputs, and returns a structured DecisionExtraction.

        Args:
            pr_body:   The PR or issue body text.
            comments:  List of comment strings (PR review comments, issue comments).
            source_id: PR number or issue number — used in conflict logging.

        Returns:
            DecisionExtraction if a confident decision was found, None otherwise.
        """
        # Combine PR body + top comments into one context block
        # Truncate to ~3000 tokens to stay within Gemini's context efficiently
        combined = self._build_context(pr_body, comments, max_chars=12000)

        log.info("extraction_start", source_id=source_id, content_len=len(combined))

        # ── Phase 1: Run Decider and Rejector in parallel ──────────────────
        # asyncio.gather fires both coroutines simultaneously.
        # Total latency = max(decider_latency, rejector_latency), not their sum.
        decider_output, rejector_output = await asyncio.gather(
            _call_agent(DECIDER_PROMPT.format(content=combined), "decider"),
            _call_agent(REJECTOR_PROMPT.format(content=combined), "rejector"),
        )

        # ── Phase 2: Skeptic reviews both outputs ─────────────────────────
        # The skeptic gets to see what both agents said before giving verdict.
        skeptic_output = await _call_agent(
            SKEPTIC_PROMPT.format(
                decider_output=json.dumps(decider_output),
                rejector_output=json.dumps(rejector_output),
            ),
            "skeptic",
        )

        recommendation = skeptic_output.get("recommendation", "discard")
        log.info(
            "skeptic_verdict",
            source_id=source_id,
            recommendation=recommendation,
            confidence=skeptic_output.get("overall_confidence"),
        )

        # ── Phase 3: Route based on skeptic's recommendation ──────────────
        if recommendation == "accept":
            result = _merge_outputs(decider_output, rejector_output, skeptic_output)
            if result:
                log.info("extraction_accepted", source_id=source_id, confidence=result.confidence)
                return result

        elif recommendation == "retry":
            # Agents disagreed or confidence was low — one reconciliation attempt
            conflict_summary = _build_conflict_summary(skeptic_output)
            _log_conflict(source_id, conflict_summary, "retry_with_reconciler")

            log.info("extraction_retry", source_id=source_id, conflict=conflict_summary)

            reconciler_output = await _call_agent(
                RECONCILER_PROMPT.format(
                    conflict_summary=conflict_summary,
                    content=combined,
                ),
                "reconciler",
            )

            result = _parse_reconciler_output(reconciler_output)
            if result:
                log.info("reconciliation_accepted", source_id=source_id, confidence=result.confidence)
                return result
            else:
                _log_conflict(source_id, conflict_summary, "discarded_after_retry")

        else:
            # recommendation == "discard"
            conflict_summary = _build_conflict_summary(skeptic_output)
            _log_conflict(source_id, conflict_summary, "discarded_by_skeptic")
            log.info("extraction_discarded", source_id=source_id)

        return None

    def _build_context(self, pr_body: str, comments: list[str], max_chars: int) -> str:
        """
        Combines PR body and comments into one context string.
        Truncates to max_chars to avoid burning excessive tokens.
        Prioritizes the PR body and top comments by length.
        """
        parts = [f"PR/Issue body:\n{pr_body}"]

        # Add comments until we hit the limit
        for i, comment in enumerate(comments):
            candidate = f"\nComment {i+1}:\n{comment}"
            if sum(len(p) for p in parts) + len(candidate) > max_chars:
                break
            parts.append(candidate)

        return "\n".join(parts)


# ── Module-level singleton ─────────────────────────────────────────────────
# Import this in orchestrator.py — don't instantiate DecisionExtractorV2 elsewhere.
decision_extractor = DecisionExtractorV2()
