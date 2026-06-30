# nodes.py
#
# The actual LangGraph node functions. Each function here matches one
# node in the compiled graph defined in graph.py.
#
# CONTRACT EVERY NODE FOLLOWS:
# Receives the full GraphState. Returns a partial dict containing only
# the fields this node is responsible for. LangGraph merges the partial
# update into the running state automatically between node calls.
#
# This file is intentionally thin on original logic. Each node mostly
# orchestrates calls to services already written: vector_store, reranker,
# critic, embedding_service. The work lives there, this file sequences it.

import json
import asyncio
import structlog
import google.generativeai as genai

from app.models.graph_state import GraphState
from app.models.chunk import Citation
from app.core.config import settings
from app.services.ingestion.embedding_service import embedding_service
from app.services.ingestion.vector_store import vector_store
from app.services.rag.reranker import reranker
from app.services.rag.critic import critic
from app.services.rag.prompts import build_rag_prompt, build_query_refinement_prompt

log = structlog.get_logger(__name__)

genai.configure(api_key=settings.GEMINI_API_KEY)

# How many candidate chunks pgvector returns before reranking narrows
# them down. Wide enough to give the reranker real choices, narrow
# enough to keep the similarity search itself fast.
RETRIEVE_TOP_K = 50

# Below this confidence, the graph loops back to retrieve with a
# refined query instead of finalizing. Matches the threshold documented
# in the original architecture design for this project.
CONFIDENCE_RETRY_THRESHOLD = 0.7

# Maximum number of retry loops before finalizing regardless of confidence.
# Prevents an infinite loop if the critic keeps flagging the same problem.
MAX_RETRIES = 2


async def retrieve(state: GraphState) -> dict:
    """
    Embeds the current query and runs similarity search against pgvector.

    On the first pass, uses state.query. On a retry, uses
    state.refined_query if the critic produced one, otherwise falls
    back to the original query.

    Writes:
        retrieved_chunks
    """
    query_to_use = state.get("refined_query") or state["query"]

    log.info(
        "node_retrieve_start",
        repo_id=state["repo_id"],
        retry_count=state["retry_count"],
        query=query_to_use[:120],
    )

    query_embedding = await embedding_service.embed_single(query_to_use)

    results = vector_store.similarity_search(
        query_embedding=query_embedding,
        repo_id=state["repo_id"],
        top_k=RETRIEVE_TOP_K,
    )

    log.info("node_retrieve_done", returned=len(results))

    return {"retrieved_chunks": results}


async def generate(state: GraphState) -> dict:
    """
    Reranks the retrieved chunks down to the top 8, then calls the LLM
    to produce a draft answer with citations and self reported confidence.

    Writes:
        answer_draft, citations, tokens_used
    """
    query_to_use = state.get("refined_query") or state["query"]
    retrieved = state["retrieved_chunks"]

    if not retrieved:
        log.warning("node_generate_no_chunks", repo_id=state["repo_id"])
        return {
            "answer_draft": (
                "I could not find any relevant information in this repository "
                "to answer your question."
            ),
            "citations": [],
            "tokens_used": 0,
        }

    top_chunks = await reranker.rerank(query_to_use, retrieved)

    prompt = build_rag_prompt(query_to_use, top_chunks)

    model = genai.GenerativeModel("gemini-2.0-flash")
    loop = asyncio.get_event_loop()

    response = await loop.run_in_executor(
        None,
        lambda: model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                response_mime_type="application/json",
                max_output_tokens=2048,
            ),
        ),
    )

    try:
        parsed = json.loads(response.text.strip())
    except json.JSONDecodeError:
        log.error("generate_json_parse_failed", raw=response.text[:200])
        parsed = {"answer": response.text, "cited_chunk_ids": [], "confidence": 0.3}

    cited_ids = set(parsed.get("cited_chunk_ids", []))

    citations = []
    for result in top_chunks:
        if result.chunk.id in cited_ids or not cited_ids:
            citations.append(_build_citation(result))

    # Store top_chunks back onto state via retrieved_chunks so the critic
    # node reviews exactly the chunks that were actually used, not the
    # full 50 from the first pass retrieval.
    log.info(
        "node_generate_done",
        cited_chunks=len(cited_ids),
        llm_confidence=parsed.get("confidence"),
    )

    return {
        "answer_draft": parsed.get("answer", ""),
        "citations": citations,
        "retrieved_chunks": top_chunks,
        "tokens_used": _estimate_tokens(response),
        "_llm_confidence": parsed.get("confidence", 0.5),
    }


async def critic_node(state: GraphState) -> dict:
    """
    Runs the critic's staleness, version, and contradiction checks
    against the chunks actually used in the answer.

    Writes:
        staleness_flags, confidence, refined_query
    """
    top_chunks = state["retrieved_chunks"]
    llm_confidence = state.get("_llm_confidence", 0.5)

    flags, confidence = await critic.review(
        query=state["query"],
        top_chunks=top_chunks,
        llm_confidence=llm_confidence,
    )

    refined_query = None
    if confidence < CONFIDENCE_RETRY_THRESHOLD and state["retry_count"] < MAX_RETRIES:
        reasons = [f.detail for f in flags]
        if reasons:
            refined_query = await _refine_query(state["query"], reasons)

    log.info(
        "node_critic_done",
        flags_count=len(flags),
        confidence=confidence,
        will_retry=refined_query is not None,
    )

    return {
        "staleness_flags": flags,
        "confidence": confidence,
        "refined_query": refined_query,
    }


def should_retry(state: GraphState) -> str:
    """
    Conditional edge function. Not an async node, LangGraph calls this
    synchronously to decide which node to route to next.

    Returns the name of the next node as a string, either "retrieve"
    to loop back, or "finalize" to produce the final answer.
    """
    if state["confidence"] >= CONFIDENCE_RETRY_THRESHOLD:
        return "finalize"

    if state["retry_count"] >= MAX_RETRIES:
        return "finalize"

    if not state.get("refined_query"):
        return "finalize"

    return "retrieve"


def increment_retry(state: GraphState) -> dict:
    """
    Small node that bumps retry_count before looping back to retrieve.
    Kept separate from critic_node so the retry counter logic is explicit
    and visible as its own step in the graph, rather than buried inside
    the critic's other responsibilities.

    Writes:
        retry_count
    """
    return {"retry_count": state["retry_count"] + 1}


async def finalize(state: GraphState) -> dict:
    """
    Produces the final answer text, appending inline staleness warnings
    if the critic found problems that were not resolved through retry.

    This is the last node in the graph. Its output is what the chat
    route streams back to the frontend.

    Writes:
        final_answer
    """
    answer = state["answer_draft"]
    flags = state["staleness_flags"]

    if flags:
        warning_lines = []
        for flag in flags:
            warning_lines.append(f"Note: {flag.detail}")
        warnings_block = "\n\n" + "\n".join(warning_lines)
        final = answer + warnings_block
    else:
        final = answer

    log.info(
        "node_finalize_done",
        repo_id=state["repo_id"],
        final_confidence=state["confidence"],
        flags_surfaced=len(flags),
    )

    return {"final_answer": final}


async def _refine_query(original_query: str, reasons: list[str]) -> str | None:
    """
    Calls the LLM to rewrite the query based on what the critic found wrong.
    Used internally by critic_node when confidence is below threshold.
    """
    prompt = build_query_refinement_prompt(original_query, reasons)

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    response_mime_type="application/json",
                ),
            ),
        )
        parsed = json.loads(response.text.strip())
        return parsed.get("refined_query")
    except Exception as e:
        log.error("query_refinement_failed", error=str(e))
        return None


def _build_citation(result) -> Citation:
    """Builds a Citation object from a ChunkResult for the API response."""
    chunk = result.chunk
    return Citation(
        chunk_id=chunk.id,
        source_type=chunk.source_type,
        source_id=chunk.source_id,
        status=chunk.status,
        version_tag=chunk.version_tag,
        url=chunk.url,
        excerpt=chunk.content[:200],
    )


def _estimate_tokens(response) -> int:
    """
    Extracts token usage from the Gemini response if available.
    Falls back to a rough character based estimate if usage metadata
    is missing, which can happen depending on the SDK version.
    """
    try:
        return response.usage_metadata.total_token_count
    except AttributeError:
        return len(response.text) // 4