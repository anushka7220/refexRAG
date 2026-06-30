# prompts.py
#
# Every prompt template used by the RAG graph lives here.
# nodes.py and critic.py import from this file rather than writing
# prompt text inline. Keeps prompts easy to find, version, and tune
# without touching control flow logic.
#
# Each function returns a fully formatted string ready to send to the LLM.
# Formatting logic, like building the chunk context block, lives here too
# since it is part of the prompt's shape, not the node's responsibility.

from app.models.chunk import ChunkResult


def build_rag_prompt(query: str, top_chunks: list[ChunkResult]) -> str:
    """
    Builds the main answer generation prompt sent to Gemini Flash.

    The prompt instructs the model to answer only from provided context,
    cite the chunks it used, and self report a confidence score.
    This confidence score feeds into the critic's overall assessment
    alongside the structural checks the critic runs independently.

    Args:
        query:      The user's question.
        top_chunks: Reranked chunks, already trimmed to the top 8.

    Returns:
        Fully formatted prompt string.
    """
    context_blocks = []
    for result in top_chunks:
        chunk = result.chunk
        block = (
            f"[CHUNK {chunk.source_type.upper()} #{chunk.source_id} "
            f"| status={chunk.status} | version={chunk.version_tag or 'unknown'} "
            f"| chunk_id={chunk.id}]\n"
            f"{chunk.content}\n"
            f"[/CHUNK]"
        )
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    return f"""You are a precise technical assistant answering questions about a GitHub repository.

Answer ONLY using the context chunks provided below. Do not use outside knowledge about this repository.
If the context is insufficient to answer confidently, say so explicitly rather than guessing.
Never invent issue numbers, PR numbers, or details not present in the context.
Always cite the chunk_id of every chunk you actually relied on.

Question: {query}

Context chunks, ordered by relevance:
{context}

Respond ONLY with valid JSON in this exact format:
{{
  "answer": "your answer here, written in clear prose",
  "cited_chunk_ids": ["chunk_id_1", "chunk_id_2"],
  "confidence": 0.0-1.0
}}

No explanation outside the JSON. No markdown formatting."""


def build_query_refinement_prompt(
    original_query: str,
    staleness_reasons: list[str],
) -> str:
    """
    Builds the prompt used to rewrite a query when the critic detects
    a problem with the first retrieval attempt.

    The refined query is what the retrieve node uses on its second pass.
    Telling the model exactly what went wrong, not just asking it to
    try again, produces meaningfully different and better retrieval.

    Args:
        original_query:    The user's original question.
        staleness_reasons: Human readable reasons the critic flagged,
                            for example "source closed in 2022, may be outdated".

    Returns:
        Fully formatted prompt string.
    """
    reasons_text = "\n".join(f"- {r}" for r in staleness_reasons)

    return f"""The first retrieval attempt for this question returned results with problems.

Original question: {original_query}

Problems detected:
{reasons_text}

Rewrite the question to retrieve better, more current information. For example,
if the issue is that retrieved sources are old or closed, add terms like
"current behavior" or "latest version" to bias retrieval toward recent content.
If the issue is a version mismatch, make the version requirement explicit in the query.

Respond ONLY with valid JSON:
{{
  "refined_query": "the rewritten question"
}}

No explanation outside the JSON."""


def build_contradiction_check_prompt(chunk_a_text: str, chunk_b_text: str) -> str:
    """
    Builds the prompt used to check whether two chunks with high embedding
    similarity actually contradict each other in content.

    High cosine similarity means two chunks discuss the same topic.
    It does not mean they agree. This prompt is the semantic check that
    catches cases where similarity is high but the chunks disagree,
    for example one says a bug was fixed, another says it still exists.

    Args:
        chunk_a_text: Content of the first chunk.
        chunk_b_text: Content of the second chunk.

    Returns:
        Fully formatted prompt string.
    """
    return f"""Compare these two passages from the same GitHub repository's history.

Passage A:
{chunk_a_text}

Passage B:
{chunk_b_text}

Do these passages contradict each other, for example one claims something
works or is fixed while the other claims it does not work or is broken?
Passages that simply discuss different aspects of the same topic are NOT
a contradiction. Only flag genuine factual disagreement.

Respond ONLY with valid JSON:
{{
  "contradicts": true or false,
  "explanation": "one sentence explaining your reasoning, or null"
}}

No explanation outside the JSON."""
