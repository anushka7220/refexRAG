# cache.py
#
# Semantic cache wrapper around GPTCache.
# Sits in front of the LangGraph, checked before the graph runs at all.
# A cache hit skips retrieval, reranking, generation, and the critic
# entirely, returning the previous answer almost instantly.
#
# WHY SEMANTIC, NOT EXACT MATCH:
# "How do I install this?" and "What's the installation process?" are
# different strings with the same meaning. An exact match cache would
# treat them as two separate queries and miss the reuse opportunity.
# A semantic cache embeds the incoming query and compares it against
# embeddings of past queries for the same repo, using a similarity
# threshold to decide what counts as the same question.
#
# SCOPING:
# Cache keys are scoped per repo_id. The same question phrased the same
# way means something different for two different repositories, so
# repo_id is part of the cache key, not just the query text.

import json
import structlog
from datetime import datetime, timezone
from typing import Optional

from app.core.config import settings
from app.services.ingestion.embedding_service import embedding_service

log = structlog.get_logger(__name__)

# Cosine similarity threshold for considering two queries the same.
# Above this, we treat them as duplicate questions and serve the cached answer.
# Tuned conservatively, a slightly too low threshold risks serving a
# wrong cached answer, which is worse than a cache miss.
SIMILARITY_THRESHOLD = 0.92

# How long a cached entry stays valid before we treat it as stale and
# fall through to a fresh generation, regardless of similarity match.
CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours


class SemanticCache:
    """
    In memory semantic cache scoped per repo.

    This implementation uses an in process dict keyed by repo_id, holding
    a list of (embedding, query_text, response, cached_at) tuples per repo.
    For a single instance deployment this is sufficient. A multi instance
    deployment would need this backed by Redis instead, since each
    FastAPI worker process would otherwise have its own disconnected cache.

    Usage:
        cache = SemanticCache()
        cached = await cache.get(query, repo_id)
        if cached:
            return cached
        result = await run_rag_graph(query, repo_id)
        await cache.set(query, repo_id, result)
    """

    def __init__(self):
        self._store: dict[str, list[dict]] = {}

    async def get(self, query: str, repo_id: str) -> Optional[dict]:
        """
        Checks for a semantically similar cached query for this repo.

        Args:
            query:   The incoming user question.
            repo_id: Scopes the cache lookup to one repo.

        Returns:
            The cached response dict if a sufficiently similar, non
            expired entry exists. None on cache miss.
        """
        entries = self._store.get(repo_id, [])
        if not entries:
            return None

        query_embedding = await embedding_service.embed_single(query)
        now = datetime.now(timezone.utc)

        best_match = None
        best_score = 0.0

        for entry in entries:
            age_seconds = (now - entry["cached_at"]).total_seconds()
            if age_seconds > CACHE_TTL_SECONDS:
                continue

            score = self._cosine_similarity(query_embedding, entry["embedding"])
            if score > best_score:
                best_score = score
                best_match = entry

        if best_match and best_score >= SIMILARITY_THRESHOLD:
            log.info(
                "cache_hit",
                repo_id=repo_id,
                similarity=round(best_score, 4),
                matched_query=best_match["query_text"][:80],
            )
            return best_match["response"]

        log.info("cache_miss", repo_id=repo_id, best_score=round(best_score, 4))
        return None

    async def set(self, query: str, repo_id: str, response: dict) -> None:
        """
        Stores a query and response pair for future semantic lookup.

        Args:
            query:    The user question that produced this response.
            repo_id:  Scopes the stored entry to this repo.
            response: The full RAG response dict to cache, including
                      answer, citations, and staleness flags.
        """
        query_embedding = await embedding_service.embed_single(query)

        entry = {
            "query_text": query,
            "embedding": query_embedding,
            "response": response,
            "cached_at": datetime.now(timezone.utc),
        }

        if repo_id not in self._store:
            self._store[repo_id] = []

        self._store[repo_id].append(entry)

        # Cap stored entries per repo to avoid unbounded memory growth.
        # Oldest entries are dropped first once the cap is reached.
        max_entries_per_repo = 500
        if len(self._store[repo_id]) > max_entries_per_repo:
            self._store[repo_id] = self._store[repo_id][-max_entries_per_repo:]

        log.info("cache_set", repo_id=repo_id, total_entries=len(self._store[repo_id]))

    def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """
        Computes cosine similarity between two vectors.
        Both vectors are already unit length from EmbeddingService,
        normalize_embeddings is True there, so this reduces to a dot product.
        """
        return sum(a * b for a, b in zip(vec_a, vec_b))

    def clear_repo(self, repo_id: str) -> None:
        """Clears all cached entries for a repo. Called when a repo is re ingested."""
        if repo_id in self._store:
            del self._store[repo_id]
            log.info("cache_cleared", repo_id=repo_id)


semantic_cache = SemanticCache()
