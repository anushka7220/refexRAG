# reranker.py
#
# Reranks the top 50 chunks from pgvector similarity search down to the
# top 8 actually sent to the LLM. Uses a local cross encoder model,
# cross-encoder/ms-marco-MiniLM-L-6-v2. No API cost, no API key.
#
# WHY RERANKING IS A SEPARATE STEP FROM RETRIEVAL:
# pgvector similarity search is fast but approximate. It compares the
# query vector against each chunk vector independently, with no awareness
# of the query and chunk text together. A cross encoder reads the query
# and chunk text together in one forward pass, which is slower but far
# more accurate at judging true relevance. Running a cross encoder against
# all chunks in a repo would be too slow. Running it against only the top
# 50 from a fast first pass gives you both speed and accuracy.
#
# This is a standard two stage retrieval pattern: a fast approximate
# first pass narrows the field, a slow accurate second pass picks the best.

import asyncio
from functools import lru_cache
from typing import Optional

import structlog
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.models.chunk import ChunkResult

log = structlog.get_logger(__name__)

TOP_K_AFTER_RERANK = 8


class Reranker:
    """
    Wraps a local cross encoder for reranking chunk results.

    Usage:
        reranker = Reranker()
        top_chunks = await reranker.rerank(query, chunk_results)
    """

    def __init__(self):
        self._model: Optional[CrossEncoder] = None

    def _load_model(self) -> CrossEncoder:
        """
        Loads the cross encoder model lazily on first use.
        Same pattern as EmbeddingService, avoids load time cost at import.
        """
        if self._model is None:
            log.info("reranker_model_loading", model=settings.RERANKER_MODEL)
            self._model = CrossEncoder(settings.RERANKER_MODEL, device="cpu")
            log.info("reranker_model_ready")
        return self._model

    def _rerank_sync(
        self,
        query: str,
        chunk_results: list[ChunkResult],
    ) -> list[ChunkResult]:
        """
        Synchronous reranking. Runs in thread pool via _run_sync.

        A cross encoder takes pairs of (query, document) and outputs a
        single relevance score per pair, unlike the embedding model which
        scores query and document independently then compares vectors.

        Args:
            query:         The user's question.
            chunk_results: Up to 50 ChunkResult objects from similarity search.

        Returns:
            Top 8 ChunkResult objects sorted by rerank_score descending,
            with rerank_score filled in on each.
        """
        if not chunk_results:
            return []

        model = self._load_model()

        pairs = [(query, result.chunk.content) for result in chunk_results]
        scores = model.predict(pairs)

        for result, score in zip(chunk_results, scores):
            result.rerank_score = float(score)

        sorted_results = sorted(
            chunk_results,
            key=lambda r: r.rerank_score,
            reverse=True,
        )

        log.info(
            "rerank_complete",
            input_count=len(chunk_results),
            output_count=min(TOP_K_AFTER_RERANK, len(sorted_results)),
            top_score=sorted_results[0].rerank_score if sorted_results else None,
        )

        return sorted_results[:TOP_K_AFTER_RERANK]

    async def rerank(
        self,
        query: str,
        chunk_results: list[ChunkResult],
    ) -> list[ChunkResult]:
        """
        Async wrapper for the reranking step.
        Runs the synchronous cross encoder in a thread pool so the
        event loop stays free.
        """
        return await self._run_sync(self._rerank_sync, query, chunk_results)

    async def _run_sync(self, fn, *args):
        """
        Runs a blocking function in a thread pool executor.
        Same pattern used throughout the codebase for sync libraries
        called from async code.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn, *args)


@lru_cache
def get_reranker() -> Reranker:
    """Returns the singleton Reranker instance."""
    return Reranker()


reranker = get_reranker()
