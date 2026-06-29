# embedding_service.py
# Converts text into 1024-dimensional float vectors using bge-large-en-v1.5.
# No API key required. Model runs locally via sentence-transformers.

# WHY bge-large-en-v1.5:
# Strong performance on retrieval benchmarks. 1024 dimensions gives good
# precision. Free to run. Already used at Memeraki so the pattern is familiar.
# Tradeoff: 1.3GB RAM, ~15s load time on first call.

# SINGLETON PATTERN:
# The model is loaded once when first used (lazy init) and reused forever.
# Loading on every request would add 15s latency to each embedding call.

# BATCHING:
# Embedding 10,000 chunks one at a time = 10,000 forward passes.
# Batching 64 chunks at a time = ~157 forward passes. Same result, 50x faster.
# Batch size 64 is a safe default that fits in 512MB RAM.

import asyncio
from functools import lru_cache
from typing import Optional

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from app.core.config import settings

log = structlog.get_logger(__name__)

BATCH_SIZE = 64


class EmbeddingService:
    """
    Wraps bge-large-en-v1.5 for single and batch embedding.

    The model is loaded lazily on first use, not at import time.
    This keeps test startup fast and lets you import the service
    without waiting for the model to load.

    Usage:
        embedder = EmbeddingService()
        vector = await embedder.embed_single("why was JWT chosen?")
        vectors = await embedder.embed_batch(["text1", "text2", ...])
    """

    def __init__(self):
        self._model: Optional[SentenceTransformer] = None

    def _load_model(self) -> SentenceTransformer:
        """
        Loads the model from HuggingFace cache (downloads on first run).
        Subsequent runs load from local cache in ~/.cache/huggingface.
        Thread-safe because Python's GIL serializes this on first call.
        """
        if self._model is None:
            log.info("embedding_model_loading", model=settings.EMBEDDING_MODEL)
            self._model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device="cpu",
            )
            log.info("embedding_model_ready", dim=settings.EMBEDDING_DIM)
        return self._model

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        """
        Synchronous batch embedding. Runs in thread pool via _run_sync.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float vectors, one per input text.
            Each vector has EMBEDDING_DIM dimensions (1024 for bge-large).

        Notes:
            normalize_embeddings=True makes vectors unit length.
            Required for cosine similarity to work correctly with pgvector.
            pgvectors cosine distance operator (<=>)  assumes unit vectors.
        """
        model = self._load_model()

        all_embeddings = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            log.debug(
                "embedding_batch",
                batch_num=i // BATCH_SIZE + 1,
                batch_size=len(batch),
                total=len(texts),
            )
            embeddings = model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=BATCH_SIZE,
            )
            all_embeddings.extend(embeddings.tolist())

        return all_embeddings

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds a list of texts. Use this for ingestion.

        Runs the synchronous model in a thread pool so FastAPIs event loop
        stays free while the CPU-intensive embedding runs.
        """
        if not texts:
            return []
        return await self._run_sync(self._embed_sync, texts)

    async def embed_single(self, text: str) -> list[float]:
        """
        Embeds one string. Use this for query-time embedding.

        bge-large has a specific query prefix that improves retrieval quality
        for question-like inputs. We prepend it here so the query vector
        lands closer to relevant document vectors in embedding space.

        The prefix is specific to bge models. Other models do not need it.
        """
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        results  = await self.embed_batch([prefixed])
        return results[0]

    async def embed_chunks(self, chunks: list) -> list:
        """
        Embeds a list of Chunk objects and fills their embedding field in place.

        Args:
            chunks: List of Chunk dataclass objects with empty embedding fields.

        Returns:
            Same list with embedding fields populated.

        Usage:
            chunks = chunker.chunk_issue(issue, version_map)
            chunks = await embedder.embed_chunks(chunks)
        """
        if not chunks:
            return chunks

        texts      = [chunk.content for chunk in chunks]
        embeddings = await self.embed_batch(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        log.info("chunks_embedded", count=len(chunks))
        return chunks

    async def _run_sync(self, fn, *args):
        """
        Runs a blocking function in a thread pool executor.
        Keeps the async event loop free during CPU-heavy embedding work.
        Same pattern as GitHubFetcher because sentence-transformers is also synchronous.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, fn, *args)


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """
    Returns the singleton EmbeddingService instance.
    lru_cache ensures only one instance is ever created.
    Import this function, not the class directly.
    """
    return EmbeddingService()


embedding_service = get_embedding_service()