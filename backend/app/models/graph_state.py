#the state object that flows through evry node in the Langraph RAG graphg
# WHY TypedDict and not a Pydantic model or dataclass:
# LangGraph requires state to be a TypedDict. This is a LangGraph convention
# LangGraph merges the partial update into the full state automatically.
# HOW IT WORKS:
#   Node 1 (retrieve) returns:  {"retrieved_chunks": [...]}
#   Node 2 (generate) returns:  {"answer_draft": "...", "citations": [...]}
#   Node 3 (critic)   returns:  {"staleness_flags": [...], "confidence": 0.8}
#   LangGraph merges all of these into one running state object.
# Every node receives the FULL state (all fields) and only writes
# back the fields it's responsible for. Clean separation of concerns.

from typing import TypeDict, Annotated
from app.models.chunk import ChunkResult, Citation, StalenessFlag
import operator

class GraphState(TypeDict):
    query: str #user question
    repo_id: str #scopes all pgvector searches
    retrieved_chunks: list[ChunkResult]
    answer_draft: str #llm answer before critic
    citations: list[Citation] # Citations built from the top-8 chunk metadata after reranking.
    staleness_flags: list[StalenessFlag]
    confidence: float
    refined_query: str | None #rewritten query used on retry
    retry_count: int
    final_answer: str | None
    token_used: int

def intial_state(query: str, repo_id: str) -> GraphState:
    #creating a fresh graphstate for a new query
     return GraphState(
        query=query,
        repo_id=repo_id,
        retrieved_chunks=[],
        answer_draft="",
        citations=[],
        staleness_flags=[],
        confidence=0.0,
        refined_query=None,
        retry_count=0,
        final_answer=None,
        tokens_used=0,
    )


