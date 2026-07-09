# main.py

#
# FastAPI application entry point.
# Creates the app, registers routers, configures middleware and CORS.
# No business logic lives here. This file is assembly only.
#
#  uvicorn app.main:app --reload --port 8000START COMMAND:
#  

import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import auth, repos, chat, decisions, contributors, webhooks

log = structlog.get_logger(__name__)


# ── Lifespan: startup and shutdown events ─────────────────────────────────
#
# Runs setup code before the first request and cleanup on shutdown.
# We warm up both ML models here so the first user request does not pay
# the 15 second load cost. Models are large files loaded from disk
# once at startup, then reused for every subsequent request.

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup_begin", env=settings.APP_ENV)

    from app.services.ingestion.embedding_service import embedding_service
    embedding_service._load_model()
    log.info("embedding_model_warmed")

    from app.services.rag.reranker import reranker
    reranker._load_model()
    log.info("reranker_model_warmed")

    # Compile LangGraph once at startup. Reused for every chat request.
    from app.services.rag.graph import rag_graph
    log.info("rag_graph_ready")

    log.info("startup_complete")
    yield
    log.info("shutdown_complete")


# ── App instance ──────────────────────────────────────────────────────────

app = FastAPI(
    title="GitMind API",
    description="Self-healing RAG pipeline for GitHub repositories.",
    version="0.1.0",
    lifespan=lifespan,
    # Disable API docs in production to avoid leaking internal structure.
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
)


# ── CORS ──────────────────────────────────────────────────────────────────
#
# Without CORS headers, the browser blocks cross-origin requests before
# they reach your route handlers. allow_credentials=True is required
# for the Authorization header to be sent across origins.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.FRONTEND_URL,
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routers ───────────────────────────────────────────────────────────────

app.include_router(auth.router,         prefix="/auth",     tags=["Auth"])
app.include_router(repos.router,        prefix="/repos",    tags=["Repos"])
app.include_router(chat.router,         prefix="",          tags=["Chat"])
app.include_router(decisions.router,    prefix="/repos",    tags=["Decisions"])
app.include_router(contributors.router, prefix="/repos",    tags=["Contributors"])
app.include_router(webhooks.router,     prefix="/webhooks", tags=["Webhooks"])


# ── Health check ──────────────────────────────────────────────────────────
#
# Render pings this endpoint after deployment to verify the service is up
# before routing traffic to it. Returns 200 or deployment is marked failed.

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "env": settings.APP_ENV, "version": "0.1.0"}


@app.get("/", tags=["Health"])
async def root():
    return {
        "name": "GitMind API",
        "docs": "/docs" if settings.is_development else "disabled in production",
    }