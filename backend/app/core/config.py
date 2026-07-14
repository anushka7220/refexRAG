# config.py
#
# WHY THIS FILE EXISTS:
# Rather than calling os.getenv("SOME_KEY") scattered across the codebase,
# we define all config in one place. Pydantic reads the .env file, validates
# types, and raises a clear error at startup if anything is missing.
#
# HOW TO USE IT ANYWHERE IN THE PROJECT:
#   from app.core.config import settings
#   print(settings.SUPABASE_URL)

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """
    Pydantic reads every field here from your .env file automatically.
    The field name must exactly match the variable name in .env.
    Types are enforced — if you put a string where an int is expected, it errors at startup.
    Fields with defaults are optional. Fields without defaults are required.
    """

    # ── App ────────────────────────────────────────────────────────────────
    APP_ENV: str = "development"
    SECRET_KEY: str                          # required — no default
    FRONTEND_URL: str = "http://localhost:5173"
    BACKEND_URL: str = "http://localhost:8000"

    # ── GitHub OAuth ───────────────────────────────────────────────────────
    GITHUB_CLIENT_ID: str
    GITHUB_CLIENT_SECRET: str
    GITHUB_WEBHOOK_SECRET: str
    GITHUB_PERSONAL_ACCESS_TOKEN: str


    # ── Supabase ───────────────────────────────────────────────────────────
    SUPABASE_URL: str
    SUPABASE_SERVICE_ROLE_KEY: str
    SUPABASE_ANON_KEY: str

    # ── LLM ────────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str
    GROQ_API_KEY: str 

    # ── Redis (Celery broker) ──────────────────────────────────────────────
    REDIS_URL: str

    # ── Local ML models (no API cost) ─────────────────────────────────────
    # These are HuggingFace model IDs — downloaded once and cached locally.
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIM: int = 1024
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Ingestion limits ───────────────────────────────────────────────────
    MAX_ISSUES_PER_REPO: int = 2000
    MAX_PRS_PER_REPO: int = 1000
    MAX_CHUNK_SIZE_TOKENS: int = 512
    CHUNK_OVERLAP_TOKENS: int = 50

    # Delay between GitHub API paginated calls to avoid hitting secondary rate limit.
    # GitHub allows max 90 requests/min — 700ms delay keeps us safely under.
    GITHUB_API_DELAY_MS: int = 700

    # ── Free tier limits ───────────────────────────────────────────────────
    FREE_TIER_REPOS_LIMIT: int = 3
    FREE_TIER_CHAT_RPM: int = 20          # requests per minute for chat endpoint

    # ── Pydantic config ────────────────────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",                  # reads from .env in the working directory
        env_file_encoding="utf-8",
        case_sensitive=True,              # SUPABASE_URL != supabase_url
        extra="ignore",                   # silently ignore unknown vars in .env
    )

    # ── Derived properties ─────────────────────────────────────────────────
    # These are computed from other fields — not read from .env.

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def is_development(self) -> bool:
        return self.APP_ENV == "development"

    @property
    def github_api_delay_seconds(self) -> float:
        """Converts GITHUB_API_DELAY_MS to seconds for use in asyncio.sleep()"""
        return self.GITHUB_API_DELAY_MS / 1000


# ── Singleton pattern ──────────────────────────────────────────────────────
#
# @lru_cache means this function only runs ONCE for the entire lifetime
# of the application. Every subsequent call returns the same Settings object.
#
# WHY: Reading and validating the .env file on every request would be slow
# and wasteful. We want one Settings instance shared across the whole app.
#
# HOW TO USE:
#   from app.core.config import settings   <- import the instance, not the class

@lru_cache
def get_settings() -> Settings:
    return Settings()


# This is what every other file imports.
# Usage: from app.core.config import settings
settings = get_settings()
