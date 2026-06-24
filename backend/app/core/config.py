#Every value in your .env file needs to be read into Python somewhere. 
#You could do os.getenv("SUPABASE_URL") scattered across every file — but that's a nightmare.
#If the variable name changes, you'd hunt it down everywhere.
#so this is one central place that reads all env vars, validates them, and exposes them as typed Python attributes. Every other file imports from here.
#The tool we use: Pydantic Settings

## HOW TO USE IT ANYWHERE IN THE PROJECT:
#   from app.core.config import settings
#   print(settings.SUPABASE_URL)
 
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    #The field name must exactly match the variable name in .env.
    #Fields with defaults are optional. Fields without defaults are required.
    #app
    APP_ENV: str = "development"
    SECRET_KEY: str
    FRONTEND_URL: str = "http://localhost:5173"
    BACKEND_URL: str = "http://localhost:8000"

    #GITHUB OAUTH
    GITHUB_CLIENT_ID: str
    GITHUB_CLIENT_SECRET: str
    GITHUB_WEBHOOK_SECRET: str

    #LLM
    GEMINI_API_KEY: str

    #REDIS
    REDIS_URL: str

    #Supabase
    SUPABASE_URL: str
    SUPABASE_SERVICE_ROLE_KEY: str
    SUPABASE_ANON_KEY: str

    #local ml models
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIM: int = 1024
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    #ingetion limits
    MAX_ISSUES_PER_REPO: int = 2000
    MAX_PRS_PER_REPO: int = 1000
    MAX_CHUNK_SIZE_TOKENS: int = 512
    CHUNK_OVERLAP_TOKENS: int = 50

    #delay between github api paginated calls to avoid hitting secondary rate limit
    #github allows max 90 requests/min  - 700ms delay keeps us safely under
    GITHUB_API_DELAY_MS: int = 700

    #free tier limits
    FREE_TIER_REPOS_LIMIT: int = 3
    FREE_TIER_CHAT_RPM: int = 20   

    #pydantic settings config
    model_config = SettingsConfigDict(
        env_file=".env",                  # reads from .env in the working directory
        env_file_encoding="utf-8",
        case_sensitive=True,              # SUPABASE_URL != supabase_url
        extra="ignore",                   # silently ignore unknown vars in .env
    )

    #derived properties 
    #these are computed from other fields - not read from .env

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"
    
    @property
    def is_development(self) -> bool:
        return self.APP_ENV == "development"
    
    @property
    def github_api_delay_seconds(self) -> float:
        return self.GITHUB_API_DELAY_MS / 1000.0
    
#singleton pattern for settings
#@lru_cache — this is a standard Python decorator that memoizes a function's return value. First call runs the function and stores the result. Every call after returns the stored result instantly. This is how you make a singleton in Python without a class pattern.
@lru_cache
def get_settings() -> Settings:
    return Settings()


# This is what every other file imports.
# Usage: from app.core.config import settings
settings = get_settings()