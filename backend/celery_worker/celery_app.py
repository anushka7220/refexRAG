# celery_worker/celery_app.py
#
# Configures the Celery application and connects it to Redis as the broker.
#
# HOW CELERY WORKS WITH FASTAPI:
# FastAPI and Celery are two separate processes. They do not share memory.
# They communicate only through Redis:
#
#   FastAPI (web process)
#       calls: ingest_repo.delay(repo_id, github_url)
#       pushes a message to Redis
#
#   Celery worker (background process)
#       reads the message from Redis
#       calls the actual task function
#       writes results to Supabase (not back to FastAPI)
#
# WHY NO RESULT BACKEND:
# Ingestion progress is written directly to Supabase ingestion_jobs by the
# orchestrator. The frontend polls that table via the status endpoint.
# Using a separate Celery result backend would be redundant.

from celery import Celery
from app.core.config import settings


def create_celery_app() -> Celery:
    app = Celery("reflexRAG")

    app.config_from_object({
        # Upstash Redis uses rediss:// (SSL). Plain Redis uses redis://.
        "broker_url": settings.REDIS_URL,

        # No result backend needed, progress goes to Supabase directly.
        "result_backend": None,

        # JSON is readable and safe. Pickle is faster but can deserialise
        # arbitrary Python objects, which is a security risk.
        "task_serializer": "json",
        "accept_content": ["json"],
        "result_serializer": "json",

        "timezone": "UTC",
        "enable_utc": True,

        # A large repo with 2000 issues at 700ms delay per call takes
        # over 20 minutes to fetch. The hard limit gives a safety margin.
        "task_acks_late": True,
        "task_time_limit":      60 * 35,    # hard kill at 35 minutes
        "task_soft_time_limit": 60 * 30,    # raises SoftTimeLimitExceeded at 30

        # Take one task at a time. Each ingestion is already heavy enough
        # that prefetching more would cause memory pressure on the free tier.
        "worker_prefetch_multiplier": 1,

        "include": ["celery_worker.tasks"],
    })

    return app


celery_app = create_celery_app()