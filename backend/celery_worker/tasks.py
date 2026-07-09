# celery_worker/tasks.py
#
# Celery task definitions. These are the functions the worker process
# actually runs when it picks a message off the Redis queue.
#
# HOW TO CALL A TASK FROM FASTAPI:
#   from celery_worker.tasks import ingest_repo
#   ingest_repo.delay(repo_id="uuid", github_url="https://github.com/...")
#
# .delay() serialises the arguments to JSON, pushes them to Redis,
# and returns immediately. The worker picks them up asynchronously.
#
# HOW ASYNC WORKS HERE:
# Celery tasks run synchronously in the worker process by default.
# Our orchestrator is async. We bridge this with asyncio.run(),
# which creates a fresh event loop, runs the coroutine to completion,
# and returns. This is the correct pattern for calling async code
# from a synchronous Celery task.

import asyncio
import structlog
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from celery_worker.celery_app import celery_app
from app.services.ingestion.orchestrator import orchestrator

log = structlog.get_logger(__name__)


class BaseTask(Task):
    """
    Base class for tasks that need shared setup logic.
    abstract = True prevents Celery from registering this as a callable task.
    """
    abstract = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called by Celery automatically when a task raises an exception."""
        log.error(
            "task_failed",
            task_id=task_id,
            exc=str(exc),
            kwargs=kwargs,
        )

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        log.warning(
            "task_retrying",
            task_id=task_id,
            exc=str(exc),
            kwargs=kwargs,
        )


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="celery_worker.tasks.ingest_repo",
    max_retries=2,
    default_retry_delay=60,    # wait 60 seconds before retrying
)
def ingest_repo(self, repo_id: str, github_url: str) -> dict:
    """
    Background task that runs the full ingestion pipeline for a repo.

    Called from: api/routes/repos.py via ingest_repo.delay()

    Args:
        repo_id:    UUID of the repo row already created in Supabase.
        github_url: Full public GitHub URL.

    Returns:
        Dict with status on completion. Not used since we have no
        result backend, but Celery expects a return value.

    Retry behaviour:
        On transient failures (network errors, GitHub rate limits),
        Celery retries up to max_retries times with default_retry_delay
        seconds between attempts. On hard failures the task is marked
        failed and the error is written to ingestion_jobs.
    """
    log.info("ingest_repo_task_start", repo_id=repo_id, github_url=github_url)

    try:
        # asyncio.run() creates a new event loop, runs the coroutine,
        # closes the loop. Safe to call from a synchronous Celery task.
        asyncio.run(orchestrator.ingest(repo_id=repo_id, github_url=github_url))
        log.info("ingest_repo_task_done", repo_id=repo_id)
        return {"status": "done", "repo_id": repo_id}

    except SoftTimeLimitExceeded:
        # Raised at 30 minutes by Celery. Write a clean failure to the
        # DB rather than letting the task hang until the hard kill at 35 min.
        log.error("ingest_repo_timeout", repo_id=repo_id)
        _mark_repo_failed(repo_id, "Ingestion timed out after 30 minutes.")
        return {"status": "timeout", "repo_id": repo_id}

    except Exception as exc:
        log.error("ingest_repo_task_error", repo_id=repo_id, error=str(exc))
        try:
            # Retry on transient errors. self.retry() raises a Retry
            # exception internally — it does not return, so the line
            # after it only runs if max_retries is already exhausted.
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            _mark_repo_failed(repo_id, str(exc))
            return {"status": "failed", "repo_id": repo_id}


@celery_app.task(
    bind=True,
    base=BaseTask,
    name="celery_worker.tasks.differential_ingest",
    max_retries=1,
    default_retry_delay=30,
)
def differential_ingest(self, repo_id: str, since_sha: str) -> dict:
    """
    Background task for re ingesting only objects changed since a commit.
    Called from: api/routes/webhooks.py when a GitHub push event arrives.

    Much faster than full ingest since it only processes changed objects.
    Lower max_retries since webhook events are time sensitive and a
    failed differential ingest is less critical than a failed full ingest.
    The next webhook event will catch anything missed.

    Args:
        repo_id:   UUID of the repo in Supabase.
        since_sha: Commit SHA to use as the differential baseline.
    """
    log.info(
        "differential_ingest_task_start",
        repo_id=repo_id,
        since_sha=since_sha[:8],
    )

    try:
        asyncio.run(
            orchestrator.differential_ingest(
                repo_id=repo_id,
                since_sha=since_sha,
            )
        )
        log.info("differential_ingest_task_done", repo_id=repo_id)
        return {"status": "done", "repo_id": repo_id}

    except SoftTimeLimitExceeded:
        log.error("differential_ingest_timeout", repo_id=repo_id)
        return {"status": "timeout", "repo_id": repo_id}

    except Exception as exc:
        log.error("differential_ingest_error", repo_id=repo_id, error=str(exc))
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            return {"status": "failed", "repo_id": repo_id}


def _mark_repo_failed(repo_id: str, error_msg: str) -> None:
    """
    Writes a failed status to Supabase when a task exhausts all retries.
    Called only after Celery gives up retrying, not on every failure.
    Imported here rather than from orchestrator to avoid circular imports
    since orchestrator imports from celery_worker indirectly.
    """
    from app.core.supabase import supabase_admin
    from datetime import datetime, timezone

    try:
        supabase_admin.table("repos").update({
            "status": "failed",
        }).eq("id", repo_id).execute()

        supabase_admin.table("ingestion_jobs").insert({
            "repo_id": repo_id,
            "stage": "failed",
            "progress_pct": 0,
            "error_msg": error_msg[:500],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }).execute()

    except Exception as e:
        log.error("mark_repo_failed_error", repo_id=repo_id, error=str(e))