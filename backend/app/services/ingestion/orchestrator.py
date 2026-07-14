# orchestrator.py
#
# Coordinates the full ingestion pipeline end to end.
# This is the only file that imports all ingestion services together.
# Everything else stays decoupled and independently testable.
#
# SEQUENCE:
#   1. Fetch repo metadata
#   2. Fetch issues, PRs, commits, releases from GitHub
#   3. Build version map from releases
#   4. Chunk everything
#   5. Embed all chunks
#   6. Upsert to pgvector
#   7. Run decision extraction and contributor building in parallel
#   8. Mark ingestion job as done
#
# PROGRESS TRACKING:
# Every stage writes to ingestion_jobs so the frontend polling endpoint
# has real data to show, not a generic spinner.

import asyncio
import structlog
from datetime import datetime, timezone

from app.core.supabase import supabase_admin, execute
from app.core.config import settings
from app.services.ingestion.github_fetcher import GitHubFetcher
from app.services.ingestion.chunker import Chunker
from app.services.ingestion.embedding_service import embedding_service
from app.services.ingestion.vector_store import vector_store
from app.services.features.decision_extractor import decision_extractor
from app.services.features.contributor_builder import ContributorBuilder
from app.utils.version_extractor import build_version_map
from app.utils.github_parser import parse_github_url

log = structlog.get_logger(__name__)


class IngestionOrchestrator:
    """
    Runs the full ingestion pipeline for one repo.

    Usage (called from a Celery task):
        orchestrator = IngestionOrchestrator()
        await orchestrator.ingest(repo_id="uuid", github_url="https://github.com/owner/repo")
    """

    def __init__(self):
        self.contributor_builder = ContributorBuilder()

    async def ingest(self, repo_id: str, github_url: str) -> None:
        """
        Runs the complete ingestion pipeline for a repo.

        Args:
            repo_id:    UUID of the repo row already created in Supabase.
            github_url: Full GitHub repo URL.

        Updates ingestion_jobs at every stage so progress is visible.
        Updates repos.status when complete or failed.
        """
        job_id = self._create_job(repo_id)

        try:
            parsed = parse_github_url(github_url)
            if not parsed:
                raise ValueError(f"Invalid GitHub URL: {github_url}")
            owner, repo_name = parsed

            fetcher = GitHubFetcher(github_token=None)
            chunker = Chunker(repo_id=repo_id, owner=owner, repo_name=repo_name)

            # Stage 1: fetch metadata
            self._update_job(job_id, stage="fetching", progress=5)
            meta = await fetcher.fetch_repo_meta(github_url)
            self._update_repo_meta(repo_id, meta)

            # Stage 2: fetch all GitHub data
            self._update_job(job_id, stage="fetching", progress=15)
            issues, prs, commits, releases = await asyncio.gather(
                fetcher.fetch_issues(owner, repo_name),
                fetcher.fetch_prs(owner, repo_name),
                fetcher.fetch_commits(owner, repo_name),
                fetcher.fetch_releases(owner, repo_name),
            )
            log.info(
                "fetch_complete",
                repo_id=repo_id,
                issues=len(issues),
                prs=len(prs),
                commits=len(commits),
                releases=len(releases),
            )

            # Stage 2b: fetch source code as ONE tarball call.
            # Runs after discussion fetching on purpose: the files that
            # commits and PRs touched become priority_paths, so when the
            # source-file cap bites, discussion-linked files are kept first.
            # Those are the files the code-to-discussion join can enrich.
            self._update_job(job_id, stage="fetching", progress=30)
            priority_paths = set()
            for commit in commits:
                priority_paths.update(commit.files)
            for pr in prs:
                priority_paths.update(getattr(pr, "files_changed", []) or [])

            try:
                source_files_raw = await fetcher.fetch_source_files(
                    owner, repo_name, priority_paths=priority_paths
                )
            except Exception as e:
                # Code indexing failing must not kill discussion indexing.
                log.warning("source_fetch_failed_continuing", error=str(e))
                source_files_raw = []

            # Stage 3: chunk everything
            self._update_job(job_id, stage="chunking", progress=40)
            version_map = build_version_map(releases)

            all_chunks = []
            for issue in issues:
                all_chunks.extend(chunker.chunk_issue(issue, version_map))
            for pr in prs:
                all_chunks.extend(chunker.chunk_pr(pr, version_map))
            for commit in commits:
                all_chunks.extend(chunker.chunk_commit(commit))
            for release in releases:
                all_chunks.extend(chunker.chunk_release(release))

            # Stage 3b: chunk source code by structure
            if source_files_raw:
                from app.services.ingestion.code_chunker import (
                    CodeChunker, SourceFile, language_for,
                )
                code_chunker = CodeChunker(
                    repo_id=repo_id,
                    owner=owner,
                    repo_name=repo_name,
                    default_branch=meta.default_branch,
                )
                source_files = [
                    SourceFile(path=f.path, content=f.content, language=language_for(f.path) or "other")
                    for f in source_files_raw
                ]
                all_chunks.extend(code_chunker.chunk_files(source_files))

            log.info("chunking_complete", repo_id=repo_id, total_chunks=len(all_chunks))

            # Stage 4: embed everything
            self._update_job(job_id, stage="embedding", progress=60)
            all_chunks = await embedding_service.embed_chunks(all_chunks)

            # Stage 5: upsert to pgvector
            self._update_job(job_id, stage="embedding", progress=80)
            inserted, skipped = vector_store.upsert_chunks(all_chunks)
            log.info(
                "upsert_complete",
                repo_id=repo_id,
                inserted=inserted,
                skipped=skipped,
            )

            # Stage 6: feature extraction, runs in parallel since neither
            # decision extraction nor contributor building depends on the other
            self._update_job(job_id, stage="extracting", progress=90)
            await asyncio.gather(
                self._run_decision_extraction(repo_id, prs),
                self._run_contributor_building(repo_id, commits, prs),
            )

            # Stage 7: mark done
            self._update_job(job_id, stage="done", progress=100)
            self._update_repo_status(repo_id, status="done", chunk_count=inserted)

            log.info("ingestion_complete", repo_id=repo_id)

        except Exception as e:
            log.error("ingestion_failed", repo_id=repo_id, error=str(e))
            self._update_job(job_id, stage="failed", progress=0, error_msg=str(e))
            self._update_repo_status(repo_id, status="failed")
            raise

    async def differential_ingest(self, repo_id: str, since_sha: str) -> None:
        """
        Re-ingests only objects changed since a given commit SHA.
        Called by the webhook handler when GitHub notifies us of repo updates.

        Much cheaper than full ingest. Only fetches and embeds what changed.
        """
        job_id = self._create_job(repo_id)

        try:
            repo_row = self._get_repo(repo_id)
            owner, repo_name = repo_row["owner"], repo_row["name"]
            github_url = repo_row["github_url"]

            fetcher = GitHubFetcher(github_token=None)
            chunker = Chunker(repo_id=repo_id, owner=owner, repo_name=repo_name)

            self._update_job(job_id, stage="fetching", progress=20)
            changed = await fetcher.fetch_changed_since(owner, repo_name, since_sha)

            self._update_job(job_id, stage="chunking", progress=50)
            version_map = self._get_cached_version_map(repo_id)

            new_chunks = []
            for issue in changed["issues"]:
                new_chunks.extend(chunker.chunk_issue(issue, version_map))
            for pr in changed["prs"]:
                new_chunks.extend(chunker.chunk_pr(pr, version_map))

            if new_chunks:
                self._update_job(job_id, stage="embedding", progress=75)
                new_chunks = await embedding_service.embed_chunks(new_chunks)
                inserted, skipped = vector_store.upsert_chunks(new_chunks)
                log.info(
                    "differential_upsert_complete",
                    repo_id=repo_id,
                    inserted=inserted,
                    skipped=skipped,
                )

            meta = await fetcher.fetch_repo_meta(github_url)
            self._update_repo_meta(repo_id, meta)

            self._update_job(job_id, stage="done", progress=100)
            log.info("differential_ingest_complete", repo_id=repo_id)

        except Exception as e:
            log.error("differential_ingest_failed", repo_id=repo_id, error=str(e))
            self._update_job(job_id, stage="failed", progress=0, error_msg=str(e))
            raise

    async def _run_decision_extraction(self, repo_id: str, prs: list) -> None:
        """
        Runs the multi agent decision extractor on every PR with a body.
        Stores accepted DecisionNodes in the decision_nodes table.
        Skips PRs with very short bodies since those rarely contain real decisions.
        """
        decision_count = 0

        for pr in prs:
            if len(pr.body or "") < 50:
                continue

            extraction = await decision_extractor.extract(
                pr_body=pr.body,
                comments=pr.comments + [r.body for r in pr.reviews if r.body],
                source_id=f"pr#{pr.number}",
            )

            if extraction:
                embedding = await embedding_service.embed_single(extraction.decision)
                self._save_decision_node(repo_id, extraction, embedding, pr.number)
                decision_count += 1

        log.info("decision_extraction_complete", repo_id=repo_id, count=decision_count)

    async def _run_contributor_building(
        self,
        repo_id: str,
        commits: list,
        prs: list,
    ) -> None:
        """
        Builds ownership scores, authority scores, and file area data.
        Pure computation, no LLM calls needed. Fast relative to other stages.
        """
        contributors = self.contributor_builder.build_ownership_scores(commits)
        authority = self.contributor_builder.build_authority_scores(prs)
        file_areas = self.contributor_builder.build_file_areas(commits)

        self._save_contributors(repo_id, contributors, authority)
        self._save_file_areas(repo_id, file_areas)

        log.info(
            "contributor_building_complete",
            repo_id=repo_id,
            contributors=len(contributors),
            file_areas=len(file_areas),
        )

    def _create_job(self, repo_id: str) -> str:
        response = (
            supabase_admin.table("ingestion_jobs")
            .insert({
                "repo_id": repo_id,
                "stage": "queued",
                "progress_pct": 0,
                "started_at": datetime.now(timezone.utc).isoformat(),
            })
            .execute()
        )
        rows = execute(response)
        return rows[0]["id"]

    def _update_job(
        self,
        job_id: str,
        stage: str,
        progress: int,
        error_msg: str | None = None,
    ) -> None:
        payload = {"stage": stage, "progress_pct": progress}
        if error_msg:
            payload["error_msg"] = error_msg
        if stage in ("done", "failed"):
            payload["finished_at"] = datetime.now(timezone.utc).isoformat()

        supabase_admin.table("ingestion_jobs").update(payload).eq("id", job_id).execute()

    def _update_repo_status(
        self,
        repo_id: str,
        status: str,
        chunk_count: int | None = None,
    ) -> None:
        payload = {
            "status": status,
            "last_ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        if chunk_count is not None:
            payload["chunk_count"] = chunk_count

        supabase_admin.table("repos").update(payload).eq("id", repo_id).execute()

    def _update_repo_meta(self, repo_id: str, meta) -> None:
        supabase_admin.table("repos").update({
            "owner": meta.owner,
            "name": meta.name,
            "latest_commit_sha": meta.latest_commit_sha,
        }).eq("id", repo_id).execute()

    def _get_repo(self, repo_id: str) -> dict:
        response = supabase_admin.table("repos").select("*").eq("id", repo_id).execute()
        rows = execute(response)
        if not rows:
            raise ValueError(f"Repo not found: {repo_id}")
        return rows[0]

    def _get_cached_version_map(self, repo_id: str) -> dict:
        """
        Rebuilds version map from existing release chunks already in the DB.
        Avoids re-fetching releases from GitHub on every differential ingest.
        """
        response = (
            supabase_admin.table("chunks")
            .select("version_tag, source_created_at")
            .eq("repo_id", repo_id)
            .eq("source_type", "release")
            .execute()
        )
        rows = execute(response)
        version_map = {}
        for row in rows:
            if row.get("version_tag"):
                dt = datetime.fromisoformat(row["source_created_at"].replace("Z", "+00:00"))
                version_map[dt] = row["version_tag"]
        return version_map

    def _save_decision_node(
        self,
        repo_id: str,
        extraction,
        embedding: list[float],
        pr_number: int,
    ) -> None:
        supabase_admin.table("decision_nodes").insert({
            "repo_id": repo_id,
            "decision": extraction.decision,
            "alternatives_rejected": [
                {"option": a.option, "reason": a.reason}
                for a in extraction.alternatives_rejected
            ],
            "reasoning": extraction.reasoning,
            "source_chunk_ids": [],
            "embedding": embedding,
        }).execute()

    def _save_contributors(
        self,
        repo_id: str,
        ownership: dict,
        authority: dict,
    ) -> None:
        rows = []
        usernames = set(ownership.keys()) | set(authority.keys())
        for username in usernames:
            rows.append({
                "repo_id": repo_id,
                "github_username": username,
                "ownership_score": ownership.get(username, 0.0),
                "authority_score": authority.get(username, 0.0),
                "top_areas": [],
            })
        if rows:
            supabase_admin.table("contributors").insert(rows).execute()

    def _save_file_areas(self, repo_id: str, file_areas: list) -> None:
        rows = [
            {
                "repo_id": repo_id,
                "area_path": area.area_path,
                "complexity_score": area.complexity_score,
                "co_changes_with": area.co_changes_with,
            }
            for area in file_areas
        ]
        if rows:
            supabase_admin.table("file_areas").insert(rows).execute()


orchestrator = IngestionOrchestrator()