// Mirrors the Pydantic models the backend actually returns.
// Keep SourceType in sync with SourceType in app/models/chunk.py.

export type SourceType =
  | "issue"
  | "pr"
  | "comment"
  | "commit"
  | "release"
  | "discussion"
  | "code";

export interface Citation {
  chunk_id: string;
  source_type: SourceType;
  source_id: string;
  status: string;
  version_tag: string | null;
  url: string;
  excerpt: string;
}

export interface StalenessFlag {
  chunk_id: string;
  reason: string;
  severity: "warn" | "error";
  detail: string;
}

export interface User {
  id: string;
  username: string;
  email: string | null;
  avatar_url: string | null;
  plan: string;
  repos_used: number;
}

export interface Repo {
  id: string;
  github_url: string;
  owner: string;
  name: string;
  status: string;
  chunk_count: number;
  decision_count: number;
  last_ingested_at: string | null;
}

export interface IngestStatus {
  repo_id: string;
  stage: string;
  progress_pct: number;
  error_msg: string | null;
}

export interface Session {
  id: string;
  repo_id: string;
  title: string | null;
  message_count: number;
  created_at: string;
}

/** One question and its answer, as rendered in the thread. */
export interface Turn {
  id: string;
  question: string;
  answer: string;
  citations: Citation[];
  flags: StalenessFlag[];
  streaming: boolean;
  error?: string;
}

/** A message as stored and returned by GET /sessions/{id}/messages. */
export interface StoredMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  staleness_flags: StalenessFlag[];
  model_used: string | null;
  tokens_used: number | null;
  created_at: string;
}
