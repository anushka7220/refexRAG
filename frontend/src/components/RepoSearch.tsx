import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { Repo } from "../lib/types";

function timeAgo(iso: string | null): string {
  if (!iso) return "not indexed";
  const secs = (Date.now() - new Date(iso).getTime()) / 1000;
  if (secs < 60) return "just now";
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}

function dotClass(status: string): string {
  if (status === "done") return "dot done";
  if (status === "failed") return "dot failed";
  return "dot queued";
}

export default function RepoSearch({ onPick }: { onPick: (r: Repo) => void }) {
  const [value, setValue] = useState("");
  const [repos, setRepos] = useState<Repo[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listRepos()
      .then(setRepos)
      .catch(() => {
        /* search still works without history, so stay quiet */
      });
  }, []);

  async function submit() {
    const url = value.trim();
    if (!url || submitting) return;

    setSubmitting(true);
    setError(null);
    try {
      const full = url.startsWith("http") ? url : `https://github.com/${url}`;
      const repo = await api.addRepo(full);
      onPick(repo);
    } catch (e) {
      setError((e as Error).message);
      setSubmitting(false);
    }
  }

  return (
    <div className="stage">
      <div className="stage-inner">
        <span className="eyebrow">Repository history, indexed</span>
        <h1 className="headline">
          Paste a repository.
          <br />
          Ask why it&rsquo;s <em>built that way</em>.
        </h1>
        <p className="sub">
          Most tools read the code. reflexRAG also reads the issues, pull
          requests, and commits around it, so it can answer the questions the
          code alone can&rsquo;t.
        </p>

        <div className="field">
          <span className="field-prefix">github.com/</span>
          <input
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void submit();
            }}
            placeholder="owner/repo"
            spellCheck={false}
            autoFocus
            aria-label="GitHub repository"
          />
          <button
            className="primary-btn"
            onClick={() => void submit()}
            disabled={!value.trim() || submitting}
          >
            {submitting ? "Starting\u2026" : "Index"}
          </button>
        </div>

        <p className="hint">
          Public repositories only. Small ones index in about a minute.
        </p>

        {error && <div className="err-note">{error}</div>}

        {repos.length > 0 && (
          <div className="recent">
            <span className="eyebrow">Already indexed</span>
            <div className="recent-list">
              {repos.map((r) => (
                <button
                  key={r.id}
                  className="recent-row"
                  onClick={() => onPick(r)}
                >
                  <span className={dotClass(r.status)} />
                  <span className="recent-name">
                    {r.owner}/{r.name}
                  </span>
                  <span className="recent-meta">
                    {r.status === "done"
                      ? `${r.chunk_count} passages \u00b7 ${timeAgo(
                          r.last_ingested_at
                        )}`
                      : r.status}
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
