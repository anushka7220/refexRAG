import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import type { Repo } from "../lib/types";

/* Backend stage names are internal vocabulary ("chunking", "extracting").
   These say what is happening to the repository instead, in the reader's
   terms. Order matches the orchestrator's actual pipeline order. */
const STAGES: { key: string; label: string }[] = [
  { key: "fetching", label: "Reading issues, pull requests, and commits" },
  { key: "chunking", label: "Splitting into passages" },
  { key: "embedding", label: "Building the search index" },
  { key: "extracting", label: "Mapping who owns what" },
  { key: "done", label: "Ready" },
];

export default function IngestProgress({
  repo,
  onReady,
  onCancel,
}: {
  repo: Repo;
  onReady: (r: Repo) => void;
  onCancel: () => void;
}) {
  const [stage, setStage] = useState(repo.status || "fetching");
  const [pct, setPct] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    let alive = true;

    async function poll() {
      try {
        const s = await api.repoStatus(repo.id);
        if (!alive) return;

        setStage(s.stage);
        setPct(s.progress_pct ?? 0);

        if (s.stage === "failed") {
          setError(s.error_msg || "Indexing failed. Try again.");
          return;
        }
        if (s.stage === "done") {
          // Re-fetch the repo so chunk_count is populated for the chat header.
          const fresh = await api.getRepo(repo.id).catch(() => repo);
          if (alive) onReady(fresh);
          return;
        }
        timer.current = window.setTimeout(poll, 2000);
      } catch (e) {
        if (!alive) return;
        setError((e as Error).message);
      }
    }

    void poll();
    return () => {
      alive = false;
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [repo.id]);

  const currentIndex = STAGES.findIndex((s) => s.key === stage);

  return (
    <div className="stage">
      <div className="stage-inner">
        <span className="eyebrow">Indexing</span>
        <h1 className="headline">
          {repo.owner}/<em>{repo.name}</em>
        </h1>
        <p className="sub">
          Reading the repository&rsquo;s history. You can chat as soon as the
          index is built; the contributor map keeps building after that.
        </p>

        {error ? (
          <>
            <div className="err-note">{error}</div>
            <div style={{ marginTop: 16 }}>
              <button className="primary-btn" onClick={onCancel}>
                Back
              </button>
            </div>
          </>
        ) : (
          <>
            <div className="prog-stages">
              {STAGES.map((s, i) => {
                const state =
                  currentIndex < 0
                    ? ""
                    : i < currentIndex
                    ? "past"
                    : i === currentIndex
                    ? "active"
                    : "";
                return (
                  <div key={s.key} className={`prog-stage ${state}`}>
                    <span className="prog-tick">
                      {state === "past" ? "\u2713" : state === "active" ? "\u203a" : "\u00b7"}
                    </span>
                    <span>{s.label}</span>
                    {state === "active" && (
                      <span style={{ marginLeft: "auto" }} className="spin" />
                    )}
                  </div>
                );
              })}
            </div>

            <div className="prog-rail">
              <div className="prog-fill" style={{ width: `${pct}%` }} />
            </div>

            <p className="hint">
              This runs in the background. Leaving this page won&rsquo;t stop it.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
