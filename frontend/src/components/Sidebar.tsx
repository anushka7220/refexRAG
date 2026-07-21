import { useEffect, useState } from "react";
import { api } from "../lib/api";
import type { Session } from "../lib/types";

/* Conversation history for the active repo. Lists past sessions from
   GET /repos/{id}/sessions, lets you open one, start a new one, or
   delete one. The selected session id is owned by the parent (Chat's
   container) so the thread and the list stay in sync. */

function sessionLabel(s: Session): string {
  if (s.title && s.title.trim()) return s.title;
  // Untitled sessions fall back to a stable, readable stamp.
  const d = new Date(s.created_at);
  return isNaN(d.getTime())
    ? "New conversation"
    : d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) +
        ", " +
        d.toLocaleTimeString(undefined, {
          hour: "numeric",
          minute: "2-digit",
        });
}

export default function Sidebar({
  repoId,
  activeSessionId,
  onSelect,
  onNew,
  refreshKey,
}: {
  repoId: string;
  activeSessionId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  refreshKey: number;
}) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    api
      .listSessions(repoId)
      .then((rows) => {
        if (!alive) return;
        // Newest first; backend order isn't guaranteed.
        rows.sort((a, b) => b.created_at.localeCompare(a.created_at));
        setSessions(rows);
      })
      .catch(() => {
        if (alive) setSessions([]);
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [repoId, refreshKey]);

  async function remove(e: React.MouseEvent, id: string) {
    e.stopPropagation();
    // Optimistic: drop it immediately, the delete is fire-and-forget.
    setSessions((s) => s.filter((x) => x.id !== id));
    try {
      await api.deleteSession(id);
    } catch {
      /* if it fails the next refresh will restore it */
    }
    if (id === activeSessionId) onNew();
  }

  return (
    <aside className="aside">
      <div className="aside-head">
        <span className="aside-title">Conversations</span>
      </div>

      <button className="aside-new" onClick={onNew}>
        <b>+</b> New conversation
      </button>

      <div className="aside-list">
        {loading ? (
          <div className="aside-empty">Loading…</div>
        ) : sessions.length === 0 ? (
          <div className="aside-empty">
            No conversations yet. Ask a question to start one.
          </div>
        ) : (
          sessions.map((s) => (
            <button
              key={s.id}
              className={
                s.id === activeSessionId ? "aside-item active" : "aside-item"
              }
              onClick={() => onSelect(s.id)}
            >
              <span className="aside-item-title">{sessionLabel(s)}</span>
              <span
                className="aside-item-del"
                onClick={(e) => remove(e, s.id)}
                role="button"
                aria-label="Delete conversation"
              >
                ✕
              </span>
            </button>
          ))
        )}
      </div>
    </aside>
  );
}
