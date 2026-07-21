import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { api, streamChat } from "../lib/api";
import type { Repo, StoredMessage, Turn } from "../lib/types";
import Citations from "./Citations";

const EXAMPLES = [
  { kind: "why", text: "Why is this built the way it is?" },
  { kind: "issues", text: "What problems do people report most?" },
  { kind: "people", text: "Who should I ask about this codebase?" },
];

function newId(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : String(Date.now() + Math.random());
}

/* Groups a flat message list (user, assistant, user, ...) into Turn shape. */
function messagesToTurns(msgs: StoredMessage[]): Turn[] {
  const turns: Turn[] = [];
  let pending: string | null = null;

  for (const m of msgs) {
    if (m.role === "user") {
      if (pending !== null) {
        turns.push({
          id: newId(),
          question: pending,
          answer: "",
          citations: [],
          flags: [],
          streaming: false,
        });
      }
      pending = m.content;
    } else {
      turns.push({
        id: m.id,
        question: pending ?? "",
        answer: m.content,
        citations: m.citations ?? [],
        flags: m.staleness_flags ?? [],
        streaming: false,
      });
      pending = null;
    }
  }
  if (pending !== null) {
    turns.push({
      id: newId(),
      question: pending,
      answer: "",
      citations: [],
      flags: [],
      streaming: false,
    });
  }
  return turns;
}

export default function Chat({
  repo,
  sessionId,
  onSessionCreated,
}: {
  repo: Repo;
  sessionId: string | null;
  onSessionCreated: (id: string) => void;
}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [localSession, setLocalSession] = useState<string | null>(sessionId);

  const threadRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let alive = true;
    abortRef.current?.abort();
    setLocalSession(sessionId);

    if (!sessionId) {
      setTurns([]);
      return;
    }

    api
      .sessionMessages(sessionId)
      .then((msgs) => {
        if (alive) setTurns(messagesToTurns(msgs));
      })
      .catch(() => {
        if (alive) setTurns([]);
      });

    return () => {
      alive = false;
    };
  }, [sessionId, repo.id]);

  useEffect(() => {
    threadRef.current?.scrollTo({
      top: threadRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [turns]);

  function autosize() {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 148)}px`;
  }

  async function ensureSession(): Promise<string | null> {
    if (localSession) return localSession;
    try {
      const s = await api.createSession(repo.id);
      setLocalSession(s.id);
      onSessionCreated(s.id);
      return s.id;
    } catch {
      return null;
    }
  }

  async function ask(question: string) {
    const q = question.trim();
    if (!q || busy) return;

    const sid = await ensureSession();
    if (!sid) return;

    const id = newId();
    setTurns((t) => [
      ...t,
      { id, question: q, answer: "", citations: [], flags: [], streaming: true },
    ]);
    setDraft("");
    setBusy(true);
    requestAnimationFrame(autosize);

    const patch = (fn: (t: Turn) => Turn) =>
      setTurns((all) => all.map((t) => (t.id === id ? fn(t) : t)));

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    await streamChat(
      sid,
      q,
      {
        onToken: (text) => patch((t) => ({ ...t, answer: t.answer + text })),
        onCitations: (c) => patch((t) => ({ ...t, citations: c })),
        onFlags: (f) => patch((t) => ({ ...t, flags: f })),
        onDone: () => patch((t) => ({ ...t, streaming: false })),
        onError: (msg) => patch((t) => ({ ...t, streaming: false, error: msg })),
      },
      ctrl.signal
    );

    patch((t) => ({ ...t, streaming: false }));
    setBusy(false);
  }

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void ask(draft);
    }
  }

  return (
    <>
      <div className="thread" ref={threadRef}>
        <div className="thread-inner">
          {turns.length === 0 && (
            <div style={{ paddingTop: 24 }}>
              <span className="eyebrow">
                Indexed &middot; {repo.chunk_count} passages
              </span>
              <h2 className="headline">
                Ask why it&rsquo;s <em>that way</em>.
              </h2>
              <p className="sub">
                Answers cite the exact code, and the pull requests and commits
                that shaped it.
              </p>
              <div className="examples">
                {EXAMPLES.map((ex) => (
                  <button
                    key={ex.text}
                    className="example"
                    onClick={() => void ask(ex.text)}
                  >
                    <b>{ex.kind}</b>
                    {ex.text}
                  </button>
                ))}
              </div>
            </div>
          )}

          {turns.map((t) => (
            <div className="turn" key={t.id}>
              <div className="turn-q">{t.question}</div>
              <div style={{ marginTop: 16 }}>
                {!t.answer && t.streaming && (
                  <div className="thinking">
                    <span className="thinking-dot" />
                    Reading the repository
                  </div>
                )}
                {t.answer && (
                  <div className="turn-a-body">
                    {t.answer}
                    {t.streaming && <span className="caret" />}
                  </div>
                )}
                {t.error && <div className="err-note">{t.error}</div>}
                {t.flags.length > 0 && (
                  <div className="flags">
                    {t.flags.map((f, i) => (
                      <div
                        key={i}
                        className={f.severity === "error" ? "flag error" : "flag"}
                      >
                        <span className="flag-mark">!</span>
                        <span>{f.detail}</span>
                      </div>
                    ))}
                  </div>
                )}
                {!t.streaming && <Citations items={t.citations} />}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="composer">
        <div className="composer-inner">
          <textarea
            ref={taRef}
            rows={1}
            value={draft}
            placeholder="Ask why&hellip;"
            disabled={busy}
            onChange={(e) => {
              setDraft(e.target.value);
              autosize();
            }}
            onKeyDown={onKey}
          />
          <button
            className="send-btn"
            onClick={() => void ask(draft)}
            disabled={!draft.trim() || busy}
            aria-label="Send question"
          >
            {busy ? <span className="spin" /> : "\u2191"}
          </button>
        </div>
      </div>
    </>
  );
}
