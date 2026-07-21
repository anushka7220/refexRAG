import type {
  Citation,
  IngestStatus,
  Repo,
  Session,
  StalenessFlag,
  StoredMessage,
  User,
} from "./types";

const BASE: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  "http://127.0.0.1:8000";

const TOKEN_KEY = "reflexrag.token";

export const auth = {
  get: () => localStorage.getItem(TOKEN_KEY),
  set: (t: string) => localStorage.setItem(TOKEN_KEY, t),
  clear: () => localStorage.removeItem(TOKEN_KEY),
  loginUrl: () => `${BASE}/auth/github`,
};

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

function headers(): Record<string, string> {
  const t = auth.get();
  return {
    "Content-Type": "application/json",
    ...(t ? { Authorization: `Bearer ${t}` } : {}),
  };
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { ...init, headers: headers() });
  } catch {
    // "Server is down" and "server said no" need completely different
    // fixes, so they get completely different messages.
    throw new ApiError(`Can't reach the backend at ${BASE}. Is it running?`, 0);
  }

  if (res.status === 401 || res.status === 403) {
    throw new ApiError("Session expired. Sign in again.", res.status);
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* body wasn't JSON, keep the status message */
    }
    throw new ApiError(detail, res.status);
  }

  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  me: () => req<User>("/auth/me"),
  listRepos: () => req<Repo[]>("/repos"),
  addRepo: (github_url: string) =>
    req<Repo>("/repos", {
      method: "POST",
      body: JSON.stringify({ github_url }),
    }),
  repoStatus: (id: string) => req<IngestStatus>(`/repos/${id}/status`),
  getRepo: (id: string) => req<Repo>(`/repos/${id}`),
  createSession: (repoId: string) =>
    req<Session>(`/repos/${repoId}/sessions`, { method: "POST" }),

  listSessions: (repoId: string) =>
    req<Session[]>(`/repos/${repoId}/sessions`),

  sessionMessages: (sessionId: string) =>
    req<StoredMessage[]>(`/sessions/${sessionId}/messages`),

  deleteSession: (sessionId: string) =>
    req<void>(`/sessions/${sessionId}`, { method: "DELETE" }),
};

/* ------------------------------------------------------------------
   Streaming chat.

   Uses fetch + ReadableStream rather than EventSource, because
   EventSource can only issue GET requests and cannot set an
   Authorization header, and this endpoint needs both.
   ------------------------------------------------------------------ */

export interface StreamHandlers {
  onToken: (text: string) => void;
  onCitations: (c: Citation[]) => void;
  onFlags: (f: StalenessFlag[]) => void;
  onDone: (messageId: string, tokens: number) => void;
  onError: (message: string) => void;
}

/**
 * Pulls the JSON payload out of one SSE frame.
 *
 * Tolerates a doubled "data: " prefix. The backend's generator yields
 * strings that already start with "data: ", and EventSourceResponse
 * adds its own prefix on top, so frames currently arrive shaped like
 * "data: data: {...}". Stripping repeatedly means this keeps working
 * whether or not that gets fixed server side. Comment frames (": ping")
 * carry no payload and are skipped.
 */
function payloadOf(frame: string): string | null {
  const dataLines = frame
    .split("\n")
    .filter((l) => l.startsWith("data:"))
    .map((l) => l.slice(5));

  if (dataLines.length === 0) return null;

  let payload = dataLines.join("").trim();
  while (payload.startsWith("data:")) payload = payload.slice(5).trim();

  return payload.length ? payload : null;
}

export async function streamChat(
  sessionId: string,
  question: string,
  h: StreamHandlers,
  signal?: AbortSignal
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/sessions/${sessionId}/chat`, {
      method: "POST",
      headers: headers(),
      body: JSON.stringify({ question }),
      signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") return;
    h.onError(`Can't reach the backend at ${BASE}. Is it running?`);
    return;
  }

  if (!res.ok || !res.body) {
    h.onError(
      res.status === 401 || res.status === 403
        ? "Session expired. Sign in again."
        : `The server returned ${res.status}.`
    );
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;

      // SSE permits CRLF, LF, or CR as line terminators. Normalise to LF so
      // frame splitting and "data:" prefix detection work regardless of what
      // the server emits.
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

      // SSE frames are separated by a blank line.
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";

      for (const frame of frames) {
        const raw = payloadOf(frame);
        if (!raw) continue;

        let evt: Record<string, unknown>;
        try {
          evt = JSON.parse(raw);
        } catch {
          continue; // partial or malformed frame, skip rather than crash
        }

        switch (evt.type) {
          case "token":
            h.onToken((evt.content as string) ?? "");
            break;
          case "citations":
            h.onCitations((evt.data as Citation[]) ?? []);
            break;
          case "staleness":
            h.onFlags((evt.data as StalenessFlag[]) ?? []);
            break;
          case "done":
            h.onDone(
              (evt.message_id as string) ?? "",
              (evt.tokens_used as number) ?? 0
            );
            break;
          case "error":
            h.onError((evt.detail as string) ?? "Something went wrong.");
            break;
          default:
            break;
        }
      }
    }
  } catch (e) {
    if ((e as Error).name !== "AbortError") {
      h.onError("The connection dropped mid-answer.");
    }
  }
}

/** Reads a JWT out of the URL fragment after the OAuth round trip. */
export function consumeTokenFromHash(): string | null {
  const hash = window.location.hash.replace(/^#/, "").trim();
  if (!hash || !hash.startsWith("ey")) return null;
  auth.set(hash);
  // Strip the token from the address bar so it doesn't linger in history.
  window.history.replaceState({}, "", window.location.pathname);
  return hash;
}
