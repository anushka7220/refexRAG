import { useEffect, useState } from "react";
import { api, auth, consumeTokenFromHash } from "./lib/api";
import type { Repo, User } from "./lib/types";
import RepoSearch from "./components/RepoSearch";
import IngestProgress from "./components/IngestProgress";
import Chat from "./components/Chat";
import Sidebar from "./components/Sidebar";

type Boot = "checking" | "signed-out" | "signed-in";

function Glyph() {
  // Two strata joined by a rail: the product's what/why split as a mark.
  return (
    <svg className="hdr-glyph" viewBox="0 0 20 20" fill="none" aria-hidden="true">
      <rect x="2" y="3" width="10" height="3.4" rx="1" fill="var(--muted)" />
      <rect x="2" y="13.6" width="10" height="3.4" rx="1" fill="var(--brass)" />
      <path d="M13.5 6.7 V13.3" stroke="var(--brass-dim)" strokeWidth="1.2" />
      <circle cx="13.5" cy="6.7" r="1.5" fill="var(--muted)" />
      <circle cx="13.5" cy="13.3" r="1.5" fill="var(--brass)" />
    </svg>
  );
}

function Header({
  user,
  repo,
  onHome,
  onSignOut,
}: {
  user: User | null;
  repo: Repo | null;
  onHome: () => void;
  onSignOut: () => void;
}) {
  return (
    <header className="hdr">
      <button className="hdr-mark" onClick={onHome} aria-label="Home">
        <Glyph />
        reflex<span>RAG</span>
      </button>

      {repo && (
        <>
          <span className="hdr-divider" />
          <div className="hdr-repo">
            <b>
              {repo.owner}/{repo.name}
            </b>
            {repo.chunk_count > 0 && (
              <span className="hdr-repo-badge">{repo.chunk_count} passages</span>
            )}
          </div>
        </>
      )}

      <span className="hdr-spacer" />

      <nav className="hdr-nav">
        <a
          href={repo ? repo.github_url : "https://github.com"}
          target="_blank"
          rel="noreferrer"
        >
          {repo ? "View on GitHub" : "GitHub"}
        </a>
        {repo && (
          <a onClick={onHome} style={{ cursor: "pointer" }}>
            Change repo
          </a>
        )}
      </nav>

      {user && (
        <div className="hdr-user">
          {user.avatar_url && <img src={user.avatar_url} alt="" />}
          <span className="hdr-user-name">{user.username}</span>
          <button className="ghost-btn" onClick={onSignOut}>
            Sign out
          </button>
        </div>
      )}
    </header>
  );
}

function Footer() {
  return (
    <footer className="ftr">
      <span className="ftr-mono">reflexRAG</span>
      <span className="ftr-dot" />
      <span>The code says what. The history says why.</span>
      <span className="ftr-spacer" />
      <a
        href="https://github.com/anushka7220/reflexRAG"
        target="_blank"
        rel="noreferrer"
      >
        Source
      </a>
      <span className="ftr-dot" />
      <span className="ftr-mono">v0.1</span>
    </footer>
  );
}

export default function App() {
  const [boot, setBoot] = useState<Boot>("checking");
  const [user, setUser] = useState<User | null>(null);
  const [repo, setRepo] = useState<Repo | null>(null);

  // Which past conversation is open. null means "new conversation".
  const [sessionId, setSessionId] = useState<string | null>(null);
  // Bumped whenever a new session is created, to refresh the sidebar list.
  const [sidebarKey, setSidebarKey] = useState(0);

  useEffect(() => {
    consumeTokenFromHash();
    if (!auth.get()) {
      setBoot("signed-out");
      return;
    }
    api
      .me()
      .then((u) => {
        setUser(u);
        setBoot("signed-in");
      })
      .catch(() => {
        auth.clear();
        setBoot("signed-out");
      });
  }, []);

  function goHome() {
    setRepo(null);
    setSessionId(null);
  }

  function signOut() {
    auth.clear();
    setUser(null);
    setRepo(null);
    setSessionId(null);
    setBoot("signed-out");
  }

  // Reset conversation state whenever the repo changes.
  useEffect(() => {
    setSessionId(null);
  }, [repo?.id]);

  if (boot === "checking") {
    return (
      <div className="app">
        <div className="stage">
          <span className="spin" />
        </div>
      </div>
    );
  }

  if (boot === "signed-out") {
    return (
      <div className="shell">
        <Header user={null} repo={null} onHome={goHome} onSignOut={signOut} />
        <div className="stage">
          <div className="stage-inner">
            <span className="eyebrow">reflexRAG</span>
            <h1 className="headline">
              The code says what.
              <br />
              The history says <em>why</em>.
            </h1>
            <p className="sub">
              Chat with any public GitHub repository, including the issues, pull
              requests, and commits that shaped it.
            </p>
            <a className="primary-btn" href={auth.loginUrl()}>
              Continue with GitHub
            </a>
            <p className="hint">Read-only access to public repositories.</p>
          </div>
        </div>
        <Footer />
      </div>
    );
  }

  const ready = repo && repo.status === "done";

  return (
    <div className="shell">
      <Header user={user} repo={repo} onHome={goHome} onSignOut={signOut} />

      {!repo && (
        <div className="body no-aside">
          <div className="main">
            <RepoSearch onPick={setRepo} />
          </div>
        </div>
      )}

      {repo && !ready && (
        <div className="body no-aside">
          <div className="main">
            <IngestProgress repo={repo} onReady={setRepo} onCancel={goHome} />
          </div>
        </div>
      )}

      {repo && ready && (
        <div className="body">
          <Sidebar
            repoId={repo.id}
            activeSessionId={sessionId}
            onSelect={setSessionId}
            onNew={() => setSessionId(null)}
            refreshKey={sidebarKey}
          />
          <div className="main">
            <Chat
              key={sessionId ?? "new"}
              repo={repo}
              sessionId={sessionId}
              onSessionCreated={(id) => {
                setSessionId(id);
                setSidebarKey((k) => k + 1);
              }}
            />
          </div>
        </div>
      )}

      <Footer />
    </div>
  );
}
