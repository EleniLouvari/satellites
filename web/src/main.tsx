import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { api, setCsrf, type Run, type User } from "./api";
import { Brand, Notice } from "./ui";
import Auth from "./Auth";
import Accounts from "./Accounts";
import Dashboard from "./Dashboard";
import Wizard from "./Wizard";
import RunDetail from "./RunDetail";
import EdaWindow from "./EdaWindow";
import "./style.css";
function App() {
  const [user, setUser] = useState<User>(),
    [loading, setLoading] = useState(true),
    [page, setPage] = useState("Overview"),
    [run, setRun] = useState<Run>(),
    [draft, setDraft] = useState<Run>(),
    [error, setError] = useState(""),
    [expired, setExpired] = useState(false);
  const [edaRun, setEdaRun] = useState<Run>();
  useEffect(() => {
    api<User>("/auth/me")
      .then((u) => {
        setUser(u);
        setCsrf(u.csrf_token);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
    const expire = () => {
      setUser(undefined);
      setCsrf("");
      setExpired(true);
      setEdaRun(undefined);
    };
    window.addEventListener("session-expired", expire);
    return () => window.removeEventListener("session-expired", expire);
  }, []);
  function navigate(p: string) {
    setPage(p);
    setError("");
    if (p === "New analysis") setDraft(undefined);
    window.scrollTo(0, 0);
  }
  function open(r: Run) {
    setRun(r);
    setPage("Run detail");
  }
  function edit(r: Run) {
    setDraft(r);
    setPage("New analysis");
  }
  if (loading)
    return (
      <div className="loading">
        <Brand />
        <p>Opening your workspace…</p>
      </div>
    );
  if (!user)
    return (
      <Auth
        message={
          expired
            ? "Your session expired. Sign in again; server-saved drafts are preserved."
            : ""
        }
        onLogin={(u) => {
          setUser(u);
          setExpired(false);
        }}
      />
    );
  return (
    <div className="shell">
      <aside className="sidebar">
        <Brand />
        <div className="navlabel">WORKSPACE</div>
        <nav aria-label="Main navigation">
          {[
            ["Overview", "◫"],
            ["New analysis", "+"],
            ["My runs", "▤"],
            ["Copernicus accounts", "◇"],
          ].map(([p, icon]) => (
            <button
              key={p}
              className={page === p ? "active" : ""}
              aria-current={page === p ? "page" : undefined}
              onClick={() => navigate(p)}
            >
              <b aria-hidden="true">{icon}</b>
              {p}
            </button>
          ))}
        </nav>
        <div className="sidebottom">
          <div className="help">
            <h3>Built around your parcels.</h3>
            <p>
              Your inputs stay reproducible. Your work keeps running when you
              leave.
            </p>
            <span className="eyebrow">Copernicus · openEO</span>
          </div>
          <div className="user">
            <span className="avatar">
              {user.username.slice(0, 2).toUpperCase()}
            </span>
            <div>
              <strong>{user.username}</strong>
              <small>Personal workspace</small>
            </div>
          </div>
        </div>
        <button
          className="signout"
          onClick={async () => {
            try {
              await api("/auth/logout", "POST");
              setUser(undefined);
              setCsrf("");
              setPage("Overview");
              setDraft(undefined);
              setEdaRun(undefined);
            } catch (e) {
              setError((e as Error).message);
            }
          }}
        >
          Sign out →
        </button>
      </aside>
      <main className="workspace">
        <header className="topbar">
          <span>
            Workspace <i>/</i> <strong>{page}</strong>
          </span>
          <span className="private-label">PERSONAL WORKSPACE</span>
        </header>
        <div className="content">
          <Notice message={error} />
          {page === "Overview" && (
            <Dashboard
              onNew={() => navigate("New analysis")}
              onOpen={open}
              onEda={setEdaRun}
            />
          )}
          {page === "My runs" && (
            <Dashboard
              history
              onNew={() => navigate("New analysis")}
              onOpen={open}
              onEda={setEdaRun}
            />
          )}
          {page === "Copernicus accounts" && <Accounts />}
          {page === "New analysis" && (
            <Wizard
              key={draft?.run_id ?? "new"}
              initial={draft}
              onOpen={open}
            />
          )}
          {page === "Run detail" && run && (
            <RunDetail
              key={run.run_id}
              initial={run}
              onDeleted={() => {
                setRun(undefined);
                setPage("My analyses");
              }}
              onOpen={open}
              onEdit={edit}
              onEda={setEdaRun}
            />
          )}
        </div>
        {edaRun && (
          <EdaWindow
            key={edaRun.run_id}
            run={edaRun}
            onClose={() => setEdaRun(undefined)}
          />
        )}
        <footer className="workspace-footer">
          PARCEL STUDIO <span>Satellite observations. Grounded insight.</span>
        </footer>
      </main>
    </div>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
