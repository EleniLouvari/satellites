import { useEffect, useState } from "react";
import {
  api,
  bytes,
  date,
  duration,
  downloadJson,
  type Artifact,
  type Event,
  type Run,
} from "./api";
import { Button, Empty, Heading, Notice, Status } from "./ui";
import DeleteAction from "./DeleteAction";
import RenameAction from "./RenameAction";
import Progress from "./Progress";
type Bundle = { status: string; artifact_id?: string; error?: string };
export default function RunDetail({
  initial,
  onEdit,
  onOpen,
  onEda,
  onDeleted,
}: {
  initial: Run;
  onEdit: (run: Run) => void;
  onOpen: (run: Run) => void;
  onEda: (run: Run) => void;
  onDeleted: () => void;
}) {
  const [run, setRun] = useState(initial),
    [events, setEvents] = useState<Event[]>([]),
    [files, setFiles] = useState<Artifact[]>([]),
    [tab, setTab] = useState("Overview"),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [pause, setPause] = useState(false),
    [level, setLevel] = useState(""),
    [stage, setStage] = useState(""),
    [updated, setUpdated] = useState<string>(),
    [notice, setNotice] = useState("");
  const [bundle, setBundle] = useState<Bundle>({ status: "unrequested" });
  const [downloadWhenReady, setDownloadWhenReady] = useState(false);
  useEffect(() => {
    let live = true;
    let after = 0;
    async function poll() {
      try {
        const [r, f, e, b] = await Promise.all([
          api<Run>(`/runs/${initial.run_id}`),
          api<Artifact[]>(`/runs/${initial.run_id}/artifacts`),
          pause
            ? Promise.resolve([])
            : api<Event[]>(`/runs/${initial.run_id}/events?after=${after}`),
          api<Bundle>(`/runs/${initial.run_id}/bundle`),
        ]);
        if (live) {
          setRun(r);
          setFiles(f);
          setBundle(b);
          setEvents((prev) => {
            const map = new Map(prev.map((v) => [v.event_id, v]));
            e.forEach((v) => map.set(v.event_id, v));
            return [...map.values()]
              .sort((a, b) => a.event_id - b.event_id)
              .slice(-2000);
          });
          if (e.length) after = Math.max(...e.map((v) => v.event_id));
          setUpdated(new Date().toISOString());
          setError("");
        }
      } catch (e) {
        if (live)
          setError(
            `${(e as Error).message} Displayed status may be stale; reconnecting automatically.`,
          );
      }
    }
    void poll();
    const timer = setInterval(poll, 5000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [initial.run_id, pause]);
  useEffect(() => {
    if (!downloadWhenReady) return;
    if (bundle.status === "ready" && bundle.artifact_id) {
      setDownloadWhenReady(false);
      setNotice(
        "Your ZIP is ready. Download it again using the same button if needed.",
      );
      window.location.assign(`/api/artifacts/${bundle.artifact_id}/download`);
    } else if (bundle.status === "failed") {
      setDownloadWhenReady(false);
      setError(bundle.error ?? "Error: ZIP preparation failed. Try again.");
      setNotice("");
    }
  }, [bundle, downloadWhenReady]);
  async function downloadAll() {
    setBusy(true);
    setError("");
    try {
      const result = await api<Bundle>(`/runs/${run.run_id}/bundle`, "POST");
      setBundle(result);
      setDownloadWhenReady(true);
      setNotice(
        "Preparing your ZIP with results, reports, configuration and logs. The download starts when ready.",
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const preparingBundle =
    downloadWhenReady || ["pending", "running"].includes(bundle.status);
  const zipControl =
    run.status === "complete" &&
    (bundle.status === "ready" && bundle.artifact_id ? (
      <a
        className="btn primary"
        href={`/api/artifacts/${bundle.artifact_id}/download`}
      >
        Download all files (.zip)
      </a>
    ) : (
      <Button
        primary
        disabled={busy || preparingBundle}
        onClick={() => void downloadAll()}
      >
        {preparingBundle ? "Preparing ZIP…" : "Download all files (.zip)"}
      </Button>
    ));
  async function action(name: string) {
    setBusy(true);
    setError("");
    try {
      const r = await api<Run>(`/runs/${run.run_id}/${name}`, "POST");
      if (name === "duplicate") onEdit(r);
      else setRun(r);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const filtered = events.filter(
    (e) => (!level || e.level === level) && (!stage || e.stage === stage),
  );
  const latest = events.at(-1);
  const editable =
    ["draft", "ready"].includes(run.status) ||
    (run.status === "failed" && !run.validated_revision);
  return (
    <>
      <Heading title={run.name} subtitle={`Created ${date(run.created_at)}`}>
        <div className="actions">
          <RenameAction run={run} onRenamed={setRun} />
          <Button
            disabled={run.status !== "complete"}
            onClick={() => onEda(run)}
          >
            EDA
          </Button>
          {zipControl}
          <DeleteAction
            kind="run"
            id={run.run_id}
            name={run.name}
            status={run.status}
            onDeleted={onDeleted}
          />
          <Button
            disabled={busy || run.status === "deleting"}
            onClick={() => void action("duplicate")}
          >
            Duplicate settings
          </Button>
          {editable && (
            <Button primary onClick={() => onEdit(run)}>
              Edit draft
            </Button>
          )}
          {["failed", "interrupted"].includes(run.status) &&
            run.validated_revision !== null && (
              <Button
                disabled={busy}
                primary
                onClick={() => void action("resume")}
              >
                Resume analysis
              </Button>
            )}
          {run.status === "queued" && (
            <Button disabled={busy} onClick={() => void action("cancel")}>
              Remove from queue
            </Button>
          )}
        </div>
      </Heading>
      <Notice message={error} />
      <Notice message={run.error ?? ""} />
      {notice && (
        <div role="status" className="note">
          {notice}
        </div>
      )}
      <div className="run-status">
        <Status value={run.status} />
        <small>Last refreshed {date(updated)}</small>
        <small>Duration {duration(run.duration_seconds)}</small>
        <Button onClick={() => onOpen(run)}>Refresh view</Button>
      </div>
      <div className="tabs">
        {["Overview", "Configuration", "Logs", "Files"].map((t) => (
          <button
            key={t}
            className={tab === t ? "active" : ""}
            onClick={() => setTab(t)}
          >
            {t}
            {t === "Files" ? ` · ${files.length}` : ""}
          </button>
        ))}
      </div>
      {tab === "Overview" && (
        <div className="columns">
          <section className="panel pad">
            <h2>
              {run.status === "complete"
                ? "Your analysis is complete."
                : run.status === "running"
                  ? "Your analysis is underway."
                  : "Analysis status"}
            </h2>
            <p className="muted sectiongap">
              {latest
                ? `${latest.stage}: ${latest.message}`
                : "Waiting for the first processing event."}
            </p>
            <div className="timeline">
              {[
                "Acquisition",
                "Download",
                "Local processing",
                "Publication",
              ].map((s) => (
                <div key={s}>
                  <strong>{s}</strong>
                  <small>See recorded events below</small>
                </div>
              ))}
            </div>
            <Progress value={run.progress} />
            {["queued", "running", "validating"].includes(run.status) && (
              <div
                className="indeterminate"
                aria-label="Processing; percentage unavailable"
              />
            )}
            <p className="footnote">
              Stages may overlap between batches. Remote completion is separate
              from verified local results.
            </p>
            {Object.keys(run.summary ?? {}).length > 0 && (
              <>
                <hr />
                <h3>Validated summary</h3>
                <pre className="json">
                  {JSON.stringify(run.summary, null, 2)}
                </pre>
              </>
            )}
            <hr />
            <h3>Latest events</h3>
            <div className="log sectiongap">
              {events.length
                ? events.slice(-8).map((e) => (
                    <div key={e.event_id}>
                      {date(e.created_at)} · {e.level} · {e.stage}
                      <br />
                      {e.message}
                    </div>
                  ))
                : "No events recorded yet."}
            </div>
          </section>
          <aside className="panel summary">
            <h3>Run details</h3>
            <div className="kv">
              <span>Started</span>
              <strong>{date(run.started_at ?? undefined)}</strong>
            </div>
            <div className="kv">
              <span>Finished</span>
              <strong>{date(run.finished_at ?? undefined)}</strong>
            </div>
            <div className="kv">
              <span>Duration</span>
              <strong>{duration(run.duration_seconds)}</strong>
            </div>
            <p className="footnote">
              Elapsed since processing first started; resumed runs include time
              between attempts.
            </p>
            <div className="kv">
              <span>Updated</span>
              <strong>{date(run.updated_at)}</strong>
            </div>
            <div className="kv">
              <span>Accounts</span>
              <strong>{run.account_ids.length} selected</strong>
            </div>
            <div className="kv">
              <span>Revision</span>
              <strong>{run.revision}</strong>
            </div>
            <div className="kv">
              <span>Parcel identifier</span>
              <strong>{run.parcel_id_field}</strong>
            </div>
            <div className="note">
              It is safe to leave this page. Return to My runs to check
              progress.
            </div>
          </aside>
        </div>
      )}
      {tab === "Configuration" && (
        <section className="panel pad">
          <div className="heading">
            <h2>Saved configuration</h2>
            <Button
              onClick={() => downloadJson(run.config, "configuration.json")}
            >
              Export configuration
            </Button>
          </div>
          <pre className="json">
            {JSON.stringify(
              Object.keys(run.config ?? {}).length
                ? run.config
                : run.processing,
              null,
              2,
            )}
          </pre>
        </section>
      )}
      {tab === "Logs" && (
        <>
          <div className="filters">
            <select
              aria-label="Log level"
              value={level}
              onChange={(e) => setLevel(e.target.value)}
            >
              <option value="">All levels</option>
              {[...new Set(events.map((e) => e.level))].map((l) => (
                <option key={l}>{l}</option>
              ))}
            </select>
            <select
              aria-label="Log source"
              value={stage}
              onChange={(e) => setStage(e.target.value)}
            >
              <option value="">All sources</option>
              {[...new Set(events.map((e) => e.stage))].map((s) => (
                <option key={s}>{s}</option>
              ))}
            </select>
            <Button onClick={() => setPause(!pause)}>
              {pause ? "Resume tail" : "Pause tail"}
            </Button>
            <Button onClick={() => downloadJson(filtered, "safe-events.json")}>
              Download displayed events
            </Button>
          </div>
          <p className="footnote">
            Showing the latest {events.length} loaded safe events.{" "}
            {pause ? "Tail paused." : "Polling every five seconds."}
          </p>
          <div className="log sectiongap">
            {filtered.length
              ? filtered.map((e) => (
                  <div key={e.event_id}>
                    {date(e.created_at)} {e.level} [{e.stage}] {e.message}
                  </div>
                ))
              : "No matching events."}
          </div>
        </>
      )}
      {tab === "Files" && (
        <>
          <div className="heading">
            <div>
              <h2>Published files</h2>
              <p>Only verified, registered deliverables are available here.</p>
            </div>
            {zipControl}
          </div>
          <section className="panel">
            {files.length ? (
              files.map((f) => (
                <div className="download" key={f.artifact_id}>
                  <span className="fileicon">↓</span>
                  <div>
                    <strong>{f.name}</strong>
                    <small>
                      {f.kind} · {bytes(f.size_bytes)} · {date(f.created_at)}
                    </small>
                  </div>
                  <a
                    className="btn"
                    href={`/api/artifacts/${f.artifact_id}/download`}
                  >
                    Download
                  </a>
                </div>
              ))
            ) : (
              <Empty>
                <h2>No published files yet.</h2>
                <p>Verified outputs appear here after publication.</p>
              </Empty>
            )}
          </section>
        </>
      )}
    </>
  );
}
