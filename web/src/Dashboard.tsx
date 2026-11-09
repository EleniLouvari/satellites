import { useEffect, useState } from "react";
import {
  api,
  date,
  duration,
  type Account,
  type Run,
  type Upload,
} from "./api";
import { Button, Empty, Heading, Notice, Status } from "./ui";
import ParcelMap from "./ParcelMap";
import DeleteAction from "./DeleteAction";
import Progress from "./Progress";
export default function Dashboard({
  history = false,
  onNew,
  onOpen,
  onEda,
}: {
  history?: boolean;
  onNew: () => void;
  onOpen: (r: Run) => void;
  onEda: (r: Run) => void;
}) {
  const [runs, setRuns] = useState<Run[]>([]),
    [accounts, setAccounts] = useState<Account[]>([]),
    [uploads, setUploads] = useState<Upload[]>([]),
    [loading, setLoading] = useState(true),
    [error, setError] = useState(""),
    [search, setSearch] = useState(""),
    [status, setStatus] = useState(""),
    [since, setSince] = useState("");
  useEffect(() => {
    let live = true;
    async function refresh() {
      try {
        const [r, a, u] = await Promise.all([
          api<Run[]>("/runs"),
          api<Account[]>("/accounts"),
          api<Upload[]>("/uploads"),
        ]);
        if (live) {
          setRuns(r);
          setAccounts(a);
          setUploads(u);
          setError("");
        }
      } catch (e) {
        if (live)
          setError(
            `${(e as Error).message} Displayed data may be stale; retrying automatically.`,
          );
      } finally {
        if (live) setLoading(false);
      }
    }
    void refresh();
    const timer = setInterval(refresh, 5000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, []);
  const filtered = runs.filter(
    (r) =>
      r.name.toLowerCase().includes(search.toLowerCase()) &&
      (!status || r.status === status) &&
      (!since || r.created_at.slice(0, 10) >= since),
  );
  const preview = uploads
    .find((u) => u.metadata?.layers?.some((l) => l.preview))
    ?.metadata.layers?.find((l) => l.preview)?.preview;
  return (
    <>
      <Heading
        title={history ? "My analyses" : "A clearer view of your land."}
        subtitle={
          history
            ? "Your inputs, progress, and results — all in one place."
            : "Welcome to your satellite analysis workspace."
        }
      >
        <Button primary onClick={onNew}>
          + New analysis
        </Button>
      </Heading>
      <Notice message={error} />
      {!history && (
        <>
          <section className="hero">
            <div className="hero-copy">
              <span className="eyebrow">From boundary to insight</span>
              <h2>
                Small parcels.
                <br />A bigger perspective.
              </h2>
              <p>
                Turn satellite observations into meaningful parcel statistics,
                in one considered workspace.
              </p>
              <Button primary onClick={onNew}>
                Start an analysis ?
              </Button>
            </div>
            <ParcelMap geojson={preview} />
          </section>
          <div className="metrics">
            {[
              [
                "Active analyses",
                runs.filter((r) =>
                  ["running", "queued", "validating"].includes(r.status),
                ).length,
                "Processing continues when you leave",
              ],
              [
                "Completed analyses",
                runs.filter((r) => r.status === "complete").length,
                "Verified outputs, ready to explore",
              ],
              [
                "Copernicus accounts",
                accounts.length,
                "Your connections to satellite imagery",
              ],
            ].map(([label, count, help]) => (
              <article className="metric" key={label}>
                <small>{label}</small>
                <strong>{loading ? "—" : count}</strong>
                <small>{help}</small>
              </article>
            ))}
          </div>
        </>
      )}
      {history && (
        <div className="filters">
          <input
            aria-label="Search analyses"
            placeholder="Search analyses…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <select
            aria-label="Filter status"
            value={status}
            onChange={(e) => setStatus(e.target.value)}
          >
            <option value="">All statuses</option>
            {[
              "draft",
              "validating",
              "ready",
              "queued",
              "running",
              "complete",
              "failed",
              "interrupted",
              "cancelled",
              "deleting",
            ].map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
          <label className="inline-label">
            Created since
            <input
              type="date"
              value={since}
              onChange={(e) => setSince(e.target.value)}
            />
          </label>
        </div>
      )}
      <section className="panel">
        <header className="panelhead">
          <div>
            <h2>{history ? "Your analyses" : "Recent analyses"}</h2>
            <p>
              {loading
                ? "Loading your workspace…"
                : "Pick up where you left off."}
            </p>
          </div>
          <span className="eyebrow">{runs.length} runs</span>
        </header>
        {!loading && !filtered.length ? (
          <Empty>
            <h2>
              {runs.length
                ? "No matching analyses."
                : "Your next discovery starts here."}
            </h2>
            <p>
              {runs.length
                ? "Try a different search or filter."
                : "Upload parcel boundaries and set up your first satellite analysis."}
            </p>
            {!runs.length && (
              <Button primary onClick={onNew}>
                Create your first analysis ?
              </Button>
            )}
          </Empty>
        ) : (
          <div className="tablewrap">
            <table>
              <thead>
                <tr>
                  <th>Analysis</th>
                  <th>Status</th>
                  <th>Parcels</th>
                  <th>Created</th>
                  <th>Duration</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {(history ? filtered : filtered.slice(0, 6)).map((r) => (
                  <tr key={r.run_id}>
                    <td>
                      <strong>{r.name}</strong>
                      <small>
                        {r.parcel_id_field} · Revision {r.revision}
                      </small>
                    </td>
                    <td>
                      <Status value={r.status} />
                      {r.status !== "complete" && (
                        <Progress value={r.progress} compact />
                      )}
                    </td>
                    <td>
                      {typeof r.summary?.parcel_count === "number"
                        ? r.summary.parcel_count.toLocaleString()
                        : "—"}
                    </td>
                    <td>{date(r.created_at)}</td>
                    <td>{duration(r.duration_seconds)}</td>
                    <td>
                      <Button onClick={() => onOpen(r)}>Open</Button>
                      <span title="EDA is available after completion">
                        <Button
                          disabled={r.status !== "complete"}
                          onClick={() => onEda(r)}
                        >
                          EDA
                        </Button>
                      </span>
                      <DeleteAction
                        kind="run"
                        id={r.run_id}
                        name={r.name}
                        status={r.status}
                        onDeleted={() =>
                          setRuns((rows) =>
                            rows.filter((row) => row.run_id !== r.run_id),
                          )
                        }
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}
