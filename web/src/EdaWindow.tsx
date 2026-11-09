import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  bytes,
  date,
  duration,
  type Run,
  type Schema,
  type ProgressState,
} from "./api";
import DeleteAction from "./DeleteAction";
import Progress from "./Progress";
import { Button, Notice, Status } from "./ui";
import ConfigForm from "./ConfigForm";
import { includeRoleColumns, newlyCompletedReport } from "./edaControls";

type Column = {
  name: string;
  dtype: string;
  numeric: boolean;
  geometry: boolean;
};
type EdaArtifact = { artifact_id: string; name: string; size_bytes: number };
type EdaRun = {
  progress?: ProgressState;
  eda_id: string;
  status: string;
  selected_columns: string[];
  config: Record<string, unknown>;
  artifacts: EdaArtifact[];
  error?: string;
  created_at: string;
  started_at?: string;
  finished_at?: string;
};
type Options = {
  columns: Column[];
  default_columns: string[];
  schema: Schema & { managed: Record<string, unknown> };
};

export default function EdaWindow({
  run,
  onClose,
}: {
  run: Run;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const requestId = useRef(crypto.randomUUID());
  const [options, setOptions] = useState<Options>();
  const [config, setConfig] = useState<Record<string, unknown>>({});
  const [selected, setSelected] = useState<string[]>([]);
  const [history, setHistory] = useState<EdaRun[]>([]);
  const previousHistory = useRef<EdaRun[]>([]);
  const [reportId, setReportId] = useState<string>();
  const [tab, setTab] = useState("Setup");
  const [search, setSearch] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [valid, setValid] = useState(true);
  const [configKey, setConfigKey] = useState(0);
  useEffect(() => {
    const completed = newlyCompletedReport(previousHistory.current, history);
    previousHistory.current = history;
    if (completed) {
      setReportId(completed);
      setTab("Report");
    }
  }, [history]);
  useEffect(() => {
    const element = dialog.current;
    element?.showModal();
    return () => element?.close();
  }, []);
  useEffect(() => {
    let live = true;
    Promise.allSettled([
      api<Options>(`/runs/${run.run_id}/eda/options`),
      api<EdaRun[]>(`/runs/${run.run_id}/eda`),
    ])
      .then(([settings, analyses]) => {
        if (!live) return;
        if (settings.status === "fulfilled") {
          setOptions(settings.value);
          setConfig(settings.value.schema.defaults);
          setSelected(settings.value.default_columns);
        } else {
          setError(settings.reason.message);
          setTab("History");
        }
        if (analyses.status === "fulfilled") setHistory(analyses.value);
        else setError(analyses.reason.message);
      })
      .catch((e) => {
        if (live) setError(e.message);
      });
    const timer = setInterval(() => {
      api<EdaRun[]>(`/runs/${run.run_id}/eda`)
        .then((rows) => {
          if (live) setHistory(rows);
        })
        .catch((e) => {
          if (live) setError(e.message);
        });
    }, 5000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [run.run_id]);
  const roles = [config.target_column, config.id_column].filter(
    (value): value is string => typeof value === "string",
  );
  const selectedSet = useMemo(() => new Set(selected), [selected]);
  const visible =
    options?.columns.filter((column) =>
      column.name.toLowerCase().includes(search.toLowerCase()),
    ) ?? [];
  function changeConfig(value: Record<string, unknown>) {
    requestId.current = crypto.randomUUID();
    setConfig(value);
    setSelected((previous) =>
      includeRoleColumns(
        previous,
        value,
        options?.columns.filter((c) => !c.geometry).map((c) => c.name) ?? [],
      ),
    );
  }
  function chooseColumns(columns: string[]) {
    requestId.current = crypto.randomUUID();
    const next = [...new Set([...columns, ...roles])];
    setSelected(next);
    setConfig((previous) => ({
      ...previous,
      feature_selection_exclude_columns: Array.isArray(
        previous.feature_selection_exclude_columns,
      )
        ? previous.feature_selection_exclude_columns.filter((c) =>
            next.includes(c),
          )
        : [],
    }));
  }
  const formSchema = options && {
    ...options.schema,
    fields: options.schema.fields.map((field) => {
      if (["target_column", "id_column"].includes(field.name))
        return {
          ...field,
          choices: [
            null,
            ...options.columns
              .filter((c) => selectedSet.has(c.name) && !c.geometry)
              .map((c) => c.name),
          ],
        };
      if (field.name === "feature_selection_exclude_columns")
        return { ...field, choices: selected };
      return field;
    }),
  };
  async function submit() {
    setBusy(true);
    setError("");
    try {
      const row = await api<EdaRun>(`/runs/${run.run_id}/eda`, "POST", {
        request_id: requestId.current,
        columns: selected,
        config,
      });
      setHistory((rows) => [
        row,
        ...rows.filter((r) => r.eda_id !== row.eda_id),
      ]);
      setTab("History");
      requestId.current = crypto.randomUUID();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <dialog
      ref={dialog}
      className="eda-dialog"
      aria-labelledby="eda-title"
      onCancel={onClose}
    >
      <header className="eda-dialog-header">
        <div>
          <span className="eyebrow">Explore your results</span>
          <h2 id="eda-title">Exploratory data analysis</h2>
          <p>{run.name}</p>
        </div>
        <Button onClick={onClose} aria-label="Close EDA window">
          Close
        </Button>
      </header>
      <div className="eda-dialog-body">
        <Notice message={error} />
        <div className="tabs">
          {["Setup", "History", ...(reportId ? ["Report"] : [])].map((name) => (
            <button
              key={name}
              className={tab === name ? "active" : ""}
              onClick={() => setTab(name)}
            >
              {name}
              {name === "History" ? ` (${history.length})` : ""}
            </button>
          ))}
        </div>
        {tab === "Report" && reportId && (
          <section className="eda-report">
            <div className="actions">
              <a
                className="btn"
                href={`/api/eda/${reportId}/report`}
                target="_blank"
                rel="noopener noreferrer"
              >
                Open report in new tab
              </a>
              <Button onClick={() => setTab("History")}>Back to history</Button>
            </div>
            <iframe
              title="EDA report"
              src={`/api/eda/${reportId}/report`}
              sandbox="allow-scripts"
            />
          </section>
        )}
        {tab === "Setup" &&
          (!options || !formSchema ? (
            <p>Loading result columns and EDA settings…</p>
          ) : (
            <>
              <section className="eda-columns">
                <div className="heading">
                  <div>
                    <h3>Columns to analyse</h3>
                    <p>
                      {selected.length} of {options.columns.length} selected
                    </p>
                  </div>
                  <div className="actions">
                    <Button
                      onClick={() =>
                        chooseColumns(options.columns.map((c) => c.name))
                      }
                    >
                      Select all
                    </Button>
                    <Button
                      onClick={() =>
                        chooseColumns(
                          options.columns
                            .filter((c) => c.numeric)
                            .map((c) => c.name),
                        )
                      }
                    >
                      Numeric only
                    </Button>
                    <Button onClick={() => chooseColumns([])}>Clear</Button>
                  </div>
                </div>
                <input
                  aria-label="Search result columns"
                  placeholder="Search columns…"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
                <div className="eda-column-list">
                  {visible.map((column) => (
                    <label key={column.name}>
                      <input
                        type="checkbox"
                        checked={selectedSet.has(column.name)}
                        disabled={roles.includes(column.name)}
                        onChange={(e) =>
                          chooseColumns(
                            e.target.checked
                              ? [...selected, column.name]
                              : selected.filter((name) => name !== column.name),
                          )
                        }
                      />
                      <span>
                        {column.name}
                        <small>
                          {column.geometry ? "Geometry" : column.dtype}
                          {roles.includes(column.name)
                            ? " · selected as target or ID"
                            : ""}
                        </small>
                      </span>
                    </label>
                  ))}
                </div>
                <p className="footnote">
                  Only selected columns enter EDA. Keep geometry selected for
                  spatial summaries. Choose target and ID roles below; numeric
                  class labels need the classification task.
                </p>
              </section>
              <ConfigForm
                key={configKey}
                schema={formSchema}
                value={config}
                onChange={changeConfig}
                onValidityChange={setValid}
              />
              <details className="sectiongap">
                <summary>Managed output settings</summary>
                <p>
                  Each analysis gets a separate output directory. Previous
                  results are preserved.
                </p>
                <pre className="json">
                  {JSON.stringify(options.schema.managed, null, 2)}
                </pre>
              </details>
              <div className="actions sectiongap">
                <Button
                  primary
                  disabled={
                    busy ||
                    !valid ||
                    !selected.length ||
                    run.status !== "complete"
                  }
                  onClick={() => void submit()}
                >
                  {busy ? "Submitting…" : "Run EDA"}
                </Button>
                <span className="muted">
                  Processing continues after you close this window.
                </span>
              </div>
            </>
          ))}
        {tab === "History" && (
          <div className="eda-history">
            {!history.length && (
              <p>
                No EDA analyses yet. Select columns and settings in Setup to
                begin.
              </p>
            )}
            {history.map((item) => (
              <section className="panel pad" key={item.eda_id}>
                <div className="heading">
                  <div>
                    <h3>{String(item.config.report_title)}</h3>
                    <small>
                      {date(item.created_at)} · {item.selected_columns.length}{" "}
                      columns
                    </small>
                  </div>
                  <Status value={item.status} />
                </div>
                <Notice message={item.error ?? ""} />
                {item.progress?.stage && <Progress value={item.progress} />}
                {item.started_at && (
                  <p className="footnote">
                    Duration{" "}
                    {duration(
                      ((item.finished_at
                        ? Date.parse(item.finished_at)
                        : Date.now()) -
                        Date.parse(item.started_at)) /
                        1000,
                    )}
                  </p>
                )}
                <div className="actions">
                  <DeleteAction
                    kind="eda"
                    id={item.eda_id}
                    name={String(item.config.report_title ?? "EDA analysis")}
                    status={item.status}
                    onDeleted={() => {
                      setHistory((rows) =>
                        rows.filter((row) => row.eda_id !== item.eda_id),
                      );
                      if (reportId === item.eda_id) {
                        setReportId(undefined);
                        setTab("History");
                      }
                    }}
                  />
                  {item.status === "complete" &&
                    item.artifacts.some((f) => f.name === "report.html") && (
                      <a
                        className="btn primary"
                        href={`/api/eda/${item.eda_id}/report`}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        Open report
                      </a>
                    )}
                  {["failed", "interrupted"].includes(item.status) &&
                    item.started_at && (
                      <a className="btn" href={`/api/eda/${item.eda_id}/log`}>
                        Download EDA log
                      </a>
                    )}
                  {item.status === "complete" &&
                    item.artifacts
                      .filter((f) => f.name === "eda-results.zip")
                      .map((file) => (
                        <a
                          key={file.artifact_id}
                          className="btn primary"
                          href={`/api/eda/${item.eda_id}/files/${file.artifact_id}`}
                        >
                          Download EDA report & files (.zip)
                        </a>
                      ))}
                  <Button
                    onClick={() => {
                      changeConfig(item.config);
                      setSelected(item.selected_columns);
                      setConfigKey((key) => key + 1);
                      setTab("Setup");
                    }}
                  >
                    Use these settings
                  </Button>
                </div>
                <details className="sectiongap">
                  <summary>Saved columns and configuration</summary>
                  <pre className="json">
                    {JSON.stringify(
                      { columns: item.selected_columns, config: item.config },
                      null,
                      2,
                    )}
                  </pre>
                </details>
                {item.status === "complete" && (
                  <details className="sectiongap">
                    <summary>
                      Individual files ({item.artifacts.length})
                    </summary>
                    <p className="footnote">
                      Extract the ZIP and open report.html to view the report
                      with its plots.
                    </p>
                    {item.artifacts.map((file) => (
                      <div className="download" key={file.artifact_id}>
                        <span className="grow">
                          {file.name}
                          <small>{bytes(file.size_bytes)}</small>
                        </span>
                        <a
                          className="btn"
                          href={`/api/eda/${item.eda_id}/files/${file.artifact_id}`}
                        >
                          Download
                        </a>
                      </div>
                    ))}
                  </details>
                )}
              </section>
            ))}
          </div>
        )}
      </div>
    </dialog>
  );
}
