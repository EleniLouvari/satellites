import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  bytes,
  date,
  duration,
  type Run,
  type Schema,
  type ProgressState,
} from "../../api";
import DeleteAction from "../../components/DeleteAction";
import RenameAction from "../../components/RenameAction";
import Progress from "../../components/Progress";
import { Button, Heading, Notice, Status } from "../../components/ui";
import ConfigForm from "../../components/ConfigForm";
import Chooser, { BackLink } from "../../components/Chooser";
import ColumnFilter, { useColumnFilter } from "../../components/ColumnFilter";
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
  name: string;
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

export default function EdaPage({
  run,
  initialOpenId,
  onBack,
  onMl,
  onFocus,
}: {
  run: Run;
  /** Open this saved analysis directly instead of the list of analyses. */
  initialOpenId?: string;
  onBack: () => void;
  onMl: (source: { edaId: string; title: string; runName: string }) => void;
  /** Reports the analysis on screen (undefined on the list or set-up views) for the page path. */
  onFocus?: (item?: { id: string; name: string }) => void;
}) {
  const requestId = useRef(crypto.randomUUID());
  const [options, setOptions] = useState<Options>();
  const [config, setConfig] = useState<Record<string, unknown>>({});
  const [selected, setSelected] = useState<string[]>([]);
  const [history, setHistory] = useState<EdaRun[]>([]);
  const [loaded, setLoaded] = useState(false);
  const previousHistory = useRef<EdaRun[]>([]);
  const [reportId, setReportId] = useState<string>();
  const [tab, setTab] = useState(initialOpenId ? "History" : "Start");
  const [openId, setOpenId] = useState<string | undefined>(initialOpenId);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [valid, setValid] = useState(true);
  const [configKey, setConfigKey] = useState(0);
  useEffect(() => {
    const completed = newlyCompletedReport(previousHistory.current, history);
    previousHistory.current = history;
    if (completed) {
      setReportId(completed);
      setOpenId(completed);
      setTab("Report");
    }
  }, [history]);
  const focusId =
    tab === "History" ? openId : tab === "Report" ? reportId : undefined;
  const focusName = history.find((item) => item.eda_id === focusId)?.name;
  useEffect(() => {
    onFocus?.(focusId && focusName ? { id: focusId, name: focusName } : undefined);
  }, [focusId, focusName, onFocus]);
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
        }
        if (analyses.status === "fulfilled") setHistory(analyses.value);
        else setError(analyses.reason.message);
        setLoaded(true);
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
  const columnNames = useMemo(
    () => options?.columns.map((c) => c.name) ?? [],
    [options],
  );
  const filter = useColumnFilter(columnNames);
  const visible =
    options?.columns.filter((column) => filter.matches(column.name)) ?? [];
  const visibleNames = visible.map((c) => c.name);
  const geometryNames =
    options?.columns.filter((c) => c.geometry).map((c) => c.name) ?? [];
  // Target, ID and geometry are always analysed and cannot be deselected.
  const locked = [...roles, ...geometryNames];
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
    const next = [...new Set([...columns, ...locked])];
    // Parcel ID and geometry are re-excluded whenever they are (re)selected.
    const automatic = (
      (options?.schema.defaults.feature_selection_exclude_columns as
        | string[]
        | undefined) ?? []
    ).filter((c) => next.includes(c) && !selected.includes(c));
    setSelected(next);
    setConfig((previous) => {
      const current = Array.isArray(previous.feature_selection_exclude_columns)
        ? (previous.feature_selection_exclude_columns as string[])
        : [];
      return {
        ...previous,
        feature_selection_exclude_columns: [
          ...new Set([...current.filter((c) => next.includes(c)), ...automatic]),
        ],
      };
    });
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
      setOpenId(row.eda_id);
      setTab("History");
      requestId.current = crypto.randomUUID();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <Heading title="Exploratory data analysis" subtitle={run.name}>
        <div className="actions">
          <Button onClick={onBack}>Back</Button>
        </div>
      </Heading>
      <div className="eda-page sectiongap">
        <Notice message={error} />
        {tab === "Start" ? (
          <Chooser
            newLabel="New EDA analysis"
            newHint="Choose the columns and settings, then run a fresh report."
            onNew={() => setTab("Setup")}
            historyLabel="Previous analyses"
            items={history.map((item) => ({
              id: item.eda_id,
              title: item.name,
              meta: `${date(item.created_at)} · ${item.selected_columns.length} columns`,
              status: item.status,
            }))}
            onOpen={(id) => {
              setOpenId(id);
              setTab("History");
            }}
            emptyText="No EDA analyses yet. Start a new one to explore this result."
          />
        ) : (
          <BackLink label="All analyses" onClick={() => setTab("Start")} />
        )}
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
              <Button
                onClick={() => {
                  setOpenId(reportId);
                  setTab("History");
                }}
              >
                Back to analysis
              </Button>
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
                <ColumnFilter
                  filter={filter}
                  visible={visibleNames}
                  selected={selected}
                  onChoose={chooseColumns}
                />
                <div className="eda-column-list">
                  {visible.map((column) => (
                    <label key={column.name}>
                      <input
                        type="checkbox"
                        checked={selectedSet.has(column.name)}
                        disabled={locked.includes(column.name)}
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
                          {column.geometry
                            ? "Geometry · always included"
                            : column.dtype}
                          {roles.includes(column.name)
                            ? " · selected as target or ID"
                            : ""}
                        </small>
                      </span>
                    </label>
                  ))}
                </div>
                <p className="footnote">
                  Only selected columns enter EDA. Geometry is always included
                  for spatial summaries. Keywords match whole name parts, e.g.
                  “sd” matches NDVI_sd but not NDVI_sdx; “Select only
                  filtered” keeps target, ID and geometry. Choose target and ID roles below; numeric
                  class labels need the classification task.
                </p>
              </section>
              <ConfigForm
                key={configKey}
                schema={formSchema}
                value={config}
                onChange={changeConfig}
                onValidityChange={setValid}
                highlight={["report_title"]}
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
                  Processing continues after you leave this page.
                </span>
              </div>
            </>
          ))}
        {tab === "History" && (
          <div className="eda-history">
            {!history.some((item) => item.eda_id === openId) && (
              <p>
                {loaded
                  ? "This analysis is no longer available."
                  : "Loading this analysis…"}
              </p>
            )}
            {history.filter((item) => item.eda_id === openId).map((item) => (
              <section className="panel pad" key={item.eda_id}>
                <div className="heading">
                  <div>
                    <h3>{item.name}</h3>
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
                  <RenameAction<EdaRun>
                    id={item.eda_id}
                    name={item.name}
                    path={`/eda/${item.eda_id}/name`}
                    status={item.status}
                    onRenamed={(renamed) =>
                      setHistory((rows) =>
                        rows.map((row) =>
                          row.eda_id === renamed.eda_id ? renamed : row,
                        ),
                      )
                    }
                  />
                  <DeleteAction
                    kind="eda"
                    id={item.eda_id}
                    name={item.name}
                    status={item.status}
                    onDeleted={() => {
                      setHistory((rows) =>
                        rows.filter((row) => row.eda_id !== item.eda_id),
                      );
                      if (reportId === item.eda_id) setReportId(undefined);
                      setTab("Start");
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
                  {item.status === "complete" && (
                    <span title="Train and compare classification models on this analysis">
                      <Button
                        primary
                        onClick={() =>
                          onMl({
                            edaId: item.eda_id,
                            title: item.name,
                            runName: run.name,
                          })
                        }
                      >
                        ML classification
                      </Button>
                    </span>
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
                      // Older analyses may lack geometry; it is now always included and excluded from screening.
                      const exclusions = Array.isArray(
                        item.config.feature_selection_exclude_columns,
                      )
                        ? (item.config.feature_selection_exclude_columns as string[])
                        : [];
                      changeConfig({
                        ...item.config,
                        feature_selection_exclude_columns: [
                          ...new Set([...exclusions, ...geometryNames]),
                        ],
                      });
                      setSelected([
                        ...new Set([...item.selected_columns, ...geometryNames]),
                      ]);
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
    </>
  );
}
