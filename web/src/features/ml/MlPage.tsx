import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  date,
  duration,
  type ProgressState,
  type Schema,
} from "../../api";
import ConfigForm from "../../components/ConfigForm";
import Chooser, { BackLink } from "../../components/Chooser";
import DeleteAction from "../../components/DeleteAction";
import Progress from "../../components/Progress";
import MlLive from "./MlLive";
import ColumnFilter, { useColumnFilter } from "../../components/ColumnFilter";
import { Button, Heading, Notice, Status } from "../../components/ui";
import { canContinue, STEPS, stepStates, withoutRoles } from "./mlControls";

export type MlSource = { edaId: string; title: string; runName: string };
type Column = {
  name: string;
  dtype: string;
  numeric: boolean;
  geometry: boolean;
  source: "eda" | "calculated" | "result";
  proposed: boolean;
};
type Options = {
  columns: Column[];
  default_features: string[];
  default_name: string;
  schema: Schema & { managed: Record<string, unknown> };
};
type MlArtifact = { artifact_id: string; name: string; size_bytes: number };
type MlRun = {
  ml_id: string;
  name: string;
  status: string;
  steps_done: number;
  feature_columns: string[];
  config: Record<string, unknown>;
  artifacts: MlArtifact[];
  progress?: ProgressState;
  error?: string;
  created_at: string;
  started_at?: string;
  finished_at?: string;
};

const SECTIONS: { source: Column["source"]; title: string; hint: string }[] = [
  {
    source: "eda",
    title: "Analysed in EDA",
    hint: "Columns you explored in this EDA analysis. The ones EDA proposed are pre-selected.",
  },
  {
    source: "calculated",
    title: "Calculated by EDA",
    hint: "Diagnostics EDA added, such as is_outlier. Not selected unless you tick them.",
  },
  {
    source: "result",
    title: "Other parcel result columns",
    hint: "Columns of the parcel result that were left out of this EDA analysis.",
  },
];
const PREDICTIONS = "05_predict/final_predictions_preview.csv";
const DASHBOARD = "final_dashboard/report.html";

export default function MlPage({
  source,
  onBack,
}: {
  source: MlSource;
  onBack: () => void;
}) {
  const requestId = useRef(crypto.randomUUID());
  const [options, setOptions] = useState<Options>();
  const [name, setName] = useState("");
  const [config, setConfig] = useState<Record<string, unknown>>({});
  const [selected, setSelected] = useState<string[]>([]);
  const [runs, setRuns] = useState<MlRun[]>([]);
  const [tab, setTab] = useState("Start");
  const [openId, setOpenId] = useState<string>();
  const [report, setReport] = useState<{ run: MlRun; page: string }>();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [valid, setValid] = useState(true);
  const [configKey, setConfigKey] = useState(0);
  const base = `/eda/${source.edaId}/ml`;

  useEffect(() => {
    let live = true;
    const refresh = () =>
      api<MlRun[]>(base)
        .then((rows) => {
          if (live) setRuns(rows);
        })
        .catch((e) => {
          if (live) setError(e.message);
        });
    api<Options>(`${base}/options`)
      .then((value) => {
        if (!live) return;
        setOptions(value);
        setConfig(value.schema.defaults);
        setSelected(value.default_features);
        setName(value.default_name);
      })
      .catch((e) => {
        if (live) setError(e.message);
      });
    void refresh();
    const timer = setInterval(refresh, 5000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [base]);

  const selectedSet = useMemo(() => new Set(selected), [selected]);
  const columnNames = useMemo(
    () => options?.columns.map((c) => c.name) ?? [],
    [options],
  );
  const filter = useColumnFilter(columnNames);
  // Geometry, the target and the parcel ID are shown but locked: they can never be features.
  const locked = useMemo(
    () =>
      new Set(
        options?.columns
          .filter(
            (c) =>
              c.geometry ||
              c.name === config.target_column ||
              c.name === config.id_column,
          )
          .map((c) => c.name) ?? [],
      ),
    [options, config.target_column, config.id_column],
  );
  const selectable = useMemo(
    () => options?.columns.filter((column) => !locked.has(column.name)) ?? [],
    [options, locked],
  );
  const proposed = selectable.filter((column) => column.proposed);
  // EDA diagnostics such as is_outlier are only ever added by ticking them.
  const bulk = selectable.filter((column) => column.source !== "calculated");
  const visibleNames = selectable
    .filter((column) => filter.matches(column.name))
    .map((column) => column.name);

  function choose(features: string[]) {
    requestId.current = crypto.randomUUID();
    setSelected(
      withoutRoles([...new Set(features)], config).filter((n) => !locked.has(n)),
    );
  }
  function changeConfig(value: Record<string, unknown>) {
    requestId.current = crypto.randomUUID();
    setConfig(value);
    setSelected((previous) => withoutRoles(previous, value));
  }
  async function create(through: number) {
    setBusy(true);
    setError("");
    try {
      const row = await api<MlRun>(base, "POST", {
        request_id: requestId.current,
        name: name.trim(),
        features: selected,
        config,
        through,
      });
      setRuns((rows) => [row, ...rows.filter((r) => r.ml_id !== row.ml_id)]);
      setOpenId(row.ml_id);
      setTab("Scenarios");
      requestId.current = crypto.randomUUID();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function resume(run: MlRun, through: number) {
    setError("");
    try {
      const row = await api<MlRun>(`/ml/${run.ml_id}/continue`, "POST", {
        through,
      });
      setRuns((rows) => rows.map((r) => (r.ml_id === row.ml_id ? row : r)));
    } catch (e) {
      setError((e as Error).message);
    }
  }
  const ready =
    !busy &&
    valid &&
    selected.length > 0 &&
    name.trim().length > 0 &&
    typeof config.target_column === "string";

  return (
    <>
      <Heading
        title="ML classification"
        subtitle={`${source.title} · ${source.runName}`}
      >
        <div className="actions">
          <Button onClick={onBack}>Back to my runs</Button>
        </div>
      </Heading>
      <Notice message={error} />
      <div className="sectiongap">
        {tab === "Start" ? (
          <Chooser
            newLabel="New scenario"
            newHint="Pick features, models and settings for a new classification run."
            onNew={() => setTab("New scenario")}
            historyLabel="Saved scenarios"
            items={runs.map((run) => ({
              id: run.ml_id,
              title: run.name,
              meta: `${date(run.created_at)} · ${run.feature_columns.length} features · ${run.steps_done} of ${STEPS.length} steps`,
              status: run.status === "paused" ? "waiting" : run.status,
            }))}
            onOpen={(id) => {
              setOpenId(id);
              setTab("Scenarios");
            }}
            emptyText="No scenarios yet. Create the first one for this EDA analysis."
          />
        ) : (
          <BackLink label="All scenarios" onClick={() => setTab("Start")} />
        )}
      </div>

      {tab === "New scenario" &&
        (!options ? (
          <p>Loading columns and settings…</p>
        ) : (
          <>
            <ol className="ml-flow" aria-label="How a scenario runs">
              {STEPS.map((step, index) => (
                <li key={step.id}>
                  <b>{index + 1}</b>
                  <span>
                    <strong>{step.label}</strong>
                    <small>{step.hint}</small>
                  </span>
                </li>
              ))}
            </ol>
            <p className="footnote">
              A scenario is one combination of features, models and settings. Create
              as many as you like for this EDA analysis and compare them. Run the
              five steps in one go, or one at a time and read each report before
              continuing.
            </p>
            <section className="panel pad ml-stage">
              <h3>
                <b>1</b> Name this scenario
              </h3>
              <input
                aria-label="Scenario name"
                value={name}
                maxLength={200}
                placeholder="For example: tree models, all indices"
                onChange={(e) => {
                  requestId.current = crypto.randomUUID();
                  setName(e.target.value);
                }}
              />
            </section>

            <section className="panel pad ml-stage">
              <div className="heading">
                <div>
                  <h3>
                    <b>2</b> Choose the features
                  </h3>
                  <p>
                    {selected.length} of {selectable.length} selected. The
                    model input is the parcel result; EDA supplies the proposal
                    and its calculated columns. The parcel ID, the target and
                    the geometry are never features.
                  </p>
                </div>
                <div className="actions">
                  <Button
                    disabled={!proposed.length}
                    onClick={() => choose(proposed.map((c) => c.name))}
                  >
                    EDA proposed ({proposed.length})
                  </Button>
                  <Button
                    onClick={() =>
                      choose(
                        bulk
                          .filter((c) => c.source === "eda" && c.numeric)
                          .map((c) => c.name),
                      )
                    }
                  >
                    All analysed in EDA
                  </Button>
                  <Button onClick={() => choose(bulk.map((c) => c.name))}>
                    Select all
                  </Button>
                  <Button
                    onClick={() =>
                      choose(bulk.filter((c) => c.numeric).map((c) => c.name))
                    }
                  >
                    Numeric only
                  </Button>
                  <Button onClick={() => choose([])}>Clear</Button>
                </div>
              </div>
              <ColumnFilter
                filter={filter}
                visible={visibleNames}
                selected={selected}
                onChoose={choose}
              />
              {SECTIONS.map((section) => {
                const members =
                  options.columns.filter(
                    (column) => column.source === section.source,
                  );
                const visible = members.filter((column) =>
                  filter.matches(column.name),
                );
                if (!members.length) return null;
                const eligible = members.filter((c) => !locked.has(c.name));
                const count = eligible.filter((c) =>
                  selectedSet.has(c.name),
                ).length;
                return (
                  <details
                    key={section.source}
                    className="ml-columns"
                    open={section.source !== "result"}
                  >
                    <summary>
                      <strong>{section.title}</strong>
                      <span className="badge">
                        {count} of {eligible.length} selected
                      </span>
                      <small>{section.hint}</small>
                    </summary>
                    <div className="eda-column-list">
                      {visible.map((column) => {
                        return (
                          <label key={column.name}>
                            <input
                              type="checkbox"
                              checked={selectedSet.has(column.name)}
                              disabled={locked.has(column.name)}
                              onChange={(e) =>
                                choose(
                                  e.target.checked
                                    ? [...selected, column.name]
                                    : selected.filter((n) => n !== column.name),
                                )
                              }
                            />
                            <span>
                              {column.name}
                              {column.proposed && (
                                <em className="ml-proposed">EDA proposed</em>
                              )}
                              <small>
                                {column.geometry ? "Geometry" : column.dtype}
                                {locked.has(column.name)
                                  ? " · target, ID or geometry, never a feature"
                                  : ""}
                              </small>
                            </span>
                          </label>
                        );
                      })}
                      {!visible.length && (
                        <p className="muted">No column matches the search.</p>
                      )}
                    </div>
                  </details>
                );
              })}
            </section>

            <section className="ml-stage">
              <h3 className="ml-stage-title">
                <b>3</b> Review the settings
              </h3>
              <p className="footnote">
                Every setting starts at its default, so you only change what
                you need. Open a group to see what each setting does. Choose
                the target column in the first group.
              </p>
              <ConfigForm
                key={configKey}
                schema={options.schema}
                value={config}
                onChange={changeConfig}
                onValidityChange={setValid}
                exportExtras={{ features: selected }}
                onImportExtras={(raw, imported) => {
                  if (!Array.isArray(raw.features)) return;
                  // Keep only columns this EDA analysis offers.
                  const known = new Set(selectable.map((column) => column.name));
                  const features = raw.features.filter(
                    (f): f is string => typeof f === "string" && known.has(f),
                  );
                  setSelected(withoutRoles([...new Set(features)], imported));
                }}
              />
            </section>

            <section className="panel pad ml-stage">
              <h3>
                <b>4</b> Run
              </h3>
              {typeof config.target_column !== "string" && (
                <p className="muted">
                  Choose the target column in “Target and parcels” to continue.
                </p>
              )}
              <div className="actions">
                <Button
                  primary
                  disabled={!ready}
                  onClick={() => void create(STEPS.length)}
                >
                  {busy ? "Submitting…" : "Create and run all steps"}
                </Button>
                <Button disabled={!ready} onClick={() => void create(1)}>
                  Create and run only the first step (Check)
                </Button>
              </div>
              <p className="footnote">
                Processing continues after you leave this page. Training can
                take a long time with many models or parcels.
              </p>
            </section>
          </>
        ))}

      {tab === "Scenarios" && (
        <div className="eda-history">
          {!runs.some((run) => run.ml_id === openId) && (
            <p>This scenario is no longer available.</p>
          )}
          {runs.filter((run) => run.ml_id === openId).map((run) => {
            const states = stepStates(run.status, run.steps_done);
            const files = new Map(run.artifacts.map((f) => [f.name, f]));
            const next = STEPS[run.steps_done];
            const retry = ["failed", "interrupted"].includes(run.status);
            const models = Array.isArray(run.config.selected_models)
              ? run.config.selected_models.length
              : 0;
            return (
              <section className="panel pad" key={run.ml_id}>
                <div className="heading">
                  <div>
                    <h3>{run.name}</h3>
                    <small>
                      {date(run.created_at)} · {run.feature_columns.length}{" "}
                      features · {models} models
                      {run.started_at &&
                        ` · ${duration(
                          ((run.finished_at
                            ? Date.parse(run.finished_at)
                            : Date.now()) -
                            Date.parse(run.started_at)) /
                            1000,
                        )}`}
                    </small>
                  </div>
                  <Status
                    value={run.status === "paused" ? "waiting" : run.status}
                  />
                </div>
                <Notice message={run.error ?? ""} />
                <ol className="ml-steps">
                  {STEPS.map((step, index) => (
                    <li key={step.id} className={states[index]}>
                      <b>
                        {states[index] === "done"
                          ? "✓"
                          : states[index] === "failed"
                            ? "!"
                            : index + 1}
                      </b>
                      <strong>{step.label}</strong>
                      <small>
                        {states[index] === "done"
                          ? "Done"
                          : states[index] === "running"
                            ? "Running…"
                            : states[index] === "queued"
                              ? "Queued"
                              : states[index] === "failed"
                                ? "Stopped here"
                                : "Not run yet"}
                      </small>
                      {states[index] === "done" && (
                        <button
                          className="linkbutton"
                          onClick={() => {
                            setReport({ run, page: step.id });
                            setTab("Report");
                          }}
                        >
                          View report
                        </button>
                      )}
                    </li>
                  ))}
                </ol>
                {["running", "queued"].includes(run.status) &&
                  run.progress?.stage && <Progress value={run.progress} />}
                {run.started_at && run.status !== "queued" && (
                  <MlLive
                    mlId={run.ml_id}
                    active={run.status === "running"}
                  />
                )}
                <div className="actions ml-actions">
                  {canContinue(run.status, run.steps_done) && (
                    <>
                      <Button
                        primary
                        onClick={() => void resume(run, run.steps_done + 1)}
                      >
                        {retry ? "Retry" : "Run next"} step: {next.label}
                      </Button>
                      {run.steps_done + 1 < STEPS.length && (
                        <Button
                          onClick={() => void resume(run, STEPS.length)}
                        >
                          Run all remaining steps
                        </Button>
                      )}
                    </>
                  )}
                  {files.has(DASHBOARD) && (
                    <Button
                      primary
                      onClick={() => {
                        setReport({ run, page: "dashboard" });
                        setTab("Report");
                      }}
                    >
                      Final dashboard
                    </Button>
                  )}
                  {run.status === "complete" && files.has(PREDICTIONS) && (
                    <a
                      className="btn primary"
                      href={`/api/ml/${run.ml_id}/files/${files.get(PREDICTIONS)!.artifact_id}`}
                    >
                      Download predictions (.csv)
                    </a>
                  )}
                  {run.status === "complete" && files.has("ml-results.zip") && (
                    <a
                      className="btn"
                      href={`/api/ml/${run.ml_id}/files/${files.get("ml-results.zip")!.artifact_id}`}
                    >
                      Download all results (.zip)
                    </a>
                  )}
                  {run.started_at && run.status !== "queued" && (
                    <a className="btn" href={`/api/ml/${run.ml_id}/log`}>
                      Download log
                    </a>
                  )}
                  <Button
                    onClick={() => {
                      requestId.current = crypto.randomUUID();
                      setConfig(run.config);
                      setSelected(withoutRoles(run.feature_columns, run.config));
                      setName(`${run.name} (copy)`);
                      setConfigKey((key) => key + 1);
                      setTab("New scenario");
                    }}
                  >
                    Use as a new scenario
                  </Button>
                  <DeleteAction
                    kind="ml"
                    id={run.ml_id}
                    name={run.name}
                    status={run.status}
                    onDeleted={() => {
                      setRuns((rows) =>
                        rows.filter((row) => row.ml_id !== run.ml_id),
                      );
                      if (report?.run.ml_id === run.ml_id) setReport(undefined);
                      setTab("Start");
                    }}
                  />
                </div>
                <details className="sectiongap">
                  <summary>Saved features and configuration</summary>
                  <pre className="json">
                    {JSON.stringify(
                      { features: run.feature_columns, config: run.config },
                      null,
                      2,
                    )}
                  </pre>
                </details>
              </section>
            );
          })}
        </div>
      )}

      {tab === "Report" && report && (
        <section className="eda-report">
          <h3>{report.run.name}</h3>
          <div className="actions ml-actions">
            {[
              ...STEPS.slice(0, report.run.steps_done),
              ...(report.run.artifacts.some((f) => f.name === DASHBOARD)
                ? [{ id: "dashboard", label: "Final dashboard" }]
                : []),
            ].map((page) => (
              <Button
                key={page.id}
                primary={report.page === page.id}
                onClick={() => setReport({ ...report, page: page.id })}
              >
                {page.label}
              </Button>
            ))}
            <a
              className="btn"
              href={`/api/ml/${report.run.ml_id}/report/${report.page}`}
              target="_blank"
              rel="noopener noreferrer"
            >
              Open in new tab
            </a>
            <Button
              onClick={() => {
                setOpenId(report.run.ml_id);
                setTab("Scenarios");
              }}
            >
              Back to scenario
            </Button>
          </div>
          <iframe
            key={`${report.run.ml_id}-${report.page}`}
            title="ML report"
            src={`/api/ml/${report.run.ml_id}/report/${report.page}`}
            sandbox="allow-scripts"
          />
        </section>
      )}
    </>
  );
}
