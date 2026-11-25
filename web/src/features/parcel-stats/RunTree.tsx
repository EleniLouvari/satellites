import { useEffect, useState } from "react";
import { api, date, duration, type Run } from "../../api";
import { Button, Status } from "../../components/ui";

/** The fields of an EDA analysis the run tree shows. */
export type EdaSummary = {
  eda_id: string;
  name: string;
  status: string;
  selected_columns: string[];
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  ml_count?: number;
};
type MlSummary = {
  ml_id: string;
  name: string;
  status: string;
  steps_done: number;
  feature_columns: string[];
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
};
const ML_STEPS = 5;
const COLUMNS = 6;

export type TreeActions = {
  onEda: (run: Run, edaId?: string) => void;
  onMl: (run: Run, eda: EdaSummary, mlId?: string) => void;
};

function elapsed(started?: string | null, finished?: string | null) {
  if (!started) return "—";
  const end = finished ? Date.parse(finished) : Date.now();
  return duration(Math.max(0, (end - Date.parse(started)) / 1000));
}

/** Poll a child list every five seconds while its parent row is expanded. */
function useChildren<T>(path: string) {
  const [rows, setRows] = useState<T[]>();
  const [error, setError] = useState("");
  useEffect(() => {
    let live = true;
    const refresh = () =>
      api<T[]>(path)
        .then((value) => {
          if (live) {
            setRows(value);
            setError("");
          }
        })
        .catch((e) => {
          if (live) setError((e as Error).message);
        });
    void refresh();
    const timer = setInterval(refresh, 5000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [path]);
  return { rows, error };
}

function MessageRow({ level, children }: { level: number; children: string }) {
  return (
    <tr className={`tree-row tree-level-${level}`}>
      <td colSpan={COLUMNS}>
        <small className="tree-message">{children}</small>
      </td>
    </tr>
  );
}

/** Expand/collapse control shown before a tree row's name; a spacer keeps names aligned. */
export function TreeToggle({
  open,
  label,
  onToggle,
  hidden = false,
}: {
  open: boolean;
  label: string;
  onToggle: () => void;
  hidden?: boolean;
}) {
  if (hidden) return <span className="tree-toggle spacer" aria-hidden="true" />;
  return (
    <button
      type="button"
      className="tree-toggle"
      aria-expanded={open}
      aria-label={`${open ? "Hide" : "Show"} ${label}`}
      onClick={onToggle}
    >
      {open ? "▾" : "▸"}
    </button>
  );
}

/** Count line shown under a run's name, e.g. "2 EDA · 5 scenarios". */
export function runChildSummary(run: Run) {
  const eda = run.eda_count ?? 0;
  const ml = run.ml_count ?? 0;
  if (!eda) return "";
  return `${eda} EDA${ml ? ` · ${ml} scenario${ml === 1 ? "" : "s"}` : ""}`;
}

/** Rows of the EDA analyses made from one run, each expandable to its ML scenarios. */
export function RunChildren({ run, actions }: { run: Run; actions: TreeActions }) {
  const { rows, error } = useChildren<EdaSummary>(`/runs/${run.run_id}/eda`);
  const [open, setOpen] = useState<Set<string>>(new Set());
  if (error && !rows) return <MessageRow level={1}>{error}</MessageRow>;
  if (!rows) return <MessageRow level={1}>Loading EDA analyses…</MessageRow>;
  if (!rows.length)
    return <MessageRow level={1}>No EDA analyses yet.</MessageRow>;
  return (
    <>
      {rows.map((eda) => {
        const expanded = open.has(eda.eda_id);
        const scenarios = eda.ml_count ?? 0;
        return [
          <tr key={eda.eda_id} className="tree-row tree-level-1">
            <td>
              <div className="tree-name">
                <TreeToggle
                  open={expanded}
                  label={`ML scenarios of ${eda.name}`}
                  hidden={!scenarios}
                  onToggle={() =>
                    setOpen((ids) => {
                      const next = new Set(ids);
                      if (!next.delete(eda.eda_id)) next.add(eda.eda_id);
                      return next;
                    })
                  }
                />
                <span className="tree-kind">EDA</span>
                <button
                  type="button"
                  className="tree-link"
                  onClick={() => actions.onEda(run, eda.eda_id)}
                >
                  <strong>{eda.name}</strong>
                  <small>
                    {eda.selected_columns.length} columns
                    {scenarios
                      ? ` · ${scenarios} scenario${scenarios === 1 ? "" : "s"}`
                      : ""}
                  </small>
                </button>
              </div>
            </td>
            <td>
              <Status value={eda.status} />
            </td>
            <td />
            <td>{date(eda.created_at)}</td>
            <td>{elapsed(eda.started_at, eda.finished_at)}</td>
            <td>
              <Button onClick={() => actions.onEda(run, eda.eda_id)}>Open</Button>
              <span title="ML classification is available after the EDA completes">
                <Button
                  disabled={eda.status !== "complete"}
                  onClick={() => actions.onMl(run, eda)}
                >
                  ML
                </Button>
              </span>
            </td>
          </tr>,
          expanded && (
            <EdaChildren
              key={`${eda.eda_id}-ml`}
              run={run}
              eda={eda}
              actions={actions}
            />
          ),
        ];
      })}
    </>
  );
}

function EdaChildren({
  run,
  eda,
  actions,
}: {
  run: Run;
  eda: EdaSummary;
  actions: TreeActions;
}) {
  const { rows, error } = useChildren<MlSummary>(`/eda/${eda.eda_id}/ml`);
  if (error && !rows) return <MessageRow level={2}>{error}</MessageRow>;
  if (!rows) return <MessageRow level={2}>Loading scenarios…</MessageRow>;
  if (!rows.length) return <MessageRow level={2}>No scenarios yet.</MessageRow>;
  return (
    <>
      {rows.map((ml) => (
        <tr key={ml.ml_id} className="tree-row tree-level-2">
          <td>
            <div className="tree-name">
              <span className="tree-toggle spacer" aria-hidden="true" />
              <span className="tree-kind ml">ML</span>
              <button
                type="button"
                className="tree-link"
                onClick={() => actions.onMl(run, eda, ml.ml_id)}
              >
                <strong>{ml.name}</strong>
                <small>
                  {ml.steps_done} of {ML_STEPS} steps · {ml.feature_columns.length}{" "}
                  features
                </small>
              </button>
            </div>
          </td>
          <td>
            <Status value={ml.status === "paused" ? "waiting" : ml.status} />
          </td>
          <td />
          <td>{date(ml.created_at)}</td>
          <td>{elapsed(ml.started_at, ml.finished_at)}</td>
          <td>
            <Button onClick={() => actions.onMl(run, eda, ml.ml_id)}>Open</Button>
          </td>
        </tr>
      ))}
    </>
  );
}
