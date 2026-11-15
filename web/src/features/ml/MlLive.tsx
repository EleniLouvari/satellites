import { useEffect, useRef, useState } from "react";

import { api, duration } from "../../api";
import { trainingCounts, type TrainingModel } from "./mlControls";

type Live = {
  training: { started_at?: string; finished_at?: string | null; models: TrainingModel[] } | null;
  log: string;
};

const LABELS: Record<string, string> = {
  pending: "Waiting",
  running: "Training…",
  finished: "Trained",
  failed: "Failed",
  reused: "Reused",
  unavailable: "Not installed",
};

/** Per-model Train progress and the end of the run log, refreshed while the run works. */
export default function MlLive({ mlId, active }: { mlId: string; active: boolean }) {
  const [live, setLive] = useState<Live>();
  const [now, setNow] = useState(Date.now());
  const logRef = useRef<HTMLPreElement>(null);
  const follow = useRef(true);

  useEffect(() => {
    let current = true;
    const load = () =>
      api<Live>(`/ml/${mlId}/live`)
        .then((value) => {
          if (current) setLive(value);
        })
        .catch(() => {
          // The live view is optional; the run card keeps showing its saved state.
        });
    void load();
    if (!active) return () => void (current = false);
    const timer = setInterval(() => {
      void load();
      setNow(Date.now());
    }, 3000);
    return () => {
      current = false;
      clearInterval(timer);
    };
  }, [mlId, active]);

  useEffect(() => {
    const box = logRef.current;
    if (box && follow.current) box.scrollTop = box.scrollHeight;
  }, [live?.log]);

  if (!live || (!live.training && !live.log)) return null;
  const models = live.training?.models ?? [];
  const counts = trainingCounts(models);

  return (
    <div className="ml-live">
      {models.length > 0 && (
        <div className="ml-train">
          <div className="ml-train-heading">
            <strong>Train step models</strong>
            <small>
              {counts.done} of {counts.total} done
              {counts.failed > 0 && ` · ${counts.failed} failed`}
              {counts.unavailable > 0 && ` · ${counts.unavailable} not installed`}
            </small>
          </div>
          <progress value={counts.done} max={Math.max(counts.total, 1)} aria-label="Models trained" />
          <ul className="ml-models">
            {models.map((model) => {
              const seconds =
                model.duration_seconds ??
                (model.status === "running" && model.started_at
                  ? (now - Date.parse(model.started_at)) / 1000
                  : undefined);
              return (
                <li key={model.name} className={`is-${model.status}`} title={model.error}>
                  <span className="ml-model-dot" aria-hidden="true" />
                  <strong>{model.name.replaceAll("_", " ")}</strong>
                  <small>
                    {LABELS[model.status] ?? model.status}
                    {seconds != null && ` · ${duration(seconds)}`}
                  </small>
                  {model.error && <small className="ml-model-error">{model.error}</small>}
                </li>
              );
            })}
          </ul>
        </div>
      )}
      {live.log && (
        <details className="ml-log" open={active}>
          <summary>
            Run log <small>(last 200 lines{active ? ", refreshed every 3 s" : ""})</small>
          </summary>
          <pre
            ref={logRef}
            onScroll={(e) => {
              const box = e.currentTarget;
              // Stop following new lines while the user reads further up.
              follow.current = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
            }}
          >
            {live.log}
          </pre>
        </details>
      )}
    </div>
  );
}
