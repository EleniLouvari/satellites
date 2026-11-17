/** The five pipeline steps, in the order the server runs them. */
export const STEPS = [
  { id: "check", label: "Check", hint: "Validate and clean the data" },
  { id: "prepare", label: "Prepare", hint: "Split train, test and folds" },
  { id: "train", label: "Train", hint: "Tune every selected model" },
  { id: "evaluate", label: "Evaluate", hint: "Score models on the test set" },
  { id: "predict", label: "Predict", hint: "Classify all parcels" },
] as const;

export type StepState = "done" | "running" | "queued" | "failed" | "pending";

/** Finished steps are done; the first unfinished one carries the run's activity or failure. */
export function stepStates(status: string, stepsDone: number): StepState[] {
  return STEPS.map((_, index) => {
    if (index < stepsDone) return "done";
    if (index > stepsDone) return "pending";
    if (status === "running" || status === "queued") return status;
    return status === "failed" || status === "interrupted"
      ? "failed"
      : "pending";
  });
}

/** A run can take more steps only while it is idle with steps left. */
export function canContinue(status: string, stepsDone: number): boolean {
  return (
    ["paused", "failed", "interrupted"].includes(status) &&
    stepsDone < STEPS.length
  );
}

/** The target and ID columns can never be features; drop them when a role changes. */
export function withoutRoles(
  features: string[],
  config: Record<string, unknown>,
): string[] {
  const roles = [config.target_column, config.id_column];
  return features.filter((name) => !roles.includes(name));
}

export type TrainingModel = {
  name: string;
  status: string;
  started_at?: string;
  finished_at?: string;
  duration_seconds?: number;
  error?: string;
};

/** Count Train step models: trained, failed and reused ones are done; missing libraries never run. */
export function trainingCounts(models: TrainingModel[]) {
  const runnable = models.filter((model) => model.status !== "unavailable");
  return {
    total: runnable.length,
    done: runnable.filter((model) => ["finished", "failed", "reused"].includes(model.status)).length,
    failed: models.filter((model) => model.status === "failed").length,
    unavailable: models.length - runnable.length,
  };
}

export type RerunStep = "train" | "evaluate" | "predict";
export type RerunRules = {
  train: string[];
  evaluate: string[];
  predict: string[];
  retrain_all_when: string[];
  retrain_model_when: Record<string, string[]>;
};

/** Train, Evaluate and Predict can run again on an idle run once the steps before them are done. */
export function canRerun(status: string, stepsDone: number, step: RerunStep): boolean {
  const index = { train: 2, evaluate: 3, predict: 4 }[step];
  return ["complete", "paused", "failed", "interrupted"].includes(status) && stepsDone >= index;
}

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

/** Selected models whose saved training no longer matches the edited settings; the server applies the same rule. */
export function requiredRetrain(
  previous: Record<string, unknown>,
  current: Record<string, unknown>,
  rules: RerunRules,
): Set<string> {
  const models = Array.isArray(current.selected_models) ? (current.selected_models as string[]) : [];
  if (rules.retrain_all_when.some((name) => !same(previous[name], current[name]))) return new Set(models);
  const changed = models.filter(
    (model) => !same(previous[`tune_params.${model}`], current[`tune_params.${model}`]),
  );
  for (const [model, names] of Object.entries(rules.retrain_model_when)) {
    if (models.includes(model) && names.some((name) => !same(previous[name], current[name]))) changed.push(model);
  }
  return new Set(changed);
}
