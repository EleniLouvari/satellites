-- ML classification runs (scenarios) of completed EDA analyses. Applied after 002_eda.sql.
-- The model input is the parcel result of run_id; eda_id supplies the proposed features and outlier flag.
-- steps_done counts the finished pipeline steps (check, prepare, train, evaluate, predict).
CREATE TABLE IF NOT EXISTS public.ml_runs (
 ml_id uuid PRIMARY KEY, run_id uuid NOT NULL REFERENCES public.analysis_runs(run_id),
 eda_id uuid NOT NULL REFERENCES public.eda_runs(eda_id),
 app_user_id bigint NOT NULL REFERENCES public.app_users(app_user_id), name text NOT NULL,
 feature_columns jsonb NOT NULL, config jsonb NOT NULL, input_sha256 text NOT NULL, eda_sha256 text NOT NULL,
 output_dir text NOT NULL, status text NOT NULL DEFAULT 'queued', steps_done integer NOT NULL DEFAULT 0, progress jsonb NOT NULL DEFAULT '{}',
 artifacts jsonb NOT NULL DEFAULT '[]', error text,
 created_at timestamptz NOT NULL DEFAULT now(), started_at timestamptz, finished_at timestamptz
);
CREATE INDEX IF NOT EXISTS ml_runs_owner_eda ON public.ml_runs(app_user_id,eda_id,created_at);
CREATE INDEX IF NOT EXISTS ml_runs_run ON public.ml_runs(run_id);
