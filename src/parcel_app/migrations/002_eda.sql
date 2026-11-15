-- EDA analyses of completed parcel-statistics runs. Applied after 001_core.sql.
CREATE TABLE IF NOT EXISTS public.eda_runs (
 eda_id uuid PRIMARY KEY, run_id uuid NOT NULL REFERENCES public.analysis_runs(run_id),
 app_user_id bigint NOT NULL REFERENCES public.app_users(app_user_id), selected_columns jsonb NOT NULL,
 config jsonb NOT NULL, input_sha256 text NOT NULL, output_dir text NOT NULL,
 status text NOT NULL DEFAULT 'queued', artifacts jsonb NOT NULL DEFAULT '[]', error text,
 created_at timestamptz NOT NULL DEFAULT now(), started_at timestamptz, finished_at timestamptz
);
CREATE INDEX IF NOT EXISTS eda_runs_owner_run ON public.eda_runs(app_user_id,run_id,created_at);
ALTER TABLE public.eda_runs ADD COLUMN IF NOT EXISTS progress jsonb NOT NULL DEFAULT '{}';
