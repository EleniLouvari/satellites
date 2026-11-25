-- Parcel inspection of the best ML scenario. Applied after 003_ml.sql.
-- One scenario per EDA analysis can be marked as the best; its parcels are then reviewed one by one.
ALTER TABLE public.ml_runs ADD COLUMN IF NOT EXISTS is_best boolean NOT NULL DEFAULT false;
CREATE UNIQUE INDEX IF NOT EXISTS ml_runs_one_best ON public.ml_runs(eda_id) WHERE is_best;
-- A row exists only once a parcel has been reviewed; missing rows mean "not checked, not approved, no comment".
-- Reviews follow the scenario: deleting the scenario (or its parcel run) removes them.
CREATE TABLE IF NOT EXISTS public.ml_inspection_reviews (
 ml_id uuid NOT NULL REFERENCES public.ml_runs(ml_id) ON DELETE CASCADE,
 parcel_id text NOT NULL,
 check_status text NOT NULL DEFAULT 'not_checked'
  CHECK (check_status IN ('not_checked','satellite','field','satellite_and_field')),
 approved boolean NOT NULL DEFAULT false,
 comment text NOT NULL DEFAULT '' CHECK (length(comment) <= 2000),
 updated_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY (ml_id, parcel_id),
 CHECK (NOT approved OR check_status <> 'not_checked')
);
