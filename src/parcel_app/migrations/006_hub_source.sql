-- Preserve every existing run as openEO; HUB runs use server-owned credentials.
ALTER TABLE public.analysis_runs ADD COLUMN IF NOT EXISTS source text NOT NULL DEFAULT 'openeo';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'analysis_runs_source_check'
          AND conrelid = 'public.analysis_runs'::regclass
    ) THEN
        ALTER TABLE public.analysis_runs ADD CONSTRAINT analysis_runs_source_check
            CHECK (source IN ('openeo', 'hub'));
    END IF;
END $$;
