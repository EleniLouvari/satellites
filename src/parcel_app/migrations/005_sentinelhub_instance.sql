-- Optional Sentinel Hub configuration instance per Copernicus account. Applied after 004_inspection.sql.
-- Inspection map tiles pick one of the owner's instances at random, spreading processing units across accounts.
-- Write-only like the password: the API reports only whether an instance is set.
ALTER TABLE public.eo_users ADD COLUMN IF NOT EXISTS sen_instance_id text;
