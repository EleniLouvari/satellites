-- Apply explicitly to the selected PostgreSQL database, using a migration role.
-- This creates new tables; it does not rename or modify the legacy TBL_USERS table.
-- Fail on existing names so incompatible schemas cannot be silently accepted.
BEGIN;

CREATE TABLE public.app_users (
    app_user_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    username TEXT NOT NULL CHECK (username = btrim(username) AND username <> ''),
    password TEXT NOT NULL CHECK (password <> ''),
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);
CREATE UNIQUE INDEX app_users_username_unique ON public.app_users (lower(username));
COMMENT ON COLUMN public.app_users.password IS
    'Application password hash (for example Argon2id), never plaintext or a Copernicus credential.';

CREATE TABLE public.eo_users (
    eo_user_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    app_user_id BIGINT NOT NULL REFERENCES public.app_users(app_user_id) ON DELETE RESTRICT,
    username TEXT NOT NULL CHECK (username = btrim(username) AND username <> ''),
    password TEXT NOT NULL CHECK (password <> '')
);
CREATE UNIQUE INDEX eo_users_owner_username_unique ON public.eo_users (app_user_id, lower(username));
COMMENT ON COLUMN public.eo_users.password IS
    'Fernet-encrypted Copernicus password; the encryption key is supplied separately at runtime.';

COMMIT;
