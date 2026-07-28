-- Migration 072: canonical asset/instrument/provider identities (BL-37)

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS reference;

CREATE TABLE IF NOT EXISTS reference.calendar (
    calendar_id TEXT PRIMARY KEY,
    timezone TEXT NOT NULL,
    calendar_kind TEXT NOT NULL,
    session_definition JSONB NOT NULL,
    version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS reference.asset (
    asset_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    quote_currency TEXT,
    annualization INTEGER NOT NULL CHECK (annualization > 0),
    calendar_id TEXT NOT NULL REFERENCES reference.calendar(calendar_id),
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS reference.instrument (
    instrument_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    canonical_symbol TEXT NOT NULL UNIQUE,
    asset_id TEXT NOT NULL REFERENCES reference.asset(asset_id),
    instrument_type TEXT NOT NULL,
    base_currency TEXT,
    quote_currency TEXT,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS reference.provider (
    provider_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    authoritative_for TEXT[] NOT NULL DEFAULT '{}',
    active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS reference.provider_symbol (
    provider_id TEXT NOT NULL REFERENCES reference.provider(provider_id),
    provider_symbol TEXT NOT NULL,
    instrument_id UUID NOT NULL REFERENCES reference.instrument(instrument_id),
    valid_from TIMESTAMPTZ,
    valid_until TIMESTAMPTZ,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (provider_id, provider_symbol),
    CHECK (valid_until IS NULL OR valid_from IS NULL OR valid_until > valid_from)
);

CREATE TABLE IF NOT EXISTS reference.instrument_alias (
    alias TEXT PRIMARY KEY,
    instrument_id UUID NOT NULL REFERENCES reference.instrument(instrument_id),
    alias_kind TEXT NOT NULL,
    deprecated BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS reference.bar_interval (
    interval_id TEXT PRIMARY KEY CHECK (interval_id ~ '^P(T[0-9]+[HMS]|[0-9]+D|[0-9]+W|[0-9]+M)$'),
    seconds INTEGER,
    calendar_aware BOOLEAN NOT NULL DEFAULT FALSE,
    CHECK ((calendar_aware AND seconds IS NULL) OR (NOT calendar_aware AND seconds > 0))
);

INSERT INTO reference.bar_interval (interval_id, seconds, calendar_aware) VALUES
    ('PT5M', 300, FALSE),
    ('PT1H', 3600, FALSE),
    ('PT4H', 14400, FALSE),
    ('P1D', 86400, FALSE),
    ('P1W', 604800, FALSE),
    ('P1M', NULL, TRUE)
ON CONFLICT (interval_id) DO NOTHING;

COMMENT ON TABLE reference.instrument IS
    'BL-37: SPX, SPY and ES are distinct instruments; aliases never collapse identity.';
COMMENT ON TABLE reference.provider_symbol IS
    'Provider-specific names resolve to one canonical instrument_id.';
