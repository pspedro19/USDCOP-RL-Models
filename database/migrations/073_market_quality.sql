-- Migration 073: immutable raw/canonical market bars and quarantine events (BL-38/40)

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS market;
CREATE SCHEMA IF NOT EXISTS quality;

CREATE TABLE IF NOT EXISTS market.raw_bar (
    raw_bar_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    instrument_id UUID NOT NULL REFERENCES reference.instrument(instrument_id),
    provider_id TEXT NOT NULL REFERENCES reference.provider(provider_id),
    provider_symbol TEXT NOT NULL,
    interval_id TEXT NOT NULL REFERENCES reference.bar_interval(interval_id),
    event_time TIMESTAMPTZ NOT NULL,
    provider_published_at TIMESTAMPTZ,
    available_at TIMESTAMPTZ NOT NULL,
    retrieved_at TIMESTAMPTZ NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume NUMERIC,
    source_payload_hash TEXT NOT NULL,
    source_uri TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (instrument_id, provider_id, interval_id, event_time, source_payload_hash),
    CHECK (available_at >= event_time),
    CHECK (retrieved_at >= available_at),
    CHECK (high >= GREATEST(open, close, low)),
    CHECK (low <= LEAST(open, close, high)),
    CHECK (volume IS NULL OR volume >= 0)
);

CREATE TABLE IF NOT EXISTS quality.quarantine_event (
    quarantine_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    instrument_id UUID REFERENCES reference.instrument(instrument_id),
    rule_id TEXT NOT NULL,
    rule_version TEXT NOT NULL,
    observed_value JSONB NOT NULL,
    source_record JSONB NOT NULL,
    quarantined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status TEXT NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN','CONFIRMED','CORRECTED','REJECTED')),
    resolution TEXT,
    correction_event_id UUID
);

CREATE TABLE IF NOT EXISTS quality.correction_event (
    correction_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    quarantine_id UUID NOT NULL REFERENCES quality.quarantine_event(quarantine_id),
    revision_type TEXT NOT NULL CHECK (
        revision_type IN ('PROVIDER_CORRECTION','PIPELINE_ERROR','SCHEMA_REINTERPRETATION')
    ),
    compared_provider_id TEXT,
    old_record JSONB NOT NULL,
    corrected_record JSONB NOT NULL,
    reason TEXT NOT NULL,
    corrected_by TEXT NOT NULL,
    corrected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE quality.quarantine_event
    DROP CONSTRAINT IF EXISTS quarantine_correction_fk;
ALTER TABLE quality.quarantine_event
    ADD CONSTRAINT quarantine_correction_fk
    FOREIGN KEY (correction_event_id)
    REFERENCES quality.correction_event(correction_event_id);

CREATE TABLE IF NOT EXISTS quality.feature_status (
    feature_status_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_id TEXT NOT NULL,
    instrument_id UUID REFERENCES reference.instrument(instrument_id),
    status TEXT NOT NULL CHECK (status IN ('AVAILABLE','STALE','UNAVAILABLE','QUARANTINED')),
    reason_code TEXT NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_feature_status_instrument
    ON quality.feature_status (feature_id, instrument_id, observed_at)
    WHERE instrument_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_feature_status_global
    ON quality.feature_status (feature_id, observed_at)
    WHERE instrument_id IS NULL;

CREATE TABLE IF NOT EXISTS market.canonical_bar (
    canonical_bar_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    instrument_id UUID NOT NULL REFERENCES reference.instrument(instrument_id),
    interval_id TEXT NOT NULL REFERENCES reference.bar_interval(interval_id),
    event_time TIMESTAMPTZ NOT NULL,
    canonical_version INTEGER NOT NULL DEFAULT 1 CHECK (canonical_version > 0),
    available_at TIMESTAMPTZ NOT NULL,
    bar_method TEXT NOT NULL CHECK (bar_method IN ('provider_official','resampled')),
    source_raw_bar_ids UUID[] NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume NUMERIC,
    semantic_hash TEXT NOT NULL,
    quality_status TEXT NOT NULL CHECK (quality_status IN ('VALID','STALE','QUARANTINED')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (instrument_id, interval_id, event_time, bar_method, canonical_version),
    UNIQUE (semantic_hash),
    CHECK (cardinality(source_raw_bar_ids) > 0),
    CHECK (available_at >= event_time),
    CHECK (high >= GREATEST(open, close, low)),
    CHECK (low <= LEAST(open, close, high)),
    CHECK (volume IS NULL OR volume >= 0)
);

CREATE TABLE IF NOT EXISTS market.canonical_bar_source (
    canonical_bar_id UUID NOT NULL REFERENCES market.canonical_bar(canonical_bar_id) ON DELETE RESTRICT,
    raw_bar_id UUID NOT NULL REFERENCES market.raw_bar(raw_bar_id) ON DELETE RESTRICT,
    source_order INTEGER NOT NULL CHECK (source_order >= 0),
    PRIMARY KEY (canonical_bar_id, raw_bar_id),
    UNIQUE (canonical_bar_id, source_order)
);

CREATE INDEX IF NOT EXISTS idx_raw_bar_instrument_time
    ON market.raw_bar (instrument_id, interval_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_canonical_bar_instrument_time
    ON market.canonical_bar (instrument_id, interval_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_quarantine_open
    ON quality.quarantine_event (rule_id, quarantined_at DESC)
    WHERE status = 'OPEN';

CREATE OR REPLACE FUNCTION market.block_raw_bar_mutation()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'market.raw_bar is immutable; publish a correction_event';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_raw_bar_immutable ON market.raw_bar;
CREATE TRIGGER trg_raw_bar_immutable
    BEFORE UPDATE OR DELETE ON market.raw_bar
    FOR EACH ROW EXECUTE FUNCTION market.block_raw_bar_mutation();

DROP TRIGGER IF EXISTS trg_canonical_bar_immutable ON market.canonical_bar;
CREATE TRIGGER trg_canonical_bar_immutable
    BEFORE UPDATE OR DELETE ON market.canonical_bar
    FOR EACH ROW EXECUTE FUNCTION market.block_raw_bar_mutation();
DROP TRIGGER IF EXISTS trg_canonical_bar_source_immutable ON market.canonical_bar_source;
CREATE TRIGGER trg_canonical_bar_source_immutable
    BEFORE UPDATE OR DELETE ON market.canonical_bar_source
    FOR EACH ROW EXECUTE FUNCTION market.block_raw_bar_mutation();
DROP TRIGGER IF EXISTS trg_quality_correction_immutable ON quality.correction_event;
CREATE TRIGGER trg_quality_correction_immutable
    BEFORE UPDATE OR DELETE ON quality.correction_event
    FOR EACH ROW EXECUTE FUNCTION market.block_raw_bar_mutation();

DROP TRIGGER IF EXISTS trg_raw_bar_no_truncate ON market.raw_bar;
CREATE TRIGGER trg_raw_bar_no_truncate
    BEFORE TRUNCATE ON market.raw_bar
    FOR EACH STATEMENT EXECUTE FUNCTION market.block_raw_bar_mutation();
DROP TRIGGER IF EXISTS trg_canonical_bar_no_truncate ON market.canonical_bar;
CREATE TRIGGER trg_canonical_bar_no_truncate
    BEFORE TRUNCATE ON market.canonical_bar
    FOR EACH STATEMENT EXECUTE FUNCTION market.block_raw_bar_mutation();
DROP TRIGGER IF EXISTS trg_canonical_bar_source_no_truncate
    ON market.canonical_bar_source;
CREATE TRIGGER trg_canonical_bar_source_no_truncate
    BEFORE TRUNCATE ON market.canonical_bar_source
    FOR EACH STATEMENT EXECUTE FUNCTION market.block_raw_bar_mutation();
DROP TRIGGER IF EXISTS trg_quality_correction_no_truncate
    ON quality.correction_event;
CREATE TRIGGER trg_quality_correction_no_truncate
    BEFORE TRUNCATE ON quality.correction_event
    FOR EACH STATEMENT EXECUTE FUNCTION market.block_raw_bar_mutation();

COMMENT ON TABLE market.raw_bar IS
    'BL-38 immutable provider observations with five explicit timestamps.';
COMMENT ON TABLE quality.quarantine_event IS
    'BL-40 anomalous records are events, never silent UPDATE/clip operations.';
