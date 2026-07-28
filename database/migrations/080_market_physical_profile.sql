-- FABRIC v1 / BL-44 + BL-38: physical market profile and canonical raw bars.
-- Additive only. Cold archive restore evidence is required before any retention.
CREATE SCHEMA IF NOT EXISTS market;
CREATE TABLE IF NOT EXISTS market.raw_bar (
    raw_bar_id TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    provider_published_at TIMESTAMPTZ,
    available_at TIMESTAMPTZ,
    retrieved_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume NUMERIC,
    provider TEXT NOT NULL,
    source_uri TEXT,
    bar_method TEXT NOT NULL DEFAULT 'provider_official',
    PRIMARY KEY (instrument_id, event_time, provider)
);
SELECT create_hypertable('market.raw_bar', by_range('event_time'), chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);
ALTER TABLE market.raw_bar SET (timescaledb.compress, timescaledb.compress_segmentby = 'instrument_id');
CREATE INDEX IF NOT EXISTS idx_raw_bar_instrument_event ON market.raw_bar (instrument_id, event_time DESC);
COMMENT ON TABLE market.raw_bar IS 'Immutable provider/raw market bars; cold archive restore evidence is mandatory before retention.';
-- Deliberately no add_retention_policy: archival evidence precedes deletion.
