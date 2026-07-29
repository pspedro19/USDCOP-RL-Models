-- =============================================================================
-- USD/COP forward-looking macro point-in-time store
-- =============================================================================
-- The existing macro_indicators_{daily,monthly,quarterly} tables remain the
-- native-frequency SSOT for non-vintaged wide indicators.  This long table is
-- the companion contract for releases where multiple vintages and an exact
-- `available_at` timestamp must survive ingestion.

CREATE TABLE IF NOT EXISTS macro_indicators_pit (
    id BIGSERIAL PRIMARY KEY,
    series_id TEXT NOT NULL,
    observation_date DATE NOT NULL,
    reference_date DATE,
    release_date DATE NOT NULL,
    available_at TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    frequency VARCHAR(16) NOT NULL CHECK (frequency IN ('daily', 'weekly', 'monthly', 'quarterly')),
    unit TEXT NOT NULL,
    source TEXT NOT NULL,
    source_url TEXT NOT NULL,
    document_sha256 CHAR(64) NOT NULL,
    retrieved_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    availability_policy TEXT NOT NULL,
    pit_vintage BOOLEAN NOT NULL DEFAULT FALSE,
    promotion_eligible BOOLEAN NOT NULL DEFAULT FALSE,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_macro_indicators_pit_vintage
        UNIQUE (series_id, observation_date, available_at),
    CONSTRAINT ck_macro_indicators_pit_available_after_observation
        CHECK (available_at >= observation_date::timestamptz)
);

CREATE INDEX IF NOT EXISTS idx_macro_pit_series_available
    ON macro_indicators_pit (series_id, available_at DESC);
CREATE INDEX IF NOT EXISTS idx_macro_pit_observation
    ON macro_indicators_pit (observation_date DESC);
CREATE INDEX IF NOT EXISTS idx_macro_pit_promotion
    ON macro_indicators_pit (series_id, available_at DESC)
    WHERE promotion_eligible = TRUE;

CREATE OR REPLACE VIEW macro_indicators_pit_latest AS
SELECT DISTINCT ON (series_id, observation_date)
    series_id,
    observation_date,
    reference_date,
    release_date,
    available_at,
    value,
    frequency,
    unit,
    source,
    source_url,
    document_sha256,
    retrieved_at,
    availability_policy,
    pit_vintage,
    promotion_eligible,
    metadata_json
FROM macro_indicators_pit
ORDER BY series_id, observation_date, available_at DESC, retrieved_at DESC;

COMMENT ON TABLE macro_indicators_pit IS
    'Long-form PIT macro/forward releases. Forecast joins must use available_at, never observation_date.';
COMMENT ON COLUMN macro_indicators_pit.promotion_eligible IS
    'True only when historical availability is supported by an official document/publication timestamp.';
