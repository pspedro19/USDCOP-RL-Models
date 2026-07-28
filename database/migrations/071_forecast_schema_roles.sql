-- Migration 071: physical ACTION/DIAGNOSTIC wall for forecast data (BL-19)
-- Additive only. Existing public forecast_h5_* tables remain compatibility sources.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS forecast;
CREATE SCHEMA IF NOT EXISTS action;
CREATE SCHEMA IF NOT EXISTS portfolio;
CREATE SCHEMA IF NOT EXISTS exec;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'forecast_writer') THEN
        CREATE ROLE forecast_writer NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS forecast.forecast_output (
    forecast_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    forecast_spec_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    horizon TEXT NOT NULL,
    as_of TIMESTAMPTZ NOT NULL,
    available_at TIMESTAMPTZ NOT NULL,
    target_time TIMESTAMPTZ NOT NULL,
    prediction_type TEXT NOT NULL,
    point DOUBLE PRECISION NOT NULL,
    lower_bound DOUBLE PRECISION,
    upper_bound DOUBLE PRECISION,
    direction_probability_up DOUBLE PRECISION,
    model_fingerprint TEXT NOT NULL CHECK (model_fingerprint ~ '^sha256:[0-9a-f]{64}$'),
    data_snapshot_id TEXT NOT NULL,
    diagnostic_only BOOLEAN NOT NULL DEFAULT TRUE CHECK (diagnostic_only),
    run_id TEXT NOT NULL,
    derivation_id TEXT NOT NULL,
    schema_version TEXT NOT NULL DEFAULT '1.0.0',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (available_at >= as_of),
    CHECK (target_time > as_of),
    CHECK (point NOT IN ('NaN'::DOUBLE PRECISION, 'Infinity'::DOUBLE PRECISION, '-Infinity'::DOUBLE PRECISION)),
    CHECK (lower_bound IS NULL OR lower_bound NOT IN ('NaN'::DOUBLE PRECISION, 'Infinity'::DOUBLE PRECISION, '-Infinity'::DOUBLE PRECISION)),
    CHECK (upper_bound IS NULL OR upper_bound NOT IN ('NaN'::DOUBLE PRECISION, 'Infinity'::DOUBLE PRECISION, '-Infinity'::DOUBLE PRECISION)),
    CHECK (
        lower_bound IS NULL OR upper_bound IS NULL
        OR (lower_bound <= point AND point <= upper_bound)
    ),
    CHECK (
        direction_probability_up IS NULL
        OR (
            direction_probability_up NOT IN ('NaN'::DOUBLE PRECISION, 'Infinity'::DOUBLE PRECISION, '-Infinity'::DOUBLE PRECISION)
            AND direction_probability_up BETWEEN 0.0 AND 1.0
        )
    ),
    UNIQUE (
        forecast_spec_id, model_id, horizon, as_of, target_time,
        model_fingerprint, data_snapshot_id
    )
);

CREATE TABLE IF NOT EXISTS forecast.forecast_score (
    score_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    forecast_id UUID NOT NULL REFERENCES forecast.forecast_output(forecast_id),
    scored_at TIMESTAMPTZ NOT NULL,
    actual_value DOUBLE PRECISION,
    error_value DOUBLE PRECISION,
    score_name TEXT NOT NULL,
    score_value DOUBLE PRECISION,
    formula_version TEXT NOT NULL,
    run_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (actual_value IS NULL OR actual_value NOT IN ('NaN'::DOUBLE PRECISION, 'Infinity'::DOUBLE PRECISION, '-Infinity'::DOUBLE PRECISION)),
    CHECK (error_value IS NULL OR error_value NOT IN ('NaN'::DOUBLE PRECISION, 'Infinity'::DOUBLE PRECISION, '-Infinity'::DOUBLE PRECISION)),
    CHECK (score_value IS NULL OR score_value NOT IN ('NaN'::DOUBLE PRECISION, 'Infinity'::DOUBLE PRECISION, '-Infinity'::DOUBLE PRECISION)),
    UNIQUE (forecast_id, scored_at, score_name, formula_version)
);

CREATE TABLE IF NOT EXISTS forecast.model_horizon_result (
    result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    forecast_spec_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    horizon TEXT NOT NULL,
    evaluation_window TSTZRANGE NOT NULL,
    n_observations INTEGER NOT NULL CHECK (n_observations >= 0),
    metrics JSONB NOT NULL,
    formula_versions JSONB NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL,
    run_id TEXT NOT NULL,
    UNIQUE (forecast_spec_id, model_id, horizon, evaluation_window)
);

CREATE TABLE IF NOT EXISTS forecast.calibration_result (
    calibration_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    forecast_spec_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    horizon TEXT NOT NULL,
    calibration_version TEXT NOT NULL,
    fitted_until TIMESTAMPTZ NOT NULL,
    method TEXT NOT NULL,
    parameters JSONB NOT NULL,
    artifact_uri TEXT NOT NULL,
    semantic_hash TEXT NOT NULL CHECK (semantic_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (forecast_spec_id, model_id, horizon, calibration_version)
);

CREATE INDEX IF NOT EXISTS idx_forecast_output_lookup
    ON forecast.forecast_output (asset_id, model_id, horizon, as_of DESC);

GRANT USAGE ON SCHEMA forecast TO forecast_writer;
REVOKE ALL ON ALL TABLES IN SCHEMA forecast FROM PUBLIC;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA forecast FROM PUBLIC;
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA forecast TO forecast_writer;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA forecast TO forecast_writer;
ALTER DEFAULT PRIVILEGES IN SCHEMA forecast
    GRANT SELECT, INSERT ON TABLES TO forecast_writer;

REVOKE ALL ON SCHEMA action FROM forecast_writer;
REVOKE ALL ON SCHEMA portfolio FROM forecast_writer;
REVOKE ALL ON SCHEMA exec FROM forecast_writer;
REVOKE ALL ON ALL TABLES IN SCHEMA action FROM forecast_writer;
REVOKE ALL ON ALL TABLES IN SCHEMA portfolio FROM forecast_writer;
REVOKE ALL ON ALL TABLES IN SCHEMA exec FROM forecast_writer;

COMMENT ON SCHEMA forecast IS
    'BL-19 DIAGNOSTIC namespace. forecast_writer has no access to ACTION/portfolio/exec.';
