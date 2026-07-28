-- FABRIC v1 / BL-43: synthetic demo models cannot enter real performance paths.
-- Additive and fail-closed; no production model rows are rewritten here.
CREATE SCHEMA IF NOT EXISTS demo;
CREATE TABLE IF NOT EXISTS demo.synthetic_model (
    model_id TEXT PRIMARY KEY,
    algorithm TEXT NOT NULL,
    environment TEXT NOT NULL,
    surface TEXT NOT NULL,
    execution_eligible BOOLEAN NOT NULL DEFAULT FALSE,
    CONSTRAINT synthetic_demo_only CHECK (algorithm = 'SYNTHETIC' AND environment = 'demo' AND surface = 'synthetic' AND execution_eligible = FALSE)
);
CREATE OR REPLACE FUNCTION demo.reject_synthetic_performance(p_algorithm TEXT, p_environment TEXT, p_surface TEXT, p_execution_eligible BOOLEAN)
RETURNS VOID LANGUAGE plpgsql AS $$
BEGIN
    IF upper(COALESCE(p_algorithm, '')) = 'SYNTHETIC'
       AND (COALESCE(p_environment, '') <> 'demo' OR COALESCE(p_surface, '') <> 'synthetic' OR COALESCE(p_execution_eligible, TRUE) IS TRUE) THEN
        RAISE EXCEPTION 'synthetic models are demo-only and execution-ineligible';
    END IF;
END;
$$;
COMMENT ON FUNCTION demo.reject_synthetic_performance(TEXT, TEXT, TEXT, BOOLEAN) IS 'Call at every performance/export boundary; synthetic models are never real performance.';
-- The legacy metrics.model_performance table remains untouched until its planned retirement.
