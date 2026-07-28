-- Migration 075: environment-aware position and PnL facts (BL-22)

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS fact;

CREATE TABLE IF NOT EXISTS fact.position (
    as_of TIMESTAMPTZ NOT NULL,
    strategy_id TEXT NOT NULL,
    sleeve_id TEXT NOT NULL,
    instrument_id UUID NOT NULL REFERENCES reference.instrument(instrument_id),
    environment TEXT NOT NULL CHECK (
        environment IN ('backtest','held_out','paper','canary','live')
    ),
    qty NUMERIC NOT NULL,
    average_cost NUMERIC,
    market_price NUMERIC,
    market_value NUMERIC,
    currency TEXT NOT NULL,
    source_fill_set_id TEXT NOT NULL,
    reconciliation_status TEXT NOT NULL CHECK (
        reconciliation_status IN ('PENDING','RECONCILED','MISMATCH','QUARANTINED')
    ),
    run_id TEXT NOT NULL,
    derivation_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (as_of, strategy_id, sleeve_id, instrument_id, environment),
    CHECK (derivation_id ~ '^sha256:[0-9a-f]{64}$'),
    CHECK (length(trim(strategy_id)) > 0),
    CHECK (length(trim(sleeve_id)) > 0),
    CHECK (length(trim(currency)) > 0),
    CHECK (length(trim(source_fill_set_id)) > 0),
    CHECK (length(trim(run_id)) > 0),
    CHECK (qty NOT IN ('NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC)),
    CHECK (average_cost IS NULL OR average_cost NOT IN ('NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC)),
    CHECK (market_price IS NULL OR market_price NOT IN ('NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC)),
    CHECK (market_value IS NULL OR market_value NOT IN ('NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC))
);

CREATE TABLE IF NOT EXISTS fact.pnl (
    as_of TIMESTAMPTZ NOT NULL,
    strategy_id TEXT NOT NULL,
    sleeve_id TEXT NOT NULL,
    instrument_id UUID NOT NULL REFERENCES reference.instrument(instrument_id),
    environment TEXT NOT NULL CHECK (
        environment IN ('backtest','held_out','paper','canary','live')
    ),
    pnl_component TEXT NOT NULL CHECK (
        pnl_component IN (
            'gross_pnl','pnl_beta','pnl_timing','pnl_carry',
            'commissions','slippage','financing','pnl_residual'
        )
    ),
    amount NUMERIC NOT NULL,
    nav_amount NUMERIC NOT NULL CHECK (nav_amount > 0),
    currency TEXT NOT NULL,
    attribution_model_version TEXT NOT NULL,
    benchmark_id TEXT NOT NULL,
    source_fill_set_id TEXT NOT NULL,
    reconciliation_status TEXT NOT NULL CHECK (
        reconciliation_status IN ('PENDING','RECONCILED','MISMATCH','QUARANTINED')
    ),
    run_id TEXT NOT NULL,
    derivation_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (
        as_of, strategy_id, sleeve_id, instrument_id, environment, pnl_component
    ),
    CHECK (derivation_id ~ '^sha256:[0-9a-f]{64}$'),
    CHECK (amount NOT IN ('NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC)),
    CHECK (nav_amount NOT IN ('NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC)),
    CHECK (length(trim(strategy_id)) > 0),
    CHECK (length(trim(sleeve_id)) > 0),
    CHECK (length(trim(currency)) > 0),
    CHECK (length(trim(attribution_model_version)) > 0),
    CHECK (length(trim(benchmark_id)) > 0),
    CHECK (length(trim(source_fill_set_id)) > 0),
    CHECK (length(trim(run_id)) > 0),
    CHECK (
        pnl_component NOT IN ('commissions','slippage','financing')
        OR amount >= 0
    )
);

CREATE OR REPLACE VIEW fact.v_pnl_identity AS
SELECT
    as_of,
    strategy_id,
    sleeve_id,
    instrument_id,
    environment,
    MAX(currency) AS currency,
    MIN(nav_amount) AS nav_amount,
    COUNT(DISTINCT nav_amount) AS nav_value_count,
    COUNT(*) FILTER (WHERE pnl_component = 'gross_pnl') AS gross_pnl_count,
    COUNT(*) FILTER (
        WHERE pnl_component = 'pnl_residual'
    ) AS reported_residual_count,
    COUNT(DISTINCT currency) AS currency_value_count,
    COUNT(DISTINCT (
        attribution_model_version,
        benchmark_id,
        source_fill_set_id,
        run_id,
        derivation_id
    )) AS lineage_value_count,
    SUM(amount) FILTER (WHERE pnl_component = 'gross_pnl') AS gross_pnl,
    COALESCE(SUM(amount) FILTER (WHERE pnl_component = 'pnl_beta'), 0)
      + COALESCE(SUM(amount) FILTER (WHERE pnl_component = 'pnl_timing'), 0)
      + COALESCE(SUM(amount) FILTER (WHERE pnl_component = 'pnl_carry'), 0)
      - COALESCE(SUM(amount) FILTER (WHERE pnl_component = 'commissions'), 0)
      - COALESCE(SUM(amount) FILTER (WHERE pnl_component = 'slippage'), 0)
      - COALESCE(SUM(amount) FILTER (WHERE pnl_component = 'financing'), 0)
        AS reconstructed_gross_pnl,
    COALESCE(SUM(amount) FILTER (WHERE pnl_component = 'pnl_residual'), 0)
        AS reported_residual,
    (
        COALESCE(SUM(amount) FILTER (WHERE pnl_component = 'gross_pnl'), 0)
        - (
          COALESCE(SUM(amount) FILTER (WHERE pnl_component = 'pnl_beta'), 0)
          + COALESCE(SUM(amount) FILTER (WHERE pnl_component = 'pnl_timing'), 0)
          + COALESCE(SUM(amount) FILTER (WHERE pnl_component = 'pnl_carry'), 0)
          - COALESCE(SUM(amount) FILTER (WHERE pnl_component = 'commissions'), 0)
          - COALESCE(SUM(amount) FILTER (WHERE pnl_component = 'slippage'), 0)
          - COALESCE(SUM(amount) FILTER (WHERE pnl_component = 'financing'), 0)
        )
    ) AS calculated_residual,
    ABS(
        (
            COALESCE(SUM(amount) FILTER (WHERE pnl_component = 'gross_pnl'), 0)
            - (
              COALESCE(SUM(amount) FILTER (WHERE pnl_component = 'pnl_beta'), 0)
              + COALESCE(SUM(amount) FILTER (WHERE pnl_component = 'pnl_timing'), 0)
              + COALESCE(SUM(amount) FILTER (WHERE pnl_component = 'pnl_carry'), 0)
              - COALESCE(SUM(amount) FILTER (WHERE pnl_component = 'commissions'), 0)
              - COALESCE(SUM(amount) FILTER (WHERE pnl_component = 'slippage'), 0)
              - COALESCE(SUM(amount) FILTER (WHERE pnl_component = 'financing'), 0)
            )
        )
        - COALESCE(SUM(amount) FILTER (WHERE pnl_component = 'pnl_residual'), 0)
    ) AS identity_error,
    ABS(COALESCE(SUM(amount) FILTER (WHERE pnl_component = 'pnl_residual'), 0))
      / NULLIF(MIN(nav_amount), 0) AS residual_nav_fraction
FROM fact.pnl
GROUP BY as_of, strategy_id, sleeve_id, instrument_id, environment;

CREATE OR REPLACE FUNCTION fact.assert_pnl_identity(
    p_as_of TIMESTAMPTZ,
    p_strategy_id TEXT,
    p_sleeve_id TEXT,
    p_instrument_id UUID,
    p_environment TEXT,
    p_tolerance NUMERIC
) RETURNS VOID AS $$
DECLARE
    r fact.v_pnl_identity%ROWTYPE;
BEGIN
    IF p_tolerance IS NULL
       OR p_tolerance IN (
            'NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC
       )
       OR p_tolerance < 0
       OR p_tolerance > 1 THEN
        RAISE EXCEPTION 'PnL tolerance must be finite and within [0,1]';
    END IF;
    SELECT * INTO r
    FROM fact.v_pnl_identity
    WHERE as_of = p_as_of
      AND strategy_id = p_strategy_id
      AND sleeve_id = p_sleeve_id
      AND instrument_id = p_instrument_id
      AND environment = p_environment;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'PnL identity grain not found';
    END IF;
    IF r.nav_value_count <> 1 THEN
        RAISE EXCEPTION 'PnL identity grain has inconsistent NAV values';
    END IF;
    IF r.gross_pnl_count <> 1 OR r.reported_residual_count <> 1 THEN
        RAISE EXCEPTION
            'PnL identity requires exactly one gross_pnl and one pnl_residual';
    END IF;
    IF r.currency_value_count <> 1 THEN
        RAISE EXCEPTION 'PnL identity grain mixes currencies';
    END IF;
    IF r.lineage_value_count <> 1 THEN
        RAISE EXCEPTION 'PnL identity grain mixes derivation lineage';
    END IF;
    IF r.identity_error > 0.00000001 * GREATEST(ABS(r.gross_pnl), 1) THEN
        RAISE EXCEPTION 'PnL declared residual does not close identity: error %',
            r.identity_error;
    END IF;
    IF r.residual_nav_fraction > p_tolerance THEN
        RAISE EXCEPTION 'PnL residual/NAV % exceeds tolerance %',
            r.residual_nav_fraction, p_tolerance;
    END IF;
END;
$$ LANGUAGE plpgsql;

CREATE INDEX IF NOT EXISTS idx_fact_position_sleeve_time
    ON fact.position (strategy_id, sleeve_id, environment, as_of DESC);
CREATE INDEX IF NOT EXISTS idx_fact_pnl_sleeve_time
    ON fact.pnl (strategy_id, sleeve_id, environment, as_of DESC);

DROP TRIGGER IF EXISTS trg_fact_position_immutable ON fact.position;
CREATE TRIGGER trg_fact_position_immutable
    BEFORE UPDATE OR DELETE ON fact.position
    FOR EACH ROW EXECUTE FUNCTION control.block_append_only_mutation();
DROP TRIGGER IF EXISTS trg_fact_pnl_immutable ON fact.pnl;
CREATE TRIGGER trg_fact_pnl_immutable
    BEFORE UPDATE OR DELETE ON fact.pnl
    FOR EACH ROW EXECUTE FUNCTION control.block_append_only_mutation();
DROP TRIGGER IF EXISTS trg_fact_position_no_truncate ON fact.position;
CREATE TRIGGER trg_fact_position_no_truncate
    BEFORE TRUNCATE ON fact.position
    FOR EACH STATEMENT EXECUTE FUNCTION control.block_append_only_mutation();
DROP TRIGGER IF EXISTS trg_fact_pnl_no_truncate ON fact.pnl;
CREATE TRIGGER trg_fact_pnl_no_truncate
    BEFORE TRUNCATE ON fact.pnl
    FOR EACH STATEMENT EXECUTE FUNCTION control.block_append_only_mutation();
