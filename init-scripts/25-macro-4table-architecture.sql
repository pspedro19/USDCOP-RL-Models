-- =============================================================================
-- USDCOP Trading System - 4-Table Macro Architecture Migration
-- =============================================================================
-- Contract: CTR-L0-4TABLE-001
--
-- Implements the 4-table architecture:
--   1. usdcop_m5_ohlcv (existing) - 5min OHLCV bars
--   2. macro_indicators_daily    - 18 daily variables
--   3. macro_indicators_monthly  - 18 monthly variables
--   4. macro_indicators_quarterly - 4 quarterly variables
--
-- Anti-Leakage Integration:
--   - FFill limits from macro_variables_ssot.yaml
--   - Publication delay tracking
--   - Data staleness monitoring
--
-- Version: 1.0.0
-- Date: 2026-02-01
-- =============================================================================

-- =============================================================================
-- TABLE 1: macro_indicators_daily (18 daily variables)
-- =============================================================================
-- This table is recreated to ONLY contain daily-frequency variables.
-- Drops monthly/quarterly columns that don't belong here.

-- First, backup existing data
CREATE TABLE IF NOT EXISTS macro_indicators_daily_backup AS
SELECT * FROM macro_indicators_daily;

-- Drop and recreate with ONLY daily columns
DROP TABLE IF EXISTS macro_indicators_daily CASCADE;

CREATE TABLE macro_indicators_daily (
    fecha DATE PRIMARY KEY,

    -- FX Indices (3 columns)
    fxrt_index_dxy_usa_d_dxy DECIMAL(8,4),           -- DXY Dollar Index
    fxrt_spot_usdmxn_mex_d_usdmxn DECIMAL(10,4),     -- USD/MXN spot
    fxrt_spot_usdclp_chl_d_usdclp DECIMAL(10,2),     -- USD/CLP spot
    fxrt_spot_usdcop_col_d_usdcop DECIMAL(10,2),     -- USD/COP spot (reference)

    -- Volatility (1 column)
    volt_vix_usa_d_vix DECIMAL(8,2),                 -- VIX Index

    -- Country Risk (1 column)
    crsk_spread_embi_col_d_embi DECIMAL(10,2),       -- EMBI Colombia spread

    -- Commodities (4 columns)
    comm_oil_wti_glb_d_wti DECIMAL(10,2),            -- WTI crude oil
    comm_oil_brent_glb_d_brent DECIMAL(10,2),        -- Brent crude oil
    comm_metal_gold_glb_d_gold DECIMAL(10,2),        -- Gold
    comm_agri_coffee_glb_d_coffee DECIMAL(10,2),     -- Coffee

    -- US Bond Yields (2 columns)
    finc_bond_yield10y_usa_d_ust10y DECIMAL(6,3),    -- UST 10Y
    finc_bond_yield2y_usa_d_dgs2 DECIMAL(6,3),       -- UST 2Y

    -- US Prime Rate (1 column)
    polr_prime_rate_usa_d_prime DECIMAL(6,3),        -- Prime rate

    -- Colombia Rates (2 columns)
    finc_rate_ibr_overnight_col_d_ibr DECIMAL(6,3),  -- IBR overnight
    polr_policy_rate_col_d_tpm DECIMAL(6,3),         -- TPM policy rate

    -- Colombia Bonds (2 columns)
    finc_bond_yield10y_col_d_col10y DECIMAL(6,3),    -- Colombia 10Y
    finc_bond_yield5y_col_d_col5y DECIMAL(6,3),      -- Colombia 5Y

    -- Colombia Equity (1 column)
    eqty_index_colcap_col_d_colcap DECIMAL(12,2),    -- COLCAP index

    -- Metadata (required for anti-leakage)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    is_complete BOOLEAN DEFAULT FALSE,              -- All critical vars present

    -- Anti-Leakage Tracking
    ffill_count INTEGER DEFAULT 0,                  -- Days forward-filled
    source_date DATE                                -- Original publication date
);

-- Indexes for daily table
CREATE INDEX idx_macro_daily_fecha ON macro_indicators_daily (fecha DESC);
CREATE INDEX idx_macro_daily_dxy ON macro_indicators_daily (fxrt_index_dxy_usa_d_dxy);
CREATE INDEX idx_macro_daily_vix ON macro_indicators_daily (volt_vix_usa_d_vix);
CREATE INDEX idx_macro_daily_complete ON macro_indicators_daily (is_complete) WHERE is_complete = TRUE;

COMMENT ON TABLE macro_indicators_daily IS
'L0 Daily macro variables (18 columns). FFill limit: 5 days. Contract: CTR-L0-4TABLE-001';


-- =============================================================================
-- TABLE 2: macro_indicators_monthly (18 monthly variables)
-- =============================================================================

CREATE TABLE IF NOT EXISTS macro_indicators_monthly (
    fecha DATE PRIMARY KEY,                          -- First day of month

    -- US Policy Rates (1 column)
    polr_fed_funds_usa_m_fedfunds DECIMAL(6,3),     -- Fed Funds Rate

    -- US Inflation (3 columns)
    infl_cpi_all_usa_m_cpiaucsl DECIMAL(10,3),      -- CPI All Items
    infl_cpi_core_usa_m_cpilfesl DECIMAL(10,3),     -- Core CPI
    infl_pce_usa_m_pcepi DECIMAL(10,3),             -- PCE Price Index

    -- US Labor (1 column)
    labr_unemployment_usa_m_unrate DECIMAL(6,2),    -- Unemployment Rate

    -- US Production (1 column)
    prod_industrial_usa_m_indpro DECIMAL(10,3),     -- Industrial Production

    -- US Money Supply (1 column)
    mnys_m2_supply_usa_m_m2sl DECIMAL(14,2),        -- M2 Money Stock

    -- US Sentiment (1 column)
    sent_consumer_usa_m_umcsent DECIMAL(8,2),       -- Consumer Sentiment

    -- Colombia REER (2 columns)
    fxrt_reer_bilateral_col_m_itcr DECIMAL(10,4),   -- ITCR (Real Exchange Rate)
    fxrt_reer_bilateral_usa_col_m_itcr_usa DECIMAL(10,4), -- ITCR vs USA

    -- Colombia Terms of Trade (1 column)
    ftrd_terms_trade_col_m_tot DECIMAL(10,2),       -- Terms of Trade

    -- Colombia Reserves (1 column)
    rsbp_reserves_international_col_m_resint DECIMAL(14,2), -- International Reserves

    -- Colombia Inflation (1 column)
    infl_cpi_total_col_m_ipccol DECIMAL(10,2),      -- CPI Colombia

    -- Colombia Expectations (1 column)
    infl_exp_eof_col_m_infexp DECIMAL(6,2),         -- Inflation Expectation (Fedesarrollo)

    -- Colombia Sentiment (2 columns)
    crsk_sentiment_cci_col_m_cci DECIMAL(10,2),     -- Consumer Confidence Index
    crsk_sentiment_ici_col_m_ici DECIMAL(10,2),     -- Industrial Confidence Index

    -- Colombia Trade (2 columns)
    ftrd_exports_total_col_m_expusd DECIMAL(12,2),  -- Exports Total (USD millions)
    ftrd_imports_total_col_m_impusd DECIMAL(12,2),  -- Imports Total (USD millions)

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    is_complete BOOLEAN DEFAULT FALSE,

    -- Anti-Leakage Tracking (FFill limit: 35 days)
    ffill_count INTEGER DEFAULT 0,
    source_date DATE,
    publication_date DATE                           -- When data was actually published
);

-- Indexes for monthly table
CREATE INDEX idx_macro_monthly_fecha ON macro_indicators_monthly (fecha DESC);
CREATE INDEX idx_macro_monthly_complete ON macro_indicators_monthly (is_complete) WHERE is_complete = TRUE;

COMMENT ON TABLE macro_indicators_monthly IS
'L0 Monthly macro variables (18 columns). FFill limit: 35 days. Contract: CTR-L0-4TABLE-001';


-- =============================================================================
-- TABLE 3: macro_indicators_quarterly (4 quarterly variables)
-- =============================================================================

CREATE TABLE IF NOT EXISTS macro_indicators_quarterly (
    fecha DATE PRIMARY KEY,                          -- First day of quarter

    -- US GDP (1 column)
    gdpp_real_gdp_usa_q_gdp_q DECIMAL(12,2),        -- Real GDP USA

    -- Colombia BOP (3 columns)
    rsbp_current_account_col_q_cacct DECIMAL(12,2), -- Current Account Balance
    rsbp_fdi_inflow_col_q_fdiin DECIMAL(12,2),      -- FDI Inflows (IED)
    rsbp_fdi_outflow_col_q_fdiout DECIMAL(12,2),    -- FDI Outflows (IDCE)

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    is_complete BOOLEAN DEFAULT FALSE,

    -- Anti-Leakage Tracking (FFill limit: 95 days)
    ffill_count INTEGER DEFAULT 0,
    source_date DATE,
    publication_date DATE                           -- ~90 days after quarter end
);

-- Indexes for quarterly table
CREATE INDEX idx_macro_quarterly_fecha ON macro_indicators_quarterly (fecha DESC);

COMMENT ON TABLE macro_indicators_quarterly IS
'L0 Quarterly macro variables (4 columns). FFill limit: 95 days. Contract: CTR-L0-4TABLE-001';


-- =============================================================================
-- MIGRATION: Restore data from backup to new tables
-- =============================================================================

-- Restore daily data (only daily columns)
INSERT INTO macro_indicators_daily (
    fecha,
    fxrt_index_dxy_usa_d_dxy,
    fxrt_spot_usdmxn_mex_d_usdmxn,
    fxrt_spot_usdclp_chl_d_usdclp,
    volt_vix_usa_d_vix,
    crsk_spread_embi_col_d_embi,
    comm_oil_wti_glb_d_wti,
    comm_oil_brent_glb_d_brent,
    comm_metal_gold_glb_d_gold,
    comm_agri_coffee_glb_d_coffee,
    finc_bond_yield10y_usa_d_ust10y,
    finc_bond_yield2y_usa_d_dgs2,
    polr_prime_rate_usa_d_prime,
    finc_rate_ibr_overnight_col_d_ibr,
    polr_policy_rate_col_d_tpm,
    finc_bond_yield10y_col_d_col10y,
    finc_bond_yield5y_col_d_col5y,
    eqty_index_colcap_col_d_colcap,
    created_at,
    updated_at
)
SELECT
    fecha,
    fxrt_index_dxy_usa_d_dxy,
    fxrt_spot_usdmxn_mex_d_usdmxn,
    fxrt_spot_usdclp_chl_d_usdclp,
    volt_vix_usa_d_vix,
    crsk_spread_embi_col_d_embi,
    comm_oil_wti_glb_d_wti,
    comm_oil_brent_glb_d_brent,
    comm_metal_gold_glb_d_gold,
    comm_agri_coffee_glb_d_coffee,
    finc_bond_yield10y_usa_d_ust10y,
    finc_bond_yield2y_usa_d_dgs2,
    polr_prime_rate_usa_d_prime,
    finc_rate_ibr_overnight_col_d_ibr,
    polr_policy_rate_col_d_tpm,
    finc_bond_yield10y_col_d_col10y,
    finc_bond_yield5y_col_d_col5y,
    eqty_index_colcap_col_d_colcap,
    COALESCE(created_at, NOW()),
    NOW()
FROM macro_indicators_daily_backup
ON CONFLICT (fecha) DO UPDATE SET
    updated_at = NOW();

-- Restore monthly data (extract unique months)
INSERT INTO macro_indicators_monthly (
    fecha,
    polr_fed_funds_usa_m_fedfunds,
    infl_cpi_all_usa_m_cpiaucsl,
    infl_cpi_core_usa_m_cpilfesl,
    infl_pce_usa_m_pcepi,
    labr_unemployment_usa_m_unrate,
    prod_industrial_usa_m_indpro,
    mnys_m2_supply_usa_m_m2sl,
    sent_consumer_usa_m_umcsent,
    fxrt_reer_bilateral_col_m_itcr,
    ftrd_terms_trade_col_m_tot,
    rsbp_reserves_international_col_m_resint,
    infl_cpi_total_col_m_ipccol,
    crsk_sentiment_cci_col_m_cci,
    crsk_sentiment_ici_col_m_ici,
    ftrd_exports_total_col_m_expusd,
    ftrd_imports_total_col_m_impusd
)
SELECT DISTINCT ON (DATE_TRUNC('month', fecha))
    DATE_TRUNC('month', fecha)::DATE,
    polr_fed_funds_usa_m_fedfunds,
    infl_cpi_all_usa_m_cpiaucsl,
    infl_cpi_core_usa_m_cpilfesl,
    infl_pce_usa_m_pcepi,
    labr_unemployment_usa_m_unrate,
    prod_industrial_usa_m_indpro,
    mnys_m2_supply_usa_m_m2sl,
    sent_consumer_usa_m_umcsent,
    fxrt_reer_bilateral_col_m_itcr,
    ftrd_terms_trade_col_m_tot,
    rsbp_reserves_international_col_m_resint,
    infl_cpi_total_col_m_ipccol,
    crsk_sentiment_cci_col_m_cci,
    crsk_sentiment_ici_col_m_ici,
    ftrd_exports_total_col_m_expusd,
    ftrd_imports_total_col_m_impusd
FROM macro_indicators_daily_backup
WHERE polr_fed_funds_usa_m_fedfunds IS NOT NULL
   OR infl_cpi_all_usa_m_cpiaucsl IS NOT NULL
   OR labr_unemployment_usa_m_unrate IS NOT NULL
ORDER BY DATE_TRUNC('month', fecha), fecha DESC
ON CONFLICT (fecha) DO NOTHING;

-- Restore quarterly data (extract unique quarters)
INSERT INTO macro_indicators_quarterly (
    fecha,
    gdpp_real_gdp_usa_q_gdp_q,
    rsbp_current_account_col_q_cacct,
    rsbp_fdi_inflow_col_q_fdiin,
    rsbp_fdi_outflow_col_q_fdiout
)
SELECT DISTINCT ON (DATE_TRUNC('quarter', fecha))
    DATE_TRUNC('quarter', fecha)::DATE,
    gdpp_real_gdp_usa_q_gdp_q,
    rsbp_current_account_col_q_cacct_q,
    rsbp_fdi_inflow_col_q_fdiin_q,
    rsbp_fdi_outflow_col_q_fdiout_q
FROM macro_indicators_daily_backup
WHERE gdpp_real_gdp_usa_q_gdp_q IS NOT NULL
   OR rsbp_current_account_col_q_cacct_q IS NOT NULL
ORDER BY DATE_TRUNC('quarter', fecha), fecha DESC
ON CONFLICT (fecha) DO NOTHING;


-- =============================================================================
-- VIEWS: Compatibility and Inference
-- =============================================================================

-- View: Combined macro data for L2 consumption (joins all 3 tables)
CREATE OR REPLACE VIEW macro_combined_for_l2 AS
SELECT
    d.fecha,
    -- Daily variables
    d.fxrt_index_dxy_usa_d_dxy,
    d.fxrt_spot_usdmxn_mex_d_usdmxn,
    d.fxrt_spot_usdclp_chl_d_usdclp,
    d.fxrt_spot_usdcop_col_d_usdcop,
    d.volt_vix_usa_d_vix,
    d.crsk_spread_embi_col_d_embi,
    d.comm_oil_wti_glb_d_wti,
    d.comm_oil_brent_glb_d_brent,
    d.comm_metal_gold_glb_d_gold,
    d.comm_agri_coffee_glb_d_coffee,
    d.finc_bond_yield10y_usa_d_ust10y,
    d.finc_bond_yield2y_usa_d_dgs2,
    d.polr_prime_rate_usa_d_prime,
    d.finc_rate_ibr_overnight_col_d_ibr,
    d.polr_policy_rate_col_d_tpm,
    d.finc_bond_yield10y_col_d_col10y,
    d.finc_bond_yield5y_col_d_col5y,
    d.eqty_index_colcap_col_d_colcap,
    -- Monthly variables (joined by month start)
    m.polr_fed_funds_usa_m_fedfunds,
    m.infl_cpi_all_usa_m_cpiaucsl,
    m.infl_cpi_core_usa_m_cpilfesl,
    m.infl_pce_usa_m_pcepi,
    m.labr_unemployment_usa_m_unrate,
    m.prod_industrial_usa_m_indpro,
    m.mnys_m2_supply_usa_m_m2sl,
    m.sent_consumer_usa_m_umcsent,
    m.fxrt_reer_bilateral_col_m_itcr,
    m.fxrt_reer_bilateral_usa_col_m_itcr_usa,
    m.ftrd_terms_trade_col_m_tot,
    m.rsbp_reserves_international_col_m_resint,
    m.infl_cpi_total_col_m_ipccol,
    m.infl_exp_eof_col_m_infexp,
    m.crsk_sentiment_cci_col_m_cci,
    m.crsk_sentiment_ici_col_m_ici,
    m.ftrd_exports_total_col_m_expusd,
    m.ftrd_imports_total_col_m_impusd,
    -- Quarterly variables (joined by quarter start)
    q.gdpp_real_gdp_usa_q_gdp_q,
    q.rsbp_current_account_col_q_cacct,
    q.rsbp_fdi_inflow_col_q_fdiin,
    q.rsbp_fdi_outflow_col_q_fdiout
FROM macro_indicators_daily d
LEFT JOIN macro_indicators_monthly m
    ON DATE_TRUNC('month', d.fecha)::DATE = m.fecha
LEFT JOIN macro_indicators_quarterly q
    ON DATE_TRUNC('quarter', d.fecha)::DATE = q.fecha;

COMMENT ON VIEW macro_combined_for_l2 IS
'Combined view of all 40 macro variables for L2 dataset builder consumption. Contract: CTR-L0-4TABLE-001';


-- View: Inference features (simplified names for realtime inference)
CREATE OR REPLACE VIEW inference_macro_features_v2 AS
SELECT
    fecha AS date,
    -- Critical features for RL model
    fxrt_index_dxy_usa_d_dxy AS dxy,
    volt_vix_usa_d_vix AS vix,
    crsk_spread_embi_col_d_embi AS embi,
    fxrt_spot_usdmxn_mex_d_usdmxn AS usdmxn,
    comm_oil_brent_glb_d_brent AS brent,
    finc_bond_yield10y_usa_d_ust10y AS ust10y,
    finc_bond_yield2y_usa_d_dgs2 AS ust2y,
    polr_policy_rate_col_d_tpm AS tpm_col,
    polr_fed_funds_usa_m_fedfunds AS fedfunds,
    is_complete
FROM macro_indicators_daily d
LEFT JOIN macro_indicators_monthly m
    ON DATE_TRUNC('month', d.fecha)::DATE = m.fecha
ORDER BY fecha DESC;

COMMENT ON VIEW inference_macro_features_v2 IS
'Simplified inference features for realtime model. Requires is_complete=TRUE for valid inference.';


-- =============================================================================
-- FUNCTIONS: Anti-Leakage Helpers
-- =============================================================================

-- Function: Get FFill limit for a table
CREATE OR REPLACE FUNCTION get_ffill_limit(table_name TEXT)
RETURNS INTEGER AS $$
BEGIN
    RETURN CASE table_name
        WHEN 'macro_indicators_daily' THEN 5
        WHEN 'macro_indicators_monthly' THEN 35
        WHEN 'macro_indicators_quarterly' THEN 95
        ELSE 5
    END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function: Check if data is stale (beyond FFill limit)
CREATE OR REPLACE FUNCTION is_data_stale(
    p_table_name TEXT,
    p_last_update DATE
) RETURNS BOOLEAN AS $$
DECLARE
    v_limit INTEGER;
    v_days_since INTEGER;
BEGIN
    v_limit := get_ffill_limit(p_table_name);
    v_days_since := CURRENT_DATE - p_last_update;
    RETURN v_days_since > v_limit;
END;
$$ LANGUAGE plpgsql STABLE;


-- =============================================================================
-- TRIGGERS: Auto-update timestamps
-- =============================================================================

CREATE OR REPLACE FUNCTION update_modified_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_macro_daily_updated ON macro_indicators_daily;
CREATE TRIGGER trg_macro_daily_updated
    BEFORE UPDATE ON macro_indicators_daily
    FOR EACH ROW EXECUTE FUNCTION update_modified_timestamp();

DROP TRIGGER IF EXISTS trg_macro_monthly_updated ON macro_indicators_monthly;
CREATE TRIGGER trg_macro_monthly_updated
    BEFORE UPDATE ON macro_indicators_monthly
    FOR EACH ROW EXECUTE FUNCTION update_modified_timestamp();

DROP TRIGGER IF EXISTS trg_macro_quarterly_updated ON macro_indicators_quarterly;
CREATE TRIGGER trg_macro_quarterly_updated
    BEFORE UPDATE ON macro_indicators_quarterly
    FOR EACH ROW EXECUTE FUNCTION update_modified_timestamp();


-- =============================================================================
-- SUMMARY
-- =============================================================================

DO $$
DECLARE
    daily_count INTEGER;
    monthly_count INTEGER;
    quarterly_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO daily_count FROM macro_indicators_daily;
    SELECT COUNT(*) INTO monthly_count FROM macro_indicators_monthly;
    SELECT COUNT(*) INTO quarterly_count FROM macro_indicators_quarterly;

    RAISE NOTICE '';
    RAISE NOTICE '============================================================';
    RAISE NOTICE '4-TABLE MACRO ARCHITECTURE - Migration Complete';
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'Contract: CTR-L0-4TABLE-001';
    RAISE NOTICE '';
    RAISE NOTICE 'Tables Created:';
    RAISE NOTICE '  1. macro_indicators_daily    (18 cols) - % rows', daily_count;
    RAISE NOTICE '  2. macro_indicators_monthly  (18 cols) - % rows', monthly_count;
    RAISE NOTICE '  3. macro_indicators_quarterly (4 cols) - % rows', quarterly_count;
    RAISE NOTICE '';
    RAISE NOTICE 'FFill Limits (anti-leakage):';
    RAISE NOTICE '  - Daily:     5 days';
    RAISE NOTICE '  - Monthly:  35 days';
    RAISE NOTICE '  - Quarterly: 95 days';
    RAISE NOTICE '';
    RAISE NOTICE 'Views Created:';
    RAISE NOTICE '  - macro_combined_for_l2: Combined view for L2 dataset builder';
    RAISE NOTICE '  - inference_macro_features_v2: Simplified inference features';
    RAISE NOTICE '';
    RAISE NOTICE 'Backup Table: macro_indicators_daily_backup';
    RAISE NOTICE '============================================================';
END $$;
