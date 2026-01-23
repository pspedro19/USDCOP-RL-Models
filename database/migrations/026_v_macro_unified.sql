-- Migration: 026_v_macro_unified
-- Description: Create unified macro view with friendly column names
-- Contract: CTR-DATA-006
-- Version: 2.0.0
--
-- Purpose:
-- This view normalizes column names from macro_indicators_daily to provide
-- consistent, friendly names for both RL and Forecasting pipelines.
--
-- Usage:
--   SELECT * FROM v_macro_unified WHERE fecha >= '2024-01-01';

-- Drop existing view if exists
DROP VIEW IF EXISTS public.v_macro_unified CASCADE;

-- Create unified macro view
CREATE OR REPLACE VIEW public.v_macro_unified AS
SELECT
    -- Date (primary key)
    fecha,

    -- =================================================================
    -- FOREX (4 columns)
    -- =================================================================
    fxrt_index_dxy_usa_d_dxy AS dxy,                    -- US Dollar Index
    fxrt_spot_usdmxn_mex_d_usdmxn AS usdmxn,           -- USD/MXN
    fxrt_spot_usdclp_chl_d_usdclp AS usdclp,           -- USD/CLP
    fxrt_reer_bilateral_col_m_itcr AS itcr,            -- Colombia REER

    -- =================================================================
    -- COMMODITIES (4 columns)
    -- =================================================================
    comm_oil_wti_glb_d_wti AS wti,                      -- WTI Crude Oil
    comm_oil_brent_glb_d_brent AS brent,                -- Brent Crude Oil
    comm_metal_gold_glb_d_gold AS gold,                 -- Gold
    comm_agri_coffee_glb_d_coffee AS coffee,            -- Coffee

    -- =================================================================
    -- VOLATILITY (1 column)
    -- =================================================================
    volt_vix_usa_d_vix AS vix,                          -- VIX Index

    -- =================================================================
    -- CREDIT RISK (1 column)
    -- =================================================================
    crsk_spread_embi_col_d_embi AS embi,               -- EMBI Colombia

    -- =================================================================
    -- INTEREST RATES (5 columns)
    -- =================================================================
    finc_bond_yield10y_usa_d_ust10y AS ust10y,         -- US 10Y Treasury
    finc_bond_yield2y_usa_d_dgs2 AS ust2y,             -- US 2Y Treasury
    finc_bond_yield10y_col_d_col10y AS col10y,         -- Colombia 10Y Bond
    finc_bond_yield5y_col_d_col5y AS col5y,            -- Colombia 5Y Bond
    finc_rate_ibr_overnight_col_d_ibr AS ibr,          -- Colombia Overnight Rate

    -- =================================================================
    -- POLICY RATES (4 columns)
    -- =================================================================
    polr_policy_rate_col_d_tpm AS tpm,                 -- Colombia Policy Rate (daily)
    polr_policy_rate_col_m_tpm AS tpm_m,               -- Colombia Policy Rate (monthly)
    polr_fed_funds_usa_m_fedfunds AS fedfunds,         -- US Fed Funds Rate
    polr_prime_rate_usa_d_prime AS prime,              -- US Prime Rate

    -- =================================================================
    -- EQUITY (1 column)
    -- =================================================================
    eqty_index_colcap_col_d_colcap AS colcap,          -- COLCAP Index

    -- =================================================================
    -- INFLATION (4 columns)
    -- =================================================================
    infl_cpi_total_col_m_ipccol AS cpi_col,            -- Colombia CPI
    infl_cpi_all_usa_m_cpiaucsl AS cpi_usa,            -- US CPI
    infl_cpi_core_usa_m_cpilfesl AS core_cpi_usa,      -- US Core CPI
    infl_pce_usa_m_pcepi AS pce,                        -- US PCE

    -- =================================================================
    -- LABOR (1 column)
    -- =================================================================
    labr_unemployment_usa_m_unrate AS unemployment,     -- US Unemployment Rate

    -- =================================================================
    -- MONEY SUPPLY (1 column)
    -- =================================================================
    mnys_m2_supply_usa_m_m2sl AS m2,                    -- US M2 Money Supply

    -- =================================================================
    -- GDP & PRODUCTION (2 columns)
    -- =================================================================
    gdpp_real_gdp_usa_q_gdp_q AS gdp_usa,              -- US Real GDP
    prod_industrial_usa_m_indpro AS indpro,             -- US Industrial Production

    -- =================================================================
    -- BALANCE OF PAYMENTS (4 columns)
    -- =================================================================
    rsbp_current_account_col_q_cacct_q AS current_account,  -- Current Account
    rsbp_fdi_inflow_col_q_fdiin_q AS fdi_inflow,           -- FDI Inflow
    rsbp_fdi_outflow_col_q_fdiout_q AS fdi_outflow,        -- FDI Outflow
    rsbp_reserves_international_col_m_resint AS reserves,   -- International Reserves

    -- =================================================================
    -- FOREIGN TRADE (3 columns)
    -- =================================================================
    ftrd_exports_total_col_m_expusd AS exports,        -- Colombia Exports
    ftrd_imports_total_col_m_impusd AS imports,        -- Colombia Imports
    ftrd_terms_trade_col_m_tot AS terms_of_trade,      -- Terms of Trade

    -- =================================================================
    -- SENTIMENT (3 columns)
    -- =================================================================
    sent_consumer_usa_m_umcsent AS consumer_sentiment, -- US Consumer Sentiment
    crsk_sentiment_cci_col_m_cci AS cci_col,           -- Colombia Consumer Confidence
    crsk_sentiment_ici_col_m_ici AS ici_col,           -- Colombia Industrial Confidence

    -- =================================================================
    -- CALCULATED COLUMNS (useful for both pipelines)
    -- =================================================================
    -- Rate Spread: Colombia 10Y - US 10Y
    COALESCE(finc_bond_yield10y_col_d_col10y, 0) -
        COALESCE(finc_bond_yield10y_usa_d_ust10y, 0) AS rate_spread,

    -- Yield Curve Slope: US 10Y - US 2Y
    COALESCE(finc_bond_yield10y_usa_d_ust10y, 0) -
        COALESCE(finc_bond_yield2y_usa_d_dgs2, 0) AS yield_curve_slope,

    -- Oil Spread: Brent - WTI
    COALESCE(comm_oil_brent_glb_d_brent, 0) -
        COALESCE(comm_oil_wti_glb_d_wti, 0) AS oil_spread,

    -- =================================================================
    -- METADATA
    -- =================================================================
    created_at,
    updated_at

FROM macro_indicators_daily
WHERE fecha >= '2015-01-01'
ORDER BY fecha;

-- Add comment
COMMENT ON VIEW public.v_macro_unified IS
    'SSOT: Unified macro data view with friendly column names for RL and Forecasting pipelines. Contract: CTR-DATA-006';

-- Grant permissions
GRANT SELECT ON public.v_macro_unified TO PUBLIC;

-- Create index hint for common queries
-- Note: Views don't support indexes, but the underlying table should have fecha indexed
-- This is just documentation
COMMENT ON COLUMN public.v_macro_unified.fecha IS
    'Primary date column. Underlying table has index on this column.';

-- =================================================================
-- VERIFICATION QUERY
-- =================================================================
-- Run after migration to verify:
-- SELECT
--     COUNT(*) as total_rows,
--     MIN(fecha) as min_date,
--     MAX(fecha) as max_date,
--     COUNT(dxy) as dxy_count,
--     COUNT(vix) as vix_count,
--     COUNT(embi) as embi_count
-- FROM v_macro_unified;
