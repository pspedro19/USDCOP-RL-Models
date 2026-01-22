-- =====================================================
-- USDCOP Trading System - Macro Indicators Schema
-- 37-column macroeconomic indicators table
-- Source: data/pipeline/05_resampling/output/MACRO_DAILY_CONSOLIDATED.csv
-- =====================================================

-- Create macro_indicators_daily table with all 37 macroeconomic columns
-- This table stores daily macroeconomic data used for feature engineering
CREATE TABLE IF NOT EXISTS macro_indicators_daily (
    fecha DATE PRIMARY KEY,

    -- Commodities (4 columns)
    comm_agri_coffee_glb_d_coffee DECIMAL(10,2),     -- Coffee price
    comm_metal_gold_glb_d_gold DECIMAL(10,2),        -- Gold price
    comm_oil_brent_glb_d_brent DECIMAL(10,2),        -- Brent crude oil
    comm_oil_wti_glb_d_wti DECIMAL(10,2),            -- WTI crude oil

    -- Country Risk/Sentiment - Colombia (3 columns)
    crsk_sentiment_cci_col_m_cci DECIMAL(10,2),      -- Consumer Confidence Index
    crsk_sentiment_ici_col_m_ici DECIMAL(10,2),      -- Industrial Confidence Index
    crsk_spread_embi_col_d_embi DECIMAL(10,2),       -- EMBI spread (sovereign risk)

    -- Equity Index - Colombia (1 column)
    eqty_index_colcap_col_d_colcap DECIMAL(12,2),    -- COLCAP index

    -- Financial - Bond Yields (4 columns)
    finc_bond_yield10y_col_d_col10y DECIMAL(6,3),    -- Colombia 10Y bond yield
    finc_bond_yield10y_usa_d_ust10y DECIMAL(6,3),    -- US Treasury 10Y yield
    finc_bond_yield2y_usa_d_dgs2 DECIMAL(6,3),       -- US Treasury 2Y yield
    finc_bond_yield5y_col_d_col5y DECIMAL(6,3),      -- Colombia 5Y bond yield

    -- Financial - Interest Rates (1 column)
    finc_rate_ibr_overnight_col_d_ibr DECIMAL(6,3),  -- Colombia IBR overnight rate

    -- Foreign Trade (3 columns)
    ftrd_exports_total_col_m_expusd DECIMAL(12,2),   -- Total exports (USD millions)
    ftrd_imports_total_col_m_impusd DECIMAL(12,2),   -- Total imports (USD millions)
    ftrd_terms_trade_col_m_tot DECIMAL(10,2),        -- Terms of trade index

    -- FX Rates (4 columns)
    fxrt_index_dxy_usa_d_dxy DECIMAL(8,4),           -- US Dollar Index (DXY)
    fxrt_reer_bilateral_col_m_itcr DECIMAL(10,4),    -- Real Effective Exchange Rate
    fxrt_spot_usdclp_chl_d_usdclp DECIMAL(10,2),     -- USD/CLP spot rate
    fxrt_spot_usdmxn_mex_d_usdmxn DECIMAL(10,4),     -- USD/MXN spot rate

    -- GDP - USA (1 column)
    gdpp_real_gdp_usa_q_gdp_q DECIMAL(12,2),         -- US Real GDP quarterly

    -- Inflation (5 columns)
    infl_cpi_all_usa_m_cpiaucsl DECIMAL(10,3),       -- US CPI All Items
    infl_cpi_core_usa_m_cpilfesl DECIMAL(10,3),      -- US Core CPI
    infl_cpi_total_col_m_ipccol DECIMAL(10,2),       -- Colombia CPI
    infl_pce_usa_m_pcepi DECIMAL(10,3),              -- US PCE Price Index

    -- Labor Market - USA (1 column)
    labr_unemployment_usa_m_unrate DECIMAL(6,2),     -- US Unemployment Rate

    -- Money Supply (1 column)
    mnys_m2_supply_usa_m_m2sl DECIMAL(14,2),         -- US M2 Money Supply

    -- Policy Rates (4 columns)
    polr_fed_funds_usa_m_fedfunds DECIMAL(6,3),      -- Federal Funds Rate
    polr_policy_rate_col_d_tpm DECIMAL(6,3),         -- Colombia Policy Rate (daily)
    polr_policy_rate_col_m_tpm DECIMAL(6,3),         -- Colombia Policy Rate (monthly)
    polr_prime_rate_usa_d_prime DECIMAL(6,3),        -- US Prime Rate

    -- Production (1 column)
    prod_industrial_usa_m_indpro DECIMAL(10,3),      -- US Industrial Production

    -- Balance of Payments (4 columns)
    rsbp_current_account_col_q_cacct_q DECIMAL(12,2),   -- Current Account Balance
    rsbp_fdi_inflow_col_q_fdiin_q DECIMAL(12,2),        -- FDI Inflows
    rsbp_fdi_outflow_col_q_fdiout_q DECIMAL(12,2),      -- FDI Outflows
    rsbp_reserves_international_col_m_resint DECIMAL(14,2), -- International Reserves

    -- Sentiment (1 column)
    sent_consumer_usa_m_umcsent DECIMAL(8,2),        -- US Consumer Sentiment

    -- Volatility (1 column)
    volt_vix_usa_d_vix DECIMAL(8,2),                 -- VIX Volatility Index

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- L0 DAG Required Fields (Data acquisition tracking)
    release_date TIMESTAMP WITH TIME ZONE,          -- When data was released/published
    ffilled_from_date DATE                          -- Source date for forward-filled values
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_macro_daily_fecha ON macro_indicators_daily (fecha DESC);
CREATE INDEX IF NOT EXISTS idx_macro_daily_dxy ON macro_indicators_daily (fxrt_index_dxy_usa_d_dxy);
CREATE INDEX IF NOT EXISTS idx_macro_daily_vix ON macro_indicators_daily (volt_vix_usa_d_vix);
CREATE INDEX IF NOT EXISTS idx_macro_daily_embi ON macro_indicators_daily (crsk_spread_embi_col_d_embi);

-- Create view for inference features (the 7 columns used in RL model)
CREATE OR REPLACE VIEW inference_macro_features AS
SELECT
    fecha,
    fxrt_index_dxy_usa_d_dxy AS dxy,
    volt_vix_usa_d_vix AS vix,
    crsk_spread_embi_col_d_embi AS embi,
    fxrt_spot_usdmxn_mex_d_usdmxn AS usdmxn,
    comm_oil_wti_glb_d_wti AS wti,
    finc_bond_yield10y_usa_d_ust10y AS ust10y,
    polr_fed_funds_usa_m_fedfunds AS fedfunds
FROM macro_indicators_daily
ORDER BY fecha DESC;

-- Create latest macro data view
CREATE OR REPLACE VIEW latest_macro AS
SELECT * FROM macro_indicators_daily
ORDER BY fecha DESC
LIMIT 1;

-- =============================================================================
-- COMPATIBILITY VIEW: macro_daily_simple
-- =============================================================================
-- This view provides simple column names for backward compatibility and
-- easier use in SQL queries. Maps long technical column names to short aliases.
-- Used by inference features views and other components.

CREATE OR REPLACE VIEW macro_daily_simple AS
SELECT
    fecha AS date,
    -- FX and Indices
    fxrt_index_dxy_usa_d_dxy AS dxy,
    volt_vix_usa_d_vix AS vix,
    fxrt_spot_usdmxn_mex_d_usdmxn AS usdmxn,
    fxrt_spot_usdclp_chl_d_usdclp AS usdclp,
    -- Country Risk
    crsk_spread_embi_col_d_embi AS embi,
    -- Commodities
    comm_oil_brent_glb_d_brent AS brent,
    comm_oil_wti_glb_d_wti AS wti,
    comm_metal_gold_glb_d_gold AS gold,
    comm_agri_coffee_glb_d_coffee AS coffee,
    -- Bond Yields
    finc_bond_yield2y_usa_d_dgs2 AS treasury_2y,
    finc_bond_yield10y_usa_d_ust10y AS treasury_10y,
    finc_bond_yield10y_col_d_col10y AS colombia_10y,
    finc_bond_yield5y_col_d_col5y AS colombia_5y,
    -- Policy Rates
    polr_fed_funds_usa_m_fedfunds AS fedfunds,
    polr_policy_rate_col_d_tpm AS colombia_rate,
    polr_prime_rate_usa_d_prime AS prime_rate,
    -- Colombia Local
    eqty_index_colcap_col_d_colcap AS colcap,
    finc_rate_ibr_overnight_col_d_ibr AS ibr,
    -- Inflation
    infl_cpi_all_usa_m_cpiaucsl AS us_cpi,
    infl_cpi_total_col_m_ipccol AS col_cpi,
    -- Other US
    labr_unemployment_usa_m_unrate AS us_unemployment,
    sent_consumer_usa_m_umcsent AS us_consumer_sentiment,
    prod_industrial_usa_m_indpro AS us_industrial_production,
    -- Balance of Payments
    rsbp_reserves_international_col_m_resint AS col_reserves,
    -- Metadata
    created_at,
    updated_at
FROM macro_indicators_daily;

COMMENT ON VIEW macro_daily_simple IS
    'Simplified column names for macro_indicators_daily. Use for queries requiring short aliases.';

-- Confirmation message
DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'Macro Indicators Table Created Successfully';
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'Table: macro_indicators_daily (37 macroeconomic columns)';
    RAISE NOTICE 'View: inference_macro_features (7 RL model features)';
    RAISE NOTICE 'View: macro_daily_simple (simplified column names)';
    RAISE NOTICE 'View: latest_macro (most recent row)';
    RAISE NOTICE 'Data Source: MACRO_DAILY_CONSOLIDATED.csv';
    RAISE NOTICE '============================================================';
    RAISE NOTICE '';
END $$;
