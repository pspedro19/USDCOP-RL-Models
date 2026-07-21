-- ============================================================================
-- Migration 062: monthly + quarterly macro wide views
-- Contract: CTR-MKT-CANON-001 (extends 060/061; operator request 2026-07-21:
--           "si hay datos macro mensuales entonces necesitamos una tabla mensual")
-- ============================================================================
-- Why monthly gets its OWN view instead of columns in the daily wide:
--
--  * Frequency routing is a governance invariant (FrequencyRoutedUpsertService):
--    one wide view per native frequency, never resampled/ffilled across
--    frequencies inside the contract layer. The daily-ffilled projection of a
--    monthly series erases the line between "the January print" and "January
--    smeared across February" — exactly the ambiguity these views exist to kill.
--
--  * Monthly is where the PIT problem BITES HARDEST: the June CPI is not
--    knowable on June 30 (DANE publishes ~July 5); Colombian trade balance
--    arrives ~7 weeks after month end. A naive month-anchor join leaks future.
--    Rule for any model-side join:
--
--        available_from = COALESCE(publication_date, conservative bound)
--
--    publication_date is stamped by ingestion going forward; historic rows are
--    NULL (unknown is honest — reconstructing it would fabricate PIT evidence),
--    so the view computes available_from_conservative as an UPPER bound on
--    knowability. Over-waiting can only cost signal freshness; under-waiting
--    fabricates alpha. Bounds (documented worst publishers):
--      monthly   fecha (month START) + 3 months   -- covers trade/ToT ~7 weeks post month-end
--      quarterly fecha (quarter END) + 3 months   -- covers COL current account
--
--  * Anchors as found in the MASTERs: monthly fecha = month START;
--    quarterly fecha = quarter END. Documented here, not silently normalized.
--
--  * _t1 = previous PERIOD's value (LAG over the native frequency), the
--    reporting-safe lag form. For model consumption at daily resolution the
--    asof-join on available_from is the rule; _t1 alone does NOT guarantee
--    availability for slow publishers.
-- ============================================================================

CREATE OR REPLACE VIEW market_macro_monthly_wide AS
SELECT
    m.fecha                                        AS month_start,
    (m.fecha + interval '1 month')::date           AS month_end,
    m.publication_date,
    COALESCE(m.publication_date,
             (m.fecha + interval '3 months')::date) AS available_from_conservative,
    m.is_complete,
    m.ffill_count,

    -- USA ---------------------------------------------------------------
    m.polr_fed_funds_usa_m_fedfunds  AS macro_fedfunds,
    LAG(m.polr_fed_funds_usa_m_fedfunds)  OVER w AS macro_fedfunds_t1,
    m.infl_cpi_all_usa_m_cpiaucsl    AS macro_cpi_usa,
    LAG(m.infl_cpi_all_usa_m_cpiaucsl)    OVER w AS macro_cpi_usa_t1,
    m.infl_pce_usa_m_pcepi           AS macro_pce_usa,
    LAG(m.infl_pce_usa_m_pcepi)           OVER w AS macro_pce_usa_t1,
    m.labr_unemployment_usa_m_unrate AS macro_unrate,
    LAG(m.labr_unemployment_usa_m_unrate) OVER w AS macro_unrate_t1,
    m.prod_industrial_usa_m_indpro   AS macro_indpro,
    LAG(m.prod_industrial_usa_m_indpro)   OVER w AS macro_indpro_t1,
    m.mnys_m2_supply_usa_m_m2sl      AS macro_m2_usa,
    LAG(m.mnys_m2_supply_usa_m_m2sl)      OVER w AS macro_m2_usa_t1,
    m.sent_consumer_usa_m_umcsent    AS macro_umcsent,
    LAG(m.sent_consumer_usa_m_umcsent)    OVER w AS macro_umcsent_t1,

    -- Colombia ------------------------------------------------------------
    m.infl_cpi_total_col_m_ipccol    AS macro_ipc_col,
    LAG(m.infl_cpi_total_col_m_ipccol)    OVER w AS macro_ipc_col_t1,
    m.fxrt_reer_bilateral_col_m_itcr AS macro_itcr,
    LAG(m.fxrt_reer_bilateral_col_m_itcr) OVER w AS macro_itcr_t1,
    m.rsbp_reserves_international_col_m_resint AS macro_resint,
    LAG(m.rsbp_reserves_international_col_m_resint) OVER w AS macro_resint_t1,
    m.ftrd_terms_trade_col_m_tot     AS macro_tot,
    LAG(m.ftrd_terms_trade_col_m_tot)     OVER w AS macro_tot_t1,
    m.ftrd_exports_total_col_m_expusd AS macro_expusd,
    LAG(m.ftrd_exports_total_col_m_expusd) OVER w AS macro_expusd_t1,
    m.ftrd_imports_total_col_m_impusd AS macro_impusd,
    LAG(m.ftrd_imports_total_col_m_impusd) OVER w AS macro_impusd_t1,
    m.crsk_sentiment_cci_col_m_cci   AS macro_cci,
    LAG(m.crsk_sentiment_cci_col_m_cci)   OVER w AS macro_cci_t1,
    m.crsk_sentiment_ici_col_m_ici   AS macro_ici,
    LAG(m.crsk_sentiment_ici_col_m_ici)   OVER w AS macro_ici_t1

FROM macro_indicators_monthly m
WINDOW w AS (ORDER BY m.fecha);

CREATE OR REPLACE VIEW market_macro_quarterly_wide AS
SELECT
    m.fecha                                        AS quarter_end,
    m.publication_date,
    COALESCE(m.publication_date,
             (m.fecha + interval '3 months')::date) AS available_from_conservative,
    m.is_complete,
    m.ffill_count,
    m.gdpp_real_gdp_usa_q_gdp_q          AS macro_gdp_usa,
    LAG(m.gdpp_real_gdp_usa_q_gdp_q)          OVER w AS macro_gdp_usa_t1,
    m.rsbp_fdi_inflow_col_q_fdiin        AS macro_fdi_in,
    LAG(m.rsbp_fdi_inflow_col_q_fdiin)        OVER w AS macro_fdi_in_t1,
    m.rsbp_fdi_outflow_col_q_fdiout      AS macro_fdi_out,
    LAG(m.rsbp_fdi_outflow_col_q_fdiout)      OVER w AS macro_fdi_out_t1,
    m.rsbp_current_account_col_q_cacct   AS macro_cacct,
    LAG(m.rsbp_current_account_col_q_cacct)   OVER w AS macro_cacct_t1
FROM macro_indicators_quarterly m
WINDOW w AS (ORDER BY m.fecha);
