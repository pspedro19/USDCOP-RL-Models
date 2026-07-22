-- ============================================================================
-- Migration 065: BanRep USD/COP forward market — monthly point-in-time series
-- TAREA C1 (2026-07-21). 0 trials: data ingestion only, no study is opened.
-- ============================================================================
-- Source (documented 2026-07-21):
--   "Informe mensual de otras series históricas del mercado de derivados",
--   Banco de la República (DOAM), sheet "2. FwdUSDCOP":
--   https://suameca.banrep.gov.co/archivos/sector_externo_tasas_cambio_derivados/mercado_derivados/series_historico_otros_derivados.xlsx
--   Catalog entry: SUAMECA > Catálogo > Sector externo, tasas de cambio y
--   derivados > Mercado de derivados > "Otras series del Mercado de forwards"
--   (menu 430502, dashboard 4160203). BanRep labels the series
--   "Disponible desde 2005" — the pre-2005 (1997-2004) monthly forward series
--   was retired together with the legacy OBIEE portal
--   (totoro.banrep.gov.co/analytics responde "Sitio en Mantenimiento") and is
--   NOT machine-readable anywhere on the current BanRep web. Coverage here is
--   therefore 2005-01 -> present (official maximum).
--
-- Grain: one row per (month, tenor bucket). Tenor buckets are BanRep's
--   "Rango" in days: '0', '1 a 3', '4 a 14', '15 a 35', '36 a 60', '61 a 90',
--   '91 a 180', 'mayor a 180', plus BanRep's own monthly aggregate 'TOTAL'.
--
-- forward_rate: BanRep does NOT publish the outright forward COP rate in this
--   series — it publishes the *implied annualized devaluation*
--   [(F/S)^(365/plazo) - 1]. forward_rate stays NULL unless a future source
--   provides the outright; the column is kept for schema compatibility.
--
-- implied_dev: MontoNegociado-weighted mean of DevaluacionImplicita across
--   Contraparte rows within (month, tenor), decimal (0.05 = 5% annualized).
--
-- published_at (PIT date, conservative): last day of month M + 60 calendar
--   days. Evidence (2026-07-21): latest month in the file = 2026-05 (present
--   at month_end+51d) while 2026-06 is absent (+21d) -> the file refreshes
--   ~mid-M+2. Repo precedent (config/usdcop_forward_macro_sources.yaml,
--   banrep_monthly_derivatives) uses observation_plus_45_calendar_days; we
--   take 60d to over-wait, never under-wait (same doctrine as migration 062:
--   under-waiting fabricates alpha, over-waiting only costs freshness).
-- ============================================================================

CREATE TABLE IF NOT EXISTS macro_banrep_forwards_monthly (
    month                   DATE        NOT NULL,   -- first day of reference month
    tenor                   TEXT        NOT NULL,   -- BanRep Rango bucket or 'TOTAL'
    forward_rate            NUMERIC,                -- outright COP forward: not published in this series (NULL)
    implied_dev             NUMERIC,                -- annualized implied devaluation, decimal, monto-weighted
    monto_negociado_usd_mn  NUMERIC,                -- traded amount in the bucket, USD millions (context for the weight)
    published_at            DATE        NOT NULL,   -- conservative PIT availability: month_end + 60d
    source                  TEXT,
    ingested_at             TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (month, tenor),
    CONSTRAINT chk_bfm_month_first_day  CHECK (month = date_trunc('month', month)::date),
    CONSTRAINT chk_bfm_monto_nonneg     CHECK (monto_negociado_usd_mn IS NULL OR monto_negociado_usd_mn >= 0),
    CONSTRAINT chk_bfm_dev_sane         CHECK (implied_dev IS NULL OR (implied_dev > -1 AND implied_dev < 2)),
    -- PIT exacto, no solo posterior: fin de mes + 60 dias (Codex verify C1 #1)
    CONSTRAINT chk_bfm_pit_exact CHECK (
        published_at = ((month + INTERVAL '1 month')::date - 1) + 60
    )
);

COMMENT ON TABLE macro_banrep_forwards_monthly IS
    'BanRep USD/COP forward market, monthly PIT (TAREA C1). Source: SUAMECA series_historico_otros_derivados.xlsx sheet 2. FwdUSDCOP. Coverage 2005-01+ (1997-2004 retired with OBIEE portal). published_at = month_end + 60d (conservative).';
COMMENT ON COLUMN macro_banrep_forwards_monthly.tenor IS
    'BanRep Rango in days: 0 | 1 a 3 | 4 a 14 | 15 a 35 | 36 a 60 | 61 a 90 | 91 a 180 | mayor a 180 | TOTAL (BanRep''s own monthly aggregate row)';
COMMENT ON COLUMN macro_banrep_forwards_monthly.implied_dev IS
    'Annualized implied devaluation [(F/S)^(365/plazo)-1], decimal, MontoNegociado-weighted across counterparties';
COMMENT ON COLUMN macro_banrep_forwards_monthly.forward_rate IS
    'Outright forward COP rate — NULL: not published in this BanRep series (only implied devaluation is)';
COMMENT ON COLUMN macro_banrep_forwards_monthly.published_at IS
    'Conservative PIT date = last day of month + 60 calendar days (observed refresh ~mid-M+2; never join on month alone)';

CREATE INDEX IF NOT EXISTS idx_bfm_published_at
    ON macro_banrep_forwards_monthly (published_at);
