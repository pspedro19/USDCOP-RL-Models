-- ============================================================================
-- Migration 066: BanRep workers' remittances (remesas) monthly series
-- ============================================================================
-- TAREA C2 (2026-07-21) — point-in-time monthly ingestion, 0 trials (data only).
--
-- Source: Banco de la República — "Ingresos de Remesas de trabajadores, mensual"
--   SUAMECA serie id 15363 (idCargue REMESAS_MENSUAL), unidad: Millones de USD.
--   Catalog page : https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/4150/remesas_trabajadores/
--   REST endpoint: https://suameca.banrep.gov.co/buscador-de-series/rest/buscadorSeriesRestService/consultaDatosSeries
--
-- PIT convention:
--   BanRep loads month M near the end of M+1 (observed: May-2026 loaded 2026-06-26,
--   June-2026 scheduled 2026-07-24). `published_at` stores the conservative rule
--   "last calendar day of M+1"; any consumer joining on published_at <= as_of_date
--   never sees the value earlier than the market could have.
--
-- Grain: monthly, instant-based (NOT COT session-bound). `month` = first day of month.
-- ============================================================================

CREATE TABLE IF NOT EXISTS macro_remesas_monthly (
    month           DATE PRIMARY KEY,
    remesas_usd_mn  NUMERIC NOT NULL,
    published_at    DATE NOT NULL,
    source          TEXT,
    ingested_at     TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT macro_remesas_monthly_month_first_day
        CHECK (month = date_trunc('month', month)::date),
    CONSTRAINT macro_remesas_monthly_non_negative
        CHECK (remesas_usd_mn >= 0),
    CONSTRAINT macro_remesas_monthly_pit_after_month
        CHECK (published_at > month)
);

CREATE INDEX IF NOT EXISTS idx_macro_remesas_monthly_published_at
    ON macro_remesas_monthly (published_at);

COMMENT ON TABLE macro_remesas_monthly IS
    'BanRep workers'' remittances to Colombia, monthly, USD millions (SUAMECA serie 15363). Point-in-time: join on published_at <= as_of.';
COMMENT ON COLUMN macro_remesas_monthly.month IS
    'Reference month (first calendar day). Instant-based monthly grain, not COT session-bound.';
COMMENT ON COLUMN macro_remesas_monthly.remesas_usd_mn IS
    'Ingresos de remesas de trabajadores, millones de USD corrientes.';
COMMENT ON COLUMN macro_remesas_monthly.published_at IS
    'Conservative PIT date: last calendar day of month+1 (BanRep loads ~day 24-26 of month+1).';
