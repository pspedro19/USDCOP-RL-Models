-- =============================================================================
-- 083 · Alinea `dim_asset.symbol` de SPX con el simbolo REAL del dato
-- =============================================================================
-- Contract: CTR-MKT-CANON-001 (correccion) · Date: 2026-08-24
--
-- ## El defecto
--
-- `market_ohlcv_daily` — la capa canonica que alimenta las vistas wide y el
-- dashboard — resuelve el activo con:
--
--     FROM asset_daily_ohlcv d JOIN dim_asset da ON da.symbol = d.symbol
--
-- `dim_asset` registraba `spx500` con `symbol = 'SPX500'`, pero
-- `asset_daily_ohlcv` guarda ese activo como **'SPX/500'** (con barra). El join
-- nunca casaba, asi que las **7.962 filas** de SPX (investing_daily, 1995-2026)
-- desaparecian de la capa canonica SIN ERROR NI AVISO.
--
-- Sintoma que lo destapo: `test_wide_views.py::test_july_20_reads_closed_not_missing`
-- fallaba con `spx='missing'` para el 2026-07-20 — un dia en que NYSE estuvo
-- abierto, la fila existia en `asset_daily_ohlcv` y `market_session_calendar`
-- decia `is_trading_day = t`. La vista no mentia: para ella SPX no existia.
--
-- ## Por que gana 'SPX/500'
--
-- `config/assets/spx500.yaml` es el AssetProfile SSOT (ver
-- `.claude/specs/assets/_onboarding-playbook.md`) y distingue dos campos:
--
--     symbol:       "SPX/500"    <- el simbolo del DATO
--     chart_symbol: "SPX500"     <- el simbolo del GRAFICO (dashboard)
--
-- `dim_asset` habia tomado el de grafico. Se corrige al de dato, que ademas es
-- donde viven las 7.962 filas. `SPX500` conserva su rol legitimo en el frontend.
--
-- ## Efecto
--
-- `market_ohlcv_daily` pasa de 3 activos (btcusdt, usdcop, xauusd) a 4.
--
-- Idempotente: el UPDATE es un no-op si ya esta corregido.
-- =============================================================================

BEGIN;

-- Guarda de seguridad: no tocar nada si el simbolo destino ya lo usa otra fila.
DO $$
DECLARE
    conflicting int;
BEGIN
    SELECT count(*) INTO conflicting
    FROM dim_asset WHERE symbol = 'SPX/500' AND asset_id <> 'spx500';
    IF conflicting > 0 THEN
        RAISE EXCEPTION
            'dim_asset ya tiene % fila(s) con symbol=''SPX/500'' y asset_id<>''spx500''; '
            'resuelve el conflicto antes de aplicar 083', conflicting;
    END IF;
END $$;

UPDATE dim_asset
   SET symbol = 'SPX/500'
 WHERE asset_id = 'spx500'
   AND symbol IS DISTINCT FROM 'SPX/500';

-- Verificacion: SPX debe quedar visible en la capa canonica.
DO $$
DECLARE
    n_assets int;
    n_spx    bigint;
BEGIN
    SELECT count(DISTINCT asset_id) INTO n_assets FROM market_ohlcv_daily;
    SELECT count(*) INTO n_spx FROM market_ohlcv_daily WHERE asset_id = 'spx500';

    IF n_spx = 0 THEN
        RAISE EXCEPTION
            '083 fallo: spx500 sigue sin filas en market_ohlcv_daily tras alinear el '
            'simbolo. Revisa si asset_daily_ohlcv usa otra variante.';
    END IF;
    RAISE NOTICE '083 OK: market_ohlcv_daily tiene % activos; spx500 aporta % filas',
                 n_assets, n_spx;
END $$;

COMMIT;
