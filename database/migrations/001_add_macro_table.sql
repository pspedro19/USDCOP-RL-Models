-- ==============================================================================
-- MIGRACIÓN 001: Añadir tabla macro_indicators_daily
-- Version: 3.1 (Migration V14)
-- ==============================================================================
-- REFERENCIA: ARQUITECTURA_INTEGRAL_V3.md (Sección 4, líneas 297-389)
-- REFERENCIA: MAPEO_MIGRACION_BIDIRECCIONAL.md (Parte 2.2)
-- ==============================================================================
-- DESCRIPCIÓN:
--   Esta migración crea la tabla macro_indicators_daily si no existe,
--   y migra datos desde macro_ohlcv si están disponibles.
--
-- EJECUCIÓN:
--   psql -h localhost -U postgres -d usdcop_trading -f 001_add_macro_table.sql
--
-- ROLLBACK:
--   DROP TABLE IF EXISTS macro_indicators_daily CASCADE;
-- ==============================================================================

-- ============================================================================
-- PASO 1: Verificar pre-requisitos
-- ============================================================================

DO $$
BEGIN
    -- Verificar que no existe la tabla (idempotente)
    IF EXISTS (SELECT 1 FROM information_schema.tables
               WHERE table_schema = 'public'
               AND table_name = 'macro_indicators_daily') THEN
        RAISE NOTICE 'La tabla macro_indicators_daily ya existe. Verificando estructura...';
    ELSE
        RAISE NOTICE 'Creando tabla macro_indicators_daily...';
    END IF;
END $$;

-- ============================================================================
-- PASO 2: Crear tabla macro_indicators_daily (37 columnas macro + 4 metadata)
-- ============================================================================

CREATE TABLE IF NOT EXISTS macro_indicators_daily (
    date DATE PRIMARY KEY,

    -- =========================================================================
    -- 37 COLUMNAS MACRO (en orden alfabético para claridad)
    -- =========================================================================

    -- GRUPO: Bonos (2 columnas)
    bond_yield5y_col      NUMERIC(8,4),     -- Bono Colombia 5Y
    bond_yield10y_col     NUMERIC(8,4),     -- Bono Colombia 10Y

    -- GRUPO: Commodities (4 columnas)
    brent                 NUMERIC(10,4),    -- Brent Crude Oil - Feature: brent_change_1d
    coffee                NUMERIC(10,4),    -- Coffee Arabica
    gold                  NUMERIC(10,4),    -- Gold Futures (XAU/USD)
    wti                   NUMERIC(10,4),    -- WTI Crude Oil

    -- GRUPO: Sentiment Colombia (2 columnas)
    cci_colombia          NUMERIC(10,4),    -- Índice Confianza Consumidor
    ici_colombia          NUMERIC(10,4),    -- Índice Confianza Industrial

    -- GRUPO: Equity/FX Indices (2 columnas)
    colcap                NUMERIC(10,4),    -- Índice COLCAP
    itcr                  NUMERIC(10,4),    -- Índice Tasa Cambio Real

    -- GRUPO: Sentiment USA (1 columna)
    consumer_sentiment    NUMERIC(10,4),    -- Michigan Sentiment (mensual)

    -- GRUPO: Inflation (3 columnas - mensual)
    cpi_usa               NUMERIC(10,4),    -- CPI USA (mensual)
    ipc_colombia          NUMERIC(10,4),    -- IPC Colombia (mensual)
    pce_usa               NUMERIC(10,4),    -- PCE USA (mensual)

    -- GRUPO: Balance of Payments (4 columnas - trimestral)
    current_account       NUMERIC(14,2),    -- Cuenta Corriente BP
    ied_inflow            NUMERIC(14,2),    -- IED Entrante
    ied_outflow           NUMERIC(14,2),    -- IED Saliente
    reserves_intl         NUMERIC(14,2),    -- Reservas Internacionales

    -- GRUPO: Dollar Index (1 columna)
    dxy                   NUMERIC(10,4),    -- US Dollar Index - Features: dxy_z, dxy_change_1d

    -- GRUPO: Risk (1 columna)
    embi                  NUMERIC(10,4),    -- EMBI Colombia - Feature: embi_z

    -- GRUPO: Trade (3 columnas - mensual)
    exports_col           NUMERIC(14,2),    -- Exportaciones Colombia
    imports_col           NUMERIC(14,2),    -- Importaciones Colombia
    terms_of_trade        NUMERIC(10,4),    -- Términos de Intercambio

    -- GRUPO: Policy Rates (2 columnas)
    fed_funds             NUMERIC(8,4),     -- Fed Funds Rate
    tpm_colombia          NUMERIC(8,4),     -- Tasa Política Monetaria BanRep

    -- GRUPO: GDP (1 columna - trimestral)
    gdp_usa               NUMERIC(14,2),    -- GDP USA Real

    -- GRUPO: Money Market (1 columna)
    ibr_overnight         NUMERIC(8,4),     -- IBR Colombia

    -- GRUPO: Labor (1 columna - mensual)
    unemployment_usa      NUMERIC(8,4),     -- Desempleo USA

    -- GRUPO: Production (2 columnas - mensual)
    industrial_prod_usa   NUMERIC(10,4),    -- Producción Industrial USA
    m2_supply_usa         NUMERIC(14,2),    -- M2 USA

    -- GRUPO: Fixed Income USA (3 columnas)
    prime_rate            NUMERIC(8,4),     -- Prime Rate USA
    treasury_2y           NUMERIC(8,4),     -- UST 2Y (DGS2) - Feature: rate_spread
    treasury_10y          NUMERIC(8,4),     -- UST 10Y (DGS10) - Feature: rate_spread

    -- GRUPO: Exchange Rates (3 columnas)
    usdclp                NUMERIC(10,4),    -- USD/CLP (peer currency)
    usdcop_spot           NUMERIC(10,4),    -- USD/COP Spot (cierre diario)
    usdmxn                NUMERIC(10,4),    -- USD/MXN - Feature: usdmxn_ret_1h

    -- GRUPO: Volatility (1 columna)
    vix                   NUMERIC(10,4),    -- CBOE VIX - Feature: vix_z

    -- =========================================================================
    -- METADATA (4 columnas)
    -- =========================================================================
    source                VARCHAR(100),                      -- Fuente principal del update
    is_complete           BOOLEAN DEFAULT FALSE,             -- TRUE si campos críticos completos
    created_at            TIMESTAMPTZ DEFAULT NOW(),
    updated_at            TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- PASO 3: Crear índices
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_macro_date
    ON macro_indicators_daily(date DESC);

CREATE INDEX IF NOT EXISTS idx_macro_complete
    ON macro_indicators_daily(is_complete, date DESC);

-- Índice compuesto para los 7 campos usados en features del modelo
CREATE INDEX IF NOT EXISTS idx_macro_model_features
    ON macro_indicators_daily(date, dxy, vix, embi, brent, usdmxn, treasury_10y, treasury_2y);

-- ============================================================================
-- PASO 4: Crear trigger para updated_at
-- ============================================================================

CREATE OR REPLACE FUNCTION update_macro_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_macro_updated_at ON macro_indicators_daily;
CREATE TRIGGER trg_macro_updated_at
    BEFORE UPDATE ON macro_indicators_daily
    FOR EACH ROW
    EXECUTE FUNCTION update_macro_timestamp();

-- ============================================================================
-- PASO 5: Migrar datos desde macro_ohlcv (si existe)
-- ============================================================================

DO $$
DECLARE
    rows_migrated INT;
    total_migrated INT := 0;
BEGIN
    -- Verificar si existe macro_ohlcv con datos
    IF EXISTS (SELECT 1 FROM information_schema.tables
               WHERE table_schema = 'public'
               AND table_name = 'macro_ohlcv') THEN

        RAISE NOTICE 'Tabla macro_ohlcv encontrada. Iniciando migración de datos...';

        -- =====================================================================
        -- Migrar DXY
        -- =====================================================================
        INSERT INTO macro_indicators_daily (date, dxy, source)
        SELECT
            DATE(time AT TIME ZONE 'America/Bogota'),
            close,
            'migration_macro_ohlcv'
        FROM macro_ohlcv
        WHERE symbol = 'DXY'
        ON CONFLICT (date) DO UPDATE SET
            dxy = EXCLUDED.dxy,
            updated_at = NOW()
        WHERE macro_indicators_daily.dxy IS NULL;

        GET DIAGNOSTICS rows_migrated = ROW_COUNT;
        total_migrated := total_migrated + rows_migrated;
        RAISE NOTICE '  DXY: % filas procesadas', rows_migrated;

        -- =====================================================================
        -- Migrar VIX
        -- =====================================================================
        UPDATE macro_indicators_daily m
        SET vix = subq.close, updated_at = NOW()
        FROM (
            SELECT DATE(time AT TIME ZONE 'America/Bogota') as date, close
            FROM macro_ohlcv
            WHERE symbol = 'VIX'
        ) subq
        WHERE m.date = subq.date AND m.vix IS NULL;

        GET DIAGNOSTICS rows_migrated = ROW_COUNT;
        total_migrated := total_migrated + rows_migrated;
        RAISE NOTICE '  VIX: % filas actualizadas', rows_migrated;

        -- =====================================================================
        -- Migrar Brent
        -- =====================================================================
        UPDATE macro_indicators_daily m
        SET brent = subq.close, updated_at = NOW()
        FROM (
            SELECT DATE(time AT TIME ZONE 'America/Bogota') as date, close
            FROM macro_ohlcv
            WHERE symbol = 'BRN'
        ) subq
        WHERE m.date = subq.date AND m.brent IS NULL;

        GET DIAGNOSTICS rows_migrated = ROW_COUNT;
        total_migrated := total_migrated + rows_migrated;
        RAISE NOTICE '  Brent: % filas actualizadas', rows_migrated;

        -- =====================================================================
        -- Migrar WTI
        -- =====================================================================
        UPDATE macro_indicators_daily m
        SET wti = subq.close, updated_at = NOW()
        FROM (
            SELECT DATE(time AT TIME ZONE 'America/Bogota') as date, close
            FROM macro_ohlcv
            WHERE symbol = 'WTI'
        ) subq
        WHERE m.date = subq.date AND m.wti IS NULL;

        GET DIAGNOSTICS rows_migrated = ROW_COUNT;
        total_migrated := total_migrated + rows_migrated;
        RAISE NOTICE '  WTI: % filas actualizadas', rows_migrated;

        -- =====================================================================
        -- Migrar Gold
        -- =====================================================================
        UPDATE macro_indicators_daily m
        SET gold = subq.close, updated_at = NOW()
        FROM (
            SELECT DATE(time AT TIME ZONE 'America/Bogota') as date, close
            FROM macro_ohlcv
            WHERE symbol = 'XAU'
        ) subq
        WHERE m.date = subq.date AND m.gold IS NULL;

        GET DIAGNOSTICS rows_migrated = ROW_COUNT;
        total_migrated := total_migrated + rows_migrated;
        RAISE NOTICE '  Gold: % filas actualizadas', rows_migrated;

        -- =====================================================================
        -- Migrar USDMXN
        -- =====================================================================
        UPDATE macro_indicators_daily m
        SET usdmxn = subq.close, updated_at = NOW()
        FROM (
            SELECT DATE(time AT TIME ZONE 'America/Bogota') as date, close
            FROM macro_ohlcv
            WHERE symbol = 'USD/MXN'
        ) subq
        WHERE m.date = subq.date AND m.usdmxn IS NULL;

        GET DIAGNOSTICS rows_migrated = ROW_COUNT;
        total_migrated := total_migrated + rows_migrated;
        RAISE NOTICE '  USDMXN: % filas actualizadas', rows_migrated;

        -- =====================================================================
        -- Migrar USDCLP
        -- =====================================================================
        UPDATE macro_indicators_daily m
        SET usdclp = subq.close, updated_at = NOW()
        FROM (
            SELECT DATE(time AT TIME ZONE 'America/Bogota') as date, close
            FROM macro_ohlcv
            WHERE symbol = 'USD/CLP'
        ) subq
        WHERE m.date = subq.date AND m.usdclp IS NULL;

        GET DIAGNOSTICS rows_migrated = ROW_COUNT;
        total_migrated := total_migrated + rows_migrated;
        RAISE NOTICE '  USDCLP: % filas actualizadas', rows_migrated;

        RAISE NOTICE 'Migración desde macro_ohlcv completada. Total: % filas', total_migrated;

    ELSE
        RAISE NOTICE 'Tabla macro_ohlcv no encontrada. Saltando migración de datos.';
    END IF;
END $$;

-- ============================================================================
-- PASO 6: Marcar registros completos
-- ============================================================================
-- is_complete = TRUE cuando todos los campos críticos están poblados
-- Campos críticos: dxy, vix, embi, brent, treasury_10y, treasury_2y

UPDATE macro_indicators_daily
SET is_complete = TRUE
WHERE dxy IS NOT NULL
  AND vix IS NOT NULL
  AND embi IS NOT NULL
  AND brent IS NOT NULL
  AND treasury_10y IS NOT NULL
  AND treasury_2y IS NOT NULL;

-- ============================================================================
-- PASO 7: Verificar migración
-- ============================================================================

DO $$
DECLARE
    total_rows INT;
    complete_rows INT;
    columns_count INT;
BEGIN
    SELECT COUNT(*) INTO total_rows FROM macro_indicators_daily;
    SELECT COUNT(*) INTO complete_rows FROM macro_indicators_daily WHERE is_complete = TRUE;
    SELECT COUNT(*) INTO columns_count
    FROM information_schema.columns
    WHERE table_name = 'macro_indicators_daily' AND table_schema = 'public';

    RAISE NOTICE '========================================';
    RAISE NOTICE 'MIGRACIÓN 001 COMPLETADA';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Tabla: macro_indicators_daily';
    RAISE NOTICE 'Total de columnas: % (37 macro + 4 metadata)', columns_count;
    RAISE NOTICE 'Total de filas: %', total_rows;
    RAISE NOTICE 'Filas completas: %', complete_rows;
    RAISE NOTICE 'Filas incompletas: %', total_rows - complete_rows;
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Campos críticos para is_complete:';
    RAISE NOTICE '  - dxy, vix, embi, brent';
    RAISE NOTICE '  - treasury_10y, treasury_2y';
    RAISE NOTICE '========================================';
END $$;

-- ============================================================================
-- PASO 8: Documentación
-- ============================================================================

COMMENT ON TABLE macro_indicators_daily IS
'SSOT v3.1: Indicadores macroeconómicos diarios para features del modelo PPO.
37 variables macro + 4 metadata = 41 columnas totales.
Actualización: 3x/día (07:55, 10:30, 12:00 COT).
FEATURES CALCULADOS EN SQL: dxy_z, vix_z, embi_z, dxy_change_1d, brent_change_1d, rate_spread.
FEATURES CALCULADOS EN PYTHON: usdmxn_ret_1h (requiere 12 periodos lookback).
Migración 001 aplicada: ' || NOW()::TEXT || '
REF: ARQUITECTURA_INTEGRAL_V3.md sección 4 y 21.';

COMMENT ON COLUMN macro_indicators_daily.date IS
    'Primary key - fecha del dato macro (formato DATE)';

COMMENT ON COLUMN macro_indicators_daily.dxy IS
    'US Dollar Index - Strong positive correlation (~+0.8) with USD/COP - Feature: dxy_z, dxy_change_1d';

COMMENT ON COLUMN macro_indicators_daily.vix IS
    'CBOE Volatility Index - Flight-to-safety indicator affecting EM currencies - Feature: vix_z';

COMMENT ON COLUMN macro_indicators_daily.embi IS
    'EMBI Colombia - Country risk premium spread over US Treasuries - Feature: embi_z';

COMMENT ON COLUMN macro_indicators_daily.brent IS
    'Brent Crude Oil - Key for Colombia as oil exporter - Feature: brent_change_1d';

COMMENT ON COLUMN macro_indicators_daily.usdmxn IS
    'USD/MXN exchange rate - High correlation with USD/COP (~0.7) - Feature: usdmxn_ret_1h';

COMMENT ON COLUMN macro_indicators_daily.treasury_10y IS
    'US Treasury 10Y yield (DGS10) - Feature: rate_spread';

COMMENT ON COLUMN macro_indicators_daily.treasury_2y IS
    'US Treasury 2Y yield (DGS2) - Feature: rate_spread';

COMMENT ON COLUMN macro_indicators_daily.is_complete IS
    'TRUE when all critical fields are populated (dxy, vix, embi, brent, treasury_10y, treasury_2y)';

COMMENT ON COLUMN macro_indicators_daily.source IS
    'Fuente principal del último update (twelvedata, fred, bcrp, investing)';

-- ============================================================================
-- FIN DE MIGRACIÓN 001
-- ============================================================================
