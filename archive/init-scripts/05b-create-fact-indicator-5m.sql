-- ============================================================================
-- Script: 05b-create-fact-indicator-5m.sql
-- Propósito: Crear tabla fact_indicator_5m faltante crítica para L2
-- Dependencias: 03-create-dimensions.sql (dim_symbol, dim_time_5m, dim_indicator)
-- Fecha: 2025-10-27
-- ============================================================================

\c usdcop_trading

SET search_path TO dw;

-- Crear tabla particionada
CREATE TABLE IF NOT EXISTS dw.fact_indicator_5m (
    symbol_id INT NOT NULL REFERENCES dw.dim_symbol(symbol_id),
    time_id INT NOT NULL REFERENCES dw.dim_time_5m(time_id),
    indicator_id INT NOT NULL REFERENCES dw.dim_indicator(indicator_id),
    ts_utc TIMESTAMPTZ NOT NULL,

    indicator_value DECIMAL(18,6) NOT NULL,
    signal VARCHAR(20),  -- 'buy', 'sell', 'neutral', 'overbought', 'oversold'

    created_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (symbol_id, time_id, indicator_id, ts_utc)
) PARTITION BY RANGE (ts_utc);

-- Crear particiones para 2020-2027
CREATE TABLE IF NOT EXISTS dw.fact_indicator_5m_2020 PARTITION OF dw.fact_indicator_5m
    FOR VALUES FROM ('2020-01-01') TO ('2021-01-01');

CREATE TABLE IF NOT EXISTS dw.fact_indicator_5m_2021 PARTITION OF dw.fact_indicator_5m
    FOR VALUES FROM ('2021-01-01') TO ('2022-01-01');

CREATE TABLE IF NOT EXISTS dw.fact_indicator_5m_2022 PARTITION OF dw.fact_indicator_5m
    FOR VALUES FROM ('2022-01-01') TO ('2023-01-01');

CREATE TABLE IF NOT EXISTS dw.fact_indicator_5m_2023 PARTITION OF dw.fact_indicator_5m
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

CREATE TABLE IF NOT EXISTS dw.fact_indicator_5m_2024 PARTITION OF dw.fact_indicator_5m
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE IF NOT EXISTS dw.fact_indicator_5m_2025 PARTITION OF dw.fact_indicator_5m
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

CREATE TABLE IF NOT EXISTS dw.fact_indicator_5m_2026 PARTITION OF dw.fact_indicator_5m
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

-- Índices para optimización
CREATE INDEX IF NOT EXISTS idx_fact_indicator_time ON dw.fact_indicator_5m(time_id);
CREATE INDEX IF NOT EXISTS idx_fact_indicator_symbol ON dw.fact_indicator_5m(symbol_id);
CREATE INDEX IF NOT EXISTS idx_fact_indicator_name ON dw.fact_indicator_5m(indicator_id);
CREATE INDEX IF NOT EXISTS idx_fact_indicator_ts ON dw.fact_indicator_5m(ts_utc DESC);

-- Comentarios
COMMENT ON TABLE dw.fact_indicator_5m IS 'Technical indicator values per bar (grain: symbol/timestamp/indicator). Critical for L2 pipeline.';
COMMENT ON COLUMN dw.fact_indicator_5m.symbol_id IS 'Foreign key to dim_symbol';
COMMENT ON COLUMN dw.fact_indicator_5m.time_id IS 'Foreign key to dim_time_5m';
COMMENT ON COLUMN dw.fact_indicator_5m.indicator_id IS 'Foreign key to dim_indicator';
COMMENT ON COLUMN dw.fact_indicator_5m.indicator_value IS 'Calculated indicator value (RSI, MACD, etc.)';
COMMENT ON COLUMN dw.fact_indicator_5m.signal IS 'Trading signal derived from indicator (buy/sell/neutral/overbought/oversold)';

-- Verificar creación
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'dw' AND tablename LIKE 'fact_indicator_5m%'
ORDER BY tablename;

\echo '✓ fact_indicator_5m table and partitions created successfully'
