-- ═══════════════════════════════════════════════════════════════════════════════════
-- SCRIPT: 11-realtime-inference-tables.sql
-- Tablas para sistema de inferencia RL en tiempo real y tracking de acciones del agente
-- ═══════════════════════════════════════════════════════════════════════════════════

-- Asegurar que estamos en el schema correcto
SET search_path TO dw, public;

-- ═══════════════════════════════════════════════════════════════════════════════════
-- TABLA 1: fact_rl_inference
-- Almacena cada inferencia del modelo RL en tiempo real
-- ═══════════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS dw.fact_rl_inference (
    inference_id BIGSERIAL PRIMARY KEY,

    -- Timing
    timestamp_utc TIMESTAMPTZ NOT NULL,
    timestamp_cot TIMESTAMPTZ NOT NULL,

    -- Modelo
    model_id VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    fold_id INT,

    -- Observación de entrada (20 features normalizadas)
    observation FLOAT[] NOT NULL,
    observation_raw JSONB,

    -- Acción del modelo
    action_raw FLOAT NOT NULL,
    action_discretized VARCHAR(10) NOT NULL CHECK (action_discretized IN ('LONG', 'SHORT', 'HOLD')),
    confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),

    -- Q-values (si disponibles)
    q_values FLOAT[],

    -- Contexto de mercado
    symbol VARCHAR(20) DEFAULT 'USD/COP',
    close_price DECIMAL(12,6) NOT NULL,
    raw_return_5m FLOAT,
    spread_bps DECIMAL(8,4),

    -- Estado del portfolio (pre-acción)
    position_before FLOAT NOT NULL CHECK (position_before >= -1 AND position_before <= 1),
    portfolio_value_before DECIMAL(15,2),
    log_portfolio_before FLOAT,

    -- Estado del portfolio (post-acción)
    position_after FLOAT NOT NULL CHECK (position_after >= -1 AND position_after <= 1),
    portfolio_value_after DECIMAL(15,2),
    log_portfolio_after FLOAT,

    -- Costos de transacción
    position_change FLOAT,
    transaction_cost_bps DECIMAL(8,4),
    transaction_cost_usd DECIMAL(12,4),

    -- Performance
    reward FLOAT,
    cumulative_reward FLOAT,

    -- Latencia y metadata
    latency_ms INT,
    inference_source VARCHAR(50) DEFAULT 'airflow',
    dag_run_id VARCHAR(100),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convertir a hypertable TimescaleDB
SELECT create_hypertable('dw.fact_rl_inference', 'timestamp_utc',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Índices optimizados
CREATE INDEX IF NOT EXISTS idx_rl_inference_model_time
    ON dw.fact_rl_inference(model_id, timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_rl_inference_action
    ON dw.fact_rl_inference(action_discretized);
CREATE INDEX IF NOT EXISTS idx_rl_inference_symbol_time
    ON dw.fact_rl_inference(symbol, timestamp_utc DESC);

COMMENT ON TABLE dw.fact_rl_inference IS 'Almacena cada inferencia del modelo RL PPO en tiempo real';


-- ═══════════════════════════════════════════════════════════════════════════════════
-- TABLA 2: fact_agent_actions
-- Vista simplificada de acciones para frontend (fácil de consumir)
-- ═══════════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS dw.fact_agent_actions (
    action_id BIGSERIAL PRIMARY KEY,

    -- Timing
    timestamp_utc TIMESTAMPTZ NOT NULL,
    timestamp_cot TIMESTAMPTZ NOT NULL,

    -- Identificación de sesión
    session_date DATE NOT NULL,
    bar_number INT NOT NULL CHECK (bar_number >= 1 AND bar_number <= 70),

    -- Acción tomada
    action_type VARCHAR(20) NOT NULL CHECK (action_type IN (
        'ENTRY_LONG', 'ENTRY_SHORT', 'EXIT_LONG', 'EXIT_SHORT',
        'INCREASE_LONG', 'INCREASE_SHORT', 'DECREASE_LONG', 'DECREASE_SHORT',
        'HOLD', 'FLIP_LONG', 'FLIP_SHORT'
    )),
    side VARCHAR(10) CHECK (side IN ('LONG', 'SHORT', NULL)),

    -- Precios
    price_at_action DECIMAL(12,6) NOT NULL,

    -- Posición
    position_before FLOAT NOT NULL,
    position_after FLOAT NOT NULL,
    position_change FLOAT NOT NULL,

    -- Portfolio
    equity_before DECIMAL(15,2),
    equity_after DECIMAL(15,2),
    pnl_action DECIMAL(15,4),
    pnl_daily DECIMAL(15,4),
    pnl_daily_pct DECIMAL(10,6),

    -- Métricas del modelo
    model_confidence FLOAT,
    model_id VARCHAR(100),

    -- Para visualización en gráfica
    marker_type VARCHAR(20) CHECK (marker_type IN (
        'triangle_up', 'triangle_down', 'circle', 'square', 'diamond'
    )),
    marker_color VARCHAR(20),
    marker_size INT DEFAULT 10,

    -- Razón de la acción
    reason_code VARCHAR(50),
    reason_description TEXT,

    -- Referencia a inferencia detallada
    inference_id BIGINT REFERENCES dw.fact_rl_inference(inference_id),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Hypertable
SELECT create_hypertable('dw.fact_agent_actions', 'timestamp_utc',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE);

-- Índices
CREATE INDEX IF NOT EXISTS idx_agent_actions_session
    ON dw.fact_agent_actions(session_date DESC, bar_number);
CREATE INDEX IF NOT EXISTS idx_agent_actions_type
    ON dw.fact_agent_actions(action_type);
CREATE INDEX IF NOT EXISTS idx_agent_actions_side
    ON dw.fact_agent_actions(side) WHERE side IS NOT NULL;

COMMENT ON TABLE dw.fact_agent_actions IS 'Acciones del agente RL para visualización en frontend';


-- ═══════════════════════════════════════════════════════════════════════════════════
-- TABLA 3: fact_session_performance
-- Rendimiento agregado por día de trading
-- ═══════════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS dw.fact_session_performance (
    session_id SERIAL PRIMARY KEY,
    session_date DATE NOT NULL UNIQUE,

    -- Timing
    session_start TIMESTAMPTZ,
    session_end TIMESTAMPTZ,

    -- Métricas de la sesión
    total_bars INT NOT NULL DEFAULT 0,
    bars_with_data INT DEFAULT 0,
    total_inferences INT DEFAULT 0,

    -- Trades
    total_trades INT NOT NULL DEFAULT 0,
    entry_trades INT DEFAULT 0,
    exit_trades INT DEFAULT 0,

    -- P&L
    starting_equity DECIMAL(15,2),
    ending_equity DECIMAL(15,2),
    high_water_mark DECIMAL(15,2),
    low_water_mark DECIMAL(15,2),
    daily_return_pct DECIMAL(10,6),
    daily_pnl DECIMAL(15,4),
    gross_profit DECIMAL(15,4),
    gross_loss DECIMAL(15,4),

    -- Win/Loss
    winning_trades INT DEFAULT 0,
    losing_trades INT DEFAULT 0,
    win_rate DECIMAL(5,4),
    profit_factor DECIMAL(10,4),
    avg_win DECIMAL(15,4),
    avg_loss DECIMAL(15,4),
    largest_win DECIMAL(15,4),
    largest_loss DECIMAL(15,4),

    -- Risk
    max_drawdown_intraday_pct DECIMAL(10,6),
    max_drawdown_intraday_usd DECIMAL(15,4),
    max_position_held FLOAT,
    avg_position_size FLOAT,

    -- Sharpe/Sortino estimado intraday
    intraday_sharpe DECIMAL(8,4),
    intraday_sortino DECIMAL(8,4),
    intraday_volatility DECIMAL(10,6),

    -- Comportamiento del agente
    total_long_bars INT DEFAULT 0,
    total_short_bars INT DEFAULT 0,
    total_flat_bars INT DEFAULT 0,
    avg_hold_time_bars DECIMAL(8,2),
    position_changes INT DEFAULT 0,

    -- Costos
    total_transaction_costs DECIMAL(12,4) DEFAULT 0,
    avg_spread_bps DECIMAL(8,4),

    -- Market conditions
    market_return_pct DECIMAL(10,6),
    market_volatility DECIMAL(10,6),
    market_range_pct DECIMAL(10,6),

    -- Modelo
    primary_model_id VARCHAR(100),
    model_avg_confidence DECIMAL(5,4),
    model_avg_latency_ms INT,

    -- Estado
    status VARCHAR(20) DEFAULT 'in_progress' CHECK (status IN (
        'in_progress', 'completed', 'partial', 'error', 'holiday'
    )),
    error_message TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_session_perf_date
    ON dw.fact_session_performance(session_date DESC);
CREATE INDEX IF NOT EXISTS idx_session_perf_status
    ON dw.fact_session_performance(status);

COMMENT ON TABLE dw.fact_session_performance IS 'Métricas de rendimiento agregadas por día de trading';


-- ═══════════════════════════════════════════════════════════════════════════════════
-- TABLA 4: fact_equity_curve_realtime
-- Curva de capital en tiempo real (granularidad 5min)
-- ═══════════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS dw.fact_equity_curve_realtime (
    equity_id BIGSERIAL PRIMARY KEY,

    -- Timing
    timestamp_utc TIMESTAMPTZ NOT NULL,
    timestamp_cot TIMESTAMPTZ NOT NULL,
    session_date DATE NOT NULL,
    bar_number INT NOT NULL,

    -- Equity
    equity_value DECIMAL(15,2) NOT NULL,
    log_equity FLOAT,

    -- Components
    cash_balance DECIMAL(15,2),
    position_value DECIMAL(15,2),
    unrealized_pnl DECIMAL(15,4),
    realized_pnl DECIMAL(15,4),

    -- Returns
    return_bar_pct DECIMAL(10,6),
    return_daily_pct DECIMAL(10,6),
    return_cumulative_pct DECIMAL(10,6),

    -- Drawdown
    high_water_mark DECIMAL(15,2),
    current_drawdown_pct DECIMAL(10,6),
    current_drawdown_usd DECIMAL(15,4),

    -- Position
    current_position FLOAT,
    position_side VARCHAR(10),

    -- Market context
    market_price DECIMAL(12,6),
    market_return_bar DECIMAL(10,6),

    -- Modelo
    model_id VARCHAR(100),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Hypertable
SELECT create_hypertable('dw.fact_equity_curve_realtime', 'timestamp_utc',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_equity_curve_rt_session
    ON dw.fact_equity_curve_realtime(session_date DESC, bar_number);

COMMENT ON TABLE dw.fact_equity_curve_realtime IS 'Curva de capital en tiempo real con granularidad de 5 minutos';


-- ═══════════════════════════════════════════════════════════════════════════════════
-- TABLA 5: fact_macro_realtime
-- Datos macro actualizados (3x/día)
-- ═══════════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS dw.fact_macro_realtime (
    macro_id BIGSERIAL PRIMARY KEY,

    -- Timing
    timestamp_utc TIMESTAMPTZ NOT NULL,
    scrape_run_id VARCHAR(100),

    -- Datos macro principales
    dxy DECIMAL(10,4),
    dxy_change_1d DECIMAL(10,6),

    vix DECIMAL(10,4),
    vix_change_1d DECIMAL(10,6),

    embi DECIMAL(10,4),
    embi_change_1d DECIMAL(10,6),

    brent DECIMAL(10,4),
    brent_change_1d DECIMAL(10,6),

    wti DECIMAL(10,4),
    gold DECIMAL(10,4),

    usdmxn DECIMAL(10,6),
    usdclp DECIMAL(10,4),

    colcap DECIMAL(10,4),

    -- Tasas
    fed_funds DECIMAL(6,4),
    tpm_colombia DECIMAL(6,4),
    ibr DECIMAL(6,4),

    ust_2y DECIMAL(6,4),
    ust_10y DECIMAL(6,4),

    -- Spreads calculados
    rate_spread DECIMAL(8,4),
    term_spread DECIMAL(8,4),

    -- Z-scores (normalizados)
    dxy_z DECIMAL(8,4),
    vix_z DECIMAL(8,4),
    embi_z DECIMAL(8,4),

    -- Régimen de mercado
    vix_regime INT CHECK (vix_regime >= 0 AND vix_regime <= 3),

    -- Metadata
    source VARCHAR(50),
    values_changed INT DEFAULT 0,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Hypertable
SELECT create_hypertable('dw.fact_macro_realtime', 'timestamp_utc',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_macro_rt_time
    ON dw.fact_macro_realtime(timestamp_utc DESC);

COMMENT ON TABLE dw.fact_macro_realtime IS 'Datos macroeconómicos actualizados 3x/día para features RL';


-- ═══════════════════════════════════════════════════════════════════════════════════
-- TABLA 6: fact_inference_alerts
-- Alertas del sistema de inferencia
-- ═══════════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS dw.fact_inference_alerts (
    alert_id BIGSERIAL PRIMARY KEY,

    -- Timing
    timestamp_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Alert info
    alert_type VARCHAR(50) NOT NULL CHECK (alert_type IN (
        'INFERENCE_ERROR', 'HIGH_LATENCY', 'DATA_MISSING', 'MODEL_ERROR',
        'POSITION_LIMIT', 'DRAWDOWN_WARNING', 'CONNECTION_LOST',
        'SCRAPER_ERROR', 'QUALITY_GATE_FAILED', 'ANOMALY_DETECTED'
    )),
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')),

    -- Context
    model_id VARCHAR(100),
    session_date DATE,
    bar_number INT,

    -- Details
    message TEXT NOT NULL,
    details JSONB,

    -- Resolution
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(100),
    resolution_notes TEXT,

    -- Notification
    notification_sent BOOLEAN DEFAULT FALSE,
    notification_channel VARCHAR(50),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_type_time
    ON dw.fact_inference_alerts(alert_type, timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_unresolved
    ON dw.fact_inference_alerts(resolved, severity) WHERE resolved = FALSE;

COMMENT ON TABLE dw.fact_inference_alerts IS 'Alertas y errores del sistema de inferencia en tiempo real';


-- ═══════════════════════════════════════════════════════════════════════════════════
-- VISTAS MATERIALIZADAS PARA DASHBOARDS
-- ═══════════════════════════════════════════════════════════════════════════════════

-- Vista: Últimas acciones del agente (para tabla en frontend)
CREATE OR REPLACE VIEW dw.v_latest_agent_actions AS
SELECT
    action_id,
    timestamp_cot,
    session_date,
    bar_number,
    action_type,
    side,
    price_at_action,
    position_before,
    position_after,
    position_change,
    pnl_action,
    pnl_daily,
    model_confidence,
    marker_type,
    marker_color,
    reason_code
FROM dw.fact_agent_actions
WHERE session_date >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY timestamp_cot DESC;

COMMENT ON VIEW dw.v_latest_agent_actions IS 'Últimas acciones del agente para dashboard (7 días)';


-- Vista: Performance de últimas sesiones
CREATE OR REPLACE VIEW dw.v_session_performance_summary AS
SELECT
    session_date,
    total_trades,
    winning_trades,
    losing_trades,
    win_rate,
    daily_pnl,
    daily_return_pct,
    max_drawdown_intraday_pct,
    intraday_sharpe,
    total_long_bars,
    total_short_bars,
    total_flat_bars,
    status
FROM dw.fact_session_performance
WHERE session_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY session_date DESC;

COMMENT ON VIEW dw.v_session_performance_summary IS 'Resumen de performance de últimas 30 sesiones';


-- Vista: Curva de equity de hoy
CREATE OR REPLACE VIEW dw.v_equity_curve_today AS
SELECT
    timestamp_cot,
    bar_number,
    equity_value,
    return_daily_pct,
    current_drawdown_pct,
    current_position,
    position_side,
    market_price
FROM dw.fact_equity_curve_realtime
WHERE session_date = CURRENT_DATE
ORDER BY bar_number;

COMMENT ON VIEW dw.v_equity_curve_today IS 'Curva de capital del día actual';


-- Vista: Alertas activas
CREATE OR REPLACE VIEW dw.v_active_alerts AS
SELECT
    alert_id,
    timestamp_utc,
    alert_type,
    severity,
    message,
    session_date,
    bar_number,
    details
FROM dw.fact_inference_alerts
WHERE resolved = FALSE
ORDER BY
    CASE severity
        WHEN 'CRITICAL' THEN 1
        WHEN 'ERROR' THEN 2
        WHEN 'WARNING' THEN 3
        ELSE 4
    END,
    timestamp_utc DESC;

COMMENT ON VIEW dw.v_active_alerts IS 'Alertas activas no resueltas';


-- ═══════════════════════════════════════════════════════════════════════════════════
-- FUNCIONES AUXILIARES
-- ═══════════════════════════════════════════════════════════════════════════════════

-- Función: Determinar tipo de acción basado en cambio de posición
CREATE OR REPLACE FUNCTION dw.determine_action_type(
    pos_before FLOAT,
    pos_after FLOAT
) RETURNS VARCHAR(20) AS $$
BEGIN
    -- Sin cambio significativo
    IF ABS(pos_after - pos_before) < 0.05 THEN
        RETURN 'HOLD';
    END IF;

    -- Flip completo
    IF pos_before < -0.3 AND pos_after > 0.3 THEN
        RETURN 'FLIP_LONG';
    END IF;
    IF pos_before > 0.3 AND pos_after < -0.3 THEN
        RETURN 'FLIP_SHORT';
    END IF;

    -- Entry
    IF ABS(pos_before) < 0.1 AND pos_after > 0.3 THEN
        RETURN 'ENTRY_LONG';
    END IF;
    IF ABS(pos_before) < 0.1 AND pos_after < -0.3 THEN
        RETURN 'ENTRY_SHORT';
    END IF;

    -- Exit
    IF pos_before > 0.3 AND ABS(pos_after) < 0.1 THEN
        RETURN 'EXIT_LONG';
    END IF;
    IF pos_before < -0.3 AND ABS(pos_after) < 0.1 THEN
        RETURN 'EXIT_SHORT';
    END IF;

    -- Increase/Decrease
    IF pos_after > pos_before THEN
        IF pos_after > 0 THEN
            RETURN 'INCREASE_LONG';
        ELSE
            RETURN 'DECREASE_SHORT';
        END IF;
    ELSE
        IF pos_after < 0 THEN
            RETURN 'INCREASE_SHORT';
        ELSE
            RETURN 'DECREASE_LONG';
        END IF;
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION dw.determine_action_type IS 'Determina el tipo de acción basado en cambio de posición';


-- Función: Determinar color del marcador
CREATE OR REPLACE FUNCTION dw.determine_marker_color(
    action_type VARCHAR(20)
) RETURNS VARCHAR(20) AS $$
BEGIN
    CASE action_type
        WHEN 'ENTRY_LONG', 'INCREASE_LONG', 'FLIP_LONG' THEN RETURN '#22c55e';  -- green
        WHEN 'ENTRY_SHORT', 'INCREASE_SHORT', 'FLIP_SHORT' THEN RETURN '#ef4444';  -- red
        WHEN 'EXIT_LONG', 'EXIT_SHORT' THEN RETURN '#f59e0b';  -- amber
        WHEN 'DECREASE_LONG', 'DECREASE_SHORT' THEN RETURN '#6b7280';  -- gray
        ELSE RETURN '#9ca3af';  -- light gray
    END CASE;
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- Función: Determinar tipo de marcador
CREATE OR REPLACE FUNCTION dw.determine_marker_type(
    action_type VARCHAR(20)
) RETURNS VARCHAR(20) AS $$
BEGIN
    CASE action_type
        WHEN 'ENTRY_LONG', 'INCREASE_LONG', 'FLIP_LONG' THEN RETURN 'triangle_up';
        WHEN 'ENTRY_SHORT', 'INCREASE_SHORT', 'FLIP_SHORT' THEN RETURN 'triangle_down';
        WHEN 'EXIT_LONG', 'EXIT_SHORT' THEN RETURN 'square';
        WHEN 'HOLD' THEN RETURN 'circle';
        ELSE RETURN 'diamond';
    END CASE;
END;
$$ LANGUAGE plpgsql IMMUTABLE;


-- Función: Actualizar métricas de sesión
CREATE OR REPLACE FUNCTION dw.update_session_metrics(p_session_date DATE)
RETURNS VOID AS $$
DECLARE
    v_actions RECORD;
    v_equity RECORD;
BEGIN
    -- Calcular métricas desde fact_agent_actions
    SELECT
        COUNT(*) FILTER (WHERE action_type != 'HOLD') as total_trades,
        COUNT(*) FILTER (WHERE action_type LIKE 'ENTRY%') as entry_trades,
        COUNT(*) FILTER (WHERE action_type LIKE 'EXIT%') as exit_trades,
        COUNT(*) FILTER (WHERE pnl_action > 0) as winning_trades,
        COUNT(*) FILTER (WHERE pnl_action < 0) as losing_trades,
        SUM(pnl_action) FILTER (WHERE pnl_action > 0) as gross_profit,
        ABS(SUM(pnl_action) FILTER (WHERE pnl_action < 0)) as gross_loss,
        MAX(pnl_action) as largest_win,
        MIN(pnl_action) as largest_loss,
        AVG(model_confidence) as avg_confidence,
        COUNT(*) FILTER (WHERE position_after > 0.1) as long_bars,
        COUNT(*) FILTER (WHERE position_after < -0.1) as short_bars,
        COUNT(*) FILTER (WHERE ABS(position_after) <= 0.1) as flat_bars
    INTO v_actions
    FROM dw.fact_agent_actions
    WHERE session_date = p_session_date;

    -- Calcular métricas de equity
    SELECT
        MIN(equity_value) as low_water_mark,
        MAX(equity_value) as high_water_mark,
        FIRST_VALUE(equity_value) OVER (ORDER BY bar_number) as starting_equity,
        LAST_VALUE(equity_value) OVER (ORDER BY bar_number
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) as ending_equity,
        MAX(current_drawdown_pct) as max_dd
    INTO v_equity
    FROM dw.fact_equity_curve_realtime
    WHERE session_date = p_session_date;

    -- Upsert en fact_session_performance
    INSERT INTO dw.fact_session_performance (
        session_date, total_trades, entry_trades, exit_trades,
        winning_trades, losing_trades,
        gross_profit, gross_loss, largest_win, largest_loss,
        starting_equity, ending_equity, high_water_mark, low_water_mark,
        max_drawdown_intraday_pct,
        total_long_bars, total_short_bars, total_flat_bars,
        model_avg_confidence,
        updated_at
    ) VALUES (
        p_session_date,
        COALESCE(v_actions.total_trades, 0),
        COALESCE(v_actions.entry_trades, 0),
        COALESCE(v_actions.exit_trades, 0),
        COALESCE(v_actions.winning_trades, 0),
        COALESCE(v_actions.losing_trades, 0),
        v_actions.gross_profit,
        v_actions.gross_loss,
        v_actions.largest_win,
        v_actions.largest_loss,
        v_equity.starting_equity,
        v_equity.ending_equity,
        v_equity.high_water_mark,
        v_equity.low_water_mark,
        v_equity.max_dd,
        COALESCE(v_actions.long_bars, 0),
        COALESCE(v_actions.short_bars, 0),
        COALESCE(v_actions.flat_bars, 0),
        v_actions.avg_confidence,
        NOW()
    )
    ON CONFLICT (session_date) DO UPDATE SET
        total_trades = EXCLUDED.total_trades,
        entry_trades = EXCLUDED.entry_trades,
        exit_trades = EXCLUDED.exit_trades,
        winning_trades = EXCLUDED.winning_trades,
        losing_trades = EXCLUDED.losing_trades,
        gross_profit = EXCLUDED.gross_profit,
        gross_loss = EXCLUDED.gross_loss,
        largest_win = EXCLUDED.largest_win,
        largest_loss = EXCLUDED.largest_loss,
        ending_equity = EXCLUDED.ending_equity,
        high_water_mark = GREATEST(dw.fact_session_performance.high_water_mark, EXCLUDED.high_water_mark),
        low_water_mark = LEAST(dw.fact_session_performance.low_water_mark, EXCLUDED.low_water_mark),
        max_drawdown_intraday_pct = GREATEST(dw.fact_session_performance.max_drawdown_intraday_pct, EXCLUDED.max_drawdown_intraday_pct),
        total_long_bars = EXCLUDED.total_long_bars,
        total_short_bars = EXCLUDED.total_short_bars,
        total_flat_bars = EXCLUDED.total_flat_bars,
        model_avg_confidence = EXCLUDED.model_avg_confidence,
        updated_at = NOW();

    -- Calcular win_rate y profit_factor
    UPDATE dw.fact_session_performance
    SET
        win_rate = CASE
            WHEN total_trades > 0 THEN winning_trades::DECIMAL / total_trades
            ELSE NULL
        END,
        profit_factor = CASE
            WHEN gross_loss > 0 THEN gross_profit / gross_loss
            ELSE NULL
        END,
        daily_pnl = ending_equity - starting_equity,
        daily_return_pct = CASE
            WHEN starting_equity > 0 THEN (ending_equity - starting_equity) / starting_equity * 100
            ELSE NULL
        END
    WHERE session_date = p_session_date;

END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION dw.update_session_metrics IS 'Actualiza métricas agregadas de una sesión de trading';


-- ═══════════════════════════════════════════════════════════════════════════════════
-- TRIGGERS
-- ═══════════════════════════════════════════════════════════════════════════════════

-- Trigger: Auto-completar campos en fact_agent_actions
CREATE OR REPLACE FUNCTION dw.trg_agent_actions_auto_fields()
RETURNS TRIGGER AS $$
BEGIN
    -- Determinar action_type si no se especificó
    IF NEW.action_type IS NULL OR NEW.action_type = '' THEN
        NEW.action_type := dw.determine_action_type(NEW.position_before, NEW.position_after);
    END IF;

    -- Determinar side
    IF NEW.side IS NULL THEN
        IF NEW.position_after > 0.1 THEN
            NEW.side := 'LONG';
        ELSIF NEW.position_after < -0.1 THEN
            NEW.side := 'SHORT';
        END IF;
    END IF;

    -- Determinar marker_type y marker_color
    IF NEW.marker_type IS NULL THEN
        NEW.marker_type := dw.determine_marker_type(NEW.action_type);
    END IF;
    IF NEW.marker_color IS NULL THEN
        NEW.marker_color := dw.determine_marker_color(NEW.action_type);
    END IF;

    -- Calcular position_change
    NEW.position_change := NEW.position_after - NEW.position_before;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_agent_actions_before_insert
    BEFORE INSERT ON dw.fact_agent_actions
    FOR EACH ROW
    EXECUTE FUNCTION dw.trg_agent_actions_auto_fields();


-- ═══════════════════════════════════════════════════════════════════════════════════
-- GRANTS
-- ═══════════════════════════════════════════════════════════════════════════════════
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA dw TO admin;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA dw TO admin;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA dw TO admin;


-- ═══════════════════════════════════════════════════════════════════════════════════
-- MENSAJE DE CONFIRMACIÓN
-- ═══════════════════════════════════════════════════════════════════════════════════
DO $$
BEGIN
    RAISE NOTICE '═══════════════════════════════════════════════════════════════';
    RAISE NOTICE 'Script 11-realtime-inference-tables.sql ejecutado exitosamente';
    RAISE NOTICE 'Tablas creadas:';
    RAISE NOTICE '  - dw.fact_rl_inference (inferencias del modelo)';
    RAISE NOTICE '  - dw.fact_agent_actions (acciones para frontend)';
    RAISE NOTICE '  - dw.fact_session_performance (métricas diarias)';
    RAISE NOTICE '  - dw.fact_equity_curve_realtime (curva de capital)';
    RAISE NOTICE '  - dw.fact_macro_realtime (datos macro)';
    RAISE NOTICE '  - dw.fact_inference_alerts (alertas)';
    RAISE NOTICE 'Vistas creadas:';
    RAISE NOTICE '  - dw.v_latest_agent_actions';
    RAISE NOTICE '  - dw.v_session_performance_summary';
    RAISE NOTICE '  - dw.v_equity_curve_today';
    RAISE NOTICE '  - dw.v_active_alerts';
    RAISE NOTICE '═══════════════════════════════════════════════════════════════';
END $$;
