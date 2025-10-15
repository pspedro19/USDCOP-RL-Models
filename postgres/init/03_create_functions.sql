-- ========================================
-- USDCOP Trading System - Database Functions
-- Funciones útiles para pipeline L0 y análisis
-- ========================================

\echo '🛠️ Creando funciones de utilidad...'

-- ========================================
-- FUNCIÓN: Verificar si una fecha está en horario de trading
-- ========================================

CREATE OR REPLACE FUNCTION is_trading_hours(input_datetime TIMESTAMPTZ)
RETURNS BOOLEAN AS $$
DECLARE
    colombia_time TIMESTAMPTZ;
    weekday INTEGER;
    hour_minute INTEGER;
BEGIN
    -- Convertir a tiempo de Colombia (COT)
    colombia_time := input_datetime AT TIME ZONE 'America/Bogota';

    -- Obtener día de la semana (0=Domingo, 1=Lunes, ..., 6=Sábado)
    weekday := EXTRACT(dow FROM colombia_time);

    -- Obtener hora y minuto como total de minutos desde medianoche
    hour_minute := EXTRACT(hour FROM colombia_time) * 60 + EXTRACT(minute FROM colombia_time);

    -- Verificar si es día laborable (Lunes=1 a Viernes=5) y horario 8:00-12:55 (480-775 minutos)
    RETURN weekday >= 1 AND weekday <= 5 AND hour_minute >= 480 AND hour_minute <= 775;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ========================================
-- FUNCIÓN: Calcular completitud de datos
-- ========================================

CREATE OR REPLACE FUNCTION calculate_data_completeness(
    start_date TIMESTAMPTZ,
    end_date TIMESTAMPTZ,
    symbol_name VARCHAR DEFAULT 'USDCOP'
)
RETURNS TABLE(
    total_expected INTEGER,
    total_actual INTEGER,
    completeness_pct DECIMAL(5,2),
    missing_count INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH expected_points AS (
        SELECT COUNT(*) as expected_count
        FROM generate_series(start_date, end_date, interval '5 minutes') ts
        WHERE is_trading_hours(ts)
    ),
    actual_points AS (
        SELECT COUNT(*) as actual_count
        FROM market_data
        WHERE symbol = symbol_name
        AND datetime BETWEEN start_date AND end_date
        AND trading_session = true
    )
    SELECT
        e.expected_count::INTEGER,
        a.actual_count::INTEGER,
        CASE
            WHEN e.expected_count > 0 THEN ROUND((a.actual_count::decimal / e.expected_count * 100), 2)
            ELSE 0.00
        END as completeness_pct,
        (e.expected_count - a.actual_count)::INTEGER as missing_count
    FROM expected_points e, actual_points a;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- FUNCIÓN: Obtener gaps de datos faltantes
-- ========================================

CREATE OR REPLACE FUNCTION find_data_gaps(
    start_date TIMESTAMPTZ,
    end_date TIMESTAMPTZ,
    symbol_name VARCHAR DEFAULT 'USDCOP'
)
RETURNS TABLE(
    gap_start TIMESTAMPTZ,
    gap_end TIMESTAMPTZ,
    gap_duration INTERVAL,
    missing_points INTEGER
) AS $$
BEGIN
    RETURN QUERY
    WITH trading_hours AS (
        SELECT ts as expected_datetime
        FROM generate_series(start_date, end_date, interval '5 minutes') ts
        WHERE is_trading_hours(ts)
    ),
    existing_data AS (
        SELECT datetime
        FROM market_data
        WHERE symbol = symbol_name
        AND datetime BETWEEN start_date AND end_date
        AND trading_session = true
    ),
    missing_points AS (
        SELECT th.expected_datetime
        FROM trading_hours th
        LEFT JOIN existing_data ed ON th.expected_datetime = ed.datetime
        WHERE ed.datetime IS NULL
        ORDER BY th.expected_datetime
    ),
    gap_groups AS (
        SELECT
            expected_datetime,
            expected_datetime - (ROW_NUMBER() OVER (ORDER BY expected_datetime) * interval '5 minutes') AS gap_group
        FROM missing_points
    ),
    gap_ranges AS (
        SELECT
            MIN(expected_datetime) as gap_start,
            MAX(expected_datetime) as gap_end,
            MAX(expected_datetime) - MIN(expected_datetime) as gap_duration,
            COUNT(*) as missing_points
        FROM gap_groups
        GROUP BY gap_group
    )
    SELECT * FROM gap_ranges ORDER BY gap_start;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- FUNCIÓN: Actualizar timestamp de updated_at automáticamente
-- ========================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- FUNCIÓN: Validar datos OHLCV
-- ========================================

CREATE OR REPLACE FUNCTION validate_ohlcv_data()
RETURNS TRIGGER AS $$
BEGIN
    -- Validar que High >= Low
    IF NEW.high < NEW.low THEN
        RAISE EXCEPTION 'High (%) no puede ser menor que Low (%)', NEW.high, NEW.low;
    END IF;

    -- Validar que Open y Close estén entre High y Low
    IF NEW.open < NEW.low OR NEW.open > NEW.high THEN
        RAISE EXCEPTION 'Open (%) debe estar entre Low (%) y High (%)', NEW.open, NEW.low, NEW.high;
    END IF;

    IF NEW.close < NEW.low OR NEW.close > NEW.high THEN
        RAISE EXCEPTION 'Close (%) debe estar entre Low (%) y High (%)', NEW.close, NEW.low, NEW.high;
    END IF;

    -- Validar que Volume sea no negativo
    IF NEW.volume < 0 THEN
        RAISE EXCEPTION 'Volume (%) no puede ser negativo', NEW.volume;
    END IF;

    -- Validar precios sean positivos
    IF NEW.open <= 0 OR NEW.high <= 0 OR NEW.low <= 0 OR NEW.close <= 0 THEN
        RAISE EXCEPTION 'Todos los precios deben ser positivos';
    END IF;

    -- Establecer trading_session basado en datetime
    NEW.trading_session := is_trading_hours(NEW.datetime);

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- FUNCIÓN: Calcular métricas de trading performance
-- ========================================

CREATE OR REPLACE FUNCTION calculate_trading_performance(
    start_period TIMESTAMPTZ,
    end_period TIMESTAMPTZ,
    symbol_name VARCHAR DEFAULT 'USDCOP'
)
RETURNS TABLE(
    total_signals INTEGER,
    winning_signals INTEGER,
    losing_signals INTEGER,
    win_rate DECIMAL(5,2),
    avg_win DECIMAL(12,4),
    avg_loss DECIMAL(12,4),
    profit_factor DECIMAL(8,4),
    total_return DECIMAL(12,4)
) AS $$
DECLARE
    wins INTEGER;
    losses INTEGER;
    total INTEGER;
    avg_win_return DECIMAL(12,4);
    avg_loss_return DECIMAL(12,4);
    total_win_return DECIMAL(12,4);
    total_loss_return DECIMAL(12,4);
BEGIN
    -- Obtener estadísticas básicas
    SELECT
        COUNT(*),
        COUNT(CASE WHEN actual_outcome = 'win' THEN 1 END),
        COUNT(CASE WHEN actual_outcome = 'loss' THEN 1 END)
    INTO total, wins, losses
    FROM trading_signals
    WHERE symbol = symbol_name
    AND signal_datetime BETWEEN start_period AND end_period
    AND actual_outcome IN ('win', 'loss');

    -- Calcular retornos promedio
    SELECT
        AVG(CASE WHEN actual_outcome = 'win' THEN actual_return END),
        AVG(CASE WHEN actual_outcome = 'loss' THEN actual_return END),
        SUM(CASE WHEN actual_outcome = 'win' THEN actual_return ELSE 0 END),
        SUM(CASE WHEN actual_outcome = 'loss' THEN actual_return ELSE 0 END)
    INTO avg_win_return, avg_loss_return, total_win_return, total_loss_return
    FROM trading_signals
    WHERE symbol = symbol_name
    AND signal_datetime BETWEEN start_period AND end_period
    AND actual_outcome IN ('win', 'loss');

    RETURN QUERY SELECT
        total,
        wins,
        losses,
        CASE WHEN total > 0 THEN ROUND((wins::decimal / total * 100), 2) ELSE 0.00 END,
        COALESCE(avg_win_return, 0.00),
        COALESCE(avg_loss_return, 0.00),
        CASE
            WHEN total_loss_return < 0 THEN ROUND((total_win_return / ABS(total_loss_return)), 4)
            ELSE 0.00
        END,
        COALESCE(total_win_return + total_loss_return, 0.00);
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- FUNCIÓN: Limpiar datos antiguos
-- ========================================

CREATE OR REPLACE FUNCTION cleanup_old_data(
    retention_days INTEGER DEFAULT 365
)
RETURNS TABLE(
    table_name TEXT,
    deleted_count INTEGER
) AS $$
DECLARE
    cutoff_date TIMESTAMPTZ;
    deleted_sessions INTEGER;
    deleted_api_usage INTEGER;
    deleted_quality_checks INTEGER;
    deleted_websocket INTEGER;
BEGIN
    cutoff_date := NOW() - (retention_days || ' days')::INTERVAL;

    -- Limpiar sesiones expiradas
    DELETE FROM user_sessions WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_sessions = ROW_COUNT;

    -- Limpiar uso de API antiguo
    DELETE FROM api_usage WHERE created_at < cutoff_date;
    GET DIAGNOSTICS deleted_api_usage = ROW_COUNT;

    -- Limpiar checks de calidad antiguos
    DELETE FROM data_quality_checks WHERE created_at < cutoff_date;
    GET DIAGNOSTICS deleted_quality_checks = ROW_COUNT;

    -- Limpiar conexiones websocket desconectadas
    DELETE FROM websocket_connections
    WHERE status = 'disconnected' AND disconnected_at < cutoff_date;
    GET DIAGNOSTICS deleted_websocket = ROW_COUNT;

    RETURN QUERY VALUES
        ('user_sessions', deleted_sessions),
        ('api_usage', deleted_api_usage),
        ('data_quality_checks', deleted_quality_checks),
        ('websocket_connections', deleted_websocket);
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- CONFIRMAR CREACIÓN DE FUNCIONES
-- ========================================

\echo '✅ Funciones creadas exitosamente'
\echo '⏰ is_trading_hours() - Verificar horarios de trading'
\echo '📊 calculate_data_completeness() - Calcular completitud de datos'
\echo '🔍 find_data_gaps() - Encontrar gaps en los datos'
\echo '🔄 update_updated_at_column() - Trigger para timestamps'
\echo '✅ validate_ohlcv_data() - Validación de datos OHLCV'
\echo '📈 calculate_trading_performance() - Métricas de performance'
\echo '🧹 cleanup_old_data() - Limpieza de datos antiguos'