-- ========================================
-- USDCOP Trading System - Database Triggers
-- Triggers automáticos para mantenimiento y validación
-- ========================================

\echo '⚡ Creando triggers automáticos...'

-- ========================================
-- TRIGGER: Auto-actualizar updated_at en market_data
-- ========================================

CREATE TRIGGER trigger_market_data_updated_at
    BEFORE UPDATE ON market_data
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ========================================
-- TRIGGER: Auto-actualizar updated_at en users
-- ========================================

CREATE TRIGGER trigger_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ========================================
-- TRIGGER: Validación automática de datos OHLCV
-- ========================================

CREATE TRIGGER trigger_validate_ohlcv_insert
    BEFORE INSERT ON market_data
    FOR EACH ROW
    EXECUTE FUNCTION validate_ohlcv_data();

CREATE TRIGGER trigger_validate_ohlcv_update
    BEFORE UPDATE ON market_data
    FOR EACH ROW
    EXECUTE FUNCTION validate_ohlcv_data();

-- ========================================
-- FUNCIÓN: Logging automático de cambios importantes
-- ========================================

CREATE OR REPLACE FUNCTION log_important_changes()
RETURNS TRIGGER AS $$
BEGIN
    -- Log cambios en trading_signals cuando se actualiza el outcome
    IF TG_TABLE_NAME = 'trading_signals' AND OLD.actual_outcome IS DISTINCT FROM NEW.actual_outcome THEN
        INSERT INTO system_metrics (
            metric_name,
            metric_value,
            metric_unit,
            category,
            subcategory,
            symbol,
            metadata,
            measured_at
        ) VALUES (
            'signal_outcome_updated',
            1,
            'count',
            'trading',
            'signal_tracking',
            NEW.symbol,
            jsonb_build_object(
                'signal_id', NEW.signal_id,
                'old_outcome', OLD.actual_outcome,
                'new_outcome', NEW.actual_outcome,
                'actual_return', NEW.actual_return
            ),
            NOW()
        );
    END IF;

    -- Log cuando se insertan nuevas señales
    IF TG_TABLE_NAME = 'trading_signals' AND TG_OP = 'INSERT' THEN
        INSERT INTO system_metrics (
            metric_name,
            metric_value,
            metric_unit,
            category,
            subcategory,
            symbol,
            metadata,
            measured_at
        ) VALUES (
            'new_signal_created',
            1,
            'count',
            'trading',
            'signal_generation',
            NEW.symbol,
            jsonb_build_object(
                'signal_id', NEW.signal_id,
                'signal_type', NEW.signal_type,
                'confidence', NEW.confidence,
                'model_source', NEW.model_source
            ),
            NOW()
        );
    END IF;

    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- TRIGGER: Logging automático de trading signals
-- ========================================

CREATE TRIGGER trigger_log_trading_signals_insert
    AFTER INSERT ON trading_signals
    FOR EACH ROW
    EXECUTE FUNCTION log_important_changes();

CREATE TRIGGER trigger_log_trading_signals_update
    AFTER UPDATE ON trading_signals
    FOR EACH ROW
    EXECUTE FUNCTION log_important_changes();

-- ========================================
-- FUNCIÓN: Validar y limpiar sesiones expiradas
-- ========================================

CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS TRIGGER AS $$
BEGIN
    -- Limpiar sesiones expiradas cuando se inserta una nueva
    DELETE FROM user_sessions WHERE expires_at < NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- TRIGGER: Limpieza automática de sesiones
-- ========================================

CREATE TRIGGER trigger_cleanup_sessions
    BEFORE INSERT ON user_sessions
    FOR EACH STATEMENT
    EXECUTE FUNCTION cleanup_expired_sessions();

-- ========================================
-- FUNCIÓN: Validar límites de API usage
-- ========================================

CREATE OR REPLACE FUNCTION validate_api_limits()
RETURNS TRIGGER AS $$
DECLARE
    daily_usage INTEGER;
    api_key_limit INTEGER := 800; -- Límite diario por API key
BEGIN
    -- Calcular uso diario actual para esta API key
    SELECT COALESCE(SUM(credits_used), 0)
    INTO daily_usage
    FROM api_usage
    WHERE api_key_name = NEW.api_key_name
    AND DATE(request_datetime) = DATE(NEW.request_datetime);

    -- Actualizar credits_remaining
    NEW.daily_credits_remaining := api_key_limit - daily_usage - NEW.credits_used;

    -- Si excede el límite, registrar warning en system_metrics
    IF daily_usage + NEW.credits_used > api_key_limit THEN
        INSERT INTO system_metrics (
            metric_name,
            metric_value,
            metric_unit,
            category,
            subcategory,
            metadata,
            measured_at
        ) VALUES (
            'api_limit_exceeded',
            daily_usage + NEW.credits_used,
            'credits',
            'api',
            'rate_limiting',
            jsonb_build_object(
                'api_key_name', NEW.api_key_name,
                'daily_limit', api_key_limit,
                'usage', daily_usage + NEW.credits_used
            ),
            NOW()
        );
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- TRIGGER: Validación de límites de API
-- ========================================

CREATE TRIGGER trigger_validate_api_limits
    BEFORE INSERT ON api_usage
    FOR EACH ROW
    EXECUTE FUNCTION validate_api_limits();

-- ========================================
-- FUNCIÓN: Actualizar estado de pipeline automáticamente
-- ========================================

CREATE OR REPLACE FUNCTION update_pipeline_status()
RETURNS TRIGGER AS $$
BEGIN
    -- Si se está actualizando el end_time, calcular duración y estado final
    IF OLD.end_time IS NULL AND NEW.end_time IS NOT NULL THEN
        -- Si no hay error_message, marcar como success
        IF NEW.error_message IS NULL OR NEW.error_message = '' THEN
            NEW.status := 'success';
        ELSE
            NEW.status := 'failed';
        END IF;

        -- Registrar métrica de duración del pipeline
        INSERT INTO system_metrics (
            metric_name,
            metric_value,
            metric_unit,
            category,
            subcategory,
            metadata,
            measured_at
        ) VALUES (
            'pipeline_duration',
            EXTRACT(EPOCH FROM (NEW.end_time - NEW.start_time)),
            'seconds',
            'pipeline',
            'execution_time',
            jsonb_build_object(
                'pipeline_name', NEW.pipeline_name,
                'run_id', NEW.run_id,
                'status', NEW.status,
                'records_processed', NEW.records_processed
            ),
            NEW.end_time
        );
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- TRIGGER: Estado automático de pipeline
-- ========================================

CREATE TRIGGER trigger_update_pipeline_status
    BEFORE UPDATE ON pipeline_runs
    FOR EACH ROW
    EXECUTE FUNCTION update_pipeline_status();

-- ========================================
-- FUNCIÓN: Actualizar last_ping para websockets
-- ========================================

CREATE OR REPLACE FUNCTION update_websocket_ping()
RETURNS TRIGGER AS $$
BEGIN
    -- Auto-actualizar last_ping si no se especifica
    IF NEW.last_ping IS NULL THEN
        NEW.last_ping := NOW();
    END IF;

    -- Si se está desconectando, marcar disconnected_at
    IF OLD.status = 'active' AND NEW.status = 'disconnected' THEN
        NEW.disconnected_at := NOW();
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- TRIGGER: Gestión de conexiones WebSocket
-- ========================================

CREATE TRIGGER trigger_update_websocket_ping
    BEFORE UPDATE ON websocket_connections
    FOR EACH ROW
    EXECUTE FUNCTION update_websocket_ping();

-- ========================================
-- CONFIRMAR CREACIÓN DE TRIGGERS
-- ========================================

\echo '✅ Triggers creados exitosamente'
\echo '🔄 Auto-actualización de updated_at en tablas principales'
\echo '✅ Validación automática de datos OHLCV'
\echo '📊 Logging automático de cambios en trading signals'
\echo '🧹 Limpieza automática de sesiones expiradas'
\echo '🔑 Validación automática de límites de API'
\echo '⏱️ Actualización automática de estado de pipelines'
\echo '🌐 Gestión automática de conexiones WebSocket'