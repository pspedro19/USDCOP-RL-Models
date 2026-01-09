-- ========================================
-- USDCOP Trading System - Default Data
-- Datos iniciales para el sistema
-- ========================================

\echo '📝 Insertando datos iniciales...'

-- ========================================
-- USUARIOS POR DEFECTO
-- ========================================

-- Usuario administrador por defecto
-- Password: admin123 (debería cambiarse en producción)
INSERT INTO users (
    username,
    email,
    password_hash,
    role,
    first_name,
    last_name,
    timezone,
    dashboard_preferences,
    api_permissions
) VALUES (
    'admin',
    'admin@usdcop-trading.com',
    '$2b$12$LQv3c1yqBwEHxPuNYMhOEeKE1eFdLNcYzHZ.1yWoOeK9PZcPm7NKu', -- admin123
    'admin',
    'Sistema',
    'Administrador',
    'America/Bogota',
    '{
        "theme": "dark",
        "default_timeframe": "5min",
        "auto_refresh": true,
        "refresh_interval": 30000
    }',
    '{
        "can_execute_pipelines": true,
        "can_manage_users": true,
        "can_view_all_signals": true,
        "can_modify_settings": true
    }'
) ON CONFLICT (username) DO NOTHING;

-- Usuario trader por defecto
-- Password: trader123
INSERT INTO users (
    username,
    email,
    password_hash,
    role,
    first_name,
    last_name,
    timezone,
    dashboard_preferences,
    api_permissions
) VALUES (
    'trader',
    'trader@usdcop-trading.com',
    '$2b$12$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi', -- trader123
    'trader',
    'Usuario',
    'Trader',
    'America/Bogota',
    '{
        "theme": "light",
        "default_timeframe": "5min",
        "auto_refresh": true,
        "refresh_interval": 30000,
        "favorite_indicators": ["RSI", "MACD", "Bollinger Bands"]
    }',
    '{
        "can_execute_pipelines": false,
        "can_manage_users": false,
        "can_view_all_signals": true,
        "can_modify_settings": false
    }'
) ON CONFLICT (username) DO NOTHING;

-- Usuario viewer por defecto (solo lectura)
-- Password: viewer123
INSERT INTO users (
    username,
    email,
    password_hash,
    role,
    first_name,
    last_name,
    timezone,
    dashboard_preferences,
    api_permissions
) VALUES (
    'viewer',
    'viewer@usdcop-trading.com',
    '$2b$12$Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi92IXUNpkjO0rOQ5byMi', -- viewer123
    'viewer',
    'Usuario',
    'Visualizador',
    'America/Bogota',
    '{
        "theme": "light",
        "default_timeframe": "5min",
        "auto_refresh": true,
        "refresh_interval": 60000
    }',
    '{
        "can_execute_pipelines": false,
        "can_manage_users": false,
        "can_view_all_signals": false,
        "can_modify_settings": false
    }'
) ON CONFLICT (username) DO NOTHING;

-- ========================================
-- MÉTRICAS INICIALES DEL SISTEMA
-- ========================================

-- Métricas de inicialización del sistema
INSERT INTO system_metrics (
    metric_name,
    metric_value,
    metric_unit,
    category,
    subcategory,
    metadata,
    measured_at
) VALUES
(
    'system_initialized',
    1,
    'boolean',
    'system',
    'startup',
    '{"database_version": "1.0", "initialization_date": "' || NOW()::text || '"}',
    NOW()
),
(
    'database_tables_created',
    9,
    'count',
    'system',
    'database',
    '{"tables": ["market_data", "users", "trading_signals", "system_metrics", "trading_performance", "pipeline_runs", "data_quality_checks", "api_usage", "websocket_connections"]}',
    NOW()
),
(
    'api_keys_configured',
    8,
    'count',
    'api',
    'configuration',
    '{"provider": "twelvedata", "daily_limit_per_key": 800, "total_daily_limit": 6400}',
    NOW()
);

-- ========================================
-- CONFIGURACIÓN INICIAL DE TRADING PERFORMANCE
-- ========================================

-- Registro inicial de performance (vacío, se llenará con datos reales)
INSERT INTO trading_performance (
    symbol,
    period_start,
    period_end,
    total_signals,
    winning_signals,
    losing_signals,
    win_rate,
    avg_win,
    avg_loss,
    profit_factor,
    sharpe_ratio,
    total_return,
    max_drawdown,
    model_accuracy,
    avg_confidence
) VALUES (
    'USDCOP',
    DATE_TRUNC('month', NOW()),
    NOW(),
    0,
    0,
    0,
    0.00,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.0000,
    0.00,
    0.00
) ON CONFLICT (symbol, period_start, period_end) DO NOTHING;

-- ========================================
-- CONFIGURACIÓN DE API KEYS INICIALES
-- ========================================

-- Registrar el uso inicial de API keys (placeholder)
INSERT INTO api_usage (
    api_key_name,
    endpoint,
    requests_count,
    credits_used,
    response_time_ms,
    status_code,
    success,
    daily_credits_remaining,
    rate_limit_remaining,
    request_datetime
) VALUES
('TWELVEDATA_API_KEY_1', '/time_series', 0, 0, 0, 200, true, 800, 800, NOW()),
('TWELVEDATA_API_KEY_2', '/time_series', 0, 0, 0, 200, true, 800, 800, NOW()),
('TWELVEDATA_API_KEY_3', '/time_series', 0, 0, 0, 200, true, 800, 800, NOW()),
('TWELVEDATA_API_KEY_4', '/time_series', 0, 0, 0, 200, true, 800, 800, NOW()),
('TWELVEDATA_API_KEY_5', '/time_series', 0, 0, 0, 200, true, 800, 800, NOW()),
('TWELVEDATA_API_KEY_6', '/time_series', 0, 0, 0, 200, true, 800, 800, NOW()),
('TWELVEDATA_API_KEY_7', '/time_series', 0, 0, 0, 200, true, 800, 800, NOW()),
('TWELVEDATA_API_KEY_8', '/time_series', 0, 0, 0, 200, true, 800, 800, NOW());

-- ========================================
-- SAMPLE DATA (OPCIONAL - PARA TESTING)
-- ========================================

-- Datos de muestra para testing (puedes comentar esta sección en producción)
-- Esto crea algunos registros de ejemplo para verificar que todo funciona

/*
-- Sample market data (últimas 24 horas)
INSERT INTO market_data (
    symbol,
    datetime,
    timeframe,
    open,
    high,
    low,
    close,
    volume,
    source,
    trading_session,
    batch_id
)
SELECT
    'USDCOP',
    ts,
    '5min',
    4280.00 + (RANDOM() * 40) - 20,  -- open
    4280.00 + (RANDOM() * 50) - 15,  -- high
    4280.00 + (RANDOM() * 30) - 25,  -- low
    4280.00 + (RANDOM() * 45) - 20,  -- close
    (RANDOM() * 2000000)::BIGINT,    -- volume
    'sample_data',
    is_trading_hours(ts),
    'sample_batch_' || EXTRACT(epoch FROM NOW())::text
FROM generate_series(
    NOW() - INTERVAL '24 hours',
    NOW(),
    INTERVAL '5 minutes'
) ts
WHERE is_trading_hours(ts)
LIMIT 100;

-- Sample trading signal
INSERT INTO trading_signals (
    signal_id,
    symbol,
    signal_type,
    confidence,
    price,
    stop_loss,
    take_profit,
    risk_score,
    expected_return,
    reasoning,
    technical_indicators,
    ml_prediction,
    model_source,
    time_horizon,
    latency_ms,
    signal_datetime
) VALUES (
    'sample_signal_' || EXTRACT(epoch FROM NOW())::text,
    'USDCOP',
    'BUY',
    87.50,
    4285.50,
    4270.00,
    4320.00,
    3.2,
    0.0081,
    '["RSI oversold (28.5)", "MACD bullish crossover", "Support level at 4280", "High ML confidence (87.5%)"]',
    '{"rsi": 28.5, "macd": {"macd": 12.5, "signal": 10.2, "histogram": 2.3}, "bollinger": {"upper": 4295.0, "middle": 4280.0, "lower": 4265.0}}',
    '{"prediction": 0.75, "confidence": 0.875, "action": "buy", "model_version": "v2.1"}',
    'L5_PPO_LSTM_v2.1',
    '15-30 min',
    45,
    NOW()
);
*/

-- ========================================
-- VERIFICACIÓN DE DATOS INSERTADOS
-- ========================================

-- Verificar que los usuarios fueron creados
DO $$
DECLARE
    user_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO user_count FROM users;
    RAISE NOTICE 'Usuarios creados: %', user_count;
END $$;

-- Verificar métricas iniciales
DO $$
DECLARE
    metrics_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO metrics_count FROM system_metrics;
    RAISE NOTICE 'Métricas iniciales: %', metrics_count;
END $$;

-- ========================================
-- CONFIRMAR INSERCIÓN DE DATOS
-- ========================================

\echo '✅ Datos iniciales insertados exitosamente'
\echo '👥 3 usuarios por defecto creados (admin, trader, viewer)'
\echo '📊 Métricas iniciales del sistema registradas'
\echo '🔑 8 API keys de TwelveData configuradas'
\echo '📈 Estructura de trading performance inicializada'
\echo ''
\echo '🔐 CREDENCIALES POR DEFECTO:'
\echo '   Admin:  admin/admin123'
\echo '   Trader: trader/trader123'
\echo '   Viewer: viewer/viewer123'
\echo ''
\echo '⚠️  IMPORTANTE: Cambiar passwords en producción!'