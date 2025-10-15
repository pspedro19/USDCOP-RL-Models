-- ========================================
-- USDCOP Trading System - Database Indexes
-- Optimización de consultas para pipeline L0
-- ========================================

\echo '🔧 Creando índices de optimización...'

-- ========================================
-- ÍNDICES PARA market_data (TABLA PRINCIPAL)
-- ========================================

-- Índice principal por datetime (queries más comunes)
CREATE INDEX idx_market_data_datetime ON market_data (datetime DESC);

-- Índice compuesto para consultas por símbolo y fecha
CREATE INDEX idx_market_data_symbol_datetime ON market_data (symbol, datetime DESC);

-- Índice por fuente de datos
CREATE INDEX idx_market_data_source ON market_data (source);

-- Índice para sesión de trading (horario 8am-12:55pm)
CREATE INDEX idx_market_data_trading_session ON market_data (trading_session, datetime DESC);

-- Índice compuesto para consultas específicas de timeframe
CREATE INDEX idx_market_data_symbol_timeframe_datetime ON market_data (symbol, timeframe, datetime DESC);

-- Índice para pipeline runs
CREATE INDEX idx_market_data_pipeline_run ON market_data (pipeline_run_id);

-- ========================================
-- ÍNDICES PARA trading_signals
-- ========================================

-- Índice principal por fecha de señal
CREATE INDEX idx_signals_datetime ON trading_signals (signal_datetime DESC);

-- Índice por tipo de señal
CREATE INDEX idx_signals_type ON trading_signals (signal_type);

-- Índice compuesto símbolo + fecha
CREATE INDEX idx_signals_symbol_datetime ON trading_signals (symbol, signal_datetime DESC);

-- Índice por resultado actual (para análisis de performance)
CREATE INDEX idx_signals_outcome ON trading_signals (actual_outcome);

-- Índice por fuente del modelo
CREATE INDEX idx_signals_model_source ON trading_signals (model_source);

-- Índice por confianza (para filtrar señales de alta confianza)
CREATE INDEX idx_signals_confidence ON trading_signals (confidence DESC);

-- ========================================
-- ÍNDICES PARA system_metrics
-- ========================================

-- Índice principal por nombre de métrica y tiempo
CREATE INDEX idx_metrics_name_time ON system_metrics (metric_name, measured_at DESC);

-- Índice por categoría
CREATE INDEX idx_metrics_category ON system_metrics (category, measured_at DESC);

-- Índice compuesto para consultas específicas
CREATE INDEX idx_metrics_category_name ON system_metrics (category, metric_name, measured_at DESC);

-- Índice por símbolo para métricas específicas
CREATE INDEX idx_metrics_symbol ON system_metrics (symbol, measured_at DESC);

-- ========================================
-- ÍNDICES PARA pipeline_runs
-- ========================================

-- Índice por nombre de pipeline y fecha de ejecución
CREATE INDEX idx_pipeline_runs_name_date ON pipeline_runs (pipeline_name, execution_date DESC);

-- Índice por estado
CREATE INDEX idx_pipeline_runs_status ON pipeline_runs (status, start_time DESC);

-- Índice por fecha de inicio
CREATE INDEX idx_pipeline_runs_start_time ON pipeline_runs (start_time DESC);

-- ========================================
-- ÍNDICES PARA trading_performance
-- ========================================

-- Índice por símbolo y período
CREATE INDEX idx_performance_symbol_period ON trading_performance (symbol, period_end DESC);

-- Índice por fecha de fin del período
CREATE INDEX idx_performance_period_end ON trading_performance (period_end DESC);

-- ========================================
-- ÍNDICES PARA api_usage
-- ========================================

-- Índice por fecha de request (para monitoreo diario)
CREATE INDEX idx_api_usage_datetime ON api_usage (request_datetime DESC);

-- Índice por API key y fecha
CREATE INDEX idx_api_usage_key_datetime ON api_usage (api_key_name, request_datetime DESC);

-- Índice por éxito/fallo
CREATE INDEX idx_api_usage_success ON api_usage (success, request_datetime DESC);

-- Índice para consultas diarias de uso
CREATE INDEX idx_api_usage_daily ON api_usage (api_key_name, DATE(request_datetime));

-- ========================================
-- ÍNDICES PARA data_quality_checks
-- ========================================

-- Índice por run_id
CREATE INDEX idx_quality_checks_run_id ON data_quality_checks (run_id);

-- Índice por tipo de check y estado
CREATE INDEX idx_quality_checks_type_status ON data_quality_checks (check_type, status, created_at DESC);

-- ========================================
-- ÍNDICES PARA users y user_sessions
-- ========================================

-- Índice por email para login
CREATE INDEX idx_users_email ON users (email);

-- Índice por username
CREATE INDEX idx_users_username ON users (username);

-- Índice por estado activo
CREATE INDEX idx_users_active ON users (is_active, created_at DESC);

-- Índice para sesiones por token
CREATE INDEX idx_sessions_token ON user_sessions (session_token);

-- Índice para sesiones por usuario
CREATE INDEX idx_sessions_user_id ON user_sessions (user_id, created_at DESC);

-- Índice para limpieza de sesiones expiradas
CREATE INDEX idx_sessions_expires ON user_sessions (expires_at);

-- ========================================
-- ÍNDICES PARA websocket_connections
-- ========================================

-- Índice por connection_id
CREATE INDEX idx_websocket_connection_id ON websocket_connections (connection_id);

-- Índice por usuario
CREATE INDEX idx_websocket_user_id ON websocket_connections (user_id, connected_at DESC);

-- Índice por estado
CREATE INDEX idx_websocket_status ON websocket_connections (status, last_ping DESC);

-- ========================================
-- CONFIRMAR CREACIÓN DE ÍNDICES
-- ========================================

\echo '✅ Índices creados exitosamente'
\echo '🚀 Base de datos optimizada para consultas de alta performance'
\echo '📊 market_data: 6 índices para consultas por tiempo, símbolo y fuente'
\echo '📈 trading_signals: 6 índices para análisis de señales'
\echo '📊 system_metrics: 4 índices para monitoreo del sistema'
\echo '🔄 pipeline_runs: 3 índices para seguimiento de ejecuciones'
\echo '🔑 api_usage: 4 índices para control de límites de API'
\echo '👥 users: 4 índices para autenticación y gestión'
\echo '🌐 websocket_connections: 3 índices para conexiones en tiempo real'