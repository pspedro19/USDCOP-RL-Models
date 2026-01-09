-- ========================================
-- USDCOP Trading System - Database Schema
-- Creación de tablas para pipeline L0 database-centric
-- ========================================

-- Extension para TimescaleDB (si está disponible)
-- CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ========================================
-- 1. TABLA DE USUARIOS Y AUTENTICACIÓN
-- ========================================

CREATE TABLE users (
    id SERIAL PRIMARY KEY,

    -- Credenciales
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'trader', -- 'admin', 'trader', 'viewer'

    -- Información personal
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    timezone VARCHAR(50) DEFAULT 'America/Bogota',

    -- Configuraciones
    dashboard_preferences JSONB DEFAULT '{}',
    api_permissions JSONB DEFAULT '{}',

    -- Estado
    is_active BOOLEAN DEFAULT true,
    last_login TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tabla de sesiones para autenticación JWT
CREATE TABLE user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT
);

-- ========================================
-- 2. TABLA PRINCIPAL: DATOS DE MERCADO (HISTÓRICO + TIEMPO REAL)
-- ========================================

CREATE TABLE market_data (
    id BIGSERIAL PRIMARY KEY,

    -- Identificación del instrumento
    symbol VARCHAR(10) NOT NULL DEFAULT 'USDCOP',
    timeframe VARCHAR(10) NOT NULL DEFAULT '5min',
    datetime TIMESTAMPTZ NOT NULL,

    -- Datos OHLCV (Open, High, Low, Close, Volume)
    open DECIMAL(12,4) NOT NULL,
    high DECIMAL(12,4) NOT NULL,
    low DECIMAL(12,4) NOT NULL,
    close DECIMAL(12,4) NOT NULL,
    volume BIGINT DEFAULT 0,

    -- Metadatos de origen
    source VARCHAR(20) NOT NULL, -- 'twelvedata', 'mt5', 'websocket'
    timezone VARCHAR(50) DEFAULT 'America/Bogota',
    trading_session BOOLEAN DEFAULT true, -- true si está en horario 8am-12:55pm COT

    -- Campos de auditoría
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    batch_id VARCHAR(100),
    pipeline_run_id VARCHAR(100),

    -- Restricción única para evitar duplicados
    UNIQUE(symbol, datetime, timeframe, source)
);

-- ========================================
-- 3. TABLA DE EJECUCIONES DE PIPELINE
-- ========================================

CREATE TABLE pipeline_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) UNIQUE NOT NULL,

    -- Identificación del pipeline
    pipeline_name VARCHAR(50) NOT NULL,
    dag_id VARCHAR(100),
    task_id VARCHAR(100),

    -- Detalles de ejecución
    execution_date TIMESTAMPTZ NOT NULL,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    status VARCHAR(20), -- 'running', 'success', 'failed', 'skipped'

    -- Procesamiento de datos
    records_processed INTEGER DEFAULT 0,
    records_inserted INTEGER DEFAULT 0,
    records_updated INTEGER DEFAULT 0,
    records_skipped INTEGER DEFAULT 0,

    -- Rango de fechas procesadas
    data_start_date TIMESTAMPTZ,
    data_end_date TIMESTAMPTZ,

    -- Metadatos
    config JSONB DEFAULT '{}',
    error_message TEXT,
    logs TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ========================================
-- 4. TABLA DE SEÑALES DE TRADING
-- ========================================

CREATE TABLE trading_signals (
    id SERIAL PRIMARY KEY,
    signal_id VARCHAR(100) UNIQUE NOT NULL,

    -- Datos de la señal
    symbol VARCHAR(10) NOT NULL DEFAULT 'USDCOP',
    signal_type VARCHAR(10) NOT NULL, -- 'BUY', 'SELL', 'HOLD'
    confidence DECIMAL(5,2) NOT NULL, -- 0-100
    price DECIMAL(12,4) NOT NULL,

    -- Gestión de riesgo
    stop_loss DECIMAL(12,4),
    take_profit DECIMAL(12,4),
    risk_score DECIMAL(3,1), -- 1-10
    expected_return DECIMAL(8,4),

    -- Metadatos y análisis
    reasoning JSONB, -- array de razones
    technical_indicators JSONB, -- RSI, MACD, etc.
    ml_prediction JSONB, -- predicción del modelo ML
    model_source VARCHAR(50),
    time_horizon VARCHAR(20),
    latency_ms INTEGER,

    -- Timestamps
    signal_datetime TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Seguimiento de rendimiento
    actual_outcome VARCHAR(10), -- 'win', 'loss', 'pending'
    actual_return DECIMAL(8,4),
    closed_at TIMESTAMPTZ
);

-- ========================================
-- 5. TABLA DE MÉTRICAS DEL SISTEMA
-- ========================================

CREATE TABLE system_metrics (
    id SERIAL PRIMARY KEY,

    -- Identificación de la métrica
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,4),
    metric_unit VARCHAR(20),
    category VARCHAR(50), -- 'pipeline', 'trading', 'system', 'api'
    subcategory VARCHAR(50),

    -- Dimensiones
    symbol VARCHAR(10),
    timeframe VARCHAR(10),
    source VARCHAR(20),

    -- Metadatos adicionales
    metadata JSONB DEFAULT '{}',
    measured_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ========================================
-- 6. TABLA DE RENDIMIENTO DE TRADING
-- ========================================

CREATE TABLE trading_performance (
    id SERIAL PRIMARY KEY,

    -- Identificación del período
    symbol VARCHAR(10) NOT NULL DEFAULT 'USDCOP',
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,

    -- Métricas de rendimiento
    total_signals INTEGER DEFAULT 0,
    winning_signals INTEGER DEFAULT 0,
    losing_signals INTEGER DEFAULT 0,
    win_rate DECIMAL(5,2),
    avg_win DECIMAL(12,4),
    avg_loss DECIMAL(12,4),
    profit_factor DECIMAL(8,4),
    sharpe_ratio DECIMAL(8,4),
    total_return DECIMAL(12,4),
    max_drawdown DECIMAL(12,4),

    -- Rendimiento del modelo ML
    model_accuracy DECIMAL(5,2),
    avg_confidence DECIMAL(5,2),

    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Evitar períodos duplicados
    UNIQUE(symbol, period_start, period_end)
);

-- ========================================
-- 7. TABLA DE CONTROLES DE CALIDAD
-- ========================================

CREATE TABLE data_quality_checks (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) REFERENCES pipeline_runs(run_id),

    -- Detalles del control
    check_name VARCHAR(100) NOT NULL,
    check_type VARCHAR(50), -- 'completeness', 'accuracy', 'timeliness', 'consistency'
    status VARCHAR(20), -- 'passed', 'failed', 'warning'

    -- Resultados
    expected_value DECIMAL(15,4),
    actual_value DECIMAL(15,4),
    threshold DECIMAL(15,4),

    -- Contexto
    symbol VARCHAR(10),
    date_range_start TIMESTAMPTZ,
    date_range_end TIMESTAMPTZ,

    details JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ========================================
-- 8. TABLA DE USO DE API
-- ========================================

CREATE TABLE api_usage (
    id SERIAL PRIMARY KEY,

    -- Identificación de la API key
    api_key_name VARCHAR(50) NOT NULL, -- 'TWELVEDATA_API_KEY_1', etc.
    endpoint VARCHAR(100),

    -- Seguimiento de uso
    requests_count INTEGER DEFAULT 1,
    credits_used INTEGER DEFAULT 1,
    response_time_ms INTEGER,

    -- Estado de la respuesta
    status_code INTEGER,
    success BOOLEAN DEFAULT true,
    error_message TEXT,

    -- Límites de velocidad
    daily_credits_remaining INTEGER,
    rate_limit_remaining INTEGER,

    -- Timestamps
    request_datetime TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ========================================
-- 9. TABLA DE CONEXIONES WEBSOCKET
-- ========================================

CREATE TABLE websocket_connections (
    id SERIAL PRIMARY KEY,
    connection_id VARCHAR(100) UNIQUE NOT NULL,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,

    -- Detalles de conexión
    ip_address INET,
    user_agent TEXT,
    subscriptions JSONB DEFAULT '[]', -- ['market_data', 'trading_signals']

    -- Estado
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'disconnected'
    last_ping TIMESTAMPTZ DEFAULT NOW(),

    -- Timestamps
    connected_at TIMESTAMPTZ DEFAULT NOW(),
    disconnected_at TIMESTAMPTZ
);

-- ========================================
-- CONFIRMAR CREACIÓN DE TABLAS
-- ========================================

\echo '✅ Tablas creadas exitosamente'
\echo '📊 market_data - Datos OHLCV históricos + tiempo real'
\echo '👥 users - Gestión de usuarios y autenticación'
\echo '📈 trading_signals - Señales BUY/SELL/HOLD del modelo ML'
\echo '📊 system_metrics - Métricas generales del sistema'
\echo '📈 trading_performance - Rendimiento de trading por períodos'
\echo '🔄 pipeline_runs - Ejecuciones del pipeline L0'
\echo '✅ data_quality_checks - Controles de calidad de datos'
\echo '🔑 api_usage - Uso y límites de las 8 API keys'
\echo '🌐 websocket_connections - Conexiones en tiempo real'