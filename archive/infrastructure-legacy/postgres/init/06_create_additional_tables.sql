-- ========================================
-- USDCOP Trading System - Additional Tables
-- Enhanced tables for backup system, ready signals, and gap management
-- ========================================

\echo '🔧 Creating additional tables for enhanced L0 pipeline...'

-- ========================================
-- 1. BACKUP METADATA TABLE
-- Track backup files, freshness, and completeness
-- ========================================

CREATE TABLE backup_metadata (
    id SERIAL PRIMARY KEY,

    -- Backup identification
    backup_id VARCHAR(100) UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    backup_type VARCHAR(50) NOT NULL, -- 'incremental', 'full', 'differential'

    -- Data coverage
    symbol VARCHAR(10) NOT NULL DEFAULT 'USDCOP',
    timeframe VARCHAR(10) NOT NULL DEFAULT '5min',
    data_start_date TIMESTAMPTZ NOT NULL,
    data_end_date TIMESTAMPTZ NOT NULL,

    -- File metadata
    file_size_bytes BIGINT NOT NULL,
    record_count INTEGER NOT NULL,
    compression_type VARCHAR(20) DEFAULT 'gzip',
    checksum VARCHAR(64), -- SHA-256 hash for integrity

    -- Quality metrics
    completeness_pct DECIMAL(5,2) NOT NULL DEFAULT 0.00,
    data_quality_score DECIMAL(5,2) DEFAULT NULL,
    validation_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'validated', 'failed'
    validation_errors JSONB DEFAULT '[]',

    -- Backup process metadata
    pipeline_run_id VARCHAR(100),
    source_table VARCHAR(100) DEFAULT 'market_data',
    backup_method VARCHAR(50) DEFAULT 'database_export',
    compression_ratio DECIMAL(6,2),

    -- Lifecycle management
    is_active BOOLEAN DEFAULT true,
    retention_days INTEGER DEFAULT 30,
    expire_date TIMESTAMPTZ,
    restore_count INTEGER DEFAULT 0,
    last_restored_at TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_backup_dates CHECK (data_end_date >= data_start_date),
    CONSTRAINT chk_backup_completeness CHECK (completeness_pct >= 0 AND completeness_pct <= 100),
    CONSTRAINT chk_backup_quality CHECK (data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100))
);

-- ========================================
-- 2. READY SIGNALS TABLE
-- Coordinate L0→WebSocket handover with status tracking
-- ========================================

CREATE TABLE ready_signals (
    id SERIAL PRIMARY KEY,

    -- Signal identification
    signal_id VARCHAR(100) UNIQUE NOT NULL,
    pipeline_run_id VARCHAR(100) NOT NULL,

    -- Data scope
    symbol VARCHAR(10) NOT NULL DEFAULT 'USDCOP',
    timeframe VARCHAR(10) NOT NULL DEFAULT '5min',
    data_start_time TIMESTAMPTZ NOT NULL,
    data_end_time TIMESTAMPTZ NOT NULL,

    -- Signal details
    signal_type VARCHAR(50) NOT NULL, -- 'data_ready', 'backup_complete', 'quality_check_passed'
    status VARCHAR(20) NOT NULL DEFAULT 'pending', -- 'pending', 'acknowledged', 'processed', 'expired'
    priority INTEGER DEFAULT 1, -- 1=highest, 5=lowest

    -- Data metrics
    records_available INTEGER NOT NULL DEFAULT 0,
    completeness_pct DECIMAL(5,2) NOT NULL DEFAULT 0.00,
    quality_score DECIMAL(5,2),
    latency_ms INTEGER, -- Time from data creation to signal

    -- Processing tracking
    acknowledged_by VARCHAR(100), -- Component that acknowledged the signal
    acknowledged_at TIMESTAMPTZ,
    processed_by VARCHAR(100), -- Component that processed the signal
    processed_at TIMESTAMPTZ,
    processing_duration_ms INTEGER,

    -- Signal metadata
    metadata JSONB DEFAULT '{}',
    dependencies JSONB DEFAULT '[]', -- List of required conditions
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,

    -- Lifecycle
    expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '1 hour'),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_ready_signal_dates CHECK (data_end_time >= data_start_time),
    CONSTRAINT chk_ready_signal_completeness CHECK (completeness_pct >= 0 AND completeness_pct <= 100),
    CONSTRAINT chk_ready_signal_quality CHECK (quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 100)),
    CONSTRAINT chk_ready_signal_priority CHECK (priority >= 1 AND priority <= 5),

    -- Foreign key
    FOREIGN KEY (pipeline_run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE
);

-- ========================================
-- 3. DATA GAPS TABLE
-- Track detected gaps and their fill status
-- ========================================

CREATE TABLE data_gaps (
    id SERIAL PRIMARY KEY,

    -- Gap identification
    gap_id VARCHAR(100) UNIQUE NOT NULL,

    -- Data scope
    symbol VARCHAR(10) NOT NULL DEFAULT 'USDCOP',
    timeframe VARCHAR(10) NOT NULL DEFAULT '5min',
    source VARCHAR(20) NOT NULL,

    -- Gap details
    gap_start TIMESTAMPTZ NOT NULL,
    gap_end TIMESTAMPTZ NOT NULL,
    gap_duration INTERVAL NOT NULL,
    missing_points INTEGER NOT NULL,
    expected_points INTEGER NOT NULL,

    -- Detection metadata
    detected_by VARCHAR(100) NOT NULL, -- 'pipeline_l0', 'quality_check', 'manual'
    detection_method VARCHAR(50) NOT NULL, -- 'missing_sequence', 'trading_hours_check', 'comparison'
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    detection_run_id VARCHAR(100),

    -- Gap classification
    gap_type VARCHAR(50) NOT NULL, -- 'trading_hours', 'data_provider', 'system_outage', 'weekend'
    severity VARCHAR(20) NOT NULL DEFAULT 'medium', -- 'low', 'medium', 'high', 'critical'
    impact_score DECIMAL(5,2) DEFAULT 0.00, -- 0-100 business impact

    -- Fill status
    fill_status VARCHAR(20) NOT NULL DEFAULT 'detected', -- 'detected', 'filling', 'filled', 'unfillable', 'ignored'
    fill_method VARCHAR(50), -- 'api_backfill', 'interpolation', 'backup_restore', 'manual'
    fill_source VARCHAR(100), -- Which system/API filled the gap
    fill_quality VARCHAR(20), -- 'perfect', 'good', 'acceptable', 'poor'

    -- Fill attempts tracking
    fill_attempts INTEGER DEFAULT 0,
    max_fill_attempts INTEGER DEFAULT 3,
    last_fill_attempt TIMESTAMPTZ,
    next_fill_attempt TIMESTAMPTZ,

    -- Fill results
    filled_points INTEGER DEFAULT 0,
    fill_success_rate DECIMAL(5,2), -- Percentage of points successfully filled
    filled_at TIMESTAMPTZ,
    fill_duration_ms INTEGER,

    -- Resolution
    resolution_status VARCHAR(20) DEFAULT 'open', -- 'open', 'resolved', 'permanent', 'ignored'
    resolution_reason TEXT,
    resolved_by VARCHAR(100),
    resolved_at TIMESTAMPTZ,

    -- Metadata and context
    context_data JSONB DEFAULT '{}', -- Additional context about the gap
    affected_sessions JSONB DEFAULT '[]', -- Trading sessions affected
    business_impact TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_gap_dates CHECK (gap_end >= gap_start),
    CONSTRAINT chk_gap_points CHECK (missing_points <= expected_points),
    CONSTRAINT chk_gap_filled_points CHECK (filled_points <= missing_points),
    CONSTRAINT chk_gap_impact CHECK (impact_score >= 0 AND impact_score <= 100),
    CONSTRAINT chk_gap_success_rate CHECK (fill_success_rate IS NULL OR (fill_success_rate >= 0 AND fill_success_rate <= 100))
);

-- ========================================
-- 4. PIPELINE HEALTH MONITORING TABLE
-- Track overall pipeline health and performance
-- ========================================

CREATE TABLE pipeline_health (
    id SERIAL PRIMARY KEY,

    -- Health check identification
    check_id VARCHAR(100) UNIQUE NOT NULL,
    pipeline_name VARCHAR(50) NOT NULL,
    component VARCHAR(50) NOT NULL, -- 'data_ingestion', 'processing', 'storage', 'api'

    -- Health metrics
    status VARCHAR(20) NOT NULL, -- 'healthy', 'degraded', 'unhealthy', 'critical'
    health_score DECIMAL(5,2) NOT NULL, -- 0-100
    response_time_ms INTEGER,
    error_rate DECIMAL(5,2) DEFAULT 0.00, -- Percentage
    throughput_records_per_min INTEGER,

    -- Resource utilization
    cpu_usage_pct DECIMAL(5,2),
    memory_usage_pct DECIMAL(5,2),
    disk_usage_pct DECIMAL(5,2),

    -- Specific checks
    connectivity_status BOOLEAN,
    data_freshness_min INTEGER, -- Minutes since last data update
    queue_length INTEGER,
    active_connections INTEGER,

    -- Alerts and issues
    alert_level VARCHAR(20) DEFAULT 'none', -- 'none', 'info', 'warning', 'error', 'critical'
    issues JSONB DEFAULT '[]',
    recommendations JSONB DEFAULT '[]',

    -- Context
    check_metadata JSONB DEFAULT '{}',
    environment VARCHAR(20) DEFAULT 'production',
    version VARCHAR(20),

    -- Timestamps
    measured_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_health_score CHECK (health_score >= 0 AND health_score <= 100),
    CONSTRAINT chk_error_rate CHECK (error_rate >= 0 AND error_rate <= 100),
    CONSTRAINT chk_resource_usage CHECK (
        (cpu_usage_pct IS NULL OR (cpu_usage_pct >= 0 AND cpu_usage_pct <= 100)) AND
        (memory_usage_pct IS NULL OR (memory_usage_pct >= 0 AND memory_usage_pct <= 100)) AND
        (disk_usage_pct IS NULL OR (disk_usage_pct >= 0 AND disk_usage_pct <= 100))
    )
);

-- ========================================
-- INDEXES FOR NEW TABLES
-- ========================================

-- Backup metadata indexes
CREATE INDEX idx_backup_metadata_backup_id ON backup_metadata (backup_id);
CREATE INDEX idx_backup_metadata_dates ON backup_metadata (data_start_date, data_end_date);
CREATE INDEX idx_backup_metadata_symbol_timeframe ON backup_metadata (symbol, timeframe, created_at DESC);
CREATE INDEX idx_backup_metadata_status ON backup_metadata (validation_status, is_active);
CREATE INDEX idx_backup_metadata_pipeline_run ON backup_metadata (pipeline_run_id);
CREATE INDEX idx_backup_metadata_expiry ON backup_metadata (expire_date) WHERE is_active = true;

-- Ready signals indexes
CREATE INDEX idx_ready_signals_signal_id ON ready_signals (signal_id);
CREATE INDEX idx_ready_signals_pipeline_run ON ready_signals (pipeline_run_id);
CREATE INDEX idx_ready_signals_status ON ready_signals (status, created_at DESC);
CREATE INDEX idx_ready_signals_type ON ready_signals (signal_type, status);
CREATE INDEX idx_ready_signals_symbol_timeframe ON ready_signals (symbol, timeframe, data_start_time DESC);
CREATE INDEX idx_ready_signals_pending ON ready_signals (status, priority, created_at) WHERE status = 'pending';
CREATE INDEX idx_ready_signals_expires ON ready_signals (expires_at) WHERE status IN ('pending', 'acknowledged');

-- Data gaps indexes
CREATE INDEX idx_data_gaps_gap_id ON data_gaps (gap_id);
CREATE INDEX idx_data_gaps_symbol_timeframe ON data_gaps (symbol, timeframe, gap_start DESC);
CREATE INDEX idx_data_gaps_status ON data_gaps (fill_status, resolution_status);
CREATE INDEX idx_data_gaps_dates ON data_gaps (gap_start, gap_end);
CREATE INDEX idx_data_gaps_detected ON data_gaps (detected_by, detected_at DESC);
CREATE INDEX idx_data_gaps_severity ON data_gaps (severity, gap_start DESC);
CREATE INDEX idx_data_gaps_unfilled ON data_gaps (fill_status, next_fill_attempt) WHERE fill_status IN ('detected', 'filling');

-- Pipeline health indexes
CREATE INDEX idx_pipeline_health_check_id ON pipeline_health (check_id);
CREATE INDEX idx_pipeline_health_pipeline_component ON pipeline_health (pipeline_name, component, measured_at DESC);
CREATE INDEX idx_pipeline_health_status ON pipeline_health (status, measured_at DESC);
CREATE INDEX idx_pipeline_health_alerts ON pipeline_health (alert_level, measured_at DESC) WHERE alert_level != 'none';
CREATE INDEX idx_pipeline_health_measured_at ON pipeline_health (measured_at DESC);

-- ========================================
-- AUTO-UPDATE TRIGGERS FOR NEW TABLES
-- ========================================

-- Auto-update updated_at for backup_metadata
CREATE TRIGGER trigger_backup_metadata_updated_at
    BEFORE UPDATE ON backup_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Auto-update updated_at for ready_signals
CREATE TRIGGER trigger_ready_signals_updated_at
    BEFORE UPDATE ON ready_signals
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Auto-update updated_at for data_gaps
CREATE TRIGGER trigger_data_gaps_updated_at
    BEFORE UPDATE ON data_gaps
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ========================================
-- CONFIRMAR CREACIÓN DE TABLAS ADICIONALES
-- ========================================

\echo '✅ Additional tables created successfully'
\echo '💾 backup_metadata - Track backup files and metadata'
\echo '🔔 ready_signals - Coordinate L0→WebSocket handover'
\echo '📊 data_gaps - Track and manage data gaps'
\echo '❤️ pipeline_health - Monitor pipeline health and performance'
\echo '🔧 Additional indexes and triggers created'