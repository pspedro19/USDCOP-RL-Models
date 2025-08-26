-- ═══════════════════════════════════════════════════════════════════════════════
-- Database Initialization Script (SQLite Version)
-- Creates database structure for USDCOP Trading RL System
-- ═══════════════════════════════════════════════════════════════════════════════

-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- Enable WAL mode for better concurrency
PRAGMA journal_mode = WAL;

-- Set synchronous mode for better performance
PRAGMA synchronous = NORMAL;

-- Set cache size for better performance
PRAGMA cache_size = 10000;

-- Set temp store to memory for better performance
PRAGMA temp_store = MEMORY;

-- Create database info table
CREATE TABLE IF NOT EXISTS database_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Insert initial database information
INSERT OR REPLACE INTO database_info (key, value, description) VALUES
    ('database_name', 'USDCOP_Trading_RL', 'Database name'),
    ('version', '1.0.0', 'Database schema version'),
    ('created_at', datetime('now'), 'Database creation timestamp'),
    ('last_updated', datetime('now'), 'Last update timestamp'),
    ('description', 'USDCOP Trading RL System Database', 'Database description');

-- Create database version tracking
CREATE TABLE IF NOT EXISTS schema_migrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL,
    description TEXT NOT NULL,
    applied_at TEXT DEFAULT (datetime('now')),
    checksum TEXT,
    UNIQUE(version)
);

-- Insert initial migration record
INSERT OR REPLACE INTO schema_migrations (version, description, checksum) VALUES
    ('1.0.0', 'Initial database schema', 'initial_schema_v1_0_0');
