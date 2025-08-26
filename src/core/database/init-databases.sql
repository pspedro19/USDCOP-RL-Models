-- ═══════════════════════════════════════════════════════════════════════════════
-- Database Initialization Script
-- Creates users and databases for trading system and Airflow
-- ═══════════════════════════════════════════════════════════════════════════════

-- Create trading user and database
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'trading_user') THEN
        CREATE USER trading_user WITH PASSWORD 'trading123';
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'trading_db') THEN
        CREATE DATABASE trading_db OWNER trading_user;
    END IF;
END
$$;

-- Grant all privileges on trading database
GRANT ALL PRIVILEGES ON DATABASE trading_db TO trading_user;

-- Create airflow user and database
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'airflow') THEN
        CREATE USER airflow WITH PASSWORD 'airflow123';
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow') THEN
        CREATE DATABASE airflow OWNER airflow;
    END IF;
END
$$;

-- Grant all privileges on airflow database
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;

-- Create read-only user for analytics (optional)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'analytics_user') THEN
        CREATE USER analytics_user WITH PASSWORD 'analytics123';
    END IF;
END
$$;

-- Connect to trading_db to set up permissions
\c trading_db

-- Grant connect privilege
GRANT CONNECT ON DATABASE trading_db TO analytics_user;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;