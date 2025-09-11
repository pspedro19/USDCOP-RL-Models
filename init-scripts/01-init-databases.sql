-- =====================================================
-- Initialize Databases for USDCOP Trading System
-- PostgreSQL initialization script
-- =====================================================

-- Connect to default postgres database
\c postgres;

-- Create databases
CREATE DATABASE trading_db;
CREATE DATABASE airflow;
CREATE DATABASE mlflow;

-- Create users
CREATE USER trading WITH PASSWORD 'trading123';
CREATE USER airflow WITH PASSWORD 'airflow123';
CREATE USER mlflow WITH PASSWORD 'mlflow123';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE trading_db TO trading;
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;

-- Connect to trading_db to create schemas
\c trading_db

-- Create schemas
CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;
CREATE SCHEMA IF NOT EXISTS models;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Grant schema privileges
GRANT ALL ON SCHEMA bronze TO trading;
GRANT ALL ON SCHEMA silver TO trading;
GRANT ALL ON SCHEMA gold TO trading;
GRANT ALL ON SCHEMA models TO trading;
GRANT ALL ON SCHEMA monitoring TO trading;