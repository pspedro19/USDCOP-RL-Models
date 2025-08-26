-- =====================================================
-- Initialize Databases for USDCOP Trading System (PostgreSQL)
-- =====================================================

-- Check if databases exist before creating
SELECT 'CREATE DATABASE trading_db' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'trading_db')\gexec
SELECT 'CREATE DATABASE airflow' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')\gexec
SELECT 'CREATE DATABASE mlflow' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec

-- Create users if they don't exist
DO
$$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'trading') THEN
      CREATE USER trading WITH PASSWORD 'trading123';
   END IF;
   IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'airflow') THEN
      CREATE USER airflow WITH PASSWORD 'airflow123';
   END IF;
   IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'mlflow') THEN
      CREATE USER mlflow WITH PASSWORD 'mlflow123';
   END IF;
END
$$;

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