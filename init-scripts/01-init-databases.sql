-- =====================================================
-- Initialize Databases for USDCOP Trading System
-- =====================================================

-- Create main trading database
CREATE DATABASE IF NOT EXISTS trading_db;

-- Create Airflow database
CREATE DATABASE IF NOT EXISTS airflow;

-- Create MLflow database  
CREATE DATABASE IF NOT EXISTS mlflow;

-- Create users
CREATE USER IF NOT EXISTS 'trading'@'%' IDENTIFIED BY 'trading123';
CREATE USER IF NOT EXISTS 'airflow'@'%' IDENTIFIED BY 'airflow123';
CREATE USER IF NOT EXISTS 'mlflow'@'%' IDENTIFIED BY 'mlflow123';

-- Grant privileges
GRANT ALL PRIVILEGES ON trading_db.* TO 'trading'@'%';
GRANT ALL PRIVILEGES ON airflow.* TO 'airflow'@'%';
GRANT ALL PRIVILEGES ON mlflow.* TO 'mlflow'@'%';

-- For PostgreSQL syntax (if using PostgreSQL instead of MySQL)
\c postgres

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