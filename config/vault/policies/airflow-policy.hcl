# =============================================================================
# USD/COP Trading System - Vault Policy for Apache Airflow
# =============================================================================
#
# This policy grants access to secrets required by Airflow for:
# - Airflow's own configuration secrets (Fernet key, secret key)
# - TwelveData API keys for data pipeline DAGs
# - Database credentials for storing DAG metadata and accessing trading DB
# - MinIO credentials for pipeline artifact storage
#
# Usage:
#   vault policy write airflow-service config/vault/policies/airflow-policy.hcl
#
# Associated AppRole: airflow-service
# Services: airflow-scheduler, airflow-webserver, airflow-worker
#
# Author: Pedro @ Lean Tech Solutions
# Version: 1.0.0
# Date: 2026-01-17
# =============================================================================

# -----------------------------------------------------------------------------
# Airflow Configuration Secrets
# -----------------------------------------------------------------------------
# Read access to Airflow-specific configuration secrets

# Airflow encryption and authentication keys
path "secret/data/airflow" {
  capabilities = ["read"]
}

path "secret/data/airflow/*" {
  capabilities = ["read"]
}

# Specific Airflow secrets
path "secret/data/airflow/fernet" {
  capabilities = ["read"]
}

path "secret/data/airflow/webserver" {
  capabilities = ["read"]
}

# Airflow connections stored in Vault
path "secret/data/airflow/connections" {
  capabilities = ["read"]
}

path "secret/data/airflow/connections/*" {
  capabilities = ["read", "list"]
}

# Airflow variables stored in Vault
path "secret/data/airflow/variables" {
  capabilities = ["read"]
}

path "secret/data/airflow/variables/*" {
  capabilities = ["read", "list"]
}

# List capability for Airflow metadata paths
path "secret/metadata/airflow" {
  capabilities = ["list"]
}

path "secret/metadata/airflow/*" {
  capabilities = ["list"]
}

# -----------------------------------------------------------------------------
# TwelveData API Keys - Data Pipeline Access
# -----------------------------------------------------------------------------
# Read access to TwelveData API keys for L0 data acquisition DAGs

# Legacy keys
path "secret/data/trading/twelvedata" {
  capabilities = ["read"]
}

# Group 1 keys (used by optimized-l0-validator)
path "secret/data/trading/twelvedata/g1" {
  capabilities = ["read"]
}

# Group 2 keys (used by gap-filling DAGs)
path "secret/data/trading/twelvedata/g2" {
  capabilities = ["read"]
}

# Wildcard for all TwelveData paths
path "secret/data/trading/twelvedata/*" {
  capabilities = ["read"]
}

# List capability to enumerate key paths
path "secret/metadata/trading/twelvedata" {
  capabilities = ["list"]
}

path "secret/metadata/trading/twelvedata/*" {
  capabilities = ["list"]
}

# -----------------------------------------------------------------------------
# Database Credentials - Pipeline and Metadata Storage
# -----------------------------------------------------------------------------
# Read access to database credentials for:
# - Airflow metadata database
# - Trading database (for DAGs that read/write trading data)

# Main trading database
path "secret/data/trading/database" {
  capabilities = ["read"]
}

path "secret/data/trading/database/*" {
  capabilities = ["read"]
}

# Airflow metadata database (if separate from trading DB)
path "secret/data/airflow/database" {
  capabilities = ["read"]
}

path "secret/data/airflow/database/*" {
  capabilities = ["read"]
}

# Read-only database user for reporting DAGs
path "secret/data/trading/database/readonly" {
  capabilities = ["read"]
}

# -----------------------------------------------------------------------------
# MinIO Credentials - Pipeline Artifact Storage
# -----------------------------------------------------------------------------
# Read access to MinIO for storing pipeline data (L0-L6 layers)

path "secret/data/trading/minio" {
  capabilities = ["read"]
}

path "secret/data/trading/minio/*" {
  capabilities = ["read"]
}

# Airflow-specific MinIO bucket credentials
path "secret/data/airflow/minio" {
  capabilities = ["read"]
}

# -----------------------------------------------------------------------------
# Redis Credentials - Task Queue and Caching
# -----------------------------------------------------------------------------
# Read access to Redis for Celery executor and result backend

path "secret/data/trading/redis" {
  capabilities = ["read"]
}

path "secret/data/trading/redis/*" {
  capabilities = ["read"]
}

# -----------------------------------------------------------------------------
# Notification Credentials
# -----------------------------------------------------------------------------
# Read access for DAG alerting (Slack webhooks, email)

path "secret/data/trading/notifications" {
  capabilities = ["read"]
}

path "secret/data/trading/notifications/*" {
  capabilities = ["read"]
}

# Airflow-specific notification settings
path "secret/data/airflow/notifications" {
  capabilities = ["read"]
}

# -----------------------------------------------------------------------------
# LLM API Keys (for Alpha Arena DAGs)
# -----------------------------------------------------------------------------
# Read access to LLM provider API keys for L5C pipeline DAGs

path "secret/data/trading/llm" {
  capabilities = ["read"]
}

path "secret/data/trading/llm/*" {
  capabilities = ["read"]
}

# -----------------------------------------------------------------------------
# Token Self-Management
# -----------------------------------------------------------------------------
# Allow Airflow to manage its own tokens

path "auth/token/lookup-self" {
  capabilities = ["read"]
}

path "auth/token/renew-self" {
  capabilities = ["update"]
}

# Allow creating child tokens for individual DAG runs (with restrictions)
path "auth/token/create" {
  capabilities = ["update"]
  allowed_parameters = {
    "policies" = ["airflow-dag-runner"]
    "ttl" = ["1h", "2h", "4h"]
    "max_ttl" = ["8h"]
  }
}

# -----------------------------------------------------------------------------
# Health Check Access
# -----------------------------------------------------------------------------
# Allow reading Vault health status for monitoring DAGs

path "sys/health" {
  capabilities = ["read"]
}
