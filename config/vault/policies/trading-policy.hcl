# =============================================================================
# USD/COP Trading System - Vault Policy for Trading Services
# =============================================================================
#
# This policy grants access to secrets required by trading services:
# - TwelveData API keys for market data
# - Database credentials for PostgreSQL/TimescaleDB
# - Redis credentials for caching
# - JWT secrets for API authentication
# - MinIO credentials for object storage
# - LLM API keys for alpha strategies
#
# Usage:
#   vault policy write trading-service config/vault/policies/trading-policy.hcl
#
# Associated AppRole: trading-service
# Services: trading-signals-service, ml-analytics-service, inference-api
#
# Author: Pedro @ Lean Tech Solutions
# Version: 1.0.0
# Date: 2026-01-17
# =============================================================================

# -----------------------------------------------------------------------------
# TwelveData API Keys - Market Data Access
# -----------------------------------------------------------------------------
# Read access to all TwelveData API keys for fetching OHLCV and macro data

# Legacy keys (TWELVEDATA_API_KEY_1 through TWELVEDATA_API_KEY_8)
path "secret/data/trading/twelvedata" {
  capabilities = ["read"]
}

# Group 1 keys (API_KEY_G1_1 through API_KEY_G1_8)
path "secret/data/trading/twelvedata/g1" {
  capabilities = ["read"]
}

# Group 2 keys (API_KEY_G2_1 through API_KEY_G2_8)
path "secret/data/trading/twelvedata/g2" {
  capabilities = ["read"]
}

# Wildcard for future TwelveData key groups
path "secret/data/trading/twelvedata/*" {
  capabilities = ["read"]
}

# List capability to enumerate available key paths
path "secret/metadata/trading/twelvedata" {
  capabilities = ["list"]
}

path "secret/metadata/trading/twelvedata/*" {
  capabilities = ["list"]
}

# -----------------------------------------------------------------------------
# Database Credentials - PostgreSQL/TimescaleDB
# -----------------------------------------------------------------------------
# Read access to database connection credentials

path "secret/data/trading/database" {
  capabilities = ["read"]
}

path "secret/data/trading/database/*" {
  capabilities = ["read"]
}

# Read-only users for specific services
path "secret/data/trading/database/readonly" {
  capabilities = ["read"]
}

# -----------------------------------------------------------------------------
# Redis Credentials - Caching Layer
# -----------------------------------------------------------------------------
# Read access to Redis authentication credentials

path "secret/data/trading/redis" {
  capabilities = ["read"]
}

path "secret/data/trading/redis/*" {
  capabilities = ["read"]
}

# -----------------------------------------------------------------------------
# JWT Secrets - API Authentication
# -----------------------------------------------------------------------------
# Read access to JWT signing keys for API authentication

path "secret/data/trading/jwt" {
  capabilities = ["read"]
}

path "secret/data/trading/jwt/*" {
  capabilities = ["read"]
}

# -----------------------------------------------------------------------------
# MinIO Credentials - Object Storage
# -----------------------------------------------------------------------------
# Read access to MinIO S3-compatible storage credentials

path "secret/data/trading/minio" {
  capabilities = ["read"]
}

path "secret/data/trading/minio/*" {
  capabilities = ["read"]
}

# -----------------------------------------------------------------------------
# LLM API Keys - Alpha Arena Strategies
# -----------------------------------------------------------------------------
# Read access to LLM provider API keys for trading strategies

path "secret/data/trading/llm" {
  capabilities = ["read"]
}

path "secret/data/trading/llm/*" {
  capabilities = ["read"]
}

# Specific providers
path "secret/data/trading/llm/deepseek" {
  capabilities = ["read"]
}

path "secret/data/trading/llm/anthropic" {
  capabilities = ["read"]
}

path "secret/data/trading/llm/openai" {
  capabilities = ["read"]
}

# -----------------------------------------------------------------------------
# MetaTrader 5 Credentials (Optional)
# -----------------------------------------------------------------------------
# Read access to MT5 broker credentials for live trading

path "secret/data/trading/mt5" {
  capabilities = ["read"]
}

path "secret/data/trading/mt5/*" {
  capabilities = ["read"]
}

# -----------------------------------------------------------------------------
# Notification Credentials
# -----------------------------------------------------------------------------
# Read access to notification service credentials (Slack, Email)

path "secret/data/trading/notifications" {
  capabilities = ["read"]
}

path "secret/data/trading/notifications/*" {
  capabilities = ["read"]
}

# -----------------------------------------------------------------------------
# Token Self-Management
# -----------------------------------------------------------------------------
# Allow services to manage their own tokens (lookup, renew)

path "auth/token/lookup-self" {
  capabilities = ["read"]
}

path "auth/token/renew-self" {
  capabilities = ["update"]
}

# -----------------------------------------------------------------------------
# Health Check Access
# -----------------------------------------------------------------------------
# Allow reading Vault health status for monitoring

path "sys/health" {
  capabilities = ["read"]
}
