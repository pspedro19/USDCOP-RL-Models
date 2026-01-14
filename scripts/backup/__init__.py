"""
Backup Scripts Package
=======================

Complete backup and restore tools for USDCOP RL Trading System.

Modules:
    backup_database: PostgreSQL/TimescaleDB backup
    backup_minio: MinIO/S3 bucket backup
    backup_redis: Redis data and streams backup
    backup_master: Full system backup orchestrator
    restore_master: Full system restore orchestrator

Usage:
    # Full backup
    python -m scripts.backup.backup_master

    # Full restore
    python -m scripts.backup.restore_master --backup-dir /path/to/backup

Author: Pedro @ Lean Tech Solutions / Claude Code
Date: 2026-01-14
"""

from pathlib import Path

BACKUP_VERSION = "1.0.0"
PROJECT_ROOT = Path(__file__).parent.parent.parent
