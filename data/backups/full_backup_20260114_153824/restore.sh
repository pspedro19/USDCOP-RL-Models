#!/bin/bash
# USDCOP Trading System - Quick Restore Script
# Generated: 2026-01-14T15:38:24.868210
# Backup: full_backup_20260114_153824

set -e

echo "========================================"
echo "USDCOP Trading System - Restore"
echo "========================================"

BACKUP_DIR="$(dirname "$0")"
PROJECT_ROOT="${BACKUP_DIR}/.."

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "Docker required"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Python3 required"; exit 1; }

echo ""
echo "Step 1: Restoring environment files..."
cp -f "$BACKUP_DIR/env/.env" "$PROJECT_ROOT/"
cp -f "$BACKUP_DIR/env/.env.example" "$PROJECT_ROOT/" 2>/dev/null || true

echo "Step 2: Restoring configuration files..."
cp -f "$BACKUP_DIR/config/"* "$PROJECT_ROOT/config/"

echo "Step 3: Restoring init scripts..."
cp -f "$BACKUP_DIR/init-scripts/"* "$PROJECT_ROOT/init-scripts/"

echo "Step 4: Restoring Docker files..."
cp -f "$BACKUP_DIR/docker/docker-compose.yml" "$PROJECT_ROOT/"

echo "Step 5: Restoring models..."
cp -rf "$BACKUP_DIR/models/"* "$PROJECT_ROOT/models/" 2>/dev/null || true

echo ""
echo "========================================"
echo "File restoration complete!"
echo ""
echo "Next steps:"
echo "1. cd $PROJECT_ROOT"
echo "2. docker-compose up -d postgres redis minio"
echo "3. Wait for services to be healthy"
echo "4. python scripts/backup/restore_master.py --backup-dir $BACKUP_DIR"
echo "5. docker-compose up -d"
echo "========================================"
