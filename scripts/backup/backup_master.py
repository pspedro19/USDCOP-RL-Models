#!/usr/bin/env python3
"""
Master Backup Script
=====================

Orchestrates complete system backup including:
- PostgreSQL/TimescaleDB database
- MinIO/S3 object storage
- Redis cache and streams
- Configuration files
- Model files
- Environment files

Author: Pedro @ Lean Tech Solutions / Claude Code
Date: 2026-01-14

Usage:
    python scripts/backup/backup_master.py [--output-dir PATH] [--skip-minio] [--skip-redis]
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import argparse

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load environment
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


class MasterBackup:
    """Master backup orchestrator."""

    # Configuration files to backup
    CONFIG_FILES = [
        "config/trading_config.yaml",
        "config/feature_config.json",
        "config/norm_stats.json",
        "config/feature_registry.yaml",
        "config/mlops.yaml",
        "config/database.yaml",
        "config/minio-buckets.yaml",
        "config/redis_streams.yaml",
        "config/trading_calendar.json",
        "config/hyperparameter_decisions.json",
        "config/mt5_config.yaml",
        "config/twelve_data_config.yaml",
        "config/storage.yaml",
        "config/pipeline_health_config.yaml",
        "config/quality_thresholds.yaml",
        "config/dashboard_config.yaml",
    ]

    # Model directories to backup
    MODEL_DIRS = [
        "models/ppo_production",
        "models/ppo_v20_production",
        "models/onnx",
    ]

    # Environment files
    ENV_FILES = [
        ".env",
        ".env.example",
        ".env.multimodel.example",
    ]

    # Init scripts
    INIT_SCRIPTS = [
        "init-scripts/00-init-extensions.sql",
        "init-scripts/01-essential-usdcop-init.sql",
        "init-scripts/02-macro-indicators-schema.sql",
        "init-scripts/03-inference-features-views.sql",
        "init-scripts/10-multi-model-schema.sql",
        "init-scripts/11-paper-trading-tables.sql",
        "init-scripts/12-trades-metadata.sql",
        "init-scripts/04-data-seeding.py",
    ]

    def __init__(self, output_dir: Path):
        """Initialize master backup."""
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = output_dir / f"full_backup_{self.timestamp}"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            "timestamp": datetime.now().isoformat(),
            "backup_dir": str(self.backup_dir),
            "components": {}
        }

    def backup_configs(self) -> Dict:
        """Backup configuration files."""
        print("\n" + "="*60)
        print("BACKING UP CONFIGURATION FILES")
        print("="*60)

        config_dir = self.backup_dir / "config"
        config_dir.mkdir(exist_ok=True)

        backed_up = []
        missing = []

        for config_file in self.CONFIG_FILES:
            src = PROJECT_ROOT / config_file
            if src.exists():
                dst = config_dir / Path(config_file).name
                shutil.copy2(src, dst)
                backed_up.append(config_file)
                print(f"  [OK] {config_file}")
            else:
                missing.append(config_file)
                print(f"  [SKIP] {config_file} (not found)")

        return {
            "backed_up": backed_up,
            "missing": missing,
            "count": len(backed_up)
        }

    def backup_models(self) -> Dict:
        """Backup model files."""
        print("\n" + "="*60)
        print("BACKING UP MODEL FILES")
        print("="*60)

        models_dir = self.backup_dir / "models"
        models_dir.mkdir(exist_ok=True)

        backed_up = []
        total_size = 0

        for model_dir in self.MODEL_DIRS:
            src = PROJECT_ROOT / model_dir
            if src.exists():
                dst = models_dir / Path(model_dir).name
                if src.is_dir():
                    shutil.copytree(src, dst)
                    size = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
                else:
                    shutil.copy2(src, dst)
                    size = src.stat().st_size

                total_size += size
                backed_up.append(model_dir)
                print(f"  [OK] {model_dir} ({size/1024/1024:.2f} MB)")
            else:
                print(f"  [SKIP] {model_dir} (not found)")

        # Also backup any .zip or .onnx files in models/
        models_root = PROJECT_ROOT / "models"
        if models_root.exists():
            for ext in ["*.zip", "*.onnx"]:
                for model_file in models_root.glob(ext):
                    dst = models_dir / model_file.name
                    if not dst.exists():
                        shutil.copy2(model_file, dst)
                        size = model_file.stat().st_size
                        total_size += size
                        backed_up.append(f"models/{model_file.name}")
                        print(f"  [OK] models/{model_file.name} ({size/1024/1024:.2f} MB)")

        return {
            "backed_up": backed_up,
            "count": len(backed_up),
            "total_size_mb": total_size / 1024 / 1024
        }

    def backup_env_files(self) -> Dict:
        """Backup environment files."""
        print("\n" + "="*60)
        print("BACKING UP ENVIRONMENT FILES")
        print("="*60)

        env_dir = self.backup_dir / "env"
        env_dir.mkdir(exist_ok=True)

        backed_up = []

        for env_file in self.ENV_FILES:
            src = PROJECT_ROOT / env_file
            if src.exists():
                dst = env_dir / env_file
                shutil.copy2(src, dst)
                backed_up.append(env_file)
                print(f"  [OK] {env_file}")
            else:
                print(f"  [SKIP] {env_file} (not found)")

        return {
            "backed_up": backed_up,
            "count": len(backed_up)
        }

    def backup_init_scripts(self) -> Dict:
        """Backup database initialization scripts."""
        print("\n" + "="*60)
        print("BACKING UP INIT SCRIPTS")
        print("="*60)

        init_dir = self.backup_dir / "init-scripts"
        init_dir.mkdir(exist_ok=True)

        backed_up = []

        for script in self.INIT_SCRIPTS:
            src = PROJECT_ROOT / script
            if src.exists():
                dst = init_dir / Path(script).name
                shutil.copy2(src, dst)
                backed_up.append(script)
                print(f"  [OK] {script}")
            else:
                print(f"  [SKIP] {script} (not found)")

        return {
            "backed_up": backed_up,
            "count": len(backed_up)
        }

    def backup_docker_compose(self) -> Dict:
        """Backup Docker Compose files."""
        print("\n" + "="*60)
        print("BACKING UP DOCKER COMPOSE FILES")
        print("="*60)

        docker_dir = self.backup_dir / "docker"
        docker_dir.mkdir(exist_ok=True)

        docker_files = [
            "docker-compose.yml",
            "docker-compose.multimodel.yml",
            "docker-compose.blue-green.yml",
            "docker-compose.canary.yml",
            "services/Dockerfile.api",
            "services/requirements.txt",
        ]

        backed_up = []

        for docker_file in docker_files:
            src = PROJECT_ROOT / docker_file
            if src.exists():
                dst = docker_dir / Path(docker_file).name
                shutil.copy2(src, dst)
                backed_up.append(docker_file)
                print(f"  [OK] {docker_file}")

        return {
            "backed_up": backed_up,
            "count": len(backed_up)
        }

    def backup_database(self, skip: bool = False) -> Dict:
        """Run database backup script."""
        if skip:
            print("\n[SKIP] Database backup")
            return {"skipped": True}

        print("\n" + "="*60)
        print("BACKING UP DATABASE")
        print("="*60)

        db_backup_dir = self.backup_dir / "database"

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "backup" / "backup_database.py"),
                    "--output-dir", str(db_backup_dir),
                    "--format", "both"
                ],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )

            if result.returncode == 0:
                print("  Database backup completed successfully")
                return {"success": True, "output": str(db_backup_dir)}
            else:
                print(f"  Database backup failed: {result.stderr}")
                return {"success": False, "error": result.stderr}

        except Exception as e:
            print(f"  Database backup error: {e}")
            return {"success": False, "error": str(e)}

    def backup_minio(self, skip: bool = False) -> Dict:
        """Run MinIO backup script."""
        if skip:
            print("\n[SKIP] MinIO backup")
            return {"skipped": True}

        print("\n" + "="*60)
        print("BACKING UP MINIO")
        print("="*60)

        minio_backup_dir = self.backup_dir / "minio"

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "backup" / "backup_minio.py"),
                    "--output-dir", str(minio_backup_dir),
                    "--priority-only"  # Only backup critical buckets
                ],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )

            if result.returncode == 0:
                print("  MinIO backup completed successfully")
                return {"success": True, "output": str(minio_backup_dir)}
            else:
                print(f"  MinIO backup warning: {result.stderr}")
                return {"success": True, "warning": result.stderr}

        except Exception as e:
            print(f"  MinIO backup error: {e}")
            return {"success": False, "error": str(e)}

    def backup_redis(self, skip: bool = False) -> Dict:
        """Run Redis backup script."""
        if skip:
            print("\n[SKIP] Redis backup")
            return {"skipped": True}

        print("\n" + "="*60)
        print("BACKING UP REDIS")
        print("="*60)

        redis_backup_dir = self.backup_dir / "redis"

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "backup" / "backup_redis.py"),
                    "--output-dir", str(redis_backup_dir)
                ],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )

            if result.returncode == 0:
                print("  Redis backup completed successfully")
                return {"success": True, "output": str(redis_backup_dir)}
            else:
                print(f"  Redis backup warning: {result.stderr}")
                return {"success": True, "warning": result.stderr}

        except Exception as e:
            print(f"  Redis backup error: {e}")
            return {"success": False, "error": str(e)}

    def create_master_manifest(self) -> Path:
        """Create master backup manifest."""
        manifest_file = self.backup_dir / "MASTER_MANIFEST.json"

        # Calculate total size
        total_size = sum(
            f.stat().st_size
            for f in self.backup_dir.rglob("*")
            if f.is_file()
        )

        manifest = {
            "created_at": datetime.now().isoformat(),
            "backup_dir": str(self.backup_dir),
            "total_size_mb": total_size / 1024 / 1024,
            "components": self.results["components"],
            "restore_instructions": [
                "="*60,
                "RESTORE INSTRUCTIONS",
                "="*60,
                "",
                "1. PREREQUISITES:",
                "   - Docker and Docker Compose installed",
                "   - Python 3.10+ installed",
                "   - MinIO client (mc) installed (optional)",
                "",
                "2. RESTORE ENVIRONMENT:",
                "   cp env/.env ../",
                "   cp env/.env.* ../ (if needed)",
                "",
                "3. RESTORE CONFIGS:",
                "   cp config/* ../config/",
                "",
                "4. RESTORE INIT SCRIPTS:",
                "   cp init-scripts/* ../init-scripts/",
                "",
                "5. RESTORE DOCKER:",
                "   cp docker/docker-compose.yml ../",
                "",
                "6. START INFRASTRUCTURE:",
                "   docker-compose up -d postgres redis minio",
                "",
                "7. RESTORE DATABASE:",
                "   python scripts/backup/restore_master.py --database-only",
                "",
                "8. RESTORE MINIO:",
                "   python scripts/backup/restore_master.py --minio-only",
                "",
                "9. START ALL SERVICES:",
                "   docker-compose up -d",
                "",
                "10. VERIFY:",
                "    curl http://localhost:5000/api/health",
                "    curl http://localhost:8080/",
                "",
                "="*60,
            ]
        }

        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        return manifest_file

    def create_restore_script(self) -> Path:
        """Create a shell script for quick restore."""
        restore_script = self.backup_dir / "restore.sh"

        script_content = f'''#!/bin/bash
# USDCOP Trading System - Quick Restore Script
# Generated: {datetime.now().isoformat()}
# Backup: {self.backup_dir.name}

set -e

echo "========================================"
echo "USDCOP Trading System - Restore"
echo "========================================"

BACKUP_DIR="$(dirname "$0")"
PROJECT_ROOT="${{BACKUP_DIR}}/.."

# Check prerequisites
command -v docker >/dev/null 2>&1 || {{ echo "Docker required"; exit 1; }}
command -v python3 >/dev/null 2>&1 || {{ echo "Python3 required"; exit 1; }}

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
'''

        with open(restore_script, "w") as f:
            f.write(script_content)

        # Make executable on Unix
        restore_script.chmod(0o755)

        return restore_script

    def run_full_backup(
        self,
        skip_minio: bool = False,
        skip_redis: bool = False,
        skip_database: bool = False
    ) -> Dict:
        """Run complete system backup."""
        print("="*60)
        print("USDCOP TRADING SYSTEM - FULL BACKUP")
        print("="*60)
        print(f"Backup directory: {self.backup_dir}")
        print(f"Timestamp: {self.timestamp}")

        # Backup components
        self.results["components"]["configs"] = self.backup_configs()
        self.results["components"]["models"] = self.backup_models()
        self.results["components"]["env"] = self.backup_env_files()
        self.results["components"]["init_scripts"] = self.backup_init_scripts()
        self.results["components"]["docker"] = self.backup_docker_compose()

        # Data backups
        self.results["components"]["database"] = self.backup_database(skip_database)
        self.results["components"]["minio"] = self.backup_minio(skip_minio)
        self.results["components"]["redis"] = self.backup_redis(skip_redis)

        # Create manifest and restore script
        manifest = self.create_master_manifest()
        restore_script = self.create_restore_script()

        # Calculate total size
        total_size = sum(
            f.stat().st_size
            for f in self.backup_dir.rglob("*")
            if f.is_file()
        )

        print("\n" + "="*60)
        print("BACKUP COMPLETE")
        print("="*60)
        print(f"Location: {self.backup_dir}")
        print(f"Total size: {total_size/1024/1024:.2f} MB")
        print(f"Manifest: {manifest}")
        print(f"Restore script: {restore_script}")
        print()
        print("To restore on another server:")
        print(f"  1. Copy {self.backup_dir} to the new server")
        print(f"  2. Run: bash {restore_script.name}")
        print("="*60)

        return self.results


def main():
    """Main backup function."""
    parser = argparse.ArgumentParser(description="Master Backup Tool")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "backups"),
        help="Output directory for backups"
    )
    parser.add_argument(
        "--skip-minio",
        action="store_true",
        help="Skip MinIO backup"
    )
    parser.add_argument(
        "--skip-redis",
        action="store_true",
        help="Skip Redis backup"
    )
    parser.add_argument(
        "--skip-database",
        action="store_true",
        help="Skip database backup"
    )
    parser.add_argument(
        "--configs-only",
        action="store_true",
        help="Only backup configuration files (no data)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    backup = MasterBackup(output_dir)

    if args.configs_only:
        args.skip_minio = True
        args.skip_redis = True
        args.skip_database = True

    try:
        results = backup.run_full_backup(
            skip_minio=args.skip_minio,
            skip_redis=args.skip_redis,
            skip_database=args.skip_database
        )

        # Save results
        results_file = backup.backup_dir / "backup_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
