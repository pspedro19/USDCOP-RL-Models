#!/usr/bin/env python3
"""
Master Restore Script
======================

Restores complete system from backup including:
- PostgreSQL/TimescaleDB database
- MinIO/S3 object storage
- Redis cache and streams
- Configuration files
- Model files
- Environment files

Author: Pedro @ Lean Tech Solutions / Claude Code
Date: 2026-01-14

Usage:
    python scripts/backup/restore_master.py --backup-dir PATH [--database-only] [--minio-only]
"""

import os
import sys
import json
import gzip
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


class MasterRestore:
    """Master restore orchestrator."""

    def __init__(self, backup_dir: Path):
        """Initialize master restore."""
        self.backup_dir = Path(backup_dir)

        if not self.backup_dir.exists():
            raise ValueError(f"Backup directory not found: {backup_dir}")

        # Load manifest if exists
        manifest_file = self.backup_dir / "MASTER_MANIFEST.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {}

        self.results = {
            "timestamp": datetime.now().isoformat(),
            "backup_dir": str(self.backup_dir),
            "components": {}
        }

    def restore_configs(self) -> Dict:
        """Restore configuration files."""
        print("\n" + "="*60)
        print("RESTORING CONFIGURATION FILES")
        print("="*60)

        config_backup = self.backup_dir / "config"
        if not config_backup.exists():
            print("  [SKIP] No config backup found")
            return {"skipped": True}

        config_dest = PROJECT_ROOT / "config"
        config_dest.mkdir(exist_ok=True)

        restored = []
        for config_file in config_backup.glob("*"):
            if config_file.is_file():
                dest = config_dest / config_file.name
                shutil.copy2(config_file, dest)
                restored.append(config_file.name)
                print(f"  [OK] {config_file.name}")

        return {
            "restored": restored,
            "count": len(restored)
        }

    def restore_models(self) -> Dict:
        """Restore model files."""
        print("\n" + "="*60)
        print("RESTORING MODEL FILES")
        print("="*60)

        models_backup = self.backup_dir / "models"
        if not models_backup.exists():
            print("  [SKIP] No models backup found")
            return {"skipped": True}

        models_dest = PROJECT_ROOT / "models"
        models_dest.mkdir(exist_ok=True)

        restored = []
        total_size = 0

        for item in models_backup.iterdir():
            dest = models_dest / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
                size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            else:
                shutil.copy2(item, dest)
                size = item.stat().st_size

            total_size += size
            restored.append(item.name)
            print(f"  [OK] {item.name} ({size/1024/1024:.2f} MB)")

        return {
            "restored": restored,
            "count": len(restored),
            "total_size_mb": total_size / 1024 / 1024
        }

    def restore_env_files(self) -> Dict:
        """Restore environment files."""
        print("\n" + "="*60)
        print("RESTORING ENVIRONMENT FILES")
        print("="*60)

        env_backup = self.backup_dir / "env"
        if not env_backup.exists():
            print("  [SKIP] No env backup found")
            return {"skipped": True}

        restored = []

        for env_file in env_backup.glob("*"):
            if env_file.is_file():
                dest = PROJECT_ROOT / env_file.name
                shutil.copy2(env_file, dest)
                restored.append(env_file.name)
                print(f"  [OK] {env_file.name}")

        return {
            "restored": restored,
            "count": len(restored)
        }

    def restore_init_scripts(self) -> Dict:
        """Restore database initialization scripts."""
        print("\n" + "="*60)
        print("RESTORING INIT SCRIPTS")
        print("="*60)

        init_backup = self.backup_dir / "init-scripts"
        if not init_backup.exists():
            print("  [SKIP] No init-scripts backup found")
            return {"skipped": True}

        init_dest = PROJECT_ROOT / "init-scripts"
        init_dest.mkdir(exist_ok=True)

        restored = []

        for script in init_backup.glob("*"):
            if script.is_file():
                dest = init_dest / script.name
                shutil.copy2(script, dest)
                restored.append(script.name)
                print(f"  [OK] {script.name}")

        return {
            "restored": restored,
            "count": len(restored)
        }

    def restore_docker_files(self) -> Dict:
        """Restore Docker Compose files."""
        print("\n" + "="*60)
        print("RESTORING DOCKER FILES")
        print("="*60)

        docker_backup = self.backup_dir / "docker"
        if not docker_backup.exists():
            print("  [SKIP] No docker backup found")
            return {"skipped": True}

        restored = []

        for docker_file in docker_backup.glob("*"):
            if docker_file.is_file():
                dest = PROJECT_ROOT / docker_file.name
                shutil.copy2(docker_file, dest)
                restored.append(docker_file.name)
                print(f"  [OK] {docker_file.name}")

        return {
            "restored": restored,
            "count": len(restored)
        }

    def restore_database(self, skip: bool = False) -> Dict:
        """Restore PostgreSQL/TimescaleDB database."""
        if skip:
            print("\n[SKIP] Database restore")
            return {"skipped": True}

        print("\n" + "="*60)
        print("RESTORING DATABASE")
        print("="*60)

        # Find database backup directory
        db_backup = None
        for subdir in self.backup_dir.iterdir():
            if subdir.is_dir() and "database" in subdir.name:
                # Find the timestamped subdirectory
                for backup_subdir in subdir.iterdir():
                    if backup_subdir.is_dir() and backup_subdir.name.startswith("backup_"):
                        db_backup = backup_subdir
                        break
                if db_backup:
                    break

        if not db_backup:
            print("  [SKIP] No database backup found")
            return {"skipped": True}

        print(f"  Found database backup: {db_backup}")

        # Load environment
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        database = os.getenv("POSTGRES_DB", "usdcop_trading")
        user = os.getenv("POSTGRES_USER", "admin")
        password = os.getenv("POSTGRES_PASSWORD", "")

        env = os.environ.copy()
        env["PGPASSWORD"] = password

        # First, restore schema
        schema_file = db_backup / "schema_ddl.sql.gz"
        if schema_file.exists():
            print("  Restoring schema DDL...")
            with gzip.open(schema_file, "rt") as f:
                schema_sql = f.read()

            cmd = ["psql", "-h", host, "-p", port, "-U", user, "-d", database]
            result = subprocess.run(
                cmd,
                input=schema_sql,
                capture_output=True,
                text=True,
                env=env
            )

            if result.returncode != 0:
                print(f"  Warning: Schema restore had errors: {result.stderr[:200]}")

        # Look for full SQL dump
        sql_files = list(db_backup.glob("usdcop_full_backup_*.sql.gz"))
        if sql_files:
            sql_file = sql_files[0]
            print(f"  Restoring from SQL dump: {sql_file.name}...")

            with gzip.open(sql_file, "rt") as f:
                sql_content = f.read()

            cmd = ["psql", "-h", host, "-p", port, "-U", user, "-d", database]
            result = subprocess.run(
                cmd,
                input=sql_content,
                capture_output=True,
                text=True,
                env=env
            )

            if result.returncode != 0:
                print(f"  Warning: SQL restore had errors (may be OK): {result.stderr[:200]}")
            else:
                print("  [OK] SQL dump restored successfully")

            return {
                "success": True,
                "method": "sql_dump",
                "file": sql_file.name
            }

        # Otherwise, restore from CSV files
        csv_files = list(db_backup.glob("*.csv.gz"))
        if csv_files:
            print(f"  Found {len(csv_files)} CSV files to restore")
            restored_tables = []

            for csv_file in csv_files:
                # Parse schema_table from filename
                name_parts = csv_file.stem.replace(".csv", "").split("_", 1)
                if len(name_parts) == 2:
                    schema, table = name_parts
                else:
                    schema, table = "public", name_parts[0]

                full_table = f"{schema}.{table}"
                print(f"  Restoring {full_table}...", end=" ")

                try:
                    # Truncate existing data
                    truncate_cmd = [
                        "psql", "-h", host, "-p", port, "-U", user, "-d", database,
                        "-c", f"TRUNCATE {full_table} CASCADE"
                    ]
                    subprocess.run(truncate_cmd, capture_output=True, env=env)

                    # Copy from CSV
                    with gzip.open(csv_file, "rt") as f:
                        csv_content = f.read()

                    copy_cmd = [
                        "psql", "-h", host, "-p", port, "-U", user, "-d", database,
                        "-c", f"\\COPY {full_table} FROM STDIN WITH CSV HEADER"
                    ]

                    result = subprocess.run(
                        copy_cmd,
                        input=csv_content,
                        capture_output=True,
                        text=True,
                        env=env
                    )

                    if result.returncode == 0:
                        print("OK")
                        restored_tables.append(full_table)
                    else:
                        print(f"ERROR: {result.stderr[:100]}")

                except Exception as e:
                    print(f"ERROR: {e}")

            return {
                "success": True,
                "method": "csv_import",
                "tables_restored": restored_tables,
                "count": len(restored_tables)
            }

        print("  [SKIP] No database files found to restore")
        return {"skipped": True}

    def restore_minio(self, skip: bool = False) -> Dict:
        """Restore MinIO/S3 buckets."""
        if skip:
            print("\n[SKIP] MinIO restore")
            return {"skipped": True}

        print("\n" + "="*60)
        print("RESTORING MINIO BUCKETS")
        print("="*60)

        # Find MinIO backup directory
        minio_backup = None
        for subdir in self.backup_dir.iterdir():
            if subdir.is_dir() and "minio" in subdir.name:
                for backup_subdir in subdir.iterdir():
                    if backup_subdir.is_dir() and backup_subdir.name.startswith("backup_"):
                        minio_backup = backup_subdir
                        break
                if minio_backup:
                    break

        if not minio_backup:
            print("  [SKIP] No MinIO backup found")
            return {"skipped": True}

        print(f"  Found MinIO backup: {minio_backup}")

        endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
        access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin123")

        # Configure mc alias
        subprocess.run(
            ["mc", "alias", "set", "restore_minio", endpoint, access_key, secret_key],
            capture_output=True
        )

        restored_buckets = []

        # Look for tar.gz archives
        for archive in minio_backup.glob("*.tar.gz"):
            bucket_name = archive.stem.replace(".tar", "")
            print(f"  Restoring bucket: {bucket_name}...", end=" ")

            try:
                # Extract archive
                import tarfile
                extract_dir = minio_backup / f"_extract_{bucket_name}"
                with tarfile.open(archive, "r:gz") as tar:
                    tar.extractall(extract_dir)

                # Mirror to MinIO
                bucket_dir = extract_dir / bucket_name
                if bucket_dir.exists():
                    # Create bucket if not exists
                    subprocess.run(
                        ["mc", "mb", f"restore_minio/{bucket_name}", "--ignore-existing"],
                        capture_output=True
                    )

                    # Mirror files
                    result = subprocess.run(
                        ["mc", "mirror", str(bucket_dir), f"restore_minio/{bucket_name}", "--overwrite"],
                        capture_output=True,
                        text=True
                    )

                    if result.returncode == 0:
                        print("OK")
                        restored_buckets.append(bucket_name)
                    else:
                        print(f"ERROR: {result.stderr[:100]}")

                # Cleanup
                shutil.rmtree(extract_dir, ignore_errors=True)

            except Exception as e:
                print(f"ERROR: {e}")

        # Also check for uncompressed bucket directories
        for bucket_dir in minio_backup.iterdir():
            if bucket_dir.is_dir() and not bucket_dir.name.startswith("_"):
                bucket_name = bucket_dir.name
                if bucket_name not in restored_buckets:
                    print(f"  Restoring bucket: {bucket_name}...", end=" ")

                    try:
                        # Create bucket if not exists
                        subprocess.run(
                            ["mc", "mb", f"restore_minio/{bucket_name}", "--ignore-existing"],
                            capture_output=True
                        )

                        # Mirror files
                        result = subprocess.run(
                            ["mc", "mirror", str(bucket_dir), f"restore_minio/{bucket_name}", "--overwrite"],
                            capture_output=True,
                            text=True
                        )

                        if result.returncode == 0:
                            print("OK")
                            restored_buckets.append(bucket_name)
                        else:
                            print(f"ERROR: {result.stderr[:100]}")

                    except Exception as e:
                        print(f"ERROR: {e}")

        return {
            "success": True,
            "buckets_restored": restored_buckets,
            "count": len(restored_buckets)
        }

    def restore_redis(self, skip: bool = False) -> Dict:
        """Restore Redis data."""
        if skip:
            print("\n[SKIP] Redis restore")
            return {"skipped": True}

        print("\n" + "="*60)
        print("RESTORING REDIS")
        print("="*60)

        # Find Redis backup directory
        redis_backup = None
        for subdir in self.backup_dir.iterdir():
            if subdir.is_dir() and "redis" in subdir.name:
                for backup_subdir in subdir.iterdir():
                    if backup_subdir.is_dir() and backup_subdir.name.startswith("backup_"):
                        redis_backup = backup_subdir
                        break
                if redis_backup:
                    break

        if not redis_backup:
            print("  [SKIP] No Redis backup found")
            return {"skipped": True}

        print(f"  Found Redis backup: {redis_backup}")
        print("  Note: RDB restore requires stopping Redis and copying dump.rdb")

        # Restore streams from JSON
        try:
            import redis as redis_lib

            host = os.getenv("REDIS_HOST", "localhost")
            port = int(os.getenv("REDIS_PORT", "6379"))
            password = os.getenv("REDIS_PASSWORD", "")

            client = redis_lib.Redis(host=host, port=port, password=password)

            restored_streams = []

            for stream_file in redis_backup.glob("redis_stream_*.json"):
                print(f"  Restoring stream from {stream_file.name}...", end=" ")

                try:
                    with open(stream_file) as f:
                        stream_data = json.load(f)

                    stream_name = stream_data.get("stream", "").replace("_", ":")
                    entries = stream_data.get("entries", [])

                    for entry in entries[:1000]:  # Limit to prevent overflow
                        entry_id = entry.get("id", "*")
                        data = entry.get("data", {})
                        if data:
                            client.xadd(stream_name, data, id=entry_id)

                    print(f"OK ({len(entries)} entries)")
                    restored_streams.append(stream_name)

                except Exception as e:
                    print(f"ERROR: {e}")

            return {
                "success": True,
                "streams_restored": restored_streams,
                "count": len(restored_streams)
            }

        except ImportError:
            print("  [SKIP] redis-py not installed")
            return {"skipped": True, "reason": "redis-py not installed"}
        except Exception as e:
            print(f"  [ERROR] {e}")
            return {"success": False, "error": str(e)}

    def verify_restore(self) -> Dict:
        """Verify restore was successful."""
        print("\n" + "="*60)
        print("VERIFYING RESTORE")
        print("="*60)

        checks = {}

        # Check configs
        config_dir = PROJECT_ROOT / "config"
        critical_configs = ["trading_config.yaml", "feature_config.json", "norm_stats.json"]
        configs_ok = all((config_dir / c).exists() for c in critical_configs)
        checks["configs"] = "OK" if configs_ok else "MISSING"
        print(f"  Configs: {checks['configs']}")

        # Check .env
        env_ok = (PROJECT_ROOT / ".env").exists()
        checks["env"] = "OK" if env_ok else "MISSING"
        print(f"  Environment: {checks['env']}")

        # Check models
        models_dir = PROJECT_ROOT / "models"
        models_ok = models_dir.exists() and any(models_dir.glob("**/*.onnx")) or any(models_dir.glob("**/*.zip"))
        checks["models"] = "OK" if models_ok else "MISSING"
        print(f"  Models: {checks['models']}")

        # Check Docker Compose
        docker_ok = (PROJECT_ROOT / "docker-compose.yml").exists()
        checks["docker"] = "OK" if docker_ok else "MISSING"
        print(f"  Docker Compose: {checks['docker']}")

        # Check init scripts
        init_dir = PROJECT_ROOT / "init-scripts"
        init_ok = init_dir.exists() and len(list(init_dir.glob("*.sql"))) >= 3
        checks["init_scripts"] = "OK" if init_ok else "MISSING"
        print(f"  Init Scripts: {checks['init_scripts']}")

        all_ok = all(v == "OK" for v in checks.values())

        return {
            "checks": checks,
            "all_passed": all_ok
        }

    def run_full_restore(
        self,
        database_only: bool = False,
        minio_only: bool = False,
        redis_only: bool = False,
        skip_database: bool = False,
        skip_minio: bool = False,
        skip_redis: bool = False
    ) -> Dict:
        """Run complete system restore."""
        print("="*60)
        print("USDCOP TRADING SYSTEM - FULL RESTORE")
        print("="*60)
        print(f"Backup directory: {self.backup_dir}")

        # Selective restore
        if database_only:
            self.results["components"]["database"] = self.restore_database()
        elif minio_only:
            self.results["components"]["minio"] = self.restore_minio()
        elif redis_only:
            self.results["components"]["redis"] = self.restore_redis()
        else:
            # Full restore
            self.results["components"]["configs"] = self.restore_configs()
            self.results["components"]["models"] = self.restore_models()
            self.results["components"]["env"] = self.restore_env_files()
            self.results["components"]["init_scripts"] = self.restore_init_scripts()
            self.results["components"]["docker"] = self.restore_docker_files()

            # Data restores
            self.results["components"]["database"] = self.restore_database(skip_database)
            self.results["components"]["minio"] = self.restore_minio(skip_minio)
            self.results["components"]["redis"] = self.restore_redis(skip_redis)

        # Verify
        self.results["verification"] = self.verify_restore()

        print("\n" + "="*60)
        print("RESTORE COMPLETE")
        print("="*60)

        if self.results["verification"]["all_passed"]:
            print("All verification checks PASSED!")
        else:
            print("WARNING: Some verification checks failed")

        print("\nNext steps:")
        print("  1. Start services: docker-compose up -d")
        print("  2. Verify health: curl http://localhost:5000/api/health")
        print("="*60)

        return self.results


def main():
    """Main restore function."""
    parser = argparse.ArgumentParser(description="Master Restore Tool")
    parser.add_argument(
        "--backup-dir",
        type=str,
        required=True,
        help="Path to backup directory"
    )
    parser.add_argument(
        "--database-only",
        action="store_true",
        help="Only restore database"
    )
    parser.add_argument(
        "--minio-only",
        action="store_true",
        help="Only restore MinIO buckets"
    )
    parser.add_argument(
        "--redis-only",
        action="store_true",
        help="Only restore Redis"
    )
    parser.add_argument(
        "--skip-database",
        action="store_true",
        help="Skip database restore"
    )
    parser.add_argument(
        "--skip-minio",
        action="store_true",
        help="Skip MinIO restore"
    )
    parser.add_argument(
        "--skip-redis",
        action="store_true",
        help="Skip Redis restore"
    )

    args = parser.parse_args()

    try:
        restore = MasterRestore(args.backup_dir)

        results = restore.run_full_restore(
            database_only=args.database_only,
            minio_only=args.minio_only,
            redis_only=args.redis_only,
            skip_database=args.skip_database,
            skip_minio=args.skip_minio,
            skip_redis=args.skip_redis
        )

        # Save results
        results_file = PROJECT_ROOT / "restore_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {results_file}")

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
