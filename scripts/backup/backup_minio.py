#!/usr/bin/env python3
"""
MinIO/S3 Backup Script
=======================

Creates complete backups of all MinIO buckets for project replication.
Uses MinIO client (mc) or boto3 for S3-compatible operations.

Author: Pedro @ Lean Tech Solutions / Claude Code
Date: 2026-01-14

Usage:
    python scripts/backup/backup_minio.py [--output-dir PATH] [--buckets BUCKET1,BUCKET2]
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import argparse

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load environment
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


class MinIOBackup:
    """MinIO/S3 backup manager."""

    # All buckets defined in the system
    ALL_BUCKETS = [
        # Data pipeline buckets (L0-L6)
        "00-raw-usdcop-marketdata",
        "01-l1-ds-usdcop-standardize",
        "02-l2-ds-usdcop-prepare",
        "03-l3-ds-usdcop-feature",
        "04-l4-ds-usdcop-rlready",
        "05-l5-ds-usdcop-serving",
        "usdcop-l6-backtest",
        # Alternative naming
        "usdcop-l4-rlready",
        "usdcop-l5-serving",
        # Common/shared buckets
        "99-common-trading-models",
        "99-common-trading-reports",
        "99-common-trading-backups",
        # MLOps
        "mlflow",
        "airflow",
    ]

    # Priority buckets (backup first, most important)
    PRIORITY_BUCKETS = [
        "99-common-trading-models",  # Trained models
        "00-raw-usdcop-marketdata",  # Raw market data
        "04-l4-ds-usdcop-rlready",   # Training datasets
        "mlflow",                     # Experiment tracking
    ]

    def __init__(
        self,
        endpoint: str = None,
        access_key: str = None,
        secret_key: str = None
    ):
        """Initialize MinIO connection parameters."""
        self.endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
        self.access_key = access_key or os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = secret_key or os.getenv("MINIO_SECRET_KEY", "minioadmin123")

        # Configure mc alias
        self._setup_mc_alias()

    def _setup_mc_alias(self):
        """Configure MinIO client alias."""
        cmd = [
            "mc", "alias", "set", "backup_minio",
            self.endpoint,
            self.access_key,
            self.secret_key
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"Warning: mc alias setup failed: {result.stderr}")
        except FileNotFoundError:
            print("Warning: mc (MinIO client) not found. Using boto3 fallback.")

    def list_buckets(self) -> List[str]:
        """List all available buckets."""
        try:
            result = subprocess.run(
                ["mc", "ls", "backup_minio"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                buckets = []
                for line in result.stdout.strip().split("\n"):
                    if line:
                        # Format: [DATETIME] 0B bucket_name/
                        parts = line.split()
                        if len(parts) >= 3:
                            bucket = parts[-1].rstrip("/")
                            buckets.append(bucket)
                return buckets
        except FileNotFoundError:
            pass

        # Fallback to boto3
        return self._list_buckets_boto3()

    def _list_buckets_boto3(self) -> List[str]:
        """List buckets using boto3."""
        try:
            import boto3
            from botocore.client import Config

            s3 = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=Config(signature_version="s3v4")
            )

            response = s3.list_buckets()
            return [b["Name"] for b in response.get("Buckets", [])]
        except Exception as e:
            print(f"Error listing buckets with boto3: {e}")
            return []

    def backup_bucket(
        self,
        bucket: str,
        output_path: Path,
        compress: bool = True
    ) -> Optional[Dict]:
        """
        Backup a single MinIO bucket.

        Args:
            bucket: Bucket name
            output_path: Directory to save backup
            compress: Whether to compress the backup

        Returns:
            Backup metadata or None if failed
        """
        bucket_dir = output_path / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Backing up bucket: {bucket}...", end=" ")

        try:
            # Use mc mirror for efficient sync
            cmd = [
                "mc", "mirror",
                f"backup_minio/{bucket}",
                str(bucket_dir),
                "--overwrite"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                # Check if bucket is empty or doesn't exist
                if "does not exist" in result.stderr or "Object does not exist" in result.stderr:
                    print("SKIP (empty or not found)")
                    return None
                print(f"ERROR: {result.stderr}")
                return None

            # Get bucket stats
            file_count = sum(1 for _ in bucket_dir.rglob("*") if _.is_file())
            total_size = sum(f.stat().st_size for f in bucket_dir.rglob("*") if f.is_file())

            # Optionally compress
            if compress and file_count > 0:
                archive_path = output_path / f"{bucket}.tar.gz"
                shutil.make_archive(
                    str(archive_path).replace(".tar.gz", ""),
                    "gztar",
                    output_path,
                    bucket
                )
                # Remove uncompressed directory
                shutil.rmtree(bucket_dir)
                final_path = archive_path
                final_size = archive_path.stat().st_size
            else:
                final_path = bucket_dir
                final_size = total_size

            print(f"OK ({file_count} files, {final_size/1024/1024:.2f} MB)")

            return {
                "bucket": bucket,
                "file_count": file_count,
                "size_bytes": final_size,
                "compressed": compress,
                "path": str(final_path.name)
            }

        except FileNotFoundError:
            # Fallback to boto3
            return self._backup_bucket_boto3(bucket, output_path)
        except Exception as e:
            print(f"ERROR: {e}")
            return None

    def _backup_bucket_boto3(self, bucket: str, output_path: Path) -> Optional[Dict]:
        """Backup bucket using boto3."""
        try:
            import boto3
            from botocore.client import Config

            s3 = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=Config(signature_version="s3v4")
            )

            bucket_dir = output_path / bucket
            bucket_dir.mkdir(parents=True, exist_ok=True)

            # List all objects
            paginator = s3.get_paginator("list_objects_v2")
            file_count = 0
            total_size = 0

            for page in paginator.paginate(Bucket=bucket):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    local_path = bucket_dir / key
                    local_path.parent.mkdir(parents=True, exist_ok=True)

                    s3.download_file(bucket, key, str(local_path))
                    file_count += 1
                    total_size += obj["Size"]

            print(f"OK ({file_count} files, {total_size/1024/1024:.2f} MB)")

            return {
                "bucket": bucket,
                "file_count": file_count,
                "size_bytes": total_size,
                "compressed": False,
                "path": str(bucket_dir.name)
            }

        except Exception as e:
            print(f"ERROR (boto3): {e}")
            return None

    def backup_all_buckets(
        self,
        output_path: Path,
        buckets: List[str] = None
    ) -> Dict:
        """
        Backup all specified buckets.

        Args:
            output_path: Directory to save backups
            buckets: List of buckets to backup (default: all)

        Returns:
            Dictionary with backup results
        """
        print("\n" + "="*60)
        print("BACKING UP MINIO BUCKETS")
        print("="*60)

        # Get available buckets
        available = set(self.list_buckets())
        target_buckets = buckets or self.ALL_BUCKETS

        # Filter to only existing buckets
        to_backup = [b for b in target_buckets if b in available]

        print(f"Available buckets: {len(available)}")
        print(f"Buckets to backup: {len(to_backup)}")
        print()

        results = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": self.endpoint,
            "buckets": {}
        }

        for bucket in to_backup:
            backup_result = self.backup_bucket(bucket, output_path)
            if backup_result:
                results["buckets"][bucket] = backup_result

        return results

    def create_manifest(self, output_path: Path, results: Dict) -> Path:
        """Create backup manifest."""
        manifest_file = output_path / "MINIO_MANIFEST.json"

        manifest = {
            "created_at": datetime.now().isoformat(),
            "minio": {
                "endpoint": self.endpoint,
            },
            "backup_contents": results,
            "restore_instructions": [
                "1. Ensure MinIO is running",
                "2. Create buckets using config/minio-buckets.yaml",
                "3. Use mc mirror to restore: mc mirror ./bucket backup_minio/bucket",
                "4. Or use restore_minio.py script",
                "5. Verify with: mc ls backup_minio/bucket"
            ]
        }

        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        return manifest_file


def main():
    """Main backup function."""
    parser = argparse.ArgumentParser(description="MinIO Backup Tool")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "backups" / "minio"),
        help="Output directory for backups"
    )
    parser.add_argument(
        "--buckets",
        type=str,
        default=None,
        help="Comma-separated list of buckets to backup (default: all)"
    )
    parser.add_argument(
        "--priority-only",
        action="store_true",
        help="Only backup priority buckets (models, raw data, training data)"
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Don't compress bucket backups"
    )

    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output_dir) / f"backup_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("MINIO BACKUP TOOL")
    print("="*60)
    print(f"Output directory: {output_path}")
    print()

    # Initialize backup manager
    backup = MinIOBackup()

    # Determine buckets to backup
    if args.buckets:
        buckets = args.buckets.split(",")
    elif args.priority_only:
        buckets = MinIOBackup.PRIORITY_BUCKETS
    else:
        buckets = None  # All buckets

    try:
        results = backup.backup_all_buckets(output_path, buckets)

        # Create manifest
        manifest = backup.create_manifest(output_path, results)

        print("\n" + "="*60)
        print("MINIO BACKUP COMPLETE")
        print("="*60)
        print(f"Location: {output_path}")
        print(f"Manifest: {manifest}")
        print(f"Buckets backed up: {len(results.get('buckets', {}))}")

    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
