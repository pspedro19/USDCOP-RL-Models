#!/usr/bin/env python3
"""
Redis Backup Script
====================

Creates backups of Redis data including streams and cached data.
Supports RDB snapshots and stream data export.

Author: Pedro @ Lean Tech Solutions / Claude Code
Date: 2026-01-14

Usage:
    python scripts/backup/backup_redis.py [--output-dir PATH]
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


class RedisBackup:
    """Redis backup manager."""

    # Key patterns to backup
    CRITICAL_KEYS = [
        "trading:*",           # Trading state and signals
        "signals:*",           # Signal streams
        "inference:*",         # Inference cache
        "features:*",          # Feature cache
        "model:*",             # Model state
        "session:*",           # Session data
    ]

    # Streams to backup
    STREAMS = [
        "signals:ppo_primary:stream",
        "signals:multi_model:stream",
        "trading:signals",
        "trading:predictions",
        "trading:actions",
    ]

    def __init__(
        self,
        host: str = None,
        port: int = None,
        password: str = None
    ):
        """Initialize Redis connection parameters."""
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", "6379"))
        self.password = password or os.getenv("REDIS_PASSWORD", "")

    def _get_redis_client(self):
        """Get Redis client."""
        try:
            import redis
            return redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                decode_responses=True
            )
        except ImportError:
            print("Warning: redis-py not installed. Some features disabled.")
            return None

    def trigger_rdb_save(self) -> bool:
        """
        Trigger Redis BGSAVE to create RDB snapshot.

        Returns:
            True if save was triggered successfully
        """
        print("Triggering Redis BGSAVE...")

        try:
            client = self._get_redis_client()
            if client:
                client.bgsave()
                print("  BGSAVE triggered successfully")
                return True
        except Exception as e:
            print(f"  Warning: BGSAVE failed: {e}")

        # Try via redis-cli
        cmd = ["redis-cli", "-h", self.host, "-p", str(self.port)]
        if self.password:
            cmd.extend(["-a", self.password])
        cmd.append("BGSAVE")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if "Background saving started" in result.stdout:
                print("  BGSAVE triggered via redis-cli")
                return True
        except FileNotFoundError:
            pass

        return False

    def backup_rdb_file(self, output_path: Path) -> Optional[Path]:
        """
        Copy Redis RDB dump file.

        Args:
            output_path: Directory to save backup

        Returns:
            Path to backup file or None
        """
        # Common RDB file locations
        rdb_locations = [
            Path("/data/dump.rdb"),  # Docker volume
            Path("/var/lib/redis/dump.rdb"),
            Path("/tmp/dump.rdb"),
            PROJECT_ROOT / "redis_data" / "dump.rdb",
        ]

        # Try to get RDB location from Redis
        try:
            client = self._get_redis_client()
            if client:
                config = client.config_get("dir")
                dbfilename = client.config_get("dbfilename")
                if config and dbfilename:
                    rdb_path = Path(config.get("dir", "/data")) / dbfilename.get("dbfilename", "dump.rdb")
                    rdb_locations.insert(0, rdb_path)
        except Exception:
            pass

        # Find RDB file
        for rdb_path in rdb_locations:
            if rdb_path.exists():
                backup_file = output_path / f"redis_dump_{datetime.now().strftime('%Y%m%d_%H%M%S')}.rdb"
                shutil.copy2(rdb_path, backup_file)
                print(f"  RDB backup: {backup_file} ({backup_file.stat().st_size/1024:.1f} KB)")
                return backup_file

        print("  Warning: RDB file not found")
        return None

    def export_keys(self, output_path: Path) -> Dict:
        """
        Export Redis keys to JSON files.

        Args:
            output_path: Directory to save exports

        Returns:
            Dictionary with export results
        """
        print("\nExporting Redis keys...")

        client = self._get_redis_client()
        if not client:
            return {"error": "Redis client not available"}

        results = {}

        for pattern in self.CRITICAL_KEYS:
            print(f"  Pattern: {pattern}...", end=" ")

            try:
                keys = list(client.scan_iter(pattern))
                if not keys:
                    print("(empty)")
                    continue

                export_data = {}
                for key in keys:
                    key_type = client.type(key)

                    if key_type == "string":
                        export_data[key] = {
                            "type": "string",
                            "value": client.get(key)
                        }
                    elif key_type == "hash":
                        export_data[key] = {
                            "type": "hash",
                            "value": client.hgetall(key)
                        }
                    elif key_type == "list":
                        export_data[key] = {
                            "type": "list",
                            "value": client.lrange(key, 0, -1)
                        }
                    elif key_type == "set":
                        export_data[key] = {
                            "type": "set",
                            "value": list(client.smembers(key))
                        }
                    elif key_type == "zset":
                        export_data[key] = {
                            "type": "zset",
                            "value": client.zrange(key, 0, -1, withscores=True)
                        }
                    elif key_type == "stream":
                        # Export stream separately
                        pass

                # Save to file
                safe_pattern = pattern.replace(":", "_").replace("*", "all")
                export_file = output_path / f"redis_keys_{safe_pattern}.json"

                with open(export_file, "w") as f:
                    json.dump(export_data, f, indent=2, default=str)

                print(f"OK ({len(keys)} keys)")
                results[pattern] = {
                    "key_count": len(keys),
                    "file": export_file.name
                }

            except Exception as e:
                print(f"ERROR: {e}")
                results[pattern] = {"error": str(e)}

        return results

    def export_streams(self, output_path: Path, max_entries: int = 10000) -> Dict:
        """
        Export Redis Streams to JSON files.

        Args:
            output_path: Directory to save exports
            max_entries: Maximum entries per stream

        Returns:
            Dictionary with export results
        """
        print("\nExporting Redis Streams...")

        client = self._get_redis_client()
        if not client:
            return {"error": "Redis client not available"}

        results = {}

        for stream in self.STREAMS:
            print(f"  Stream: {stream}...", end=" ")

            try:
                # Check if stream exists
                if not client.exists(stream):
                    print("(not found)")
                    continue

                # Get stream info
                info = client.xinfo_stream(stream)
                length = info.get("length", 0)

                # Read entries
                entries = client.xrange(stream, "-", "+", count=max_entries)

                # Export
                export_data = {
                    "stream": stream,
                    "info": {
                        "length": length,
                        "first_entry": info.get("first-entry"),
                        "last_entry": info.get("last-entry"),
                    },
                    "entries": [
                        {"id": entry_id, "data": data}
                        for entry_id, data in entries
                    ]
                }

                safe_name = stream.replace(":", "_")
                export_file = output_path / f"redis_stream_{safe_name}.json"

                with open(export_file, "w") as f:
                    json.dump(export_data, f, indent=2, default=str)

                print(f"OK ({len(entries)} entries)")
                results[stream] = {
                    "entry_count": len(entries),
                    "file": export_file.name
                }

            except Exception as e:
                print(f"ERROR: {e}")
                results[stream] = {"error": str(e)}

        return results

    def get_memory_info(self) -> Dict:
        """Get Redis memory usage information."""
        client = self._get_redis_client()
        if not client:
            return {}

        try:
            info = client.info("memory")
            return {
                "used_memory_human": info.get("used_memory_human"),
                "used_memory_peak_human": info.get("used_memory_peak_human"),
                "total_keys": client.dbsize()
            }
        except Exception:
            return {}

    def create_manifest(self, output_path: Path, results: Dict) -> Path:
        """Create backup manifest."""
        manifest_file = output_path / "REDIS_MANIFEST.json"

        manifest = {
            "created_at": datetime.now().isoformat(),
            "redis": {
                "host": self.host,
                "port": self.port,
            },
            "memory_info": self.get_memory_info(),
            "backup_contents": results,
            "restore_instructions": [
                "1. Ensure Redis is running",
                "2. Copy RDB file to Redis data directory",
                "3. Restart Redis to load RDB",
                "4. Or use restore_redis.py for stream restoration",
                "5. Verify with: redis-cli DBSIZE"
            ]
        }

        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        return manifest_file


def main():
    """Main backup function."""
    parser = argparse.ArgumentParser(description="Redis Backup Tool")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "backups" / "redis"),
        help="Output directory for backups"
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Run inside Docker container"
    )
    parser.add_argument(
        "--rdb-only",
        action="store_true",
        help="Only backup RDB file, skip key exports"
    )

    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output_dir) / f"backup_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("REDIS BACKUP TOOL")
    print("="*60)
    print(f"Output directory: {output_path}")
    print()

    # Initialize backup manager
    if args.docker:
        backup = RedisBackup(host="redis")
    else:
        backup = RedisBackup()

    results = {}

    try:
        # Trigger BGSAVE
        backup.trigger_rdb_save()

        # Backup RDB file
        rdb_file = backup.backup_rdb_file(output_path)
        if rdb_file:
            results["rdb_file"] = rdb_file.name

        if not args.rdb_only:
            # Export keys
            key_results = backup.export_keys(output_path)
            results["keys"] = key_results

            # Export streams
            stream_results = backup.export_streams(output_path)
            results["streams"] = stream_results

        # Create manifest
        manifest = backup.create_manifest(output_path, results)

        print("\n" + "="*60)
        print("REDIS BACKUP COMPLETE")
        print("="*60)
        print(f"Location: {output_path}")
        print(f"Manifest: {manifest}")

    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
