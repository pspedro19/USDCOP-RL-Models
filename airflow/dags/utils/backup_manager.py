"""
USDCOP Trading System - Comprehensive Backup Management
======================================================

Provides intelligent backup management for the USDCOP trading pipeline with:
- Local filesystem and S3 storage support
- Metadata tracking and validation
- Compression and optimization for parquet files
- Backup integrity checks and incremental updates

BACKUP STRATEGY:
- Local backups in /data/backups/ (or fallback to /tmp/backups/)
- Optional S3 remote storage for redundancy
- Metadata tracking for completeness verification
- Compression for storage efficiency
"""

import os
import json
import gzip
import shutil
import hashlib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import uuid

from .datetime_handler import UnifiedDatetimeHandler
from .db_manager import DatabaseManager

logger = logging.getLogger(__name__)

class BackupManager:
    """
    Comprehensive backup management for USDCOP trading data.
    Handles both local filesystem and S3 storage with metadata tracking.
    """

    def __init__(self,
                 local_backup_path: str = None,
                 s3_bucket: str = None,
                 s3_prefix: str = "usdcop-backups",
                 compression_level: int = 6,
                 enable_s3: bool = False):
        """
        Initialize backup manager with storage configuration.

        Args:
            local_backup_path: Local backup directory (default: /data/backups/)
            s3_bucket: S3 bucket name for remote backups
            s3_prefix: S3 prefix for backup organization
            compression_level: Compression level for parquet files (1-9)
            enable_s3: Whether to enable S3 storage operations
        """
        # Local storage configuration
        self.local_backup_path = local_backup_path or self._get_default_backup_path()
        self.compression_level = compression_level

        # S3 storage configuration
        self.enable_s3 = enable_s3
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_client = None

        # Initialize storage
        self._setup_local_storage()
        if self.enable_s3:
            self._setup_s3_storage()

        # Datetime handler for timezone consistency
        self.datetime_handler = UnifiedDatetimeHandler()

        logger.info(f"BackupManager initialized - Local: {self.local_backup_path}, S3: {self.enable_s3}")

    def _get_default_backup_path(self) -> str:
        """Get default backup path with fallback options."""
        preferred_paths = ["/data/backups", "/tmp/backups", "./backups"]

        for path in preferred_paths:
            try:
                os.makedirs(path, exist_ok=True)
                # Test write permissions
                test_file = os.path.join(path, ".write_test")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                logger.info(f"Using backup path: {path}")
                return path
            except (PermissionError, OSError) as e:
                logger.warning(f"Cannot use backup path {path}: {e}")
                continue

        raise RuntimeError("No suitable backup directory found")

    def _setup_local_storage(self):
        """Setup local storage directories."""
        try:
            os.makedirs(self.local_backup_path, exist_ok=True)

            # Create subdirectories for organization
            subdirs = ["data", "metadata", "temp", "incremental"]
            for subdir in subdirs:
                os.makedirs(os.path.join(self.local_backup_path, subdir), exist_ok=True)

            logger.info(f"✅ Local backup storage initialized: {self.local_backup_path}")

        except Exception as e:
            logger.error(f"❌ Failed to setup local storage: {e}")
            raise

    def _setup_s3_storage(self):
        """Setup S3 storage connection."""
        if not self.s3_bucket:
            logger.warning("S3 enabled but no bucket specified")
            self.enable_s3 = False
            return

        try:
            # Initialize S3 client with credentials from environment
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
            )

            # Test S3 connection
            self.s3_client.head_bucket(Bucket=self.s3_bucket)
            logger.info(f"✅ S3 backup storage connected: s3://{self.s3_bucket}/{self.s3_prefix}")

        except NoCredentialsError:
            logger.warning("No AWS credentials found, disabling S3 storage")
            self.enable_s3 = False
        except ClientError as e:
            logger.error(f"❌ S3 connection failed: {e}")
            self.enable_s3 = False
        except Exception as e:
            logger.error(f"❌ Unexpected S3 error: {e}")
            self.enable_s3 = False

    def check_backup_exists(self, backup_name: str, storage_type: str = "local") -> bool:
        """
        Check if a backup exists in the specified storage.

        Args:
            backup_name: Name of the backup to check
            storage_type: "local" or "s3"

        Returns:
            True if backup exists, False otherwise
        """
        try:
            if storage_type == "local":
                backup_path = os.path.join(self.local_backup_path, "data", f"{backup_name}.parquet")
                metadata_path = os.path.join(self.local_backup_path, "metadata", f"{backup_name}.json")
                return os.path.exists(backup_path) and os.path.exists(metadata_path)

            elif storage_type == "s3" and self.enable_s3:
                try:
                    data_key = f"{self.s3_prefix}/data/{backup_name}.parquet"
                    metadata_key = f"{self.s3_prefix}/metadata/{backup_name}.json"

                    self.s3_client.head_object(Bucket=self.s3_bucket, Key=data_key)
                    self.s3_client.head_object(Bucket=self.s3_bucket, Key=metadata_key)
                    return True
                except ClientError:
                    return False

            else:
                logger.warning(f"Invalid storage type: {storage_type}")
                return False

        except Exception as e:
            logger.error(f"Error checking backup existence: {e}")
            return False

    def get_backup_metadata(self, backup_name: str, storage_type: str = "local") -> Optional[Dict]:
        """
        Get metadata for a specific backup.

        Args:
            backup_name: Name of the backup
            storage_type: "local" or "s3"

        Returns:
            Metadata dictionary or None if not found
        """
        try:
            if storage_type == "local":
                metadata_path = os.path.join(self.local_backup_path, "metadata", f"{backup_name}.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        return json.load(f)

            elif storage_type == "s3" and self.enable_s3:
                try:
                    metadata_key = f"{self.s3_prefix}/metadata/{backup_name}.json"
                    response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=metadata_key)
                    return json.loads(response['Body'].read().decode('utf-8'))
                except ClientError:
                    pass

            return None

        except Exception as e:
            logger.error(f"Error getting backup metadata: {e}")
            return None

    def save_backup(self,
                   df: pd.DataFrame,
                   backup_name: str,
                   description: str = "",
                   pipeline_run_id: str = None,
                   date_range: Tuple[datetime, datetime] = None,
                   storage_types: List[str] = ["local"]) -> Dict[str, bool]:
        """
        Save DataFrame as a backup with comprehensive metadata.

        Args:
            df: DataFrame to backup
            backup_name: Unique name for the backup
            description: Description of the backup
            pipeline_run_id: Associated pipeline run ID
            date_range: Date range of the data (start, end)
            storage_types: List of storage types ["local", "s3"]

        Returns:
            Dictionary with success status for each storage type
        """
        results = {}

        if df.empty:
            logger.warning(f"Empty DataFrame provided for backup: {backup_name}")
            return {storage: False for storage in storage_types}

        try:
            # Generate backup metadata
            metadata = self._generate_backup_metadata(
                df, backup_name, description, pipeline_run_id, date_range
            )

            # Save to each requested storage type
            for storage_type in storage_types:
                try:
                    if storage_type == "local":
                        results["local"] = self._save_local_backup(df, backup_name, metadata)
                    elif storage_type == "s3" and self.enable_s3:
                        results["s3"] = self._save_s3_backup(df, backup_name, metadata)
                    else:
                        logger.warning(f"Skipping unsupported storage type: {storage_type}")
                        results[storage_type] = False

                except Exception as e:
                    logger.error(f"Failed to save backup to {storage_type}: {e}")
                    results[storage_type] = False

            # Log summary
            successful_saves = sum(results.values())
            logger.info(f"✅ Backup '{backup_name}' saved to {successful_saves}/{len(storage_types)} storage(s)")

            return results

        except Exception as e:
            logger.error(f"❌ Error saving backup '{backup_name}': {e}")
            return {storage: False for storage in storage_types}

    def load_backup(self, backup_name: str, storage_type: str = "local") -> Optional[pd.DataFrame]:
        """
        Load a backup DataFrame from storage.

        Args:
            backup_name: Name of the backup to load
            storage_type: "local" or "s3"

        Returns:
            DataFrame or None if not found
        """
        try:
            if storage_type == "local":
                backup_path = os.path.join(self.local_backup_path, "data", f"{backup_name}.parquet")
                if os.path.exists(backup_path):
                    df = pd.read_parquet(backup_path)
                    logger.info(f"✅ Loaded backup '{backup_name}' from local storage: {len(df)} records")
                    return self._standardize_loaded_data(df)

            elif storage_type == "s3" and self.enable_s3:
                try:
                    # Download to temporary file and load
                    temp_path = os.path.join(self.local_backup_path, "temp", f"{backup_name}_{uuid.uuid4().hex}.parquet")
                    data_key = f"{self.s3_prefix}/data/{backup_name}.parquet"

                    self.s3_client.download_file(self.s3_bucket, data_key, temp_path)
                    df = pd.read_parquet(temp_path)
                    os.remove(temp_path)  # Cleanup

                    logger.info(f"✅ Loaded backup '{backup_name}' from S3: {len(df)} records")
                    return self._standardize_loaded_data(df)

                except ClientError as e:
                    logger.warning(f"Backup '{backup_name}' not found in S3: {e}")
                    return None

            else:
                logger.warning(f"Invalid storage type: {storage_type}")
                return None

        except Exception as e:
            logger.error(f"❌ Error loading backup '{backup_name}': {e}")
            return None

    def validate_backup_integrity(self, backup_name: str, storage_type: str = "local") -> Dict[str, Any]:
        """
        Validate backup integrity and completeness.

        Args:
            backup_name: Name of the backup to validate
            storage_type: "local" or "s3"

        Returns:
            Validation results dictionary
        """
        validation_results = {
            "is_valid": False,
            "errors": [],
            "warnings": [],
            "metadata": None,
            "data_stats": None
        }

        try:
            # Check if backup exists
            if not self.check_backup_exists(backup_name, storage_type):
                validation_results["errors"].append("Backup does not exist")
                return validation_results

            # Load metadata
            metadata = self.get_backup_metadata(backup_name, storage_type)
            if not metadata:
                validation_results["errors"].append("Metadata not found")
                return validation_results

            validation_results["metadata"] = metadata

            # Load and validate data
            df = self.load_backup(backup_name, storage_type)
            if df is None:
                validation_results["errors"].append("Failed to load backup data")
                return validation_results

            # Validate data integrity
            current_stats = self._calculate_data_stats(df)
            validation_results["data_stats"] = current_stats

            # Compare with stored metadata
            stored_stats = metadata.get("data_stats", {})

            # Check record count
            if stored_stats.get("record_count") != current_stats.get("record_count"):
                validation_results["errors"].append(
                    f"Record count mismatch: expected {stored_stats.get('record_count')}, "
                    f"got {current_stats.get('record_count')}"
                )

            # Check data hash if available
            if "data_hash" in metadata and "data_hash" in current_stats:
                if metadata["data_hash"] != current_stats["data_hash"]:
                    validation_results["errors"].append("Data hash mismatch - possible corruption")

            # Check date range consistency
            if stored_stats.get("date_range") and current_stats.get("date_range"):
                stored_range = stored_stats["date_range"]
                current_range = current_stats["date_range"]
                if stored_range != current_range:
                    validation_results["warnings"].append(
                        f"Date range mismatch: expected {stored_range}, got {current_range}"
                    )

            # Check for missing timestamps in business hours
            if "completeness_score" in current_stats:
                if current_stats["completeness_score"] < 0.95:  # Less than 95% complete
                    validation_results["warnings"].append(
                        f"Low data completeness: {current_stats['completeness_score']:.1%}"
                    )

            # Set validation status
            validation_results["is_valid"] = len(validation_results["errors"]) == 0

            if validation_results["is_valid"]:
                logger.info(f"✅ Backup '{backup_name}' validation passed")
            else:
                logger.error(f"❌ Backup '{backup_name}' validation failed: {validation_results['errors']}")

            return validation_results

        except Exception as e:
            logger.error(f"❌ Error validating backup '{backup_name}': {e}")
            validation_results["errors"].append(f"Validation error: {str(e)}")
            return validation_results

    def list_backups(self, storage_type: str = "local") -> List[Dict]:
        """
        List all available backups with metadata.

        Args:
            storage_type: "local" or "s3"

        Returns:
            List of backup information dictionaries
        """
        backups = []

        try:
            if storage_type == "local":
                metadata_dir = os.path.join(self.local_backup_path, "metadata")
                if os.path.exists(metadata_dir):
                    for filename in os.listdir(metadata_dir):
                        if filename.endswith(".json"):
                            backup_name = filename[:-5]  # Remove .json extension
                            metadata = self.get_backup_metadata(backup_name, "local")
                            if metadata:
                                backups.append({
                                    "name": backup_name,
                                    "metadata": metadata,
                                    "storage": "local"
                                })

            elif storage_type == "s3" and self.enable_s3:
                try:
                    prefix = f"{self.s3_prefix}/metadata/"
                    response = self.s3_client.list_objects_v2(
                        Bucket=self.s3_bucket,
                        Prefix=prefix
                    )

                    for obj in response.get('Contents', []):
                        if obj['Key'].endswith('.json'):
                            backup_name = os.path.basename(obj['Key'])[:-5]  # Remove .json
                            metadata = self.get_backup_metadata(backup_name, "s3")
                            if metadata:
                                backups.append({
                                    "name": backup_name,
                                    "metadata": metadata,
                                    "storage": "s3"
                                })

                except ClientError as e:
                    logger.error(f"Error listing S3 backups: {e}")

            # Sort by creation date (newest first)
            backups.sort(key=lambda x: x["metadata"].get("created_at", ""), reverse=True)

            logger.info(f"Found {len(backups)} backups in {storage_type} storage")
            return backups

        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            return []

    def cleanup_old_backups(self,
                           max_age_days: int = 30,
                           max_count: int = 50,
                           storage_type: str = "local") -> int:
        """
        Clean up old backups based on age and count limits.

        Args:
            max_age_days: Maximum age in days to keep backups
            max_count: Maximum number of backups to keep
            storage_type: "local" or "s3"

        Returns:
            Number of backups cleaned up
        """
        try:
            backups = self.list_backups(storage_type)

            # Filter backups to delete
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            deleted_count = 0

            # Sort by creation date (oldest first for deletion)
            backups.sort(key=lambda x: x["metadata"].get("created_at", ""))

            for i, backup in enumerate(backups):
                should_delete = False
                backup_name = backup["name"]
                created_at = backup["metadata"].get("created_at")

                # Delete if too old
                if created_at:
                    backup_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    if backup_date < cutoff_date:
                        should_delete = True

                # Delete if exceeding count limit (keep most recent)
                if i < len(backups) - max_count:
                    should_delete = True

                if should_delete:
                    if self._delete_backup(backup_name, storage_type):
                        deleted_count += 1
                        logger.info(f"🗑️  Deleted old backup: {backup_name}")

            logger.info(f"✅ Cleanup completed: {deleted_count} backups removed from {storage_type}")
            return deleted_count

        except Exception as e:
            logger.error(f"❌ Error during backup cleanup: {e}")
            return 0

    def _save_local_backup(self, df: pd.DataFrame, backup_name: str, metadata: Dict) -> bool:
        """Save backup to local storage."""
        try:
            # Save data as compressed parquet
            data_path = os.path.join(self.local_backup_path, "data", f"{backup_name}.parquet")
            df.to_parquet(
                data_path,
                compression='snappy',
                index=False,
                engine='pyarrow'
            )

            # Save metadata
            metadata_path = os.path.join(self.local_backup_path, "metadata", f"{backup_name}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

            logger.info(f"✅ Local backup saved: {backup_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to save local backup: {e}")
            return False

    def _save_s3_backup(self, df: pd.DataFrame, backup_name: str, metadata: Dict) -> bool:
        """Save backup to S3 storage."""
        try:
            # Save data to temporary file then upload
            temp_data_path = os.path.join(self.local_backup_path, "temp", f"{backup_name}_{uuid.uuid4().hex}.parquet")
            df.to_parquet(temp_data_path, compression='snappy', index=False)

            # Upload data file
            data_key = f"{self.s3_prefix}/data/{backup_name}.parquet"
            self.s3_client.upload_file(temp_data_path, self.s3_bucket, data_key)
            os.remove(temp_data_path)  # Cleanup

            # Upload metadata
            metadata_key = f"{self.s3_prefix}/metadata/{backup_name}.json"
            metadata_json = json.dumps(metadata, indent=2, default=str)
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=metadata_key,
                Body=metadata_json.encode('utf-8'),
                ContentType='application/json'
            )

            logger.info(f"✅ S3 backup saved: {backup_name}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to save S3 backup: {e}")
            return False

    def _delete_backup(self, backup_name: str, storage_type: str) -> bool:
        """Delete a backup from storage."""
        try:
            if storage_type == "local":
                data_path = os.path.join(self.local_backup_path, "data", f"{backup_name}.parquet")
                metadata_path = os.path.join(self.local_backup_path, "metadata", f"{backup_name}.json")

                if os.path.exists(data_path):
                    os.remove(data_path)
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)

                return True

            elif storage_type == "s3" and self.enable_s3:
                data_key = f"{self.s3_prefix}/data/{backup_name}.parquet"
                metadata_key = f"{self.s3_prefix}/metadata/{backup_name}.json"

                # Delete both files
                self.s3_client.delete_objects(
                    Bucket=self.s3_bucket,
                    Delete={
                        'Objects': [
                            {'Key': data_key},
                            {'Key': metadata_key}
                        ]
                    }
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Error deleting backup {backup_name}: {e}")
            return False

    def _generate_backup_metadata(self,
                                 df: pd.DataFrame,
                                 backup_name: str,
                                 description: str,
                                 pipeline_run_id: str,
                                 date_range: Tuple[datetime, datetime]) -> Dict:
        """Generate comprehensive backup metadata."""

        # Calculate data statistics
        data_stats = self._calculate_data_stats(df)

        # Auto-detect date range if not provided
        if date_range is None and 'timestamp' in df.columns:
            timestamps = self.datetime_handler.ensure_timezone_aware(df['timestamp'])
            date_range = (timestamps.min(), timestamps.max())

        metadata = {
            "backup_name": backup_name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "pipeline_run_id": pipeline_run_id,
            "data_stats": data_stats,
            "date_range": {
                "start": date_range[0].isoformat() if date_range else None,
                "end": date_range[1].isoformat() if date_range else None
            },
            "schema": {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
            },
            "backup_version": "1.0",
            "compression": "snappy"
        }

        return metadata

    def _calculate_data_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive data statistics."""
        stats = {
            "record_count": len(df),
            "column_count": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
        }

        # Calculate data hash for integrity checking
        data_string = df.to_csv(index=False)
        stats["data_hash"] = hashlib.md5(data_string.encode()).hexdigest()

        # Timestamp-specific statistics
        if 'timestamp' in df.columns:
            timestamps = self.datetime_handler.ensure_timezone_aware(df['timestamp'])
            stats["date_range"] = [timestamps.min().isoformat(), timestamps.max().isoformat()]

            # Calculate completeness score for business hours
            if len(df) > 0:
                stats["completeness_score"] = self._calculate_completeness_score(timestamps)

        # Price data statistics
        price_cols = ['open', 'high', 'low', 'close']
        available_price_cols = [col for col in price_cols if col in df.columns]

        if available_price_cols:
            for col in available_price_cols:
                stats[f"{col}_mean"] = float(df[col].mean())
                stats[f"{col}_std"] = float(df[col].std())
                stats[f"{col}_min"] = float(df[col].min())
                stats[f"{col}_max"] = float(df[col].max())

        return stats

    def _calculate_completeness_score(self, timestamps: pd.Series) -> float:
        """Calculate data completeness score for business hours."""
        try:
            if len(timestamps) < 2:
                return 0.0

            # Get expected timestamps for the period
            start_time = timestamps.min()
            end_time = timestamps.max()

            expected_timestamps = self.datetime_handler.generate_expected_timestamps(
                start_time, end_time, interval_minutes=5, business_hours_only=True
            )

            if not expected_timestamps:
                return 0.0

            # Calculate coverage
            actual_count = len(timestamps)
            expected_count = len(expected_timestamps)

            return min(actual_count / expected_count, 1.0)

        except Exception as e:
            logger.warning(f"Error calculating completeness score: {e}")
            return 0.0

    def _standardize_loaded_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize loaded data with proper timezone handling."""
        # Ensure timezone awareness for timestamp columns
        timestamp_cols = [col for col in df.columns if 'time' in col.lower()]
        if timestamp_cols:
            df = self.datetime_handler.standardize_dataframe_timestamps(df, timestamp_cols)

        return df


# Convenience functions
def get_backup_manager(enable_s3: bool = False, s3_bucket: str = None) -> BackupManager:
    """Get a backup manager instance with standard configuration."""
    return BackupManager(
        enable_s3=enable_s3,
        s3_bucket=s3_bucket or os.getenv('USDCOP_S3_BUCKET')
    )


# Example usage and testing
if __name__ == "__main__":
    # Test backup functionality
    import pandas as pd
    from datetime import datetime, timedelta

    # Create test data
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-15 08:00:00', periods=100, freq='5min'),
        'open': [4200.0 + i * 0.1 for i in range(100)],
        'high': [4200.5 + i * 0.1 for i in range(100)],
        'low': [4199.5 + i * 0.1 for i in range(100)],
        'close': [4200.2 + i * 0.1 for i in range(100)],
        'volume': [1000 + i * 10 for i in range(100)]
    })

    # Test backup manager
    try:
        backup_mgr = BackupManager()

        # Save backup
        backup_name = f"test_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results = backup_mgr.save_backup(
            df=test_data,
            backup_name=backup_name,
            description="Test backup for validation",
            pipeline_run_id="test_run_001"
        )

        print(f"Backup save results: {results}")

        # Load backup
        loaded_df = backup_mgr.load_backup(backup_name)
        if loaded_df is not None:
            print(f"Loaded backup: {len(loaded_df)} records")

        # Validate backup
        validation = backup_mgr.validate_backup_integrity(backup_name)
        print(f"Validation results: {validation['is_valid']}")

        # List backups
        backups = backup_mgr.list_backups()
        print(f"Available backups: {len(backups)}")

        print("✅ BackupManager test completed successfully!")

    except Exception as e:
        print(f"❌ BackupManager test failed: {e}")