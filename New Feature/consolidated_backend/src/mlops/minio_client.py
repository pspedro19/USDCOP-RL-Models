# backend/src/mlops/minio_client.py
"""
MinIO Client for S3-compatible object storage.

This module provides a wrapper around the MinIO SDK for model artifact storage,
including upload, download, listing, and cleanup operations.

Environment Variables:
    MINIO_ENDPOINT: MinIO server endpoint (default: localhost:9000)
    MINIO_ACCESS_KEY: Access key for authentication
    MINIO_SECRET_KEY: Secret key for authentication
    MINIO_SECURE: Use HTTPS (default: false)
    MINIO_REGION: Region for the bucket (optional)

Example:
    client = MinioClient()
    client.upload_model("ml-models", "model.pkl", "models/v1/model.pkl")
    client.download_model("ml-models", "models/v1/model.pkl", "./model.pkl")
    models = client.list_models("ml-models", "models/")
    client.delete_old_models("ml-models", days_to_keep=30)
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Generator
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

try:
    from minio import Minio
    from minio.error import S3Error
    from minio.commonconfig import Tags
    from minio.lifecycleconfig import LifecycleConfig, Rule, Expiration
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    Minio = None
    S3Error = Exception
    Tags = None
    LifecycleConfig = None

import json
import io
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bucket constants
FORECASTS_BUCKET = "forecasts"
MODELS_BUCKET = "ml-models"


@dataclass
class ModelInfo:
    """Information about a stored model."""
    bucket: str
    path: str
    size: int
    last_modified: datetime
    etag: str
    content_type: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None
    tags: Optional[Dict[str, str]] = None

    def __str__(self) -> str:
        size_mb = self.size / (1024 * 1024)
        return f"{self.path} ({size_mb:.2f} MB, modified: {self.last_modified})"


class MinioClient:
    """
    A wrapper client for MinIO S3-compatible object storage.

    This class provides a simplified interface for:
    - Uploading and downloading model artifacts
    - Listing models in a bucket
    - Managing model lifecycle (cleanup old models)
    - Setting metadata and tags on objects

    Attributes:
        endpoint (str): MinIO server endpoint
        access_key (str): Access key for authentication
        secret_key (str): Secret key for authentication
        secure (bool): Whether to use HTTPS
        client (Minio): MinIO client instance

    Example:
        >>> client = MinioClient()
        >>> client.upload_model("models", "model.pkl", "v1/model.pkl")
        >>> client.download_model("models", "v1/model.pkl", "./model.pkl")
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        secure: Optional[bool] = None,
        region: Optional[str] = None
    ):
        """
        Initialize the MinIO client.

        Args:
            endpoint: MinIO server endpoint. Defaults to MINIO_ENDPOINT env var
                     or "localhost:9000"
            access_key: Access key. Defaults to MINIO_ACCESS_KEY env var
            secret_key: Secret key. Defaults to MINIO_SECRET_KEY env var
            secure: Use HTTPS. Defaults to MINIO_SECURE env var or False
            region: Region for bucket. Defaults to MINIO_REGION env var
        """
        if not MINIO_AVAILABLE:
            logger.warning(
                "MinIO SDK is not installed. Install with: pip install minio"
            )
            self._mock_mode = True
            self._client = None
        else:
            self._mock_mode = False

        # Load configuration from parameters or environment
        self.endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "localhost:9000")
        self.access_key = access_key or os.getenv("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = secret_key or os.getenv("MINIO_SECRET_KEY", "minioadmin")
        self.region = region or os.getenv("MINIO_REGION", None)

        # Parse secure flag
        if secure is not None:
            self.secure = secure
        else:
            secure_env = os.getenv("MINIO_SECURE", "false").lower()
            self.secure = secure_env in ("true", "1", "yes")

        self._client: Optional[Minio] = None
        self._initialized = False

        logger.info(
            f"MinIO client created for endpoint: {self.endpoint} "
            f"(secure: {self.secure})"
        )

    def _get_client(self) -> Minio:
        """Get or create the MinIO client."""
        if self._mock_mode:
            return None

        if self._client is None:
            self._client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
                region=self.region
            )
            self._initialized = True
            logger.debug("MinIO client initialized")

        return self._client

    def ensure_bucket(
        self,
        bucket: str,
        make_public: bool = False
    ) -> bool:
        """
        Ensure a bucket exists, creating it if necessary.

        Args:
            bucket: Bucket name
            make_public: If True, set bucket policy to public read

        Returns:
            True if bucket was created, False if it already existed

        Example:
            >>> client.ensure_bucket("ml-models")
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Ensure bucket: {bucket}")
            return True

        client = self._get_client()

        try:
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket, location=self.region)
                logger.info(f"Created bucket: {bucket}")

                if make_public:
                    # Set public read policy
                    policy = {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {"AWS": "*"},
                                "Action": ["s3:GetObject"],
                                "Resource": [f"arn:aws:s3:::{bucket}/*"]
                            }
                        ]
                    }
                    client.set_bucket_policy(bucket, json.dumps(policy))
                    logger.info(f"Set public read policy for bucket: {bucket}")

                return True
            else:
                logger.debug(f"Bucket already exists: {bucket}")
                return False

        except S3Error as e:
            logger.error(f"Error ensuring bucket: {e}")
            raise

    def upload_model(
        self,
        bucket: str,
        model_path: str,
        s3_path: str,
        metadata: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None,
        content_type: str = "application/octet-stream"
    ) -> str:
        """
        Upload a model file to MinIO.

        Args:
            bucket: Target bucket name
            model_path: Local path to the model file
            s3_path: Destination path in the bucket
            metadata: Optional metadata dictionary
            tags: Optional tags dictionary
            content_type: MIME type (default: application/octet-stream)

        Returns:
            The S3 URI of the uploaded model

        Example:
            >>> uri = client.upload_model(
            ...     "ml-models",
            ...     "models/ridge_h15.pkl",
            ...     "production/v1/ridge_h15.pkl",
            ...     metadata={"horizon": "15", "model_type": "ridge"},
            ...     tags={"environment": "production"}
            ... )
        """
        if self._mock_mode:
            s3_uri = f"s3://{bucket}/{s3_path}"
            logger.info(f"[MOCK] Uploaded model: {model_path} -> {s3_uri}")
            return s3_uri

        client = self._get_client()

        # Ensure bucket exists
        self.ensure_bucket(bucket)

        local_path = Path(model_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Upload the file
        try:
            result = client.fput_object(
                bucket_name=bucket,
                object_name=s3_path,
                file_path=str(local_path),
                content_type=content_type,
                metadata=metadata
            )

            s3_uri = f"s3://{bucket}/{s3_path}"
            logger.info(
                f"Uploaded model: {model_path} -> {s3_uri} "
                f"(etag: {result.etag})"
            )

            # Set tags if provided
            if tags and Tags is not None:
                tag_obj = Tags.new_object_tags()
                for key, value in tags.items():
                    tag_obj[key] = value
                client.set_object_tags(bucket, s3_path, tag_obj)
                logger.debug(f"Set tags on {s3_path}: {tags}")

            return s3_uri

        except S3Error as e:
            logger.error(f"Error uploading model: {e}")
            raise

    def upload_directory(
        self,
        bucket: str,
        local_dir: str,
        s3_prefix: str,
        pattern: str = "*",
        metadata: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """
        Upload an entire directory to MinIO.

        Args:
            bucket: Target bucket name
            local_dir: Local directory path
            s3_prefix: Prefix for all uploaded files in the bucket
            pattern: Glob pattern to filter files (default: "*")
            metadata: Optional metadata for all files

        Returns:
            List of S3 URIs for uploaded files

        Example:
            >>> uris = client.upload_directory(
            ...     "ml-models",
            ...     "results/models/",
            ...     "training/2024-01-15/"
            ... )
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Upload directory: {local_dir} -> s3://{bucket}/{s3_prefix}")
            return []

        local_path = Path(local_dir)
        if not local_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {local_dir}")

        uploaded = []
        for file_path in local_path.rglob(pattern):
            if file_path.is_file():
                relative = file_path.relative_to(local_path)
                s3_path = f"{s3_prefix.rstrip('/')}/{relative}"
                uri = self.upload_model(bucket, str(file_path), s3_path, metadata)
                uploaded.append(uri)

        logger.info(f"Uploaded {len(uploaded)} files from {local_dir}")
        return uploaded

    def download_model(
        self,
        bucket: str,
        s3_path: str,
        local_path: str
    ) -> str:
        """
        Download a model file from MinIO.

        Args:
            bucket: Source bucket name
            s3_path: Path in the bucket
            local_path: Local destination path

        Returns:
            The local path of the downloaded file

        Example:
            >>> local = client.download_model(
            ...     "ml-models",
            ...     "production/v1/model.pkl",
            ...     "./downloaded_model.pkl"
            ... )
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Downloaded: s3://{bucket}/{s3_path} -> {local_path}")
            return local_path

        client = self._get_client()

        # Create parent directory if needed
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)

        try:
            client.fget_object(bucket, s3_path, str(local))
            logger.info(f"Downloaded: s3://{bucket}/{s3_path} -> {local_path}")
            return str(local)

        except S3Error as e:
            logger.error(f"Error downloading model: {e}")
            raise

    def download_directory(
        self,
        bucket: str,
        s3_prefix: str,
        local_dir: str
    ) -> List[str]:
        """
        Download all files under a prefix from MinIO.

        Args:
            bucket: Source bucket name
            s3_prefix: Prefix in the bucket
            local_dir: Local destination directory

        Returns:
            List of local paths of downloaded files

        Example:
            >>> files = client.download_directory(
            ...     "ml-models",
            ...     "production/v1/",
            ...     "./models/"
            ... )
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Download directory: s3://{bucket}/{s3_prefix} -> {local_dir}")
            return []

        downloaded = []
        for model in self.list_models(bucket, s3_prefix):
            relative = model.path[len(s3_prefix):].lstrip('/')
            local_path = str(Path(local_dir) / relative)
            self.download_model(bucket, model.path, local_path)
            downloaded.append(local_path)

        logger.info(f"Downloaded {len(downloaded)} files to {local_dir}")
        return downloaded

    def list_models(
        self,
        bucket: str,
        prefix: str = "",
        recursive: bool = True
    ) -> Generator[ModelInfo, None, None]:
        """
        List models in a bucket.

        Args:
            bucket: Bucket name
            prefix: Optional prefix to filter results
            recursive: If True, list recursively (default: True)

        Yields:
            ModelInfo objects for each model found

        Example:
            >>> for model in client.list_models("ml-models", "production/"):
            ...     print(f"{model.path}: {model.size} bytes")
        """
        if self._mock_mode:
            logger.info(f"[MOCK] List models: s3://{bucket}/{prefix}")
            return

        client = self._get_client()

        try:
            objects = client.list_objects(
                bucket,
                prefix=prefix,
                recursive=recursive
            )

            for obj in objects:
                yield ModelInfo(
                    bucket=bucket,
                    path=obj.object_name,
                    size=obj.size,
                    last_modified=obj.last_modified,
                    etag=obj.etag,
                    content_type=obj.content_type
                )

        except S3Error as e:
            logger.error(f"Error listing models: {e}")
            raise

    def get_model_info(
        self,
        bucket: str,
        s3_path: str
    ) -> Optional[ModelInfo]:
        """
        Get information about a specific model.

        Args:
            bucket: Bucket name
            s3_path: Path in the bucket

        Returns:
            ModelInfo object or None if not found

        Example:
            >>> info = client.get_model_info("ml-models", "v1/model.pkl")
            >>> print(f"Size: {info.size}, Modified: {info.last_modified}")
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Get model info: s3://{bucket}/{s3_path}")
            return None

        client = self._get_client()

        try:
            stat = client.stat_object(bucket, s3_path)

            # Get tags if available
            tags = None
            try:
                tag_obj = client.get_object_tags(bucket, s3_path)
                if tag_obj:
                    tags = dict(tag_obj)
            except S3Error:
                pass

            return ModelInfo(
                bucket=bucket,
                path=s3_path,
                size=stat.size,
                last_modified=stat.last_modified,
                etag=stat.etag,
                content_type=stat.content_type,
                metadata=stat.metadata,
                tags=tags
            )

        except S3Error as e:
            if e.code == "NoSuchKey":
                return None
            logger.error(f"Error getting model info: {e}")
            raise

    def delete_model(
        self,
        bucket: str,
        s3_path: str
    ) -> bool:
        """
        Delete a model from MinIO.

        Args:
            bucket: Bucket name
            s3_path: Path in the bucket

        Returns:
            True if deleted successfully

        Example:
            >>> client.delete_model("ml-models", "old/model.pkl")
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Delete model: s3://{bucket}/{s3_path}")
            return True

        client = self._get_client()

        try:
            client.remove_object(bucket, s3_path)
            logger.info(f"Deleted: s3://{bucket}/{s3_path}")
            return True

        except S3Error as e:
            logger.error(f"Error deleting model: {e}")
            raise

    def delete_old_models(
        self,
        bucket: str,
        days_to_keep: int = 30,
        prefix: str = "",
        dry_run: bool = False
    ) -> List[str]:
        """
        Delete models older than a specified number of days.

        Args:
            bucket: Bucket name
            days_to_keep: Number of days to keep models (default: 30)
            prefix: Optional prefix to filter models
            dry_run: If True, only list what would be deleted

        Returns:
            List of deleted (or would-be-deleted) model paths

        Example:
            >>> # Delete models older than 30 days
            >>> deleted = client.delete_old_models("ml-models", days_to_keep=30)
            >>>
            >>> # Dry run to see what would be deleted
            >>> to_delete = client.delete_old_models(
            ...     "ml-models", days_to_keep=7, dry_run=True
            ... )
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Delete old models: bucket={bucket}, days={days_to_keep}")
            return []

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        deleted = []

        for model in self.list_models(bucket, prefix):
            if model.last_modified < cutoff_date:
                if dry_run:
                    logger.info(f"[DRY RUN] Would delete: {model.path}")
                else:
                    self.delete_model(bucket, model.path)
                deleted.append(model.path)

        action = "Would delete" if dry_run else "Deleted"
        logger.info(
            f"{action} {len(deleted)} models older than {days_to_keep} days"
        )

        return deleted

    def set_lifecycle_policy(
        self,
        bucket: str,
        days_to_expire: int = 90,
        prefix: str = ""
    ) -> None:
        """
        Set a lifecycle policy to automatically expire old objects.

        Args:
            bucket: Bucket name
            days_to_expire: Days after which objects are deleted
            prefix: Optional prefix to apply policy to

        Example:
            >>> # Expire models after 90 days
            >>> client.set_lifecycle_policy("ml-models", days_to_expire=90)
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Set lifecycle policy: bucket={bucket}, days={days_to_expire}")
            return

        if LifecycleConfig is None:
            logger.warning("Lifecycle configuration not available in this MinIO version")
            return

        client = self._get_client()

        try:
            config = LifecycleConfig(
                [
                    Rule(
                        rule_id="expire-old-models",
                        status="Enabled",
                        rule_filter={"prefix": prefix} if prefix else None,
                        expiration=Expiration(days=days_to_expire)
                    )
                ]
            )

            client.set_bucket_lifecycle(bucket, config)
            logger.info(
                f"Set lifecycle policy for {bucket}: "
                f"expire after {days_to_expire} days"
            )

        except S3Error as e:
            logger.error(f"Error setting lifecycle policy: {e}")
            raise

    def copy_model(
        self,
        source_bucket: str,
        source_path: str,
        dest_bucket: str,
        dest_path: str
    ) -> str:
        """
        Copy a model from one location to another.

        Args:
            source_bucket: Source bucket name
            source_path: Source path
            dest_bucket: Destination bucket name
            dest_path: Destination path

        Returns:
            S3 URI of the copied model

        Example:
            >>> client.copy_model(
            ...     "staging", "model.pkl",
            ...     "production", "v1/model.pkl"
            ... )
        """
        if self._mock_mode:
            dest_uri = f"s3://{dest_bucket}/{dest_path}"
            logger.info(f"[MOCK] Copy: s3://{source_bucket}/{source_path} -> {dest_uri}")
            return dest_uri

        client = self._get_client()

        # Ensure destination bucket exists
        self.ensure_bucket(dest_bucket)

        try:
            from minio.commonconfig import CopySource

            result = client.copy_object(
                dest_bucket,
                dest_path,
                CopySource(source_bucket, source_path)
            )

            dest_uri = f"s3://{dest_bucket}/{dest_path}"
            logger.info(
                f"Copied: s3://{source_bucket}/{source_path} -> {dest_uri}"
            )
            return dest_uri

        except S3Error as e:
            logger.error(f"Error copying model: {e}")
            raise

    def get_presigned_url(
        self,
        bucket: str,
        s3_path: str,
        expires: timedelta = timedelta(hours=1)
    ) -> str:
        """
        Generate a presigned URL for temporary access to a model.

        Args:
            bucket: Bucket name
            s3_path: Path in the bucket
            expires: URL expiration time (default: 1 hour)

        Returns:
            Presigned URL string

        Example:
            >>> url = client.get_presigned_url("ml-models", "v1/model.pkl")
            >>> print(f"Download URL (valid 1 hour): {url}")
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Presigned URL: s3://{bucket}/{s3_path}")
            return f"https://{self.endpoint}/{bucket}/{s3_path}?mock=true"

        client = self._get_client()

        try:
            url = client.presigned_get_object(bucket, s3_path, expires=expires)
            logger.debug(f"Generated presigned URL for {s3_path}")
            return url

        except S3Error as e:
            logger.error(f"Error generating presigned URL: {e}")
            raise

    def health_check(self) -> bool:
        """
        Check if MinIO server is accessible.

        Returns:
            True if server is accessible, False otherwise
        """
        if self._mock_mode:
            return True

        try:
            client = self._get_client()
            # Try to list buckets as a health check
            list(client.list_buckets())
            return True
        except Exception as e:
            logger.error(f"MinIO health check failed: {e}")
            return False

    def __enter__(self) -> 'MinioClient':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        pass  # No cleanup needed for MinIO client

    # =========================================================================
    # WEEKLY FORECAST METHODS
    # =========================================================================

    def upload_weekly_forecast(
        self,
        year: int,
        week: int,
        forecast_data: Dict[str, Any],
        quality_report: Optional[Dict[str, Any]] = None,
        images: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Upload weekly forecast data with structured organization.

        Structure:
            forecasts/{year}/week{week:02d}/
                ├── forecast.json
                ├── quality_report.json
                ├── metadata.json
                └── figures/{model}_{horizon}.png

        Args:
            year: Forecast year (e.g., 2024)
            week: Week number (1-52)
            forecast_data: Dictionary with forecast predictions
            quality_report: Optional quality report dictionary
            images: Optional dict mapping image names to local file paths
            metadata: Optional metadata dictionary

        Returns:
            Dictionary with uploaded paths:
            {
                'forecast': 's3://forecasts/2024/week01/forecast.json',
                'quality_report': 's3://forecasts/2024/week01/quality_report.json',
                'images': ['s3://forecasts/2024/week01/figures/ridge_h15.png', ...]
            }

        Example:
            >>> result = client.upload_weekly_forecast(
            ...     year=2024,
            ...     week=15,
            ...     forecast_data={'predictions': [...]},
            ...     images={'ridge_h15': '/tmp/ridge_h15.png'}
            ... )
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Upload weekly forecast: {year}/week{week:02d}")
            return {
                'forecast': f's3://{FORECASTS_BUCKET}/{year}/week{week:02d}/forecast.json',
                'quality_report': None,
                'images': []
            }

        # Ensure bucket exists
        self.ensure_bucket(FORECASTS_BUCKET)

        base_path = f"{year}/week{week:02d}"
        uploaded_paths = {'images': []}

        # Add upload timestamp to forecast data
        forecast_data['_upload_timestamp'] = datetime.now(timezone.utc).isoformat()
        forecast_data['_week'] = week
        forecast_data['_year'] = year

        # 1. Upload forecast.json
        forecast_json = json.dumps(forecast_data, indent=2, default=str)
        forecast_path = f"{base_path}/forecast.json"
        self._upload_bytes(
            FORECASTS_BUCKET,
            forecast_path,
            forecast_json.encode('utf-8'),
            content_type='application/json'
        )
        uploaded_paths['forecast'] = f"s3://{FORECASTS_BUCKET}/{forecast_path}"
        logger.info(f"Uploaded forecast: {forecast_path}")

        # 2. Upload quality_report.json if provided
        if quality_report:
            quality_report['_upload_timestamp'] = datetime.now(timezone.utc).isoformat()
            quality_json = json.dumps(quality_report, indent=2, default=str)
            quality_path = f"{base_path}/quality_report.json"
            self._upload_bytes(
                FORECASTS_BUCKET,
                quality_path,
                quality_json.encode('utf-8'),
                content_type='application/json'
            )
            uploaded_paths['quality_report'] = f"s3://{FORECASTS_BUCKET}/{quality_path}"
            logger.info(f"Uploaded quality report: {quality_path}")
        else:
            uploaded_paths['quality_report'] = None

        # 3. Upload images if provided
        if images:
            for image_name, image_path in images.items():
                try:
                    local_path = Path(image_path)
                    if local_path.exists():
                        # Determine extension
                        ext = local_path.suffix or '.png'
                        s3_path = f"{base_path}/figures/{image_name}{ext}"

                        self.upload_model(
                            FORECASTS_BUCKET,
                            str(local_path),
                            s3_path,
                            content_type=f'image/{ext[1:]}'
                        )
                        uploaded_paths['images'].append(f"s3://{FORECASTS_BUCKET}/{s3_path}")
                        logger.info(f"Uploaded image: {s3_path}")
                    else:
                        logger.warning(f"Image file not found: {image_path}")
                except Exception as e:
                    logger.error(f"Error uploading image {image_name}: {e}")

        # 4. Upload metadata.json
        run_metadata = {
            'year': year,
            'week': week,
            'upload_timestamp': datetime.now(timezone.utc).isoformat(),
            'forecast_file': uploaded_paths['forecast'],
            'quality_report_file': uploaded_paths.get('quality_report'),
            'n_images': len(uploaded_paths['images']),
            'image_files': uploaded_paths['images'],
            **(metadata or {})
        }
        metadata_json = json.dumps(run_metadata, indent=2, default=str)
        metadata_path = f"{base_path}/metadata.json"
        self._upload_bytes(
            FORECASTS_BUCKET,
            metadata_path,
            metadata_json.encode('utf-8'),
            content_type='application/json'
        )
        uploaded_paths['metadata'] = f"s3://{FORECASTS_BUCKET}/{metadata_path}"

        logger.info(f"Uploaded weekly forecast for {year}/week{week:02d}")
        return uploaded_paths

    def _upload_bytes(
        self,
        bucket: str,
        s3_path: str,
        data: bytes,
        content_type: str = 'application/octet-stream'
    ) -> str:
        """
        Upload bytes data directly to MinIO.

        Args:
            bucket: Target bucket name
            s3_path: Destination path in the bucket
            data: Bytes data to upload
            content_type: MIME type

        Returns:
            S3 URI of uploaded object
        """
        if self._mock_mode:
            return f"s3://{bucket}/{s3_path}"

        client = self._get_client()

        try:
            data_stream = BytesIO(data)
            client.put_object(
                bucket_name=bucket,
                object_name=s3_path,
                data=data_stream,
                length=len(data),
                content_type=content_type
            )
            return f"s3://{bucket}/{s3_path}"
        except S3Error as e:
            logger.error(f"Error uploading bytes to {s3_path}: {e}")
            raise

    # =========================================================================
    # MONTHLY MODEL METHODS
    # =========================================================================

    def upload_monthly_model(
        self,
        year: int,
        month: int,
        model_name: str,
        model_bytes: bytes,
        horizon: int,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Upload a trained model with structured organization by month.

        Structure:
            ml-models/{year}/month{month:02d}/{timestamp}/
                ├── {model}_h{horizon}.pkl
                └── metadata.json

        Args:
            year: Training year
            month: Training month (1-12)
            model_name: Name of the model (e.g., 'ridge', 'xgboost')
            model_bytes: Serialized model as bytes (pickle, joblib, etc.)
            horizon: Forecast horizon in days
            metrics: Optional metrics dictionary (DA, RMSE, etc.)
            metadata: Optional additional metadata

        Returns:
            Dictionary with paths:
            {
                'model': 's3://ml-models/2024/month01/20240115_143022/ridge_h15.pkl',
                'metadata': 's3://ml-models/2024/month01/20240115_143022/metadata.json'
            }

        Example:
            >>> with open('model.pkl', 'rb') as f:
            ...     model_bytes = f.read()
            >>> result = client.upload_monthly_model(
            ...     year=2024,
            ...     month=1,
            ...     model_name='ridge',
            ...     model_bytes=model_bytes,
            ...     horizon=15,
            ...     metrics={'direction_accuracy': 0.58, 'rmse': 0.015}
            ... )
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Upload monthly model: {model_name}_h{horizon}")
            return {
                'model': f's3://{MODELS_BUCKET}/{year}/month{month:02d}/mock/{model_name}_h{horizon}.pkl',
                'metadata': f's3://{MODELS_BUCKET}/{year}/month{month:02d}/mock/metadata.json'
            }

        # Ensure bucket exists
        self.ensure_bucket(MODELS_BUCKET)

        # Create timestamp for versioning
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_path = f"{year}/month{month:02d}/{timestamp}"

        uploaded_paths = {}

        # 1. Upload model file
        model_filename = f"{model_name}_h{horizon}.pkl"
        model_path = f"{base_path}/{model_filename}"

        self._upload_bytes(
            MODELS_BUCKET,
            model_path,
            model_bytes,
            content_type='application/octet-stream'
        )
        uploaded_paths['model'] = f"s3://{MODELS_BUCKET}/{model_path}"
        logger.info(f"Uploaded model: {model_path}")

        # 2. Create and upload metadata
        model_metadata = {
            'model_name': model_name,
            'horizon': horizon,
            'year': year,
            'month': month,
            'timestamp': timestamp,
            'upload_timestamp': datetime.now(timezone.utc).isoformat(),
            'model_file': uploaded_paths['model'],
            'model_size_bytes': len(model_bytes),
            'metrics': metrics or {},
            **(metadata or {})
        }

        metadata_json = json.dumps(model_metadata, indent=2, default=str)
        metadata_path = f"{base_path}/metadata.json"
        self._upload_bytes(
            MODELS_BUCKET,
            metadata_path,
            metadata_json.encode('utf-8'),
            content_type='application/json'
        )
        uploaded_paths['metadata'] = f"s3://{MODELS_BUCKET}/{metadata_path}"

        logger.info(f"Uploaded monthly model {model_name}_h{horizon} to {base_path}")
        return uploaded_paths

    def upload_monthly_models_batch(
        self,
        year: int,
        month: int,
        models: List[Dict[str, Any]],
        run_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upload multiple models from a training run with shared metadata.

        Args:
            year: Training year
            month: Training month (1-12)
            models: List of model dictionaries, each containing:
                - model_name: str
                - model_bytes: bytes
                - horizon: int
                - metrics: Optional[Dict]
            run_metadata: Optional shared metadata for the entire run

        Returns:
            Dictionary with all uploaded paths and run summary

        Example:
            >>> models = [
            ...     {'model_name': 'ridge', 'model_bytes': b'...', 'horizon': 15, 'metrics': {...}},
            ...     {'model_name': 'xgboost', 'model_bytes': b'...', 'horizon': 15, 'metrics': {...}},
            ... ]
            >>> result = client.upload_monthly_models_batch(2024, 1, models)
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Upload batch: {len(models)} models")
            return {'uploaded': [], 'run_metadata': None}

        # Ensure bucket exists
        self.ensure_bucket(MODELS_BUCKET)

        # Create shared timestamp for this training run
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_path = f"{year}/month{month:02d}/{timestamp}"

        uploaded = []
        all_metrics = {}

        # Upload each model
        for model_info in models:
            model_name = model_info['model_name']
            model_bytes = model_info['model_bytes']
            horizon = model_info['horizon']
            metrics = model_info.get('metrics', {})

            model_filename = f"{model_name}_h{horizon}.pkl"
            model_path = f"{base_path}/{model_filename}"

            self._upload_bytes(
                MODELS_BUCKET,
                model_path,
                model_bytes,
                content_type='application/octet-stream'
            )

            model_entry = {
                'model_name': model_name,
                'horizon': horizon,
                'path': f"s3://{MODELS_BUCKET}/{model_path}",
                'size_bytes': len(model_bytes),
                'metrics': metrics
            }
            uploaded.append(model_entry)

            # Aggregate metrics
            key = f"{model_name}_h{horizon}"
            all_metrics[key] = metrics

            logger.info(f"Uploaded: {model_filename}")

        # Create run metadata
        full_run_metadata = {
            'year': year,
            'month': month,
            'timestamp': timestamp,
            'upload_timestamp': datetime.now(timezone.utc).isoformat(),
            'n_models': len(uploaded),
            'models': uploaded,
            'aggregated_metrics': all_metrics,
            **(run_metadata or {})
        }

        metadata_json = json.dumps(full_run_metadata, indent=2, default=str)
        metadata_path = f"{base_path}/metadata.json"
        self._upload_bytes(
            MODELS_BUCKET,
            metadata_path,
            metadata_json.encode('utf-8'),
            content_type='application/json'
        )

        logger.info(f"Uploaded {len(uploaded)} models to {base_path}")

        return {
            'base_path': f"s3://{MODELS_BUCKET}/{base_path}",
            'uploaded': uploaded,
            'run_metadata': f"s3://{MODELS_BUCKET}/{metadata_path}"
        }

    # =========================================================================
    # GET LATEST MODELS
    # =========================================================================

    def get_latest_models(
        self,
        bucket: str = MODELS_BUCKET,
        model_name: Optional[str] = None,
        horizon: Optional[int] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get the most recent models from the bucket.

        Args:
            bucket: Bucket name (default: ml-models)
            model_name: Optional filter by model name (e.g., 'ridge')
            horizon: Optional filter by horizon
            limit: Maximum number of results

        Returns:
            List of model info dictionaries, sorted by most recent first:
            [
                {
                    'path': 's3://ml-models/2024/month01/...',
                    'model_name': 'ridge',
                    'horizon': 15,
                    'last_modified': datetime,
                    'size': int,
                    'year': 2024,
                    'month': 1
                },
                ...
            ]

        Example:
            >>> models = client.get_latest_models(model_name='ridge', horizon=15)
            >>> latest = models[0]  # Most recent ridge h15 model
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Get latest models from {bucket}")
            return []

        client = self._get_client()

        try:
            # List all objects in bucket
            objects = list(client.list_objects(bucket, recursive=True))

            # Filter for .pkl files
            models = []
            for obj in objects:
                if obj.object_name.endswith('.pkl'):
                    # Parse path: {year}/month{month:02d}/{timestamp}/{model}_h{horizon}.pkl
                    path_parts = obj.object_name.split('/')

                    if len(path_parts) >= 4:
                        try:
                            year = int(path_parts[0])
                            month = int(path_parts[1].replace('month', ''))
                            filename = path_parts[-1]

                            # Parse model name and horizon from filename
                            name_parts = filename.replace('.pkl', '').rsplit('_h', 1)
                            if len(name_parts) == 2:
                                m_name = name_parts[0]
                                m_horizon = int(name_parts[1])

                                # Apply filters
                                if model_name and m_name != model_name:
                                    continue
                                if horizon and m_horizon != horizon:
                                    continue

                                models.append({
                                    'path': f"s3://{bucket}/{obj.object_name}",
                                    'object_name': obj.object_name,
                                    'model_name': m_name,
                                    'horizon': m_horizon,
                                    'last_modified': obj.last_modified,
                                    'size': obj.size,
                                    'year': year,
                                    'month': month
                                })
                        except (ValueError, IndexError) as e:
                            logger.debug(f"Could not parse model path {obj.object_name}: {e}")
                            continue

            # Sort by last_modified descending
            models.sort(key=lambda x: x['last_modified'], reverse=True)

            return models[:limit]

        except S3Error as e:
            logger.error(f"Error listing models: {e}")
            return []

    def get_latest_model_by_type(
        self,
        model_name: str,
        horizon: int,
        bucket: str = MODELS_BUCKET
    ) -> Optional[Dict[str, Any]]:
        """
        Get the most recent model of a specific type and horizon.

        Args:
            model_name: Model type (e.g., 'ridge', 'xgboost')
            horizon: Forecast horizon
            bucket: Bucket name

        Returns:
            Model info dictionary or None if not found

        Example:
            >>> model = client.get_latest_model_by_type('ridge', 15)
            >>> if model:
            ...     local_path = client.download_model(
            ...         bucket, model['object_name'], './ridge_h15.pkl'
            ...     )
        """
        models = self.get_latest_models(
            bucket=bucket,
            model_name=model_name,
            horizon=horizon,
            limit=1
        )
        return models[0] if models else None

    def get_latest_forecasts(
        self,
        bucket: str = FORECASTS_BUCKET,
        year: Optional[int] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get the most recent forecast uploads.

        Args:
            bucket: Bucket name (default: forecasts)
            year: Optional filter by year
            limit: Maximum number of results

        Returns:
            List of forecast info dictionaries

        Example:
            >>> forecasts = client.get_latest_forecasts(year=2024)
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Get latest forecasts from {bucket}")
            return []

        client = self._get_client()

        try:
            objects = list(client.list_objects(bucket, recursive=True))

            forecasts = []
            seen_weeks = set()

            for obj in objects:
                if obj.object_name.endswith('forecast.json'):
                    path_parts = obj.object_name.split('/')

                    if len(path_parts) >= 3:
                        try:
                            f_year = int(path_parts[0])
                            f_week = int(path_parts[1].replace('week', ''))

                            if year and f_year != year:
                                continue

                            week_key = (f_year, f_week)
                            if week_key in seen_weeks:
                                continue
                            seen_weeks.add(week_key)

                            forecasts.append({
                                'path': f"s3://{bucket}/{obj.object_name}",
                                'object_name': obj.object_name,
                                'year': f_year,
                                'week': f_week,
                                'last_modified': obj.last_modified,
                                'size': obj.size
                            })
                        except (ValueError, IndexError):
                            continue

            forecasts.sort(key=lambda x: (x['year'], x['week']), reverse=True)
            return forecasts[:limit]

        except S3Error as e:
            logger.error(f"Error listing forecasts: {e}")
            return []

    # =========================================================================
    # CLEANUP METHODS
    # =========================================================================

    def cleanup_old_forecasts(
        self,
        weeks_to_keep: int = 52,
        bucket: str = FORECASTS_BUCKET,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Delete forecasts older than specified weeks.

        Args:
            weeks_to_keep: Number of weeks to retain (default: 52 = 1 year)
            bucket: Bucket name
            dry_run: If True, only list what would be deleted

        Returns:
            Dictionary with deletion summary:
            {
                'deleted_count': int,
                'deleted_weeks': [(year, week), ...],
                'bytes_freed': int
            }

        Example:
            >>> # Delete forecasts older than 1 year
            >>> result = client.cleanup_old_forecasts(weeks_to_keep=52)
            >>>
            >>> # Dry run to preview
            >>> result = client.cleanup_old_forecasts(weeks_to_keep=26, dry_run=True)
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Cleanup old forecasts: keep {weeks_to_keep} weeks")
            return {'deleted_count': 0, 'deleted_weeks': [], 'bytes_freed': 0}

        client = self._get_client()

        # Calculate cutoff date
        now = datetime.now(timezone.utc)
        cutoff_date = now - timedelta(weeks=weeks_to_keep)
        cutoff_year = cutoff_date.year
        cutoff_week = cutoff_date.isocalendar()[1]

        logger.info(f"Cleanup cutoff: {cutoff_year} week {cutoff_week}")

        deleted_count = 0
        deleted_weeks = []
        bytes_freed = 0

        try:
            objects = list(client.list_objects(bucket, recursive=True))

            # Group objects by week
            week_objects = {}
            for obj in objects:
                path_parts = obj.object_name.split('/')
                if len(path_parts) >= 2:
                    try:
                        f_year = int(path_parts[0])
                        f_week = int(path_parts[1].replace('week', ''))

                        # Check if older than cutoff
                        if (f_year < cutoff_year) or \
                           (f_year == cutoff_year and f_week < cutoff_week):
                            key = (f_year, f_week)
                            if key not in week_objects:
                                week_objects[key] = []
                            week_objects[key].append(obj)
                    except ValueError:
                        continue

            # Delete old weeks
            for (year, week), objs in week_objects.items():
                for obj in objs:
                    if dry_run:
                        logger.info(f"[DRY RUN] Would delete: {obj.object_name}")
                    else:
                        client.remove_object(bucket, obj.object_name)
                        logger.debug(f"Deleted: {obj.object_name}")

                    deleted_count += 1
                    bytes_freed += obj.size

                deleted_weeks.append((year, week))

            action = "Would delete" if dry_run else "Deleted"
            logger.info(
                f"{action} {deleted_count} objects from {len(deleted_weeks)} weeks, "
                f"freeing {bytes_freed / (1024*1024):.2f} MB"
            )

        except S3Error as e:
            logger.error(f"Error during cleanup: {e}")

        return {
            'deleted_count': deleted_count,
            'deleted_weeks': sorted(deleted_weeks),
            'bytes_freed': bytes_freed,
            'dry_run': dry_run
        }

    def cleanup_old_models(
        self,
        months_to_keep: int = 6,
        bucket: str = MODELS_BUCKET,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Delete models older than specified months.

        Args:
            months_to_keep: Number of months to retain (default: 6)
            bucket: Bucket name
            dry_run: If True, only list what would be deleted

        Returns:
            Dictionary with deletion summary

        Example:
            >>> # Delete models older than 6 months
            >>> result = client.cleanup_old_models(months_to_keep=6)
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Cleanup old models: keep {months_to_keep} months")
            return {'deleted_count': 0, 'deleted_months': [], 'bytes_freed': 0}

        client = self._get_client()

        # Calculate cutoff date
        now = datetime.now(timezone.utc)
        cutoff_date = now - timedelta(days=months_to_keep * 30)
        cutoff_year = cutoff_date.year
        cutoff_month = cutoff_date.month

        logger.info(f"Cleanup cutoff: {cutoff_year}-{cutoff_month:02d}")

        deleted_count = 0
        deleted_months = set()
        bytes_freed = 0

        try:
            objects = list(client.list_objects(bucket, recursive=True))

            for obj in objects:
                path_parts = obj.object_name.split('/')
                if len(path_parts) >= 2:
                    try:
                        m_year = int(path_parts[0])
                        m_month = int(path_parts[1].replace('month', ''))

                        # Check if older than cutoff
                        if (m_year < cutoff_year) or \
                           (m_year == cutoff_year and m_month < cutoff_month):

                            if dry_run:
                                logger.info(f"[DRY RUN] Would delete: {obj.object_name}")
                            else:
                                client.remove_object(bucket, obj.object_name)
                                logger.debug(f"Deleted: {obj.object_name}")

                            deleted_count += 1
                            bytes_freed += obj.size
                            deleted_months.add((m_year, m_month))
                    except ValueError:
                        continue

            action = "Would delete" if dry_run else "Deleted"
            logger.info(
                f"{action} {deleted_count} objects from {len(deleted_months)} months, "
                f"freeing {bytes_freed / (1024*1024):.2f} MB"
            )

        except S3Error as e:
            logger.error(f"Error during cleanup: {e}")

        return {
            'deleted_count': deleted_count,
            'deleted_months': sorted(list(deleted_months)),
            'bytes_freed': bytes_freed,
            'dry_run': dry_run
        }

    def run_scheduled_cleanup(
        self,
        forecast_weeks: int = 52,
        model_months: int = 6,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Run complete cleanup of old forecasts and models.

        This is meant to be called from a scheduled job (e.g., monthly).

        Args:
            forecast_weeks: Keep forecasts from last N weeks (default: 52)
            model_months: Keep models from last N months (default: 6)
            dry_run: If True, only preview what would be deleted

        Returns:
            Combined cleanup results

        Example:
            >>> # Monthly cleanup job
            >>> result = client.run_scheduled_cleanup()
            >>> print(f"Freed {result['total_bytes_freed'] / 1024**2:.2f} MB")
        """
        logger.info("=" * 60)
        logger.info("RUNNING SCHEDULED CLEANUP")
        logger.info(f"Forecast retention: {forecast_weeks} weeks")
        logger.info(f"Model retention: {model_months} months")
        logger.info(f"Dry run: {dry_run}")
        logger.info("=" * 60)

        # Cleanup forecasts
        forecast_result = self.cleanup_old_forecasts(
            weeks_to_keep=forecast_weeks,
            dry_run=dry_run
        )

        # Cleanup models
        model_result = self.cleanup_old_models(
            months_to_keep=model_months,
            dry_run=dry_run
        )

        total_deleted = forecast_result['deleted_count'] + model_result['deleted_count']
        total_bytes = forecast_result['bytes_freed'] + model_result['bytes_freed']

        logger.info("=" * 60)
        logger.info("CLEANUP COMPLETE")
        logger.info(f"Total objects {'would be ' if dry_run else ''}deleted: {total_deleted}")
        logger.info(f"Total space {'would be ' if dry_run else ''}freed: {total_bytes / (1024*1024):.2f} MB")
        logger.info("=" * 60)

        return {
            'forecasts': forecast_result,
            'models': model_result,
            'total_deleted': total_deleted,
            'total_bytes_freed': total_bytes,
            'dry_run': dry_run,
            'cleanup_timestamp': datetime.now(timezone.utc).isoformat()
        }
