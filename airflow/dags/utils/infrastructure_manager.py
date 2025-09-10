"""
Infrastructure Management for MLflow and MinIO
==============================================
Handles infrastructure setup, validation, and fixes
"""

import os
import sys
import json
import logging
import subprocess
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
import mlflow
from mlflow.tracking import MlflowClient
import requests

logger = logging.getLogger(__name__)

class MLflowManager:
    """
    Manages MLflow infrastructure and configuration
    """
    
    def __init__(self, tracking_uri: str = None):
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", 
            "http://trading-mlflow:5000"
        )
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient()
    
    def setup_environment(self) -> Dict[str, Any]:
        """Setup MLflow environment variables"""
        env_vars = {
            "MLFLOW_TRACKING_URI": self.tracking_uri,
            "MLFLOW_S3_ENDPOINT_URL": os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://trading-minio:9000"),
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123"),
            "AWS_S3_ADDRESSING_STYLE": "path",
            "AWS_DEFAULT_REGION": "us-east-1",
            "GIT_PYTHON_GIT_EXECUTABLE": "/usr/bin/git",
            "GIT_PYTHON_REFRESH": "quiet"
        }
        
        # Set environment variables
        for key, value in env_vars.items():
            os.environ[key] = value
        
        logger.info(f"MLflow environment configured: {self.tracking_uri}")
        return env_vars
    
    def validate_connection(self) -> bool:
        """Validate MLflow server connection"""
        try:
            # Try to list experiments
            experiments = self.client.list_experiments()
            logger.info(f"MLflow connection successful. Found {len(experiments)} experiments")
            return True
        except Exception as e:
            logger.error(f"MLflow connection failed: {e}")
            return False
    
    def create_experiment(self, name: str, tags: Dict[str, str] = None) -> str:
        """Create or get MLflow experiment"""
        try:
            experiment = self.client.get_experiment_by_name(name)
            if experiment:
                logger.info(f"Using existing experiment: {name}")
                return experiment.experiment_id
            else:
                experiment_id = self.client.create_experiment(
                    name=name,
                    tags=tags or {}
                )
                logger.info(f"Created new experiment: {name} (ID: {experiment_id})")
                return experiment_id
        except Exception as e:
            logger.error(f"Failed to create/get experiment: {e}")
            raise
    
    def log_model_lineage(
        self, 
        run_id: str, 
        lineage_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Log comprehensive model lineage"""
        try:
            with mlflow.start_run(run_id=run_id):
                # Log Git information
                git_info = self._get_git_info()
                mlflow.set_tags({
                    "git.sha": git_info.get("sha", "unknown"),
                    "git.branch": git_info.get("branch", "unknown"),
                    "git.remote": git_info.get("remote", "unknown")
                })
                
                # Log container information
                container_info = self._get_container_info()
                mlflow.set_tags({
                    "container.image": container_info.get("image", "unknown"),
                    "container.digest": container_info.get("digest", "unknown")
                })
                
                # Log dataset hash
                mlflow.set_tag("dataset.hash", lineage_info.get("dataset_hash", "unknown"))
                
                # Log dependencies
                deps = self._get_dependencies()
                mlflow.log_text(deps, "requirements_frozen.txt")
                
                # Log complete lineage
                mlflow.log_dict(lineage_info, "lineage.json")
                
            logger.info(f"Model lineage logged for run {run_id}")
            return {
                "run_id": run_id,
                "git": git_info,
                "container": container_info,
                "lineage": lineage_info
            }
            
        except Exception as e:
            logger.error(f"Failed to log model lineage: {e}")
            return {}
    
    def _get_git_info(self) -> Dict[str, str]:
        """Get Git repository information"""
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                text=True
            ).strip()
            
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], 
                text=True
            ).strip()
            
            remote = subprocess.check_output(
                ["git", "config", "--get", "remote.origin.url"], 
                text=True
            ).strip()
            
            return {
                "sha": sha,
                "branch": branch,
                "remote": remote
            }
        except Exception as e:
            logger.warning(f"Could not get Git info: {e}")
            return {}
    
    def _get_container_info(self) -> Dict[str, str]:
        """Get container information"""
        try:
            # Get hostname (container ID in Docker)
            hostname = subprocess.check_output(
                ["hostname"], 
                text=True
            ).strip()
            
            # Try to get image info from environment
            image = os.getenv("DOCKER_IMAGE", "unknown")
            digest = os.getenv("DOCKER_DIGEST", "unknown")
            
            return {
                "hostname": hostname,
                "image": image,
                "digest": digest
            }
        except Exception as e:
            logger.warning(f"Could not get container info: {e}")
            return {}
    
    def _get_dependencies(self) -> str:
        """Get frozen dependencies"""
        try:
            deps = subprocess.check_output(
                ["pip", "freeze"], 
                text=True
            )
            return deps
        except Exception as e:
            logger.warning(f"Could not get dependencies: {e}")
            return ""
    
    def cleanup_old_runs(self, experiment_id: str, keep_best_n: int = 5):
        """Clean up old MLflow runs, keeping best N"""
        try:
            # Get all runs
            runs = self.client.search_runs(
                experiment_ids=[experiment_id],
                order_by=["metrics.eval/mean_reward DESC"]
            )
            
            if len(runs) <= keep_best_n:
                logger.info(f"No cleanup needed. Only {len(runs)} runs exist")
                return
            
            # Delete older runs
            runs_to_delete = runs[keep_best_n:]
            for run in runs_to_delete:
                self.client.delete_run(run.info.run_id)
                logger.info(f"Deleted run {run.info.run_id}")
            
            logger.info(f"Cleaned up {len(runs_to_delete)} old runs")
            
        except Exception as e:
            logger.error(f"Failed to cleanup runs: {e}")

class MinIOManager:
    """
    Manages MinIO/S3 infrastructure
    """
    
    def __init__(
        self, 
        endpoint_url: str = None,
        access_key: str = None,
        secret_key: str = None
    ):
        self.endpoint_url = endpoint_url or os.getenv(
            "MLFLOW_S3_ENDPOINT_URL", 
            "http://trading-minio:9000"
        )
        self.access_key = access_key or os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
        self.secret_key = secret_key or os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
        
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name='us-east-1'
        )
    
    def validate_connection(self) -> bool:
        """Validate MinIO connection"""
        try:
            self.s3_client.list_buckets()
            logger.info("MinIO connection successful")
            return True
        except Exception as e:
            logger.error(f"MinIO connection failed: {e}")
            return False
    
    def ensure_bucket_exists(self, bucket_name: str) -> bool:
        """Ensure bucket exists, create if not"""
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"Bucket {bucket_name} exists")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                try:
                    self.s3_client.create_bucket(Bucket=bucket_name)
                    logger.info(f"Created bucket {bucket_name}")
                    
                    # Enable versioning
                    self.s3_client.put_bucket_versioning(
                        Bucket=bucket_name,
                        VersioningConfiguration={'Status': 'Enabled'}
                    )
                    logger.info(f"Enabled versioning for {bucket_name}")
                    
                    return True
                except Exception as create_error:
                    logger.error(f"Failed to create bucket: {create_error}")
                    return False
            else:
                logger.error(f"Bucket check failed: {e}")
                return False
    
    def setup_bucket_policy(self, bucket_name: str) -> bool:
        """Setup bucket policy for proper access"""
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": "*"},
                    "Action": [
                        "s3:GetBucketLocation",
                        "s3:ListBucket",
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:DeleteObject"
                    ],
                    "Resource": [
                        f"arn:aws:s3:::{bucket_name}",
                        f"arn:aws:s3:::{bucket_name}/*"
                    ]
                }
            ]
        }
        
        try:
            self.s3_client.put_bucket_policy(
                Bucket=bucket_name,
                Policy=json.dumps(policy)
            )
            logger.info(f"Bucket policy set for {bucket_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to set bucket policy: {e}")
            return False
    
    def get_bucket_stats(self, bucket_name: str) -> Dict[str, Any]:
        """Get bucket statistics"""
        try:
            # List objects
            response = self.s3_client.list_objects_v2(Bucket=bucket_name)
            
            total_size = 0
            object_count = 0
            
            if 'Contents' in response:
                object_count = len(response['Contents'])
                total_size = sum(obj['Size'] for obj in response['Contents'])
            
            return {
                "bucket": bucket_name,
                "object_count": object_count,
                "total_size_mb": total_size / (1024 * 1024),
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get bucket stats: {e}")
            return {
                "bucket": bucket_name,
                "error": str(e)
            }
    
    def cleanup_old_artifacts(
        self, 
        bucket_name: str, 
        prefix: str, 
        keep_days: int = 7
    ) -> int:
        """Clean up old artifacts"""
        try:
            # List objects with prefix
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                logger.info(f"No objects found with prefix {prefix}")
                return 0
            
            # Calculate cutoff time
            cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 3600)
            
            # Delete old objects
            deleted_count = 0
            for obj in response['Contents']:
                obj_time = obj['LastModified'].timestamp()
                if obj_time < cutoff_time:
                    self.s3_client.delete_object(
                        Bucket=bucket_name,
                        Key=obj['Key']
                    )
                    deleted_count += 1
                    logger.debug(f"Deleted {obj['Key']}")
            
            logger.info(f"Deleted {deleted_count} old objects from {bucket_name}/{prefix}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup artifacts: {e}")
            return 0

class InfrastructureValidator:
    """
    Validates entire infrastructure setup
    """
    
    def __init__(self):
        self.mlflow_manager = MLflowManager()
        self.minio_manager = MinIOManager()
        self.validation_results = {}
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all infrastructure validations"""
        
        # Validate MLflow
        mlflow_valid = self.mlflow_manager.validate_connection()
        self.validation_results['mlflow'] = {
            "status": "healthy" if mlflow_valid else "unhealthy",
            "tracking_uri": self.mlflow_manager.tracking_uri,
            "validated": mlflow_valid
        }
        
        # Validate MinIO
        minio_valid = self.minio_manager.validate_connection()
        self.validation_results['minio'] = {
            "status": "healthy" if minio_valid else "unhealthy",
            "endpoint": self.minio_manager.endpoint_url,
            "validated": minio_valid
        }
        
        # Check required buckets
        required_buckets = [
            "00-raw-usdcop-marketdata",
            "01-l1-ds-usdcop-standardize",
            "02-l2-ds-usdcop-prepare",
            "03-l3-ds-usdcop-feature",
            "04-l4-ds-usdcop-rlready",
            "05-l5-ds-usdcop-serving",
            "mlflow",
            "airflow"
        ]
        
        bucket_status = {}
        for bucket in required_buckets:
            exists = self.minio_manager.ensure_bucket_exists(bucket)
            bucket_status[bucket] = exists
            if exists:
                stats = self.minio_manager.get_bucket_stats(bucket)
                bucket_status[f"{bucket}_stats"] = stats
        
        self.validation_results['buckets'] = bucket_status
        
        # Check Git
        git_info = self.mlflow_manager._get_git_info()
        self.validation_results['git'] = {
            "available": bool(git_info),
            "info": git_info
        }
        
        # Overall health
        all_healthy = (
            mlflow_valid and 
            minio_valid and 
            all(bucket_status.get(b, False) for b in required_buckets)
        )
        
        self.validation_results['overall'] = {
            "healthy": all_healthy,
            "timestamp": datetime.now().isoformat()
        }
        
        return self.validation_results
    
    def fix_issues(self) -> Dict[str, Any]:
        """Attempt to fix identified issues"""
        fixes_applied = {}
        
        # Fix MLflow environment
        if not self.validation_results.get('mlflow', {}).get('validated'):
            env_vars = self.mlflow_manager.setup_environment()
            fixes_applied['mlflow_env'] = env_vars
        
        # Create missing buckets
        for bucket, exists in self.validation_results.get('buckets', {}).items():
            if not exists and not bucket.endswith('_stats'):
                created = self.minio_manager.ensure_bucket_exists(bucket)
                if created:
                    self.minio_manager.setup_bucket_policy(bucket)
                fixes_applied[f"bucket_{bucket}"] = created
        
        # Re-validate after fixes
        final_validation = self.validate_all()
        
        return {
            "fixes_applied": fixes_applied,
            "final_validation": final_validation
        }

def setup_infrastructure() -> Dict[str, Any]:
    """
    Main function to setup and validate infrastructure
    """
    logger.info("Setting up infrastructure...")
    
    # Initialize validator
    validator = InfrastructureValidator()
    
    # Validate current state
    initial_validation = validator.validate_all()
    logger.info(f"Initial validation: {initial_validation['overall']}")
    
    # Apply fixes if needed
    if not initial_validation['overall']['healthy']:
        logger.info("Applying infrastructure fixes...")
        fix_results = validator.fix_issues()
        
        if fix_results['final_validation']['overall']['healthy']:
            logger.info("Infrastructure fixed successfully!")
        else:
            logger.error("Some infrastructure issues remain")
        
        return fix_results['final_validation']
    
    logger.info("Infrastructure is healthy")
    return initial_validation

if __name__ == "__main__":
    # Run infrastructure setup
    results = setup_infrastructure()
    print(json.dumps(results, indent=2))