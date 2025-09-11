#!/usr/bin/env python3
"""
MLOps Bucket Provisioner for USDCOP Trading System
==================================================
Automated bucket creation and management using YAML configuration
"""

import os
import sys
import json
import yaml
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time
import hashlib
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from minio import Minio
from minio.error import S3Error

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BucketProvisionerError(Exception):
    """Custom exception for bucket provisioning errors"""
    pass

class MLOpsBucketProvisioner:
    """
    Automated bucket provisioner using YAML configuration
    """
    
    def __init__(self, config_path: str, environment: str = "production"):
        """
        Initialize the bucket provisioner
        
        Args:
            config_path: Path to YAML configuration file
            environment: Environment name (development, staging, production)
        """
        self.config_path = config_path
        self.environment = environment
        self.config = self._load_config()
        self.minio_client = None
        self.s3_client = None
        self.provision_results = {
            "created": [],
            "updated": [],
            "skipped": [],
            "failed": [],
            "errors": []
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate YAML configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Apply environment-specific overrides
            if self.environment in config.get('environments', {}):
                env_config = config['environments'][self.environment]
                config = self._merge_configs(config, env_config)
            
            self._validate_config(config)
            return config
            
        except FileNotFoundError:
            raise BucketProvisionerError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise BucketProvisionerError(f"Invalid YAML configuration: {e}")
    
    def _merge_configs(self, base_config: Dict, env_config: Dict) -> Dict:
        """Merge environment-specific configuration with base configuration"""
        merged = base_config.copy()
        
        # Update MinIO config
        if 'minio_config' in env_config:
            merged['minio_config'].update(env_config['minio_config'])
        
        # Update retention days if specified
        if 'retention_days_override' in env_config:
            for group_name, group_config in merged['bucket_groups'].items():
                group_config['retention_days'] = env_config['retention_days_override']
        
        # Update backup config
        if 'backup_config' in env_config:
            merged.setdefault('backup_config', {}).update(env_config['backup_config'])
        
        return merged
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration structure"""
        required_sections = ['metadata', 'minio_config', 'bucket_groups']
        for section in required_sections:
            if section not in config:
                raise BucketProvisionerError(f"Missing required configuration section: {section}")
        
        # Validate bucket groups
        for group_name, group_config in config['bucket_groups'].items():
            if 'buckets' not in group_config:
                raise BucketProvisionerError(f"Bucket group '{group_name}' missing 'buckets' section")
    
    def _setup_clients(self):
        """Setup MinIO and S3 clients"""
        minio_config = self.config['minio_config']
        
        # Expand environment variables
        endpoint = os.path.expandvars(minio_config['endpoint'])
        access_key = os.path.expandvars(minio_config['access_key'])
        secret_key = os.path.expandvars(minio_config['secret_key'])
        
        # Setup MinIO client
        try:
            self.minio_client = Minio(
                endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=minio_config.get('secure', False)
            )
            # Test connection
            self.minio_client.list_buckets()
            logger.info(f"MinIO client connected to {endpoint}")
            
        except Exception as e:
            raise BucketProvisionerError(f"Failed to connect to MinIO: {e}")
        
        # Setup S3 client for advanced operations
        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=f"http://{endpoint}" if not minio_config.get('secure') else f"https://{endpoint}",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=minio_config.get('region', 'us-east-1')
            )
            logger.info("S3 client initialized for advanced operations")
            
        except Exception as e:
            logger.warning(f"S3 client initialization failed: {e}")
            self.s3_client = None
    
    def provision_all_buckets(self) -> Dict[str, Any]:
        """Provision all buckets defined in configuration"""
        logger.info(f"Starting bucket provisioning for environment: {self.environment}")
        
        self._setup_clients()
        
        total_buckets = 0
        
        # Process each bucket group
        for group_name, group_config in self.config['bucket_groups'].items():
            logger.info(f"Processing bucket group: {group_name}")
            
            for bucket_config in group_config['buckets']:
                total_buckets += 1
                self._provision_bucket(bucket_config, group_config)
        
        # Generate provision report
        report = self._generate_provision_report(total_buckets)
        logger.info(f"Bucket provisioning completed: {report['summary']}")
        
        return report
    
    def _provision_bucket(self, bucket_config: Dict, group_config: Dict):
        """Provision a single bucket"""
        bucket_name = bucket_config['name']
        
        try:
            # Check if bucket exists
            bucket_exists = self._bucket_exists(bucket_name)
            
            if bucket_exists:
                logger.info(f"Bucket {bucket_name} already exists")
                self.provision_results['skipped'].append(bucket_name)
                
                # Update bucket configuration if needed
                self._update_bucket_config(bucket_name, bucket_config, group_config)
                
            else:
                # Create bucket
                self._create_bucket(bucket_name, bucket_config, group_config)
                logger.info(f"Created bucket: {bucket_name}")
                self.provision_results['created'].append(bucket_name)
            
            # Setup bucket features
            self._setup_bucket_features(bucket_name, bucket_config, group_config)
            
        except Exception as e:
            logger.error(f"Failed to provision bucket {bucket_name}: {e}")
            self.provision_results['failed'].append(bucket_name)
            self.provision_results['errors'].append(f"{bucket_name}: {str(e)}")
    
    def _bucket_exists(self, bucket_name: str) -> bool:
        """Check if bucket exists"""
        try:
            self.minio_client.bucket_exists(bucket_name)
            return True
        except S3Error:
            return False
    
    def _create_bucket(self, bucket_name: str, bucket_config: Dict, group_config: Dict):
        """Create a new bucket"""
        try:
            self.minio_client.make_bucket(bucket_name)
            
            # Add creation metadata
            creation_metadata = {
                "created_by": "MLOps Bucket Provisioner",
                "created_at": datetime.now().isoformat(),
                "environment": self.environment,
                "configuration_version": self.config['metadata']['version']
            }
            
            # Upload metadata file
            metadata_content = json.dumps(creation_metadata, indent=2).encode('utf-8')
            self.minio_client.put_object(
                bucket_name,
                "_metadata/bucket_creation.json",
                data=metadata_content,
                length=len(metadata_content),
                content_type="application/json"
            )
            
        except S3Error as e:
            raise BucketProvisionerError(f"Failed to create bucket {bucket_name}: {e}")
    
    def _setup_bucket_features(self, bucket_name: str, bucket_config: Dict, group_config: Dict):
        """Setup bucket features like versioning, lifecycle, etc."""
        
        # Enable versioning if specified
        if group_config.get('versioning', False):
            self._enable_versioning(bucket_name)
        
        # Setup lifecycle policies
        if 'lifecycle_rules' in group_config:
            self._setup_lifecycle_policy(bucket_name, group_config['lifecycle_rules'])
        
        # Setup bucket policy
        self._setup_bucket_policy(bucket_name, bucket_config)
        
        # Setup bucket tags
        if 'tags' in bucket_config:
            self._setup_bucket_tags(bucket_name, bucket_config['tags'])
    
    def _enable_versioning(self, bucket_name: str):
        """Enable bucket versioning"""
        if not self.s3_client:
            logger.warning(f"Cannot enable versioning for {bucket_name}: S3 client not available")
            return
        
        try:
            self.s3_client.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            logger.debug(f"Enabled versioning for bucket: {bucket_name}")
        except Exception as e:
            logger.warning(f"Failed to enable versioning for {bucket_name}: {e}")
    
    def _setup_lifecycle_policy(self, bucket_name: str, lifecycle_rules: Dict):
        """Setup lifecycle policy for bucket"""
        if not self.s3_client:
            logger.warning(f"Cannot setup lifecycle for {bucket_name}: S3 client not available")
            return
        
        try:
            # Create lifecycle configuration
            lifecycle_config = {
                'Rules': []
            }
            
            # Add incomplete upload cleanup rule
            if 'delete_incomplete_uploads' in lifecycle_rules:
                lifecycle_config['Rules'].append({
                    'ID': 'delete-incomplete-uploads',
                    'Status': 'Enabled',
                    'AbortIncompleteMultipartUpload': {
                        'DaysAfterInitiation': lifecycle_rules['delete_incomplete_uploads']
                    }
                })
            
            # Add transition to IA rule
            if 'transition_to_ia' in lifecycle_rules:
                lifecycle_config['Rules'].append({
                    'ID': 'transition-to-ia',
                    'Status': 'Enabled',
                    'Transitions': [{
                        'Days': lifecycle_rules['transition_to_ia'],
                        'StorageClass': 'STANDARD_IA'
                    }]
                })
            
            if lifecycle_config['Rules']:
                self.s3_client.put_bucket_lifecycle_configuration(
                    Bucket=bucket_name,
                    LifecycleConfiguration=lifecycle_config
                )
                logger.debug(f"Setup lifecycle policy for bucket: {bucket_name}")
                
        except Exception as e:
            logger.warning(f"Failed to setup lifecycle for {bucket_name}: {e}")
    
    def _setup_bucket_policy(self, bucket_name: str, bucket_config: Dict):
        """Setup bucket access policy"""
        if not self.s3_client:
            logger.warning(f"Cannot setup policy for {bucket_name}: S3 client not available")
            return
        
        try:
            # Use default policy if none specified
            policy_template = self.config['bucket_policies']['default_policy']
            
            # Replace bucket name placeholder
            policy_str = json.dumps(policy_template)
            policy_str = policy_str.replace('{bucket_name}', bucket_name)
            policy = json.loads(policy_str)
            
            self.s3_client.put_bucket_policy(
                Bucket=bucket_name,
                Policy=json.dumps(policy)
            )
            logger.debug(f"Setup bucket policy for: {bucket_name}")
            
        except Exception as e:
            logger.warning(f"Failed to setup policy for {bucket_name}: {e}")
    
    def _setup_bucket_tags(self, bucket_name: str, tags: Dict[str, str]):
        """Setup bucket tags"""
        if not self.s3_client:
            logger.warning(f"Cannot setup tags for {bucket_name}: S3 client not available")
            return
        
        try:
            # Convert tags to S3 format
            tag_set = [{'Key': k, 'Value': str(v)} for k, v in tags.items()]
            
            # Add additional metadata tags
            tag_set.extend([
                {'Key': 'Environment', 'Value': self.environment},
                {'Key': 'ManagedBy', 'Value': 'MLOps-Bucket-Provisioner'},
                {'Key': 'ConfigVersion', 'Value': self.config['metadata']['version']}
            ])
            
            self.s3_client.put_bucket_tagging(
                Bucket=bucket_name,
                Tagging={'TagSet': tag_set}
            )
            logger.debug(f"Setup tags for bucket: {bucket_name}")
            
        except Exception as e:
            logger.warning(f"Failed to setup tags for {bucket_name}: {e}")
    
    def _update_bucket_config(self, bucket_name: str, bucket_config: Dict, group_config: Dict):
        """Update existing bucket configuration"""
        try:
            # Always try to update features for existing buckets
            self._setup_bucket_features(bucket_name, bucket_config, group_config)
            self.provision_results['updated'].append(bucket_name)
            
        except Exception as e:
            logger.warning(f"Failed to update bucket {bucket_name}: {e}")
    
    def _generate_provision_report(self, total_buckets: int) -> Dict[str, Any]:
        """Generate provisioning report"""
        return {
            "summary": {
                "total_buckets": total_buckets,
                "created": len(self.provision_results['created']),
                "updated": len(self.provision_results['updated']),
                "skipped": len(self.provision_results['skipped']),
                "failed": len(self.provision_results['failed'])
            },
            "details": self.provision_results,
            "environment": self.environment,
            "config_version": self.config['metadata']['version'],
            "timestamp": datetime.now().isoformat()
        }
    
    def validate_buckets(self) -> Dict[str, Any]:
        """Validate all buckets exist and are accessible"""
        logger.info("Validating bucket configuration...")
        
        self._setup_clients()
        
        validation_results = {
            "accessible": [],
            "inaccessible": [],
            "missing": [],
            "total_checked": 0
        }
        
        # Check each bucket
        for group_name, group_config in self.config['bucket_groups'].items():
            for bucket_config in group_config['buckets']:
                bucket_name = bucket_config['name']
                validation_results['total_checked'] += 1
                
                try:
                    # Check if bucket exists and is accessible
                    if self.minio_client.bucket_exists(bucket_name):
                        # Try to list objects to verify access
                        objects = self.minio_client.list_objects(bucket_name, max_keys=1)
                        list(objects)  # Consume the iterator to trigger any errors
                        validation_results['accessible'].append(bucket_name)
                        logger.debug(f"Bucket {bucket_name} is accessible")
                    else:
                        validation_results['missing'].append(bucket_name)
                        logger.warning(f"Bucket {bucket_name} does not exist")
                        
                except Exception as e:
                    validation_results['inaccessible'].append({
                        "bucket": bucket_name,
                        "error": str(e)
                    })
                    logger.error(f"Bucket {bucket_name} is not accessible: {e}")
        
        # Generate validation summary
        validation_results['summary'] = {
            "healthy": len(validation_results['accessible']),
            "unhealthy": len(validation_results['inaccessible']) + len(validation_results['missing']),
            "success_rate": len(validation_results['accessible']) / validation_results['total_checked'] * 100
        }
        
        return validation_results
    
    def cleanup_failed_buckets(self) -> Dict[str, Any]:
        """Clean up any buckets that failed to provision properly"""
        logger.info("Cleaning up failed bucket provisioning...")
        
        cleanup_results = {
            "cleaned": [],
            "failed_cleanup": [],
            "skipped": []
        }
        
        for bucket_name in self.provision_results['failed']:
            try:
                if self.minio_client.bucket_exists(bucket_name):
                    # Try to clean up partial bucket
                    self._cleanup_bucket(bucket_name)
                    cleanup_results['cleaned'].append(bucket_name)
                else:
                    cleanup_results['skipped'].append(bucket_name)
                    
            except Exception as e:
                cleanup_results['failed_cleanup'].append({
                    "bucket": bucket_name,
                    "error": str(e)
                })
        
        return cleanup_results
    
    def _cleanup_bucket(self, bucket_name: str):
        """Clean up a partially created bucket"""
        try:
            # Remove all objects first
            objects = self.minio_client.list_objects(bucket_name, recursive=True)
            for obj in objects:
                self.minio_client.remove_object(bucket_name, obj.object_name)
            
            # Remove bucket
            self.minio_client.remove_bucket(bucket_name)
            logger.info(f"Cleaned up bucket: {bucket_name}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup bucket {bucket_name}: {e}")
            raise

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="MLOps Bucket Provisioner for USDCOP Trading System"
    )
    parser.add_argument(
        '--config', 
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--environment',
        default='production',
        choices=['development', 'staging', 'production'],
        help='Environment to provision (default: production)'
    )
    parser.add_argument(
        '--action',
        default='provision',
        choices=['provision', 'validate', 'cleanup'],
        help='Action to perform (default: provision)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate provisioning without making changes'
    )
    parser.add_argument(
        '--output',
        help='Output file for results (JSON format)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize provisioner
        provisioner = MLOpsBucketProvisioner(args.config, args.environment)
        
        # Perform requested action
        if args.action == 'provision':
            if args.dry_run:
                logger.info("DRY RUN MODE: No actual changes will be made")
                # TODO: Implement dry-run logic
                results = {"dry_run": True, "message": "Dry run completed"}
            else:
                results = provisioner.provision_all_buckets()
                
        elif args.action == 'validate':
            results = provisioner.validate_buckets()
            
        elif args.action == 'cleanup':
            results = provisioner.cleanup_failed_buckets()
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results written to: {args.output}")
        else:
            print(json.dumps(results, indent=2))
        
        # Return appropriate exit code
        if args.action == 'provision':
            exit_code = 0 if results['summary']['failed'] == 0 else 1
        elif args.action == 'validate':
            exit_code = 0 if results['summary']['unhealthy'] == 0 else 1
        else:
            exit_code = 0
            
        return exit_code
        
    except BucketProvisionerError as e:
        logger.error(f"Provisioning error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 2

if __name__ == "__main__":
    sys.exit(main())