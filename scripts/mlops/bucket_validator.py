#!/usr/bin/env python3
"""
Comprehensive Bucket Validation System for USDCOP Trading System
================================================================
Advanced validation, monitoring, and health checking for MinIO buckets
"""

import os
import sys
import json
import yaml
import logging
import argparse
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib

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

@dataclass
class BucketHealthStatus:
    """Data class for bucket health status"""
    name: str
    exists: bool
    accessible: bool
    object_count: int
    total_size_bytes: int
    last_modified: Optional[datetime]
    versioning_enabled: bool
    lifecycle_configured: bool
    policy_configured: bool
    tags_configured: bool
    errors: List[str]
    warnings: List[str]
    health_score: float
    status: str  # healthy, degraded, unhealthy

@dataclass
class ValidationReport:
    """Complete validation report"""
    timestamp: datetime
    environment: str
    total_buckets: int
    healthy_buckets: int
    degraded_buckets: int
    unhealthy_buckets: int
    overall_health_score: float
    bucket_statuses: List[BucketHealthStatus]
    infrastructure_status: Dict[str, Any]
    recommendations: List[str]

class BucketValidator:
    """
    Comprehensive bucket validation and health monitoring
    """
    
    def __init__(self, config_path: str, environment: str = "production"):
        """
        Initialize bucket validator
        
        Args:
            config_path: Path to YAML configuration file
            environment: Environment name
        """
        self.config_path = config_path
        self.environment = environment
        self.config = self._load_config()
        self.minio_client = None
        self.s3_client = None
        self.validation_start_time = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate YAML configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Apply environment-specific overrides
            if self.environment in config.get('environments', {}):
                env_config = config['environments'][self.environment]
                config = self._merge_configs(config, env_config)
            
            return config
            
        except FileNotFoundError:
            raise ValueError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _merge_configs(self, base_config: Dict, env_config: Dict) -> Dict:
        """Merge environment-specific configuration"""
        merged = base_config.copy()
        
        if 'minio_config' in env_config:
            merged['minio_config'].update(env_config['minio_config'])
        
        return merged
    
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
            
        except Exception as e:
            raise ConnectionError(f"Failed to setup MinIO client: {e}")
        
        # Setup S3 client for advanced operations
        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=f"http://{endpoint}" if not minio_config.get('secure') else f"https://{endpoint}",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=minio_config.get('region', 'us-east-1')
            )
            
        except Exception as e:
            logger.warning(f"S3 client setup failed: {e}")
            self.s3_client = None
    
    def validate_all_buckets(self) -> ValidationReport:
        """
        Perform comprehensive validation of all buckets
        
        Returns:
            Complete validation report
        """
        logger.info("Starting comprehensive bucket validation...")
        self.validation_start_time = datetime.now()
        
        self._setup_clients()
        
        bucket_statuses = []
        recommendations = []
        
        # Validate each bucket
        for group_name, group_config in self.config['bucket_groups'].items():
            for bucket_config in group_config['buckets']:
                logger.info(f"Validating bucket: {bucket_config['name']}")
                status = self._validate_single_bucket(bucket_config, group_config)
                bucket_statuses.append(status)
        
        # Calculate overall metrics
        total_buckets = len(bucket_statuses)
        healthy_buckets = sum(1 for s in bucket_statuses if s.status == 'healthy')
        degraded_buckets = sum(1 for s in bucket_statuses if s.status == 'degraded')
        unhealthy_buckets = sum(1 for s in bucket_statuses if s.status == 'unhealthy')
        
        overall_health_score = sum(s.health_score for s in bucket_statuses) / total_buckets if total_buckets > 0 else 0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(bucket_statuses)
        
        # Validate infrastructure
        infrastructure_status = self._validate_infrastructure()
        
        # Create validation report
        report = ValidationReport(
            timestamp=self.validation_start_time,
            environment=self.environment,
            total_buckets=total_buckets,
            healthy_buckets=healthy_buckets,
            degraded_buckets=degraded_buckets,
            unhealthy_buckets=unhealthy_buckets,
            overall_health_score=overall_health_score,
            bucket_statuses=bucket_statuses,
            infrastructure_status=infrastructure_status,
            recommendations=recommendations
        )
        
        validation_duration = datetime.now() - self.validation_start_time
        logger.info(f"Validation completed in {validation_duration.total_seconds():.2f} seconds")
        
        return report
    
    def _validate_single_bucket(self, bucket_config: Dict, group_config: Dict) -> BucketHealthStatus:
        """Validate a single bucket comprehensively"""
        bucket_name = bucket_config['name']
        errors = []
        warnings = []
        
        # Initialize status
        status = BucketHealthStatus(
            name=bucket_name,
            exists=False,
            accessible=False,
            object_count=0,
            total_size_bytes=0,
            last_modified=None,
            versioning_enabled=False,
            lifecycle_configured=False,
            policy_configured=False,
            tags_configured=False,
            errors=errors,
            warnings=warnings,
            health_score=0.0,
            status='unhealthy'
        )
        
        try:
            # Check bucket existence
            if not self.minio_client.bucket_exists(bucket_name):
                errors.append("Bucket does not exist")
                return status
            
            status.exists = True
            
            # Check bucket accessibility
            try:
                objects = list(self.minio_client.list_objects(bucket_name, max_keys=1))
                status.accessible = True
            except Exception as e:
                errors.append(f"Bucket not accessible: {e}")
                return status
            
            # Get bucket statistics
            self._get_bucket_statistics(status)
            
            # Check versioning
            status.versioning_enabled = self._check_versioning(bucket_name)
            if group_config.get('versioning', False) and not status.versioning_enabled:
                warnings.append("Versioning not enabled but required by configuration")
            
            # Check lifecycle configuration
            status.lifecycle_configured = self._check_lifecycle(bucket_name)
            if 'lifecycle_rules' in group_config and not status.lifecycle_configured:
                warnings.append("Lifecycle rules not configured")
            
            # Check bucket policy
            status.policy_configured = self._check_bucket_policy(bucket_name)
            if not status.policy_configured:
                warnings.append("Bucket policy not configured")
            
            # Check bucket tags
            status.tags_configured = self._check_bucket_tags(bucket_name)
            if 'tags' in bucket_config and not status.tags_configured:
                warnings.append("Bucket tags not configured")
            
            # Calculate health score
            status.health_score = self._calculate_health_score(status)
            
            # Determine overall status
            if status.health_score >= 0.9 and len(errors) == 0:
                status.status = 'healthy'
            elif status.health_score >= 0.7 and len(errors) == 0:
                status.status = 'degraded'
            else:
                status.status = 'unhealthy'
                
        except Exception as e:
            errors.append(f"Validation failed: {e}")
            logger.error(f"Failed to validate bucket {bucket_name}: {e}")
        
        return status
    
    def _get_bucket_statistics(self, status: BucketHealthStatus):
        """Get detailed bucket statistics"""
        try:
            objects = self.minio_client.list_objects(status.name, recursive=True)
            
            total_size = 0
            object_count = 0
            last_modified = None
            
            for obj in objects:
                object_count += 1
                total_size += obj.size or 0
                
                if last_modified is None or (obj.last_modified and obj.last_modified > last_modified):
                    last_modified = obj.last_modified
            
            status.object_count = object_count
            status.total_size_bytes = total_size
            status.last_modified = last_modified
            
        except Exception as e:
            status.warnings.append(f"Failed to get bucket statistics: {e}")
    
    def _check_versioning(self, bucket_name: str) -> bool:
        """Check if bucket versioning is enabled"""
        if not self.s3_client:
            return False
        
        try:
            response = self.s3_client.get_bucket_versioning(Bucket=bucket_name)
            return response.get('Status') == 'Enabled'
        except Exception:
            return False
    
    def _check_lifecycle(self, bucket_name: str) -> bool:
        """Check if lifecycle configuration exists"""
        if not self.s3_client:
            return False
        
        try:
            self.s3_client.get_bucket_lifecycle_configuration(Bucket=bucket_name)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchLifecycleConfiguration':
                return False
            return False
        except Exception:
            return False
    
    def _check_bucket_policy(self, bucket_name: str) -> bool:
        """Check if bucket policy is configured"""
        if not self.s3_client:
            return False
        
        try:
            self.s3_client.get_bucket_policy(Bucket=bucket_name)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchBucketPolicy':
                return False
            return False
        except Exception:
            return False
    
    def _check_bucket_tags(self, bucket_name: str) -> bool:
        """Check if bucket tags are configured"""
        if not self.s3_client:
            return False
        
        try:
            response = self.s3_client.get_bucket_tagging(Bucket=bucket_name)
            return len(response.get('TagSet', [])) > 0
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchTagSet':
                return False
            return False
        except Exception:
            return False
    
    def _calculate_health_score(self, status: BucketHealthStatus) -> float:
        """Calculate health score for a bucket"""
        score = 0.0
        
        # Base score for existence and accessibility
        if status.exists:
            score += 0.3
        if status.accessible:
            score += 0.3
        
        # Configuration completeness
        if status.versioning_enabled:
            score += 0.1
        if status.lifecycle_configured:
            score += 0.1
        if status.policy_configured:
            score += 0.1
        if status.tags_configured:
            score += 0.1
        
        # Penalty for errors and warnings
        score -= len(status.errors) * 0.2
        score -= len(status.warnings) * 0.05
        
        return max(0.0, min(1.0, score))
    
    def _generate_recommendations(self, bucket_statuses: List[BucketHealthStatus]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Analyze common issues
        unhealthy_buckets = [s for s in bucket_statuses if s.status == 'unhealthy']
        if unhealthy_buckets:
            recommendations.append(f"Immediate attention required for {len(unhealthy_buckets)} unhealthy buckets")
        
        # Check for missing configurations
        missing_versioning = [s for s in bucket_statuses if not s.versioning_enabled]
        if missing_versioning:
            recommendations.append(f"Enable versioning for {len(missing_versioning)} buckets")
        
        missing_lifecycle = [s for s in bucket_statuses if not s.lifecycle_configured]
        if missing_lifecycle:
            recommendations.append(f"Configure lifecycle policies for {len(missing_lifecycle)} buckets")
        
        missing_policies = [s for s in bucket_statuses if not s.policy_configured]
        if missing_policies:
            recommendations.append(f"Configure access policies for {len(missing_policies)} buckets")
        
        # Storage optimization
        large_buckets = [s for s in bucket_statuses if s.total_size_bytes > 1024**3]  # > 1GB
        if large_buckets:
            recommendations.append(f"Consider archiving policies for {len(large_buckets)} large buckets")
        
        # Empty buckets
        empty_buckets = [s for s in bucket_statuses if s.object_count == 0]
        if empty_buckets:
            recommendations.append(f"Review necessity of {len(empty_buckets)} empty buckets")
        
        return recommendations
    
    def _validate_infrastructure(self) -> Dict[str, Any]:
        """Validate underlying infrastructure"""
        infrastructure_status = {
            "minio_connectivity": False,
            "s3_api_availability": False,
            "response_time_ms": None,
            "disk_space_available": None,
            "concurrent_connections": None
        }
        
        try:
            # Test MinIO connectivity
            start_time = time.time()
            self.minio_client.list_buckets()
            response_time = (time.time() - start_time) * 1000
            
            infrastructure_status["minio_connectivity"] = True
            infrastructure_status["response_time_ms"] = response_time
            
            # Test S3 API
            if self.s3_client:
                self.s3_client.list_buckets()
                infrastructure_status["s3_api_availability"] = True
            
        except Exception as e:
            logger.error(f"Infrastructure validation failed: {e}")
        
        return infrastructure_status
    
    def monitor_bucket_health(self, duration_minutes: int = 5) -> Dict[str, Any]:
        """
        Monitor bucket health over time
        
        Args:
            duration_minutes: How long to monitor
            
        Returns:
            Monitoring results with trends
        """
        logger.info(f"Starting {duration_minutes}-minute bucket health monitoring...")
        
        monitoring_results = {
            "start_time": datetime.now(),
            "duration_minutes": duration_minutes,
            "samples": [],
            "trends": {},
            "alerts": []
        }
        
        sample_interval = 30  # seconds
        total_samples = (duration_minutes * 60) // sample_interval
        
        for sample_num in range(total_samples):
            sample_time = datetime.now()
            logger.info(f"Taking sample {sample_num + 1}/{total_samples}")
            
            # Quick health check
            try:
                buckets = self.minio_client.list_buckets()
                bucket_count = len(buckets)
                
                # Test connectivity
                start_time = time.time()
                self.minio_client.list_objects("00-raw-usdcop-marketdata", max_keys=1)
                response_time = (time.time() - start_time) * 1000
                
                sample = {
                    "timestamp": sample_time,
                    "bucket_count": bucket_count,
                    "response_time_ms": response_time,
                    "connectivity": True
                }
                
            except Exception as e:
                sample = {
                    "timestamp": sample_time,
                    "bucket_count": 0,
                    "response_time_ms": None,
                    "connectivity": False,
                    "error": str(e)
                }
                
                monitoring_results["alerts"].append({
                    "timestamp": sample_time,
                    "severity": "critical",
                    "message": f"Connectivity lost: {e}"
                })
            
            monitoring_results["samples"].append(sample)
            
            if sample_num < total_samples - 1:  # Don't sleep after the last sample
                time.sleep(sample_interval)
        
        # Calculate trends
        monitoring_results["trends"] = self._calculate_trends(monitoring_results["samples"])
        
        return monitoring_results
    
    def _calculate_trends(self, samples: List[Dict]) -> Dict[str, Any]:
        """Calculate trends from monitoring samples"""
        if len(samples) < 2:
            return {}
        
        response_times = [s.get("response_time_ms") for s in samples if s.get("response_time_ms")]
        connectivity_rate = sum(1 for s in samples if s.get("connectivity", False)) / len(samples)
        
        trends = {
            "connectivity_rate": connectivity_rate,
            "avg_response_time_ms": sum(response_times) / len(response_times) if response_times else None,
            "max_response_time_ms": max(response_times) if response_times else None,
            "min_response_time_ms": min(response_times) if response_times else None,
            "response_time_trend": "stable"  # Could implement trend analysis
        }
        
        return trends
    
    def export_validation_report(self, report: ValidationReport, output_path: str, format: str = "json"):
        """
        Export validation report to file
        
        Args:
            report: Validation report to export
            output_path: Output file path
            format: Export format (json, yaml, html)
        """
        if format.lower() == "json":
            self._export_json_report(report, output_path)
        elif format.lower() == "yaml":
            self._export_yaml_report(report, output_path)
        elif format.lower() == "html":
            self._export_html_report(report, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json_report(self, report: ValidationReport, output_path: str):
        """Export report as JSON"""
        # Convert dataclasses to dictionaries
        report_dict = {
            "timestamp": report.timestamp.isoformat(),
            "environment": report.environment,
            "summary": {
                "total_buckets": report.total_buckets,
                "healthy_buckets": report.healthy_buckets,
                "degraded_buckets": report.degraded_buckets,
                "unhealthy_buckets": report.unhealthy_buckets,
                "overall_health_score": report.overall_health_score
            },
            "bucket_statuses": [
                {
                    **asdict(status),
                    "last_modified": status.last_modified.isoformat() if status.last_modified else None
                }
                for status in report.bucket_statuses
            ],
            "infrastructure_status": report.infrastructure_status,
            "recommendations": report.recommendations
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"JSON report exported to: {output_path}")
    
    def _export_yaml_report(self, report: ValidationReport, output_path: str):
        """Export report as YAML"""
        # Similar to JSON export but save as YAML
        report_dict = {
            "timestamp": report.timestamp.isoformat(),
            "environment": report.environment,
            "summary": {
                "total_buckets": report.total_buckets,
                "healthy_buckets": report.healthy_buckets,
                "degraded_buckets": report.degraded_buckets,
                "unhealthy_buckets": report.unhealthy_buckets,
                "overall_health_score": report.overall_health_score
            },
            "recommendations": report.recommendations
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(report_dict, f, default_flow_style=False)
        
        logger.info(f"YAML report exported to: {output_path}")
    
    def _export_html_report(self, report: ValidationReport, output_path: str):
        """Export report as HTML"""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Bucket Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .summary { display: flex; justify-content: space-around; margin: 20px 0; }
        .metric { text-align: center; padding: 15px; background-color: #e8f4f8; border-radius: 5px; }
        .bucket { margin: 10px 0; padding: 15px; border-left: 4px solid #ccc; }
        .healthy { border-left-color: #4CAF50; }
        .degraded { border-left-color: #FF9800; }
        .unhealthy { border-left-color: #F44336; }
        .recommendations { background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .health-score { font-weight: bold; font-size: 1.2em; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Bucket Validation Report</h1>
        <p>Environment: {environment}</p>
        <p>Generated: {timestamp}</p>
        <p class="health-score">Overall Health Score: {health_score:.2f}</p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>{total_buckets}</h3>
            <p>Total Buckets</p>
        </div>
        <div class="metric">
            <h3>{healthy_buckets}</h3>
            <p>Healthy</p>
        </div>
        <div class="metric">
            <h3>{degraded_buckets}</h3>
            <p>Degraded</p>
        </div>
        <div class="metric">
            <h3>{unhealthy_buckets}</h3>
            <p>Unhealthy</p>
        </div>
    </div>
    
    <h2>Bucket Details</h2>
    {bucket_details}
    
    <div class="recommendations">
        <h2>Recommendations</h2>
        <ul>
            {recommendations}
        </ul>
    </div>
</body>
</html>
        """
        
        # Generate bucket details HTML
        bucket_details = ""
        for status in report.bucket_statuses:
            bucket_details += f"""
            <div class="bucket {status.status}">
                <h3>{status.name}</h3>
                <p>Status: {status.status.title()} (Score: {status.health_score:.2f})</p>
                <p>Objects: {status.object_count} | Size: {status.total_size_bytes / (1024**2):.2f} MB</p>
                {f"<p>Errors: {', '.join(status.errors)}</p>" if status.errors else ""}
                {f"<p>Warnings: {', '.join(status.warnings)}</p>" if status.warnings else ""}
            </div>
            """
        
        # Generate recommendations HTML
        recommendations_html = "\n".join(f"<li>{rec}</li>" for rec in report.recommendations)
        
        # Fill template
        html_content = html_template.format(
            environment=report.environment,
            timestamp=report.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            health_score=report.overall_health_score,
            total_buckets=report.total_buckets,
            healthy_buckets=report.healthy_buckets,
            degraded_buckets=report.degraded_buckets,
            unhealthy_buckets=report.unhealthy_buckets,
            bucket_details=bucket_details,
            recommendations=recommendations_html
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report exported to: {output_path}")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Bucket Validator for USDCOP Trading System"
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
        help='Environment to validate (default: production)'
    )
    parser.add_argument(
        '--action',
        default='validate',
        choices=['validate', 'monitor'],
        help='Action to perform (default: validate)'
    )
    parser.add_argument(
        '--output',
        help='Output file for results'
    )
    parser.add_argument(
        '--format',
        default='json',
        choices=['json', 'yaml', 'html'],
        help='Output format (default: json)'
    )
    parser.add_argument(
        '--monitor-duration',
        type=int,
        default=5,
        help='Monitoring duration in minutes (default: 5)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        validator = BucketValidator(args.config, args.environment)
        
        if args.action == 'validate':
            report = validator.validate_all_buckets()
            
            if args.output:
                validator.export_validation_report(report, args.output, args.format)
            else:
                # Print summary to console
                print(f"\nValidation Summary:")
                print(f"  Environment: {report.environment}")
                print(f"  Total Buckets: {report.total_buckets}")
                print(f"  Healthy: {report.healthy_buckets}")
                print(f"  Degraded: {report.degraded_buckets}")
                print(f"  Unhealthy: {report.unhealthy_buckets}")
                print(f"  Overall Health Score: {report.overall_health_score:.2f}")
                
                if report.recommendations:
                    print(f"\nRecommendations:")
                    for rec in report.recommendations:
                        print(f"  - {rec}")
            
            # Return appropriate exit code
            exit_code = 0 if report.unhealthy_buckets == 0 else 1
            
        elif args.action == 'monitor':
            results = validator.monitor_bucket_health(args.monitor_duration)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            else:
                print(json.dumps(results, indent=2, default=str))
            
            exit_code = 0
        
        return exit_code
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())