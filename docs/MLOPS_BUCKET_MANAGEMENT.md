# MLOps Bucket Management System for USDCOP Trading
=================================================

## Overview

This document describes the comprehensive MLOps bucket management system designed for the USDCOP trading pipeline. The system provides Infrastructure-as-Code (IaC) for automated bucket provisioning, validation, and management across different environments.

## Architecture

### Components

1. **YAML Configuration Schema** (`config/minio-buckets.yaml`)
   - Declarative bucket definitions
   - Environment-specific overrides
   - Lifecycle and access policies
   - Monitoring and alerting rules

2. **Bucket Provisioner** (`scripts/mlops/bucket_provisioner.py`)
   - Automated bucket creation and configuration
   - Policy and lifecycle management
   - Rollback and cleanup capabilities

3. **Validation System** (`scripts/mlops/bucket_validator.py`)
   - Comprehensive health checks
   - Monitoring and alerting
   - Trend analysis and reporting

4. **Init Container** (`docker/Dockerfile.bucket-init`)
   - Pre-startup bucket provisioning
   - Dependency management for services
   - Health check integration

5. **Docker Compose Integration** (`docker-compose.mlops.yml`)
   - Orchestrated infrastructure deployment
   - Service dependencies and health checks
   - Environment configuration management

## Bucket Organization

### Data Layer Structure

```
L0: Raw Data Acquisition
├── 00-raw-usdcop-marketdata/
│   ├── source=mt5/market=usdcop/timeframe=m5/date=20250911/
│   ├── source=twelvedata/market=usdcop/timeframe=m5/date=20250911/
│   └── _control/date=20250911/run_id=20250911123456-abc123/

L1: Standardization
├── 01-l1-ds-usdcop-standardize/
│   ├── market=usdcop/timeframe=m5/date=20250911/session=premium/
│   ├── _schemas/version=v1/
│   └── _control/date=20250911/run_id=20250911123456-def456/

L2: Data Preparation
├── 02-l2-ds-usdcop-prepare/
│   ├── process=clean/market=usdcop/timeframe=m5/date=20250911/
│   ├── process=premium_filter/market=usdcop/timeframe=m5/session=premium/
│   └── _quality/process=clean/date=20250911/run_id=20250911123456-ghi789/

L3: Feature Engineering
├── 03-l3-ds-usdcop-feature/
│   ├── market=usdcop/timeframe=m5/feature_set=v1/date=20250911/
│   ├── _specs/feature_set=v1/
│   └── _control/feature_set=v1/date=20250911/run_id=20250911123456-jkl012/

L4: ML-Ready Data
├── 04-l4-ds-usdcop-rlready/
│   ├── market=usdcop/timeframe=m5/version=v20250911/run_id=20250911123456-mno345/
│   ├── scalers/
│   └── _schemas/version=v20250911/

L5: Serving & Predictions
├── 05-l5-ds-usdcop-serving/
│   ├── market=usdcop/timeframe=m5/date=20250911/predictions.parquet
│   ├── exports/market=usdcop/timeframe=m5/date=20250911/predictions.csv
│   └── dashboards/market=usdcop/timeframe=m5/date=20250911/summary.parquet

L6: Backtesting
├── usdcop-l6-backtest/
│   ├── strategy=rl_ppo/period=2024q4/run_id=20250911123456-pqr678/
│   ├── reports/strategy=rl_ppo/period=2024q4/
│   └── artifacts/strategy=rl_ppo/model_version=v1.2.0/

Common/Shared
├── 99-common-trading-models/
│   ├── model_type=rl_ppo/version=v1.2.0/run_id=20250911123456-stu901/
│   └── checkpoints/model_type=rl_ppo/version=v1.2.0/
├── 99-common-trading-reports/
│   ├── type=daily/date=20250911/
│   └── type=weekly/week=202537/
└── 99-common-trading-backups/
    ├── type=daily/date=20250911/
    └── type=emergency/timestamp=20250911123456/
```

## Usage Guide

### 1. Environment Setup

Create environment-specific configuration:

```bash
# Copy base configuration
cp config/minio-buckets.yaml config/minio-buckets-dev.yaml

# Edit for development
vim config/minio-buckets-dev.yaml
```

### 2. Standalone Bucket Provisioning

```bash
# Provision buckets for production
python3 scripts/mlops/bucket_provisioner.py \
  --config config/minio-buckets.yaml \
  --environment production \
  --action provision \
  --verbose

# Validate bucket configuration
python3 scripts/mlops/bucket_provisioner.py \
  --config config/minio-buckets.yaml \
  --environment production \
  --action validate \
  --output validation_report.json

# Cleanup failed buckets
python3 scripts/mlops/bucket_provisioner.py \
  --config config/minio-buckets.yaml \
  --environment production \
  --action cleanup
```

### 3. Docker Compose Deployment

```bash
# Set environment variables
export MINIO_ACCESS_KEY=your_access_key
export MINIO_SECRET_KEY=your_secret_key
export ENVIRONMENT=production

# Deploy with bucket provisioning
docker-compose -f docker-compose.mlops.yml up -d

# Check bucket initialization status
docker logs usdcop-bucket-init

# Validate infrastructure
docker-compose -f docker-compose.mlops.yml run --rm infrastructure-validator
```

### 4. Comprehensive Validation

```bash
# Full validation with HTML report
python3 scripts/mlops/bucket_validator.py \
  --config config/minio-buckets.yaml \
  --environment production \
  --action validate \
  --output validation_report.html \
  --format html

# Continuous monitoring
python3 scripts/mlops/bucket_validator.py \
  --config config/minio-buckets.yaml \
  --environment production \
  --action monitor \
  --monitor-duration 60 \
  --output monitoring_results.json
```

## Configuration Reference

### Environment Variables

```bash
# MinIO Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin123

# Bucket Provisioner
ENVIRONMENT=production
CONFIG_DIR=/app/config
BUCKET_CONFIG_FILE=minio-buckets.yaml
LOG_LEVEL=INFO

# Validation
VALIDATION_INTERVAL=300  # seconds
HEALTH_CHECK_TIMEOUT=30  # seconds
```

### YAML Configuration Structure

```yaml
metadata:
  version: "v1.0.0"
  description: "Bucket configuration"
  environment: "production"

minio_config:
  endpoint: "${MINIO_ENDPOINT:-localhost:9000}"
  access_key: "${MINIO_ACCESS_KEY:-minioadmin}"
  secret_key: "${MINIO_SECRET_KEY:-minioadmin123}"
  secure: false
  region: "us-east-1"

bucket_groups:
  l0_raw_data:
    description: "Raw market data"
    retention_days: 90
    versioning: true
    buckets:
      - name: "00-raw-usdcop-marketdata"
        description: "Raw USDCOP market data"
        tags:
          layer: "L0"
          data_type: "market_data"
        paths:
          mt5: "source=mt5/market=usdcop/timeframe={timeframe}/date={date}/"
          cache: "_cache/source={source}/date={date}/"
```

## Best Practices

### 1. Naming Conventions

- **Bucket Names**: Use kebab-case with numerical prefixes for ordering
- **Path Structure**: Use partition-style paths (key=value format)
- **Control Files**: Use `_control/`, `_metadata/`, `_schemas/` prefixes
- **Run IDs**: Use timestamp + UUID format: `20250911123456-abc123`

### 2. Security

- **Access Policies**: Implement least-privilege access policies
- **Encryption**: Enable server-side encryption for sensitive data
- **Versioning**: Enable versioning for critical data layers
- **Lifecycle**: Configure automatic cleanup for temporary data

### 3. Monitoring

- **Health Checks**: Regular validation of bucket accessibility
- **Metrics**: Monitor storage usage, access patterns, and performance
- **Alerting**: Set up alerts for failures, capacity, and compliance issues
- **Logging**: Comprehensive logging of all bucket operations

### 4. Disaster Recovery

- **Backups**: Regular backups of critical data
- **Replication**: Cross-region replication for production
- **Recovery**: Automated recovery procedures and testing
- **Documentation**: Clear recovery runbooks and procedures

## Troubleshooting

### Common Issues

1. **Bucket Creation Failures**
   ```bash
   # Check MinIO connectivity
   curl -f http://localhost:9000/minio/health/live
   
   # Verify credentials
   aws --endpoint-url http://localhost:9000 s3 ls
   
   # Check bucket provisioner logs
   docker logs usdcop-bucket-init
   ```

2. **Permission Errors**
   ```bash
   # Check bucket policies
   aws --endpoint-url http://localhost:9000 s3api get-bucket-policy --bucket bucket-name
   
   # Verify IAM configuration
   aws --endpoint-url http://localhost:9000 iam list-users
   ```

3. **Validation Failures**
   ```bash
   # Run detailed validation
   python3 scripts/mlops/bucket_validator.py \
     --config config/minio-buckets.yaml \
     --environment production \
     --action validate \
     --verbose
   
   # Check infrastructure status
   python3 -c "
   from scripts.mlops.bucket_validator import BucketValidator
   validator = BucketValidator('config/minio-buckets.yaml')
   validator._setup_clients()
   print('MinIO connectivity:', validator.minio_client.list_buckets())
   "
   ```

### Performance Optimization

1. **Parallel Operations**: Use concurrent bucket operations for faster provisioning
2. **Caching**: Implement caching for frequently accessed configuration
3. **Batching**: Batch multiple operations to reduce API calls
4. **Connection Pooling**: Use connection pooling for better performance

## Integration Examples

### Airflow DAG Integration

```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from scripts.mlops.bucket_validator import BucketValidator

def validate_buckets(**context):
    validator = BucketValidator('config/minio-buckets.yaml')
    report = validator.validate_all_buckets()
    
    if report.unhealthy_buckets > 0:
        raise Exception(f"Unhealthy buckets detected: {report.unhealthy_buckets}")
    
    return report.overall_health_score

dag = DAG('bucket_health_check', schedule_interval='@hourly')

validate_task = PythonOperator(
    task_id='validate_buckets',
    python_callable=validate_buckets,
    dag=dag
)
```

### Kubernetes Integration

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: bucket-provisioner
spec:
  template:
    spec:
      initContainers:
      - name: bucket-init
        image: usdcop/bucket-provisioner:latest
        env:
        - name: MINIO_ENDPOINT
          value: "minio-service:9000"
        - name: ENVIRONMENT
          value: "production"
        volumeMounts:
        - name: config
          mountPath: /app/config
      restartPolicy: Never
      volumes:
      - name: config
        configMap:
          name: bucket-config
```

## API Reference

### Bucket Provisioner API

```python
from scripts.mlops.bucket_provisioner import MLOpsBucketProvisioner

# Initialize provisioner
provisioner = MLOpsBucketProvisioner(
    config_path='config/minio-buckets.yaml',
    environment='production'
)

# Provision all buckets
results = provisioner.provision_all_buckets()

# Validate configuration
validation = provisioner.validate_buckets()

# Cleanup failed buckets
cleanup = provisioner.cleanup_failed_buckets()
```

### Bucket Validator API

```python
from scripts.mlops.bucket_validator import BucketValidator

# Initialize validator
validator = BucketValidator(
    config_path='config/minio-buckets.yaml',
    environment='production'
)

# Comprehensive validation
report = validator.validate_all_buckets()

# Continuous monitoring
monitoring = validator.monitor_bucket_health(duration_minutes=30)

# Export reports
validator.export_validation_report(report, 'report.html', 'html')
```

## Maintenance

### Regular Tasks

1. **Weekly**: Run comprehensive validation and review reports
2. **Monthly**: Review storage usage and optimize lifecycle policies
3. **Quarterly**: Update bucket configurations and security policies
4. **Annually**: Review and update disaster recovery procedures

### Automation

- Set up automated validation in CI/CD pipelines
- Implement automated cleanup of old data
- Configure automated alerting for critical issues
- Schedule regular backups and recovery testing

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review Docker and MinIO logs
3. Run validation with verbose output
4. Contact the MLOps team with detailed error information

## Changelog

- **v1.0.0** (2025-09-11): Initial release with comprehensive bucket management
- **v1.1.0** (Planned): Kubernetes integration and advanced monitoring
- **v1.2.0** (Planned): Multi-region support and enhanced security