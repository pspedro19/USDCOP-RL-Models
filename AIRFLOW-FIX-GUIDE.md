# Airflow DAG Import Errors - Fix Guide

## Overview
This guide addresses the DAG import errors encountered in the USDCOP RL Trading System Airflow setup, specifically the missing dependencies (`gymnasium`, `scipy`) that were causing DAG failures.

## Problem Description

### Original Errors
```
Broken DAG: [/opt/airflow/dags/usdcop_m5__06_l5_serving.py]
ModuleNotFoundError: No module named 'gymnasium'

Broken DAG: [/opt/airflow/dags/usdcop_m5__05_l4_rlready.py]
ModuleNotFoundError: No module named 'scipy'

Broken DAG: [/opt/airflow/dags/usdcop_m5__04_l3_feature.py]
ModuleNotFoundError: No module named 'scipy'
```

### Root Cause
The Airflow container was missing essential machine learning and scientific computing dependencies required by the USDCOP RL trading DAGs.

## Solution Implemented

### 1. Enhanced Airflow Dockerfile (`airflow/Dockerfile.prod`)

Created a production-ready Dockerfile that:
- **Installs system dependencies** for ML libraries (gcc, gfortran, BLAS, LAPACK)
- **Upgrades pip and setuptools** for better package management
- **Installs all required Python packages** with proper error handling
- **Verifies critical imports** during build process
- **Sets optimal environment variables** for production

**Key Features:**
```dockerfile
# System dependencies for ML libraries
RUN apt-get install -y gcc g++ gfortran libffi-dev libssl-dev libpq-dev libblas-dev liblapack-dev

# Install Python dependencies with retries and timeouts
RUN pip install --timeout 1000 --retries 5 -r /requirements.txt

# Verify critical imports work
RUN python -c "import scipy; import gymnasium; import torch; print('All dependencies imported successfully')"
```

### 2. Comprehensive Requirements File (`airflow/requirements.txt`)

Updated with all necessary dependencies and compatible versions:

**Core ML Libraries:**
- `scipy>=1.11.0` - Scientific computing (fixes L3/L4 DAG errors)
- `gymnasium>=0.29.0` - RL environment framework (fixes L5 DAG errors)
- `torch>=2.1.0` - Deep learning framework
- `stable-baselines3>=2.2.0` - RL algorithms
- `pandas>=2.0.0`, `numpy>=1.24.0` - Data manipulation

**Trading-Specific Libraries:**
- `ta>=0.10.0` - Technical analysis
- `yfinance>=0.2.0` - Financial data
- `pandas-ta>=0.3.0` - Technical indicators

**Infrastructure Libraries:**
- `minio>=7.2.0` - Object storage client
- `mlflow>=2.8.0` - ML model management
- `psycopg2-binary>=2.9.0` - PostgreSQL adapter
- `redis>=5.0.0` - Redis client

### 3. Airflow Initialization Script (`airflow/entrypoint.sh`)

Created an intelligent entrypoint that:
- **Verifies all dependencies** before starting services
- **Waits for database** connection to be ready
- **Initializes Airflow database** automatically
- **Creates admin user** if it doesn't exist
- **Provides detailed logging** for troubleshooting

**Key Functions:**
```bash
verify_dependencies() {
    REQUIRED_MODULES=("scipy" "gymnasium" "torch" "pandas" "numpy" "sklearn")
    for module in "${REQUIRED_MODULES[@]}"; do
        if python -c "import ${module}"; then
            echo "✓ ${module} - OK"
        else
            echo "✗ ${module} - FAILED"
            exit 1
        fi
    done
}
```

### 4. DAG Import Testing (`airflow/test_dag_imports.py`)

Created a comprehensive testing script that:
- **Tests all DAG files** for import errors
- **Verifies critical dependencies** availability
- **Provides detailed error reporting** with stack traces
- **Integrates with deployment process** for validation

### 5. Enhanced Deployment Process

Updated the deployment script (`deploy.sh`) to:
- **Test DAG imports** before deployment
- **Build Airflow images** with dependency verification
- **Initialize services** in proper order
- **Provide clear error messages** for troubleshooting

## How to Deploy the Fix

### 1. Rebuild Airflow Containers
```bash
# Clean existing containers and images
docker-compose -f docker-compose.prod.yml down
docker rmi $(docker images | grep airflow | awk '{print $3}')

# Deploy with updated configuration
./deploy.sh deploy
```

### 2. Verify the Fix
```bash
# Test DAG imports manually
docker-compose -f docker-compose.prod.yml run --rm airflow-webserver python /opt/airflow/test_dag_imports.py

# Check Airflow web UI
# Navigate to http://localhost:8080
# Verify no DAG import errors are shown
```

### 3. Monitor Airflow Logs
```bash
# Check Airflow webserver logs
docker-compose -f docker-compose.prod.yml logs -f airflow-webserver

# Check Airflow scheduler logs
docker-compose -f docker-compose.prod.yml logs -f airflow-scheduler
```

## Verification Steps

### 1. DAG Import Status
After deployment, verify in Airflow Web UI:
- Navigate to **DAGs** page
- Ensure no red error icons appear
- All DAGs should show as importable

### 2. Dependency Verification
```bash
# Test critical imports in container
docker-compose -f docker-compose.prod.yml exec airflow-webserver python -c "
import scipy
import gymnasium
import torch
import pandas
import numpy
print('✅ All critical dependencies imported successfully')
"
```

### 3. DAG Execution Test
```bash
# Enable and trigger a test DAG
docker-compose -f docker-compose.prod.yml exec airflow-webserver airflow dags unpause usdcop_m5__04_l3_feature
docker-compose -f docker-compose.prod.yml exec airflow-webserver airflow dags trigger usdcop_m5__04_l3_feature
```

## Troubleshooting

### If DAG Import Errors Persist

1. **Check Container Logs:**
```bash
docker-compose -f docker-compose.prod.yml logs airflow-webserver | grep -i error
```

2. **Verify Requirements Installation:**
```bash
docker-compose -f docker-compose.prod.yml exec airflow-webserver pip list | grep -E "(scipy|gymnasium|torch)"
```

3. **Manual Dependency Installation:**
```bash
# Enter container and install manually
docker-compose -f docker-compose.prod.yml exec airflow-webserver bash
pip install scipy gymnasium torch stable-baselines3
```

4. **Check Python Path:**
```bash
docker-compose -f docker-compose.prod.yml exec airflow-webserver python -c "
import sys
print('Python path:')
for path in sys.path:
    print(f'  {path}')
"
```

### Common Issues and Solutions

#### Issue: Build Timeout
```bash
# Increase Docker build timeout
export DOCKER_CLIENT_TIMEOUT=300
export COMPOSE_HTTP_TIMEOUT=300
```

#### Issue: Memory Errors During Build
```bash
# Increase Docker memory limit (Docker Desktop)
# Or use smaller batch installations
```

#### Issue: Network Timeouts
```bash
# Use Docker build with no cache and specify registry
docker-compose -f docker-compose.prod.yml build --no-cache --pull airflow-webserver
```

## Prevention Measures

### 1. Regular Dependency Updates
- Update `requirements.txt` monthly
- Test dependency compatibility before deployment
- Pin critical package versions

### 2. Automated Testing
- Include DAG import tests in CI/CD pipeline
- Run dependency verification before deployment
- Monitor Airflow logs for import errors

### 3. Container Health Monitoring
- Implement health checks for Airflow services
- Monitor resource usage and performance
- Set up alerts for container failures

## Additional Resources

### Docker Commands Reference
```bash
# Rebuild specific service
docker-compose -f docker-compose.prod.yml build --no-cache airflow-webserver

# Check service status
docker-compose -f docker-compose.prod.yml ps

# View container resource usage
docker stats $(docker-compose -f docker-compose.prod.yml ps -q)

# Access container shell
docker-compose -f docker-compose.prod.yml exec airflow-webserver bash
```

### Airflow Commands Reference
```bash
# List DAGs
airflow dags list

# Test DAG
airflow dags test [dag_id] [execution_date]

# Check connections
airflow connections list

# View task instances
airflow tasks list [dag_id]
```

## Conclusion

The DAG import errors have been resolved through:
1. **Comprehensive dependency management** in the Airflow container
2. **Robust initialization scripts** with verification
3. **Automated testing** integration in deployment
4. **Clear documentation** for maintenance and troubleshooting

The USDCOP RL Trading System Airflow setup is now production-ready with all required ML dependencies properly installed and verified.