# Fast Airflow Setup for USDCOP Trading System

This guide helps you get Airflow UI running quickly at `http://localhost:8080` for L0 pipeline execution.

## Quick Start (Recommended)

### Option 1: Automated Fast Setup

```bash
# Run the automated setup script
./start-airflow-fast.sh
```

This script will:
- Check prerequisites (Docker, docker-compose)
- Clean up any existing containers
- Start minimal Airflow services with fast configuration
- Wait for UI to be ready
- Provide access information

### Option 2: Manual Fast Setup

```bash
# Stop any existing containers
docker-compose down

# Start with fast configuration
docker-compose -f docker-compose.yml -f docker-compose.fast.yml up -d

# Wait for services to start (2-3 minutes)
# Check status
docker-compose -f docker-compose.yml -f docker-compose.fast.yml ps
```

## Access Information

- **Airflow UI**: http://localhost:8080
- **Username**: `admin`
- **Password**: `admin123`

## What's Different in Fast Mode?

### Optimizations Applied:

1. **Minimal Dependencies**: Uses `requirements.minimal.txt` instead of full ML stack
2. **LocalExecutor**: Uses LocalExecutor instead of CeleryExecutor (no worker needed)
3. **Reduced Build Time**: Lightweight Dockerfile with only essential packages
4. **Faster Health Checks**: Shorter intervals and timeouts
5. **Disabled Heavy Services**: Disables monitoring, dashboard, and other non-essential services
6. **Dependency Fallbacks**: Graceful handling of missing ML dependencies

### Services Enabled in Fast Mode:
- PostgreSQL (database)
- Redis (caching)
- MinIO (object storage)
- Airflow Scheduler
- Airflow Webserver
- Airflow Init

### Services Disabled in Fast Mode:
- Airflow Worker (using LocalExecutor instead)
- Real-time orchestrator
- WebSocket service
- Dashboard
- Prometheus/Grafana monitoring
- Nginx reverse proxy

## Installing Full ML Dependencies Later

If you need full ML functionality after fast startup:

```bash
# Option 1: Install in running container
docker-compose exec airflow-webserver pip install -r /opt/airflow/requirements.txt

# Option 2: Use full configuration
docker-compose down
docker-compose up -d  # Uses Dockerfile.prod with full dependencies
```

## Troubleshooting

### Port 8080 Already in Use

```bash
# Check what's using port 8080
lsof -i :8080

# Stop existing containers
docker-compose down

# Or kill specific process
sudo kill -9 <PID>
```

### Slow Startup

```bash
# Check container logs
docker-compose -f docker-compose.yml -f docker-compose.fast.yml logs airflow-webserver
docker-compose -f docker-compose.yml -f docker-compose.fast.yml logs airflow-scheduler

# Check container status
docker-compose -f docker-compose.yml -f docker-compose.fast.yml ps
```

### Database Connection Issues

```bash
# Restart database
docker-compose -f docker-compose.yml -f docker-compose.fast.yml restart postgres

# Check database logs
docker-compose -f docker-compose.yml -f docker-compose.fast.yml logs postgres
```

### DAG Import Errors (Missing Dependencies)

DAGs are designed to handle missing ML dependencies gracefully:
- Missing dependencies are logged as warnings
- DAGs using unavailable dependencies show helpful error messages
- Core L0 pipeline should work with minimal dependencies

## Expected Startup Time

- **Fast Mode**: 1-3 minutes
- **Full Mode**: 10-15 minutes (due to ML dependency compilation)

## Configuration Files

### Fast Mode Files:
- `docker-compose.fast.yml` - Fast override configuration
- `airflow/Dockerfile.minimal` - Lightweight Airflow image
- `airflow/requirements.minimal.txt` - Essential dependencies only
- `airflow/entrypoint.minimal.sh` - Fast initialization script
- `start-airflow-fast.sh` - Automated setup script

### Full Mode Files:
- `docker-compose.yml` - Full configuration
- `airflow/Dockerfile.prod` - Complete Airflow image with ML stack
- `airflow/requirements.txt` - All dependencies including ML libraries
- `airflow/entrypoint.sh` - Full initialization script

## Next Steps

Once Airflow is running:

1. **Access the UI**: http://localhost:8080
2. **Enable L0 DAG**: Find `usdcop_m5__01_l0_acquire` and enable it
3. **Configure API Keys**: Set TwelveData API keys in environment or DAG configuration
4. **Trigger L0 Pipeline**: Manually trigger or wait for scheduled run

## Stopping Services

```bash
# Stop fast mode services
docker-compose -f docker-compose.yml -f docker-compose.fast.yml down

# Stop all services and remove volumes (clean slate)
docker-compose down -v
```

## Performance Notes

- Fast mode uses ~2GB RAM vs ~8GB for full mode
- L0 pipeline should work fully in fast mode
- L4/L5 pipelines require full ML dependencies
- You can switch between modes as needed