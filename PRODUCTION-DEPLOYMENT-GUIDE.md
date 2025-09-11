# USDCOP Trading System - Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the USDCOP Reinforcement Learning Trading System in a production environment using Docker Compose. The system includes all necessary services for a complete trading infrastructure with monitoring, observability, and data persistence.

## Architecture Overview

### Service Stack

**Infrastructure Layer:**
- **Consul**: Service discovery and configuration management
- **PostgreSQL**: Primary database with initialized schemas
- **Redis**: Caching and message broker
- **MinIO**: S3-compatible object storage
- **Nginx**: Reverse proxy and load balancer

**Processing Layer:**
- **Airflow**: Workflow orchestration (webserver, scheduler, worker)
- **Trading App**: Core trading application with RL models
- **Dashboard**: Next.js frontend for monitoring and control

**Observability Layer:**
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Jaeger**: Distributed tracing
- **Loki + Promtail**: Log aggregation and analysis

**Administration:**
- **PgAdmin**: Database administration interface

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended) or macOS
- **RAM**: Minimum 16GB, Recommended 32GB
- **CPU**: Minimum 8 cores, Recommended 16 cores
- **Storage**: Minimum 100GB SSD, Recommended 500GB SSD
- **Network**: Stable internet connection for market data

### Software Dependencies

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installations
docker --version
docker-compose --version
```

### Required Tools

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y curl jq git

# CentOS/RHEL
sudo yum install -y curl jq git

# macOS
brew install curl jq git
```

## Quick Start Deployment

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd USDCOP-RL-Models

# Copy production environment template
cp .env.production.example .env.production

# Edit environment variables (see Configuration section)
nano .env.production
```

### 2. Configure Environment

Edit `.env.production` and update the following critical variables:

```bash
# Security - Generate strong passwords
POSTGRES_PASSWORD=your_secure_postgres_password
REDIS_PASSWORD=your_secure_redis_password
MINIO_ROOT_PASSWORD=your_secure_minio_password
GRAFANA_ADMIN_PASSWORD=your_secure_grafana_password

# Trading Configuration
MT5_LOGIN=your_mt5_login
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_mt5_server

# API Keys
TWELVE_API=your_twelve_data_api_key
```

### 3. Deploy the System

```bash
# Make deployment script executable
chmod +x deploy-production.sh

# Validate environment
./deploy-production.sh validate

# Deploy all services
./deploy-production.sh start
```

### 4. Verify Deployment

```bash
# Check service status
./deploy-production.sh status

# View service logs
./deploy-production.sh logs

# Access health checks
curl http://localhost:3000  # Dashboard
curl http://localhost:9090/-/healthy  # Prometheus
```

## Detailed Configuration

### Environment Variables

#### Core System

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Deployment environment | `production` | Yes |
| `TZ` | Timezone | `UTC` | No |
| `DATA_DIR` | Data persistence directory | `./data` | No |
| `LOG_DIR` | Log directory | `./logs` | No |
| `BACKUP_DIR` | Backup directory | `./backups` | No |

#### Database Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `POSTGRES_PASSWORD` | PostgreSQL root password | - | Yes |
| `DB_PASSWORD` | Trading database password | - | Yes |
| `AIRFLOW_DB_PASSWORD` | Airflow database password | - | Yes |

#### Security Keys

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `AIRFLOW_FERNET_KEY` | Airflow encryption key | - | Yes |
| `AIRFLOW_SECRET_KEY` | Airflow web secret | - | Yes |
| `REDIS_PASSWORD` | Redis authentication | - | Yes |

#### Service Ports

| Service | Port | Variable | Description |
|---------|------|----------|-------------|
| Dashboard | 3000 | `DASHBOARD_PORT` | Trading dashboard |
| API | 8000 | `APP_PORT` | Trading API |
| Airflow | 8081 | `AIRFLOW_PORT` | Workflow management |
| Grafana | 3001 | `GRAFANA_PORT` | Monitoring dashboards |
| Prometheus | 9090 | `PROMETHEUS_PORT` | Metrics collection |
| PostgreSQL | 5432 | `POSTGRES_PORT` | Database |
| Redis | 6379 | `REDIS_PORT` | Cache/message broker |
| MinIO API | 9000 | `MINIO_API_PORT` | Object storage API |
| MinIO Console | 9001 | `MINIO_CONSOLE_PORT` | Storage admin UI |
| PgAdmin | 5050 | `PGADMIN_PORT` | Database admin |

### Volume Management

The system uses persistent volumes for data retention:

```yaml
# Data Volumes
./data/postgres     # Database data
./data/redis        # Redis persistence
./data/minio        # Object storage
./data/prometheus   # Metrics data
./data/grafana      # Dashboard configs
./data/app          # Application data
./data/models       # ML model cache

# Log Volumes
./logs/app          # Application logs
./logs/nginx        # Proxy logs

# Backup Volumes
./backups/postgres  # Database backups
```

### Network Configuration

The system uses a dedicated Docker network (`trading-network`) with:

- **Subnet**: `172.28.0.0/16` (configurable)
- **Gateway**: `172.28.0.1`
- **DNS**: Automatic service discovery via container names

## Service Management

### Using the Deployment Script

```bash
# Start all services
./deploy-production.sh start

# Stop all services
./deploy-production.sh stop

# Restart all services
./deploy-production.sh restart

# Check status
./deploy-production.sh status

# View logs (all services)
./deploy-production.sh logs

# View logs (specific service)
./deploy-production.sh logs trading-app

# Create backup
./deploy-production.sh backup

# Clean up system
./deploy-production.sh cleanup

# Validate configuration
./deploy-production.sh validate
```

### Manual Docker Compose Commands

```bash
# Start services
docker-compose -f docker-compose.production.yml --env-file .env.production up -d

# View logs
docker-compose -f docker-compose.production.yml logs -f [service-name]

# Stop services
docker-compose -f docker-compose.production.yml down

# Scale services
docker-compose -f docker-compose.production.yml up -d --scale trading-app=3

# Update services
docker-compose -f docker-compose.production.yml pull
docker-compose -f docker-compose.production.yml up -d
```

## Service Access URLs

After successful deployment, access services at:

| Service | URL | Credentials |
|---------|-----|-------------|
| **Trading Dashboard** | http://localhost:3000 | None |
| **Grafana Monitoring** | http://localhost:3001 | admin / [GRAFANA_ADMIN_PASSWORD] |
| **Airflow Workflows** | http://localhost:8081 | [AIRFLOW_ADMIN_USER] / [AIRFLOW_ADMIN_PASSWORD] |
| **PgAdmin Database** | http://localhost:5050 | [PGADMIN_EMAIL] / [PGADMIN_PASSWORD] |
| **Prometheus Metrics** | http://localhost:9090 | None |
| **MinIO Console** | http://localhost:9001 | [MINIO_ROOT_USER] / [MINIO_ROOT_PASSWORD] |
| **Jaeger Tracing** | http://localhost:16686 | None |
| **Consul Service Discovery** | http://localhost:8500 | None |

## Monitoring and Observability

### Health Checks

All services include built-in health checks:

```bash
# Check individual service health
curl http://localhost:3000/api/health        # Dashboard
curl http://localhost:8000/health            # Trading App
curl http://localhost:9090/-/healthy         # Prometheus
curl http://localhost:3001/api/health        # Grafana
```

### Metrics and Alerting

**Prometheus Targets**:
- Trading Application: `trading-app:8000/metrics`
- Dashboard: `trading-dashboard:3000/metrics`
- System metrics: Various exporters

**Key Metrics**:
- Application performance (response time, throughput)
- Trading metrics (PnL, positions, signals)
- System metrics (CPU, memory, disk)
- Business metrics (trades, returns, risk)

### Log Aggregation

**Loki Configuration**:
- Collects logs from all containers
- Structured logging with labels
- Integration with Grafana for visualization

**Log Sources**:
- Application logs: `/var/log/app/`
- Nginx logs: `/var/log/nginx/`
- Container logs: Docker container stdout/stderr

## Data Management

### Database Schema

The system automatically initializes PostgreSQL with:

**Databases**:
- `trading_db`: Main trading data
- `airflow`: Workflow metadata
- `mlflow`: ML experiment tracking

**Schemas** (in trading_db):
- `bronze`: Raw data ingestion
- `silver`: Cleaned/standardized data
- `gold`: Analytics-ready data
- `models`: ML model registry
- `monitoring`: System metrics

### Backup and Recovery

#### Automated Backups

```bash
# Create system backup
./deploy-production.sh backup

# Backups include:
# - PostgreSQL database dump
# - Configuration files
# - Application data
# - Backup metadata
```

#### Manual Backup Procedures

```bash
# Database backup
docker-compose -f docker-compose.production.yml exec postgres \
  pg_dumpall -U postgres > backup_$(date +%Y%m%d_%H%M%S).sql

# Volume backup
docker run --rm -v postgres_data:/data -v $(pwd):/backup alpine \
  tar czf /backup/postgres_$(date +%Y%m%d).tar.gz -C /data .

# Configuration backup
tar czf config_backup_$(date +%Y%m%d).tar.gz \
  .env.production docker-compose.production.yml config/
```

#### Recovery Procedures

```bash
# Stop services
./deploy-production.sh stop

# Restore database
docker-compose -f docker-compose.production.yml up -d postgres
cat backup_file.sql | docker-compose -f docker-compose.production.yml exec -T postgres \
  psql -U postgres

# Restore volumes
docker run --rm -v postgres_data:/data -v $(pwd):/backup alpine \
  tar xzf /backup/postgres_backup.tar.gz -C /data

# Restart services
./deploy-production.sh start
```

## Security Considerations

### Network Security

- All services run in isolated Docker network
- External access only through defined ports
- Internal service communication via container names
- Rate limiting on API endpoints

### Authentication and Authorization

- Strong passwords for all services
- Service-to-service authentication via environment variables
- Admin interfaces protected with authentication
- API rate limiting and access controls

### Data Security

- Database encryption at rest (configurable)
- Secure credential management via environment variables
- Regular security updates via container updates
- Backup encryption (recommended)

### Best Practices

1. **Change Default Passwords**: Always use strong, unique passwords
2. **Regular Updates**: Keep container images updated
3. **Monitor Access**: Review access logs regularly
4. **Backup Strategy**: Implement regular backup schedule
5. **Network Isolation**: Use firewalls to restrict external access
6. **SSL/TLS**: Configure HTTPS for production (see SSL section)

## Performance Optimization

### Resource Allocation

**Minimum Production Resources**:
```yaml
# CPU Allocation (cores)
postgres: 2
redis: 1
trading-app: 4
dashboard: 2
airflow-scheduler: 2
airflow-worker: 4
prometheus: 2
grafana: 1

# Memory Allocation (GB)
postgres: 4
redis: 2
trading-app: 8
dashboard: 2
airflow-scheduler: 2
airflow-worker: 8
prometheus: 4
grafana: 2
```

### Database Optimization

PostgreSQL configuration optimizations:

```sql
-- Connection settings
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB

-- Performance settings
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
random_page_cost = 1.1
effective_io_concurrency = 200

-- Memory settings
work_mem = 4MB
min_wal_size = 1GB
max_wal_size = 4GB
```

### Application Optimization

1. **Connection Pooling**: Configured for all database connections
2. **Caching**: Redis caching for frequently accessed data
3. **Async Processing**: Celery workers for background tasks
4. **Load Balancing**: Nginx upstream configuration
5. **Resource Limits**: Docker resource constraints

## Troubleshooting

### Common Issues

#### Service Won't Start

```bash
# Check service logs
./deploy-production.sh logs [service-name]

# Check container status
docker-compose -f docker-compose.production.yml ps

# Check resource usage
docker stats

# Validate environment
./deploy-production.sh validate
```

#### Database Connection Issues

```bash
# Check PostgreSQL logs
./deploy-production.sh logs postgres

# Test database connectivity
docker-compose -f docker-compose.production.yml exec postgres \
  psql -U postgres -c "SELECT version();"

# Check database initialization
docker-compose -f docker-compose.production.yml logs postgres-init
```

#### Performance Issues

```bash
# Monitor resource usage
docker stats

# Check application metrics
curl http://localhost:9090/metrics

# Review slow queries
docker-compose -f docker-compose.production.yml exec postgres \
  psql -U postgres -d trading_db -c "SELECT * FROM pg_stat_activity;"
```

### Debugging Commands

```bash
# Enter container for debugging
docker-compose -f docker-compose.production.yml exec trading-app bash

# Check environment variables
docker-compose -f docker-compose.production.yml exec trading-app env

# Test service connectivity
docker-compose -f docker-compose.production.yml exec trading-app \
  curl http://postgres:5432

# Check disk usage
docker system df
```

### Log Analysis

```bash
# View real-time logs
./deploy-production.sh logs

# Search logs for errors
docker-compose -f docker-compose.production.yml logs | grep ERROR

# Export logs for analysis
docker-compose -f docker-compose.production.yml logs > system_logs.txt
```

## Maintenance

### Regular Tasks

#### Daily
- Monitor system health via Grafana dashboards
- Check error logs for anomalies
- Verify trading performance metrics
- Monitor disk space usage

#### Weekly
- Create system backups
- Review and rotate logs
- Update container images
- Performance analysis

#### Monthly
- Database maintenance (VACUUM, ANALYZE)
- Security updates
- Capacity planning review
- Disaster recovery testing

### Update Procedures

```bash
# Update container images
docker-compose -f docker-compose.production.yml pull

# Rolling update (minimal downtime)
docker-compose -f docker-compose.production.yml up -d --no-deps trading-app

# Full system update
./deploy-production.sh stop
docker-compose -f docker-compose.production.yml pull
./deploy-production.sh start
```

## Advanced Configuration

### SSL/HTTPS Setup

1. **Obtain SSL certificates**:
```bash
# Using Let's Encrypt
certbot certonly --standalone -d your-domain.com
```

2. **Configure Nginx**:
```bash
# Uncomment HTTPS server block in nginx/nginx.conf
# Update certificate paths
# Restart nginx service
```

3. **Update environment**:
```bash
# Set ENABLE_SSL=true in .env.production
# Configure SSL_CERT_PATH and SSL_KEY_PATH
```

### Scaling Configuration

**Horizontal Scaling**:
```bash
# Scale trading workers
docker-compose -f docker-compose.production.yml up -d --scale airflow-worker=4

# Scale application instances
docker-compose -f docker-compose.production.yml up -d --scale trading-app=3
```

**Load Balancer Configuration**:
```nginx
# Update nginx upstream configuration
upstream trading_app {
    server trading-app-1:8000;
    server trading-app-2:8000;
    server trading-app-3:8000;
}
```

### Custom Dashboards

Create custom Grafana dashboards:

1. Access Grafana at http://localhost:3001
2. Import dashboard templates from `config/grafana/dashboards/`
3. Configure data sources
4. Create custom panels for specific metrics

## Support and Documentation

### Getting Help

1. **Check logs**: Use deployment script log commands
2. **Review documentation**: See inline comments in configuration files
3. **Community**: Submit issues to project repository
4. **Professional support**: Contact development team

### Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Airflow Documentation](https://airflow.apache.org/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)

---

## Appendix

### Environment Template

```bash
# Copy this template to .env.production and customize
cp .env.production.example .env.production
```

### Port Reference

| Port | Service | Description |
|------|---------|-------------|
| 3000 | Dashboard | Trading interface |
| 3001 | Grafana | Monitoring dashboards |
| 5050 | PgAdmin | Database admin |
| 5432 | PostgreSQL | Database |
| 6379 | Redis | Cache/broker |
| 8000 | Trading App | API server |
| 8081 | Airflow | Workflow UI |
| 8500 | Consul | Service discovery |
| 9000 | MinIO API | Object storage |
| 9001 | MinIO Console | Storage admin |
| 9090 | Prometheus | Metrics |
| 16686 | Jaeger | Tracing UI |

This completes the comprehensive production deployment guide for the USDCOP Trading System.