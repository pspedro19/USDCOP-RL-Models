# USDCOP RL Trading System - Complete Setup Guide

## Overview
This guide provides comprehensive instructions for setting up and accessing the USDCOP Reinforcement Learning Trading System. The system consists of a modern Next.js 15.5.2 dashboard, backend API, data pipeline, and supporting infrastructure.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Development Setup](#development-setup)
- [Production Deployment](#production-deployment)
- [Accessing Services](#accessing-services)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)
- [Security](#security)

---

## Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended), macOS 10.15+, or Windows 10+ with WSL2
- **Memory**: Minimum 16GB RAM (32GB recommended for production)
- **CPU**: 4+ cores (8+ cores recommended for production)
- **Storage**: 100GB+ free space
- **Network**: Stable internet connection for data feeds

### Software Dependencies
- **Docker**: Version 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose**: Version 2.0+ ([Install Docker Compose](https://docs.docker.com/compose/install/))
- **Git**: Version 2.20+ ([Install Git](https://git-scm.com/downloads))
- **Node.js**: Version 20+ (for development only) ([Install Node.js](https://nodejs.org/))

### Hardware for Production
- **CPU**: Intel Xeon or AMD EPYC with 16+ cores
- **Memory**: 64GB+ RAM
- **Storage**: NVMe SSD with 500GB+ available space
- **Network**: 1Gbps+ connection with low latency to forex data providers

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/USDCOP-RL-Models.git
cd USDCOP-RL-Models
```

### 2. Configure Environment
```bash
# Copy environment template
cp .env.prod.example .env.prod

# Edit the environment file with your specific values
nano .env.prod
```

**Critical Environment Variables to Set:**
```bash
# Database passwords (use strong passwords)
POSTGRES_PASSWORD=your_secure_postgres_password_here
AIRFLOW_PASSWORD=your_secure_airflow_password_here
INFLUXDB_PASSWORD=your_secure_influxdb_password_here

# MinIO credentials
MINIO_ACCESS_KEY=your_minio_access_key_here
MINIO_SECRET_KEY=your_secure_minio_secret_key_here

# Security keys (generate strong keys)
JWT_SECRET=your_jwt_secret_key_minimum_32_characters_long
AIRFLOW_FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
AIRFLOW_SECRET_KEY=your_airflow_webserver_secret_key_here

# Monitoring
GRAFANA_PASSWORD=your_secure_grafana_password_here
```

### 3. Deploy the System
```bash
# Make deployment script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh deploy
```

### 4. Access the Dashboard
Once deployment is complete, access the trading dashboard at:
- **Local Development**: http://localhost:3000
- **Production**: https://dashboard.usdcop.local

---

## Development Setup

### Frontend Development (Next.js Dashboard)

1. **Navigate to dashboard directory:**
```bash
cd usdcop-trading-dashboard
```

2. **Install dependencies:**
```bash
npm install
```

3. **Start development server:**
```bash
npm run dev
```

4. **Access development dashboard:**
- Open http://localhost:3000 in your browser

### Backend Development

1. **Set up Python environment:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Start backend services:**
```bash
# Start database services first
docker-compose up -d postgres redis minio

# Run backend API
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

3. **Access API documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Data Pipeline Development (Airflow)

1. **Start Airflow services:**
```bash
docker-compose up -d postgres redis
export AIRFLOW__CORE__EXECUTOR=LocalExecutor
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql://airflow:password@localhost:5432/airflow

# Initialize Airflow database
airflow db init

# Create admin user
airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin

# Start Airflow webserver and scheduler
airflow webserver --port 8080 &
airflow scheduler &
```

2. **Access Airflow UI:**
- Open http://localhost:8080 in your browser
- Login with admin/admin

---

## Production Deployment

### Pre-deployment Checklist

- [ ] All environment variables configured in `.env.prod`
- [ ] SSL certificates obtained and placed in `nginx/ssl/`
- [ ] DNS records configured for subdomains
- [ ] Firewall rules configured
- [ ] Backup strategy implemented
- [ ] Monitoring and alerting configured

### Deployment Steps

1. **Prepare the server:**
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

2. **Configure DNS (if using custom domains):**
```bash
# Add to /etc/hosts for local testing
echo "127.0.0.1 dashboard.usdcop.local" | sudo tee -a /etc/hosts
echo "127.0.0.1 api.usdcop.local" | sudo tee -a /etc/hosts
echo "127.0.0.1 airflow.usdcop.local" | sudo tee -a /etc/hosts
echo "127.0.0.1 monitoring.usdcop.local" | sudo tee -a /etc/hosts
```

3. **Deploy using the script:**
```bash
./deploy.sh deploy
```

4. **Verify deployment:**
```bash
./deploy.sh status
```

### SSL Certificate Setup (Production)

For production, replace self-signed certificates with proper SSL certificates:

**Using Let's Encrypt (Recommended):**
```bash
# Install Certbot
sudo apt install certbot

# Generate certificates
sudo certbot certonly --standalone -d dashboard.usdcop.local -d api.usdcop.local

# Copy certificates to nginx directory
sudo cp /etc/letsencrypt/live/dashboard.usdcop.local/fullchain.pem nginx/ssl/usdcop.crt
sudo cp /etc/letsencrypt/live/dashboard.usdcop.local/privkey.pem nginx/ssl/usdcop.key
```

---

## Accessing Services

### Service URLs and Ports

| Service | Development URL | Production URL | Default Port |
|---------|----------------|----------------|--------------|
| Trading Dashboard | http://localhost:3000 | https://dashboard.usdcop.local | 3000 |
| Backend API | http://localhost:8000 | https://api.usdcop.local | 8000 |
| API Documentation | http://localhost:8000/docs | https://api.usdcop.local/docs | 8000 |
| Airflow Web UI | http://localhost:8080 | https://airflow.usdcop.local | 8080 |
| Grafana Dashboard | http://localhost:3001 | https://monitoring.usdcop.local/grafana | 3001 |
| Prometheus | http://localhost:9090 | https://monitoring.usdcop.local/prometheus | 9090 |
| MinIO Console | http://localhost:9001 | http://localhost:9001 | 9001 |
| PostgreSQL | localhost:5432 | localhost:5432 | 5432 |
| Redis | localhost:6379 | localhost:6379 | 6379 |
| InfluxDB | http://localhost:8086 | http://localhost:8086 | 8086 |

### Default Credentials

**Airflow:**
- Username: `admin`
- Password: Set in `AIRFLOW_PASSWORD` environment variable

**Grafana:**
- Username: `admin`
- Password: Set in `GRAFANA_PASSWORD` environment variable

**MinIO:**
- Access Key: Set in `MINIO_ACCESS_KEY` environment variable
- Secret Key: Set in `MINIO_SECRET_KEY` environment variable

**PostgreSQL:**
- Username: `postgres`
- Password: Set in `POSTGRES_PASSWORD` environment variable
- Database: `usdcop_trading`

### Dashboard Features Access

#### 1. Executive Overview
- **Path**: `/` (default landing page)
- **Features**: KPIs, performance charts, production gates
- **Data**: Real-time trading metrics, Sortino/Calmar ratios

#### 2. Live Trading Terminal
- **Path**: `/trading`
- **Features**: Real-time USD/COP data, RL actions, manual override
- **Data**: Price feeds, technical indicators, model decisions

#### 3. RL Model Health
- **Path**: `/models`
- **Features**: Model performance, training metrics, action distribution
- **Data**: PPO-LSTM and QR-DQN health status, convergence data

#### 4. Risk Management
- **Path**: `/risk`
- **Features**: VaR/CVaR analysis, stress testing, exposure monitoring
- **Data**: Risk metrics, scenario analysis, compliance status

#### 5. Data Pipeline Quality
- **Path**: `/pipeline`
- **Features**: L0-L4 quality gates, anti-leakage checks, system resources
- **Data**: Pipeline health, data quality scores, resource utilization

#### 6. Audit & Compliance
- **Path**: `/audit`
- **Features**: Traceability, regulatory compliance, security status
- **Data**: SHA256 hash chains, SFC Colombia/Basel III compliance

---

## Configuration

### Environment Configuration

**Key Configuration Files:**
- `.env.prod` - Production environment variables
- `docker-compose.prod.yml` - Production Docker Compose configuration
- `nginx/nginx.conf` - Nginx reverse proxy configuration
- `redis.conf` - Redis performance configuration
- `prometheus/prometheus.yml` - Monitoring configuration

### Dashboard Configuration

**Frontend Configuration (`usdcop-trading-dashboard/`):**
- `next.config.ts` - Next.js performance optimizations
- `tailwind.config.js` - 2025 fintech UI styling
- `package.json` - Dependencies and scripts

### API Configuration

**Backend Configuration:**
- Environment variables for database connections
- JWT secret for authentication
- Rate limiting and security headers
- WebSocket configuration for real-time data

### Database Configuration

**PostgreSQL Optimization:**
```sql
-- Performance tuning parameters in docker-compose.prod.yml
max_connections=200
shared_buffers=256MB
effective_cache_size=1GB
maintenance_work_mem=64MB
checkpoint_completion_target=0.9
```

**Redis Optimization:**
```conf
# Key settings in redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
appendonly yes
appendfsync everysec
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Services Not Starting

**Problem**: Docker containers fail to start
```bash
# Check service status
docker-compose -f docker-compose.prod.yml ps

# Check logs for specific service
docker-compose -f docker-compose.prod.yml logs [service_name]

# Restart specific service
docker-compose -f docker-compose.prod.yml restart [service_name]
```

**Solution**: Check environment variables and ensure all required values are set.

#### 2. Database Connection Issues

**Problem**: Application cannot connect to PostgreSQL
```bash
# Check PostgreSQL logs
docker-compose -f docker-compose.prod.yml logs postgres

# Test connection manually
docker-compose -f docker-compose.prod.yml exec postgres psql -U postgres -d usdcop_trading
```

**Solution**: Verify `POSTGRES_PASSWORD` and ensure PostgreSQL is fully initialized.

#### 3. Frontend Build Errors

**Problem**: Next.js build fails during deployment
```bash
# Check build logs
docker-compose -f docker-compose.prod.yml logs trading-dashboard

# Build manually for debugging
cd usdcop-trading-dashboard
npm run build
```

**Solution**: Check for TypeScript errors and ensure all dependencies are installed.

#### 4. SSL Certificate Issues

**Problem**: HTTPS not working with custom domains
```bash
# Check certificate files
ls -la nginx/ssl/

# Test certificate validity
openssl x509 -in nginx/ssl/usdcop.crt -text -noout
```

**Solution**: Ensure certificates are valid and properly configured in Nginx.

#### 5. Performance Issues

**Problem**: Slow dashboard response times
```bash
# Check resource usage
docker stats

# Monitor Prometheus metrics
curl http://localhost:9090/api/v1/query?query=up
```

**Solution**: Scale services, optimize queries, or increase server resources.

#### 6. Airflow DAG Import Errors

**Problem**: DAGs showing import errors (ModuleNotFoundError for scipy, gymnasium, etc.)
```bash
# Check DAG import status
docker-compose -f docker-compose.prod.yml logs airflow-webserver | grep -i "import error"

# Test DAG imports manually
docker-compose -f docker-compose.prod.yml run --rm airflow-webserver python /opt/airflow/test_dag_imports.py
```

**Solution**: Rebuild Airflow containers with updated dependencies. See [AIRFLOW-FIX-GUIDE.md](./AIRFLOW-FIX-GUIDE.md) for detailed instructions.

### Log Locations

**Container Logs:**
```bash
# All services
docker-compose -f docker-compose.prod.yml logs

# Specific service with timestamps
docker-compose -f docker-compose.prod.yml logs -f -t trading-dashboard

# Last 100 lines
docker-compose -f docker-compose.prod.yml logs --tail=100 backend-api
```

**Host System Logs:**
- Nginx: `/var/log/nginx/`
- Application: `logs/` directory in project root
- System: `/var/log/syslog` or `journalctl -u docker`

### Health Checks

**Service Health Endpoints:**
```bash
# Dashboard health
curl http://localhost:3000/api/health

# Backend API health
curl http://localhost:8000/health

# Database health
docker-compose -f docker-compose.prod.yml exec postgres pg_isready -U postgres

# Redis health
docker-compose -f docker-compose.prod.yml exec redis redis-cli ping
```

---

## Maintenance

### Regular Maintenance Tasks

#### Daily Tasks
- [ ] Monitor dashboard for alerts
- [ ] Check trading performance metrics
- [ ] Verify data pipeline quality gates
- [ ] Review system resource usage

#### Weekly Tasks
- [ ] Update security patches
- [ ] Backup databases and configurations
- [ ] Review and rotate logs
- [ ] Check disk space usage
- [ ] Validate SSL certificate expiration

#### Monthly Tasks
- [ ] Update dependencies and Docker images
- [ ] Performance optimization review
- [ ] Security audit
- [ ] Disaster recovery testing
- [ ] Documentation updates

### Backup Procedures

**Automated Backup:**
```bash
# Create backup using deployment script
./deploy.sh backup
```

**Manual Backup:**
```bash
# Database backup
docker-compose -f docker-compose.prod.yml exec -T postgres pg_dump -U postgres usdcop_trading > backup_$(date +%Y%m%d).sql

# Configuration backup
tar -czf config_backup_$(date +%Y%m%d).tar.gz .env.prod nginx/ prometheus/ redis.conf

# Data volumes backup
docker run --rm -v usdcop-rl-trading_minio-data:/data -v $(pwd):/backup alpine tar czf /backup/minio_backup_$(date +%Y%m%d).tar.gz -C /data .
```

**Restore Procedures:**
```bash
# Restore database
docker-compose -f docker-compose.prod.yml exec -T postgres psql -U postgres usdcop_trading < backup_20250115.sql

# Restore data volumes
docker run --rm -v usdcop-rl-trading_minio-data:/data -v $(pwd):/backup alpine tar xzf /backup/minio_backup_20250115.tar.gz -C /data
```

### Update Procedures

**System Updates:**
```bash
# Update the system using deployment script
./deploy.sh update

# Manual update process
git pull origin main
docker-compose -f docker-compose.prod.yml build --no-cache
docker-compose -f docker-compose.prod.yml up -d
```

**Security Updates:**
```bash
# Update base images
docker-compose -f docker-compose.prod.yml pull

# Rebuild with latest security patches
docker-compose -f docker-compose.prod.yml build --pull --no-cache
```

---

## Security

### Security Best Practices

#### 1. Authentication and Authorization
- Use strong passwords (minimum 32 characters for secrets)
- Enable two-factor authentication where available
- Regularly rotate API keys and secrets
- Implement proper role-based access control

#### 2. Network Security
```bash
# Configure firewall (Ubuntu/Debian)
sudo ufw enable
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw deny 5432/tcp   # Block direct database access
sudo ufw deny 6379/tcp   # Block direct Redis access
```

#### 3. SSL/TLS Configuration
- Use TLS 1.2 or higher
- Implement proper certificate validation
- Use strong cipher suites
- Enable HTTP Strict Transport Security (HSTS)

#### 4. Container Security
```bash
# Scan images for vulnerabilities
docker scan usdcop-trading-dashboard:latest

# Run containers as non-root users (already configured)
# Regularly update base images
# Use minimal base images (Alpine Linux)
```

#### 5. Data Protection
- Encrypt data at rest and in transit
- Implement proper backup encryption
- Use secure key management
- Regular security audits

### Security Monitoring

**Prometheus Alerts:**
- Unauthorized access attempts
- Failed authentication attempts
- Unusual traffic patterns
- Security vulnerability detection

**Log Monitoring:**
```bash
# Monitor authentication failures
grep "authentication failed" logs/api/*.log

# Monitor suspicious activities
grep "ERROR\|CRITICAL" logs/*/*.log | tail -50
```

### Compliance

**Regulatory Compliance:**
- SFC Colombia financial regulations
- Basel III capital requirements
- GDPR data protection (if applicable)
- SOX compliance for financial reporting

**Audit Requirements:**
- Complete transaction traceability with SHA256 hashes
- Immutable audit logs
- Regular compliance reporting
- Independent security assessments

---

## Support and Resources

### Documentation
- [Next.js Documentation](https://nextjs.org/docs)
- [Docker Documentation](https://docs.docker.com/)
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)

### Community Resources
- [USDCOP Trading System Wiki](https://github.com/your-org/USDCOP-RL-Models/wiki)
- [Issue Tracker](https://github.com/your-org/USDCOP-RL-Models/issues)
- [Discussion Forum](https://github.com/your-org/USDCOP-RL-Models/discussions)

### Getting Help
1. Check this documentation first
2. Search existing issues on GitHub
3. Review application logs for error messages
4. Create detailed issue reports with reproduction steps
5. Contact the development team for critical issues

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes with proper testing
4. Submit a pull request with detailed description
5. Follow code review feedback

---

## Conclusion

This setup guide provides everything needed to deploy and maintain the USDCOP RL Trading System. The system is designed for high performance, security, and compliance with financial regulations.

For questions or issues not covered in this guide, please refer to the support resources above or contact the development team.

**Happy Trading! 📈**