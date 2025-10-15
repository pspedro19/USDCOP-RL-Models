# USD/COP Trading System - L0 Pipeline Monitoring

## Overview

This comprehensive monitoring system provides real-time visibility into the L0 pipeline, backup health, WebSocket connections, API usage, and alerting for the USD/COP trading system. The system includes both React-based dashboard components and Grafana configurations for enterprise-grade monitoring.

## System Components

### 1. Enhanced L0 Dashboard (`/components/views/L0RawDataDashboard.tsx`)

**Features:**
- **Pipeline Status Monitoring**: Real-time status of L0 pipeline (running, completed, failed, idle)
- **Backup Health Tracking**: Backup existence, integrity, gaps, and freshness
- **Ready Signal Monitoring**: L0→WebSocket handover status and pending records
- **Data Quality Metrics**: Completeness, latency, gaps, duplicates, and outliers
- **API Usage Tracking**: Rate limits, remaining calls, key rotation status
- **Real-time Alerting**: Critical and warning alerts with timestamps

**Key Metrics Displayed:**
- Records processed per minute
- Data completeness percentage
- API usage rates and limits
- WebSocket connection status
- Backup integrity scores
- Processing latency

### 2. Health Check API Endpoints

#### `/api/l0/health` - L0 Pipeline Health
- **GET**: Returns comprehensive L0 pipeline health status
- **POST**: Triggers manual health checks and alert resets
- **Metrics**: Pipeline status, backup health, ready signals, data quality, API usage

#### `/api/pipeline/health` - Overall Pipeline Health
- **GET**: Health status for all pipeline components (L0-L5)
- **POST**: Component restart, error clearing, force sync operations
- **Metrics**: Component uptime, error rates, processing latency

#### `/api/backup/status` - Backup System Health
- **GET**: Backup status for all pipeline layers
- **POST**: Trigger backups, test recovery, verify integrity
- **Metrics**: Backup existence, gaps, integrity, retention policies

#### `/api/websocket/status` - WebSocket Monitoring
- **GET**: WebSocket connection and ready signal status
- **POST**: Restart WebSocket, force handover, clear buffers
- **Metrics**: Client connections, latency, data flow rates

#### `/api/usage/monitoring` - API Usage Tracking
- **GET**: API rate limits and key management
- **POST**: Key rotation, quota management, backup key activation
- **Metrics**: Usage percentages, key age, rotation schedules

#### `/api/alerts/system` - Alert Management
- **GET**: Active alerts, notification channels, response metrics
- **POST**: Acknowledge/resolve alerts, test notifications
- **Metrics**: Alert counts, MTTR, notification success rates

### 3. Grafana Dashboard Configurations

#### L0 Pipeline Monitoring (`/grafana-dashboards/l0-pipeline-monitoring.json`)
- Pipeline status overview with color-coded indicators
- Records processing rate with time-series visualization
- Data quality gauge showing completeness percentage
- API usage rate limits with threshold alerts
- Backup health status indicators
- WebSocket connection monitoring
- Active alerts summary table

#### Pipeline Health Overview (`/grafana-dashboards/pipeline-health-overview.json`)
- Component health heatmap for all layers (L0-L5)
- Data flow rate across all components
- Error rate distribution by component
- Resource utilization (CPU, memory)
- Data latency distribution with percentiles
- Alert distribution by component

#### Alerting Dashboard (`/grafana-dashboards/alerting-dashboard.json`)
- Alert status overview by severity
- Alert rate over time with stacking
- Mean Time to Resolution (MTTR) gauge
- Active alerts table with filtering
- Alert distribution by component (pie chart)
- Notification channel health status
- Alert response time percentiles

## Alert Rules and Thresholds

### Critical Alerts
- **L0 Pipeline Down**: Pipeline status = failed for >2 minutes
- **Data Gap Detected**: Any gaps in L0 data stream
- **Backup Integrity Failed**: Backup verification failure
- **API Rate Limit Critical**: >95% API usage
- **WebSocket Down**: No connected clients for >3 minutes

### Warning Alerts
- **High Latency**: Processing latency >30 seconds
- **API Rate Limit Warning**: >80% API usage
- **Backup Gaps**: Gaps detected in backup data
- **Key Rotation Due**: API keys approaching expiration

## Installation and Configuration

### 1. Dashboard Setup
```bash
# The L0 dashboard is already integrated into the main application
# Access via navigation: Pipeline → L0 Raw Data
```

### 2. Grafana Setup
```bash
# Copy dashboard configurations
cp grafana-dashboards/*.json /var/lib/grafana/dashboards/

# Apply provisioning configuration
cp grafana-dashboards/provisioning.yml /etc/grafana/provisioning/dashboards/
```

### 3. Environment Variables
```bash
# Required for notifications
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
export WEBHOOK_URL="https://your-webhook-endpoint.com/alerts"
```

## API Usage Examples

### Check L0 Pipeline Health
```bash
curl -X GET http://localhost:3000/api/l0/health
```

### Trigger Manual Backup
```bash
curl -X POST http://localhost:3000/api/backup/status \
  -H "Content-Type: application/json" \
  -d '{"action": "trigger-backup", "layer": "l0"}'
```

### Restart WebSocket Server
```bash
curl -X POST http://localhost:3000/api/websocket/status \
  -H "Content-Type: application/json" \
  -d '{"action": "restart-websocket"}'
```

### Acknowledge Alert
```bash
curl -X POST http://localhost:3000/api/alerts/system \
  -H "Content-Type: application/json" \
  -d '{"action": "acknowledge-alert", "alertId": "alert-123", "user": "admin"}'
```

## Monitoring Best Practices

### 1. Regular Health Checks
- Monitor the L0 dashboard every 15 minutes during market hours
- Check backup integrity daily
- Verify API key rotation monthly

### 2. Alert Response Procedures
- **Critical Alerts**: Respond within 5 minutes
- **Warning Alerts**: Respond within 30 minutes
- **Info Alerts**: Review during business hours

### 3. Maintenance Windows
- Schedule pipeline maintenance during market closure
- Test backup recovery monthly
- Rotate API keys before expiration

## Troubleshooting Guide

### Common Issues

#### 1. Pipeline Status Shows Failed
```bash
# Check pipeline logs
curl -X GET http://localhost:3000/api/l0/health

# Restart pipeline if necessary
curl -X POST http://localhost:3000/api/pipeline/health \
  -d '{"action": "restart-component", "component": "l0"}'
```

#### 2. High API Usage
```bash
# Check current usage
curl -X GET http://localhost:3000/api/usage/monitoring

# Activate backup key if needed
curl -X POST http://localhost:3000/api/usage/monitoring \
  -d '{"action": "activate-backup-key", "provider": "twelveData"}'
```

#### 3. Backup Integrity Issues
```bash
# Verify backup integrity
curl -X POST http://localhost:3000/api/backup/status \
  -d '{"action": "verify-integrity"}'

# Trigger new backup if corrupted
curl -X POST http://localhost:3000/api/backup/status \
  -d '{"action": "trigger-backup", "layer": "l0"}'
```

## Performance Metrics

### Target SLAs
- **Pipeline Uptime**: 99.9%
- **Data Completeness**: >99.5%
- **Processing Latency**: <30 seconds
- **API Response Time**: <2 seconds
- **Alert Response Time**: <5 minutes (critical), <30 minutes (warning)

### Key Performance Indicators
- Records processed per second
- Data quality score
- System availability percentage
- Mean time to detection (MTTD)
- Mean time to resolution (MTTR)

## Security Considerations

### 1. API Security
- All endpoints require authentication
- Rate limiting on health check endpoints
- Audit logging for all API calls

### 2. Data Protection
- Encrypted backup storage
- Secure key rotation procedures
- Access control for monitoring endpoints

### 3. Alert Security
- Encrypted notification channels
- Secure webhook endpoints
- Authentication for alert management

## Integration Points

### 1. External Systems
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and alerting
- **Slack**: Team notifications
- **Email**: Critical alert notifications

### 2. Internal Components
- **L0 Pipeline**: Direct health monitoring
- **WebSocket Server**: Connection status
- **Backup System**: Integrity verification
- **API Gateway**: Usage tracking

## Future Enhancements

### 1. Machine Learning Integration
- Anomaly detection for data patterns
- Predictive alerting for potential failures
- Automated threshold adjustment

### 2. Advanced Analytics
- Performance trend analysis
- Capacity planning metrics
- Cost optimization insights

### 3. Enhanced Automation
- Self-healing pipeline components
- Automated backup recovery
- Dynamic resource scaling

## Support and Maintenance

### Contact Information
- **Primary**: Trading Operations Team
- **Secondary**: DevOps Team
- **Escalation**: System Architecture Team

### Maintenance Schedule
- **Daily**: Health check verification
- **Weekly**: Performance review
- **Monthly**: Capacity planning review
- **Quarterly**: System optimization

---

## Quick Reference

### Dashboard URLs
- **Main Dashboard**: `/` (Navigate to Pipeline → L0)
- **Grafana L0**: `/d/l0-pipeline/l0-pipeline-monitoring`
- **Grafana Health**: `/d/pipeline-health/pipeline-health-overview`
- **Grafana Alerts**: `/d/alerts/alerting-dashboard`

### Key Endpoints
- **L0 Health**: `GET /api/l0/health`
- **Pipeline Health**: `GET /api/pipeline/health`
- **Backup Status**: `GET /api/backup/status`
- **WebSocket Status**: `GET /api/websocket/status`
- **API Usage**: `GET /api/usage/monitoring`
- **Alerts**: `GET /api/alerts/system`

### Critical Thresholds
- **Pipeline Downtime**: >2 minutes
- **Data Gaps**: Any gap detected
- **API Usage**: >95%
- **Processing Latency**: >30 seconds
- **Backup Age**: >6 hours