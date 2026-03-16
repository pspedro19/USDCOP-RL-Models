# INCIDENT RESPONSE PLAYBOOK
## USD/COP RL Trading System

**Version**: 1.0.0
**Last Updated**: 2026-01-17
**Owner**: Trading Operations Team
**Status**: APPROVED

---

## Table of Contents

1. [Incident Severity Levels](#1-incident-severity-levels)
2. [Escalation Contacts](#2-escalation-contacts)
3. [Incident Procedures](#3-incident-procedures)
4. [Communication Templates](#4-communication-templates)
5. [Runbook Quick Reference](#5-runbook-quick-reference)
6. [Post-Incident Process](#6-post-incident-process)

---

## 1. INCIDENT SEVERITY LEVELS

### 1.1 Severity Definitions

| Level | Name | Criteria | Response Time | Resolution Target |
|-------|------|----------|---------------|-------------------|
| **P0** | Critical | System down, active financial losses | Immediate (24/7) | 30 minutes |
| **P1** | High | Major feature broken, degraded performance | 15 minutes | 2 hours |
| **P2** | Medium | Partial feature broken, workaround exists | 1 hour | 8 hours |
| **P3** | Low | Minor bug, no trading impact | Next business day | 5 business days |

### 1.2 Severity Examples

#### P0 - Critical
- Trading API completely down (no trades executing)
- Kill switch triggered unexpectedly
- Database unavailable
- All predictions returning errors
- Security breach detected
- Significant unauthorized financial losses

#### P1 - High
- Model performance severely degraded (Sharpe < 0.3)
- Primary data source (TwelveData) down for >10 minutes
- Inference latency >5 seconds
- Error rate >10% sustained
- Feature drift detected with PSI >0.3
- Single component failure affecting trading

#### P2 - Medium
- Dashboard not loading but trading continues
- Backup data source in use
- Non-critical alerts not firing
- Scheduled jobs failing (non-critical)
- Monitoring gaps identified

#### P3 - Low
- UI cosmetic issues
- Documentation errors
- Non-blocking enhancement requests
- Performance optimization opportunities

### 1.3 Severity Decision Tree

```
Is trading stopped or losing money unexpectedly?
    |
    +-- YES --> P0 (Critical)
    |
    NO
    |
Is a major feature broken or severely degraded?
    |
    +-- YES --> P1 (High)
    |
    NO
    |
Is there a workaround available?
    |
    +-- YES --> P2 (Medium)
    |
    +-- NO --> P1 (High)
    |
Does it affect production trading?
    |
    +-- YES --> P2 (Medium)
    |
    +-- NO --> P3 (Low)
```

---

## 2. ESCALATION CONTACTS

### 2.1 Primary Contacts

| Role | Name | Phone | Email | Slack | Hours |
|------|------|-------|-------|-------|-------|
| On-Call Primary | [Rotation] | +57-XXX-XXX-XXXX | oncall-primary@company.com | @oncall-primary | 24/7 |
| On-Call Secondary | [Rotation] | +57-XXX-XXX-XXXX | oncall-secondary@company.com | @oncall-secondary | 24/7 |
| Engineering Lead | [Name] | +57-XXX-XXX-XXXX | eng-lead@company.com | @eng-lead | 9-18 COT |
| Trading Lead | [Name] | +57-XXX-XXX-XXXX | trading-lead@company.com | @trading-lead | 8-17 COT |
| ML Lead | [Name] | +57-XXX-XXX-XXXX | ml-lead@company.com | @ml-lead | 9-18 COT |
| CTO | [Name] | +57-XXX-XXX-XXXX | cto@company.com | @cto | Emergency Only |

### 2.2 External Contacts

| Service | Contact | Purpose |
|---------|---------|---------|
| TwelveData Support | support@twelvedata.com | Data source issues |
| AWS Support | AWS Console | Infrastructure issues |
| PagerDuty | app.pagerduty.com | Alert management |

### 2.3 Escalation Matrix

| Severity | Initial Response | 15 min | 30 min | 1 hour | 2 hours |
|----------|-----------------|--------|--------|--------|---------|
| P0 | On-Call Primary | + Eng Lead | + CTO | - | - |
| P1 | On-Call Primary | + On-Call Secondary | + Eng Lead | + Trading Lead | - |
| P2 | On-Call Primary | - | - | + On-Call Secondary | + Eng Lead |
| P3 | Team Lead | - | - | - | - |

---

## 3. INCIDENT PROCEDURES

### 3.1 P0: Trading System Down

#### Symptoms
- No trades executing for >5 minutes during trading hours
- API returning 5xx errors consistently
- Dashboard shows "System Down" or fails to load
- Kill switch triggered unexpectedly
- PagerDuty critical alert received

#### Immediate Actions (0-5 minutes)

```bash
# 1. Acknowledge incident in Slack
Post to #trading-incidents: "ACK: Investigating trading system down"

# 2. Check system status
curl -s http://trading-api:8000/api/v1/health | jq .
curl -s http://inference-api:8000/api/v1/health | jq .

# 3. Check kill switch status
curl -s http://trading-api:8000/api/v1/operations/status | jq .

# 4. Check container status
docker ps -a | grep -E "(trading|inference|postgres|redis)"

# 5. If kill switch active unintentionally
# Use dashboard to resume OR:
curl -X POST "http://trading-api:8000/api/v1/operations/resume" \
  -H "Content-Type: application/json" \
  -d '{"resumed_by":"oncall","confirmation_code":"CONFIRM_RESUME"}'
```

#### Diagnosis (5-15 minutes)

```bash
# Check service logs
docker logs trading-api --tail 200 2>&1 | grep -E "(ERROR|CRITICAL|Exception)"
docker logs inference-api --tail 200 2>&1 | grep -E "(ERROR|CRITICAL|Exception)"

# Check database connectivity
psql -h postgres -U trading_user -d trading_db -c "SELECT 1 as health_check;"

# Check Redis
redis-cli -h redis ping

# Check model status
curl -s http://inference-api:8000/api/v1/models | jq .

# Check Airflow DAGs
airflow dags list-runs -d l1_feature_refresh --state running
```

#### Resolution Actions

| Cause | Resolution |
|-------|------------|
| API container crashed | `docker-compose restart trading-api inference-api` |
| Database connection lost | Check PostgreSQL, restart if needed |
| Model file corrupted | Rollback to previous model version |
| Memory exhaustion | Restart containers, check for memory leaks |
| Network partition | Check Docker network, recreate if needed |

#### Escalation Triggers
- 15 minutes without resolution: Engage Engineering Lead
- 30 minutes without resolution: Engage CTO
- Financial losses >$1000: Immediate CTO notification

---

### 3.2 P1: Model Performance Degradation

#### Symptoms
- Sharpe ratio dropped significantly (<0.5)
- Unusual loss streak (>5 consecutive losses)
- High error rate in predictions (>5%)
- Feature drift alerts triggered (PSI >0.2)
- Win rate below 40% for recent window

#### Immediate Actions (0-5 minutes)

```bash
# 1. Acknowledge in Slack
Post to #trading-incidents: "ACK: Investigating model degradation"

# 2. Check current model status
curl -s http://inference-api:8000/api/v1/models/router/status | jq .

# 3. Check recent predictions
curl -s http://inference-api:8000/api/v1/models/router/history?limit=20 | jq .

# 4. Check drift metrics
curl -s http://inference-api:8000/api/v1/models/router/drift | jq .
```

#### Diagnosis (5-30 minutes)

```bash
# 1. Compare to historical metrics
curl -s http://inference-api:8000/api/v1/models/ppo_primary/metrics?window=7d | jq .

# 2. Check feature quality
curl -s http://inference-api:8000/api/v1/health/consistency/ppo_primary | jq .

# 3. Review recent trades
psql -c "
SELECT trade_id, signal, pnl, created_at
FROM trades_history
WHERE created_at > NOW() - INTERVAL '24 hours'
ORDER BY created_at DESC
LIMIT 20;
"

# 4. Check data source freshness
curl -s http://trading-api:8000/api/v1/data/status | jq .

# 5. Compare against shadow model if available
curl -s http://inference-api:8000/api/v1/models/shadow/comparison | jq .
```

#### Resolution Options

| Severity | Action |
|----------|--------|
| Minor degradation (Sharpe 0.5-0.8) | Monitor closely, no immediate action |
| Significant degradation (Sharpe 0.3-0.5) | Activate shadow mode comparison |
| Severe degradation (Sharpe <0.3) | Execute rollback to previous model |

#### Rollback Procedure

```bash
# 1. Get available rollback targets
curl -s http://inference-api:8000/api/v1/models/rollback-targets | jq .

# 2. Execute rollback via dashboard OR API:
curl -X POST "http://inference-api:8000/api/v1/models/rollback" \
  -H "Content-Type: application/json" \
  -d '{
    "reason": "Performance degradation - Sharpe dropped to 0.25",
    "initiated_by": "oncall_engineer"
  }'

# 3. Verify rollback success
curl -s http://inference-api:8000/api/v1/models/router/status | jq .

# 4. Monitor for 30 minutes post-rollback
```

---

### 3.3 P1: Data Source Down (TwelveData)

#### Symptoms
- OHLCV data stale (>10 minutes old)
- Macro indicators not updating
- API timeout errors in logs
- `l1_feature_refresh` DAG failing

#### Immediate Actions (0-5 minutes)

```bash
# 1. Acknowledge in Slack
Post to #trading-incidents: "ACK: Investigating data source outage"

# 2. Check TwelveData status
curl -s https://status.twelvedata.com/api/v2/status.json | jq .

# 3. Check last data timestamp
psql -c "SELECT MAX(timestamp) as last_data FROM ohlcv_usdcop_5m;"

# 4. Check API key validity
curl -s "https://api.twelvedata.com/time_series?symbol=USD/COP&interval=5min&apikey=$TWELVEDATA_API_KEY&outputsize=1" | jq .

# 5. Verify trading auto-paused
curl -s http://trading-api:8000/api/v1/operations/status | jq .
```

#### Diagnosis (5-15 minutes)

```bash
# Check rate limits
curl -s "https://api.twelvedata.com/api_usage?apikey=$TWELVEDATA_API_KEY" | jq .

# Check data pipeline logs
docker logs airflow-scheduler --tail 100 | grep -i twelvedata

# Verify fallback data source if available
curl -s http://trading-api:8000/api/v1/data/sources | jq .
```

#### Mitigation Actions

| Duration | Action |
|----------|--------|
| <10 minutes | Wait for recovery, trading auto-paused |
| 10-60 minutes | Manual pause confirmation, monitor status |
| >60 minutes | Activate kill switch, notify team |

#### Recovery Verification

```bash
# 1. Check data freshness restored
psql -c "SELECT MAX(timestamp), NOW() - MAX(timestamp) as staleness FROM ohlcv_usdcop_5m;"

# 2. Verify feature quality
curl -s http://inference-api:8000/api/v1/health/consistency/ppo_primary | jq .

# 3. Resume trading if quality OK
curl -X POST "http://trading-api:8000/api/v1/operations/resume" \
  -H "Content-Type: application/json" \
  -d '{"resumed_by":"oncall","confirmation_code":"CONFIRM_RESUME"}'
```

---

### 3.4 P2: Database Connection Issues

#### Symptoms
- Intermittent 500 errors
- Slow queries reported
- Connection pool exhausted warnings
- PostgreSQL not responding

#### Immediate Actions

```bash
# Check PostgreSQL status
docker exec postgres pg_isready -U trading_user -d trading_db

# Check active connections
psql -c "SELECT count(*) FROM pg_stat_activity WHERE datname = 'trading_db';"

# Check for long-running queries
psql -c "
SELECT pid, now() - pg_stat_activity.query_start AS duration, query
FROM pg_stat_activity
WHERE (now() - pg_stat_activity.query_start) > interval '1 minute'
AND state != 'idle';
"

# Kill long-running queries if necessary
# psql -c "SELECT pg_terminate_backend(PID);"
```

#### Resolution

| Cause | Resolution |
|-------|------------|
| Connection pool exhausted | Restart application containers |
| Long-running queries | Identify and terminate |
| Database overload | Scale up or optimize queries |
| Disk full | Clear logs, archive old data |

---

## 4. COMMUNICATION TEMPLATES

### 4.1 Incident Start Notification

```markdown
:rotating_light: **INCIDENT: [Title]**

**Severity**: P[0/1/2/3]
**Status**: Investigating
**Impact**: [Brief description of user/system impact]
**Started**: [Time] COT
**On-Call**: @[name]

**Initial Symptoms**:
- [Symptom 1]
- [Symptom 2]

**Actions Being Taken**:
- [Action 1]

**Updates**: This thread / #trading-incidents
```

### 4.2 Incident Update Notification

```markdown
:loudspeaker: **UPDATE: [Title]**

**Status**: [Investigating / Mitigating / Monitoring / Resolved]
**Current Impact**: [Updated impact description]

**Progress Since Last Update**:
- [What was tried]
- [What was found]

**Current Theory**: [Best guess at root cause]

**Next Steps**:
- [Planned action]

**ETA**: [If known, otherwise "Investigating"]
```

### 4.3 Incident Resolved Notification

```markdown
:white_check_mark: **RESOLVED: [Title]**

**Duration**: [X hours Y minutes]
**Impact Summary**:
- Trading downtime: [X minutes]
- Missed trades: [N]
- Financial impact: [$X or "None"]

**Root Cause**: [One sentence summary]

**Resolution**: [What fixed it]

**Follow-up Actions**:
- [ ] Post-mortem scheduled for [Date]
- [ ] [Action item 1]
- [ ] [Action item 2]

**Post-mortem**: [Link when available]
```

### 4.4 External Communication (if needed)

```markdown
**Status Update - USD/COP Trading System**

**Current Status**: [Operational / Degraded / Outage]
**Last Updated**: [Time] COT

**Summary**:
[Brief, non-technical description of the issue]

**Impact**:
[What users/systems are affected]

**Resolution ETA**: [If known]

**Next Update**: [Time] or when status changes
```

---

## 5. RUNBOOK QUICK REFERENCE

### 5.1 Common Operations

| Scenario | Command / Action |
|----------|------------------|
| Kill all trading | Dashboard: Click Kill Switch OR `POST /operations/kill-switch` |
| Resume trading | Dashboard: Enter CONFIRM_RESUME OR `POST /operations/resume` |
| Rollback model | Dashboard: Models > Rollback OR `POST /models/rollback` |
| Pause trading (soft) | `POST /operations/pause` |
| Restart trading API | `docker-compose restart trading-api` |
| Restart inference API | `docker-compose restart inference-api` |
| Force model reload | `POST /api/v1/models/reload` |

### 5.2 Health Checks

| Check | Command |
|-------|---------|
| System health | `curl http://trading-api:8000/api/v1/health` |
| Model status | `curl http://inference-api:8000/api/v1/models` |
| Operations status | `curl http://trading-api:8000/api/v1/operations/status` |
| Database | `psql -c "SELECT 1"` |
| Redis | `redis-cli ping` |
| Feature consistency | `curl http://inference-api:8000/api/v1/health/consistency/ppo_primary` |

### 5.3 Log Locations

| Service | Log Command |
|---------|-------------|
| Trading API | `docker logs trading-api --tail 200` |
| Inference API | `docker logs inference-api --tail 200` |
| Airflow | `docker logs airflow-scheduler --tail 200` |
| PostgreSQL | `docker logs postgres --tail 200` |
| All errors | `docker-compose logs --tail 100 \| grep -i error` |

### 5.4 Recovery Commands

| Recovery | Commands |
|----------|----------|
| Restart all services | `docker-compose down && docker-compose up -d` |
| Clear Redis cache | `redis-cli FLUSHALL` |
| Rebuild containers | `docker-compose build --no-cache && docker-compose up -d` |
| Database failover | See disaster recovery playbook |
| Restore from backup | `./scripts/backup_restore_system.py restore --latest` |

---

## 6. POST-INCIDENT PROCESS

### 6.1 Post-Mortem Requirements

| Severity | Post-Mortem Required | Timeline |
|----------|---------------------|----------|
| P0 | Yes (mandatory) | Within 48 hours |
| P1 | Yes (mandatory) | Within 1 week |
| P2 | Optional (recommended) | Within 2 weeks |
| P3 | No | - |

### 6.2 Post-Mortem Meeting

**Attendees**: On-call, Eng Lead, affected team members
**Duration**: 30-60 minutes
**Agenda**:
1. Timeline review (5 min)
2. Root cause analysis (15 min)
3. What went well (5 min)
4. What could be improved (10 min)
5. Action items (10 min)

### 6.3 Action Item Tracking

All post-mortem action items must be:
- Assigned to a specific owner
- Given a due date
- Tracked in project management system
- Reviewed in weekly team meeting

### 6.4 Metrics to Track

| Metric | Definition | Target |
|--------|------------|--------|
| MTTD (Mean Time to Detect) | Time from issue start to first alert | < 5 min |
| MTTA (Mean Time to Acknowledge) | Time from alert to acknowledgment | < 5 min |
| MTTR (Mean Time to Resolve) | Time from alert to resolution | < 30 min (P0), < 2h (P1) |
| Incidents per Month | Total incident count | Trending down |

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-17 | Trading Operations | Initial release |

**Document Owner**: Trading Operations Team
**Last Review**: 2026-01-17
**Next Review**: 2026-04-17

---

*This playbook is the authoritative guide for incident response in the USD/COP RL Trading System. All on-call engineers must be familiar with these procedures. Updates require review by Engineering Lead.*
