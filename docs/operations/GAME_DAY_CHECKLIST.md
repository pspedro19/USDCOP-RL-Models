# Game Day Checklist
## Trading System Disaster Recovery Drill

**Version**: 1.0.0
**Last Updated**: 2026-01-17
**Owner**: Trading Operations Team

---

## Overview

**Purpose**: Validate the team's ability to respond to production incidents through realistic simulation exercises.

**Frequency**: Monthly (first Friday of each month)

**Duration**: 2-4 hours

**Environment**: Staging (mirrors production configuration)

**Participants**:
- On-call Engineer (primary responder)
- Trading Lead (decision authority)
- ML Engineer (model expertise)
- Observer/Facilitator (runs scenarios, takes notes)

---

## Pre-Game Day Preparation (1 Day Before)

### Environment Setup

- [ ] Staging environment deployed and mirrors production configuration
- [ ] All services healthy: `docker-compose -f docker-compose.staging.yml ps`
- [ ] Database seeded with realistic test data
- [ ] Monitoring dashboards accessible
- [ ] Alerting configured to test channels only

### Team Coordination

- [ ] Calendar block scheduled for all participants (2-4 hours)
- [ ] Backup responders identified in case of real incidents
- [ ] Facilitator has scenario cards prepared (sealed until game day)
- [ ] Meeting room or video call set up

### Documentation Check

- [ ] All runbooks accessible and up-to-date
- [ ] Incident response playbook printed/accessible
- [ ] Escalation contacts list verified
- [ ] Slack channels #trading-gameday and #trading-incidents-test ready

### Technical Verification

- [ ] Kill switch UI functional in staging
- [ ] Rollback mechanism tested (dry-run)
- [ ] PagerDuty test integration working
- [ ] Grafana dashboards loading correctly
- [ ] Log aggregation accessible

---

## Game Day Execution

### Opening (15 minutes)

- [ ] Gather all participants
- [ ] Review rules of engagement:
  - Treat as if it's a real incident
  - Use staging environment only
  - Document all actions with timestamps
  - No advance knowledge of scenarios
- [ ] Verify communication channels working
- [ ] Start recording session (optional but recommended)

---

## Scenario 1: Kill Switch Activation

**Trigger**: Simulated 10% drawdown in 5 minutes (injected via test script)

**Time Allotted**: 30 minutes

### Steps

| Step | Action | Expected Outcome | Actual Outcome | Time |
|------|--------|------------------|----------------|------|
| 1 | Alert fires in #trading-incidents-test | Team notified within 30 seconds | | |
| 2 | On-call acknowledges alert | Acknowledgment recorded | | |
| 3 | On-call identifies kill switch need | Decision within 2 minutes | | |
| 4 | Activate kill switch from dashboard | Trading stops within 10 seconds | | |
| 5 | Verify Slack notification received | Kill switch notification in channel | | |
| 6 | Verify positions closed (simulated) | Position close confirmed | | |
| 7 | Document incident start time | Timestamp recorded | | |
| 8 | Investigate simulated cause | Root cause identified | | |
| 9 | Decision to resume trading | Approval obtained | | |
| 10 | Enter CONFIRM_RESUME code | Trading resumes successfully | | |
| 11 | Verify normal operation | Health checks pass | | |

### Success Criteria

| Criterion | Target | Actual | Pass/Fail |
|-----------|--------|--------|-----------|
| Kill switch activates in < 10 seconds | < 10s | | |
| Slack notification sent | Yes | | |
| Resume requires confirmation code | Yes | | |
| Audit log entry created | Yes | | |
| Total response time | < 5 minutes | | |

### Metrics to Record

| Metric | Value |
|--------|-------|
| Time to first alert | |
| Time to acknowledgment | |
| Time to kill switch activation | |
| Time to investigate cause | |
| Time to resume trading | |
| Total incident duration | |

### Notes

```
{Space for facilitator notes}
```

---

## Scenario 2: Model Rollback

**Trigger**: Simulated model degradation (Sharpe ratio dropped to 0.25)

**Time Allotted**: 30 minutes

### Steps

| Step | Action | Expected Outcome | Actual Outcome | Time |
|------|--------|------------------|----------------|------|
| 1 | Degradation alert triggered | Alert in Slack within 1 minute | | |
| 2 | On-call acknowledges and investigates | Metrics reviewed within 5 minutes | | |
| 3 | Decision to rollback | Decision documented | | |
| 4 | Open rollback panel in dashboard | Previous versions visible | | |
| 5 | Select target version | Version with better metrics chosen | | |
| 6 | Provide rollback reason | Reason documented | | |
| 7 | Execute rollback | Rollback completes < 60 seconds | | |
| 8 | Verify new model loaded | Model version changed | | |
| 9 | Verify predictions using new model | Health check passes | | |
| 10 | Document in incident log | Entry created | | |
| 11 | Notify team of rollback | Slack notification sent | | |

### Success Criteria

| Criterion | Target | Actual | Pass/Fail |
|-----------|--------|--------|-----------|
| Rollback completes in < 60 seconds | < 60s | | |
| No trades during rollback | 0 trades | | |
| Previous model metrics visible | Yes | | |
| Notification sent | Yes | | |
| Service uninterrupted | Yes | | |

### Metrics to Record

| Metric | Value |
|--------|-------|
| Time to detect degradation | |
| Time to decision | |
| Rollback execution time | |
| Inference downtime | |
| First prediction after rollback | |

### Notes

```
{Space for facilitator notes}
```

---

## Scenario 3: Data Source Outage

**Trigger**: Simulated TwelveData API failure (mock returns 503 errors)

**Time Allotted**: 30 minutes

### Steps

| Step | Action | Expected Outcome | Actual Outcome | Time |
|------|--------|------------------|----------------|------|
| 1 | Stale data alert triggered | Alert within 5 minutes of staleness | | |
| 2 | On-call acknowledges | Acknowledgment recorded | | |
| 3 | Verify trading auto-paused | System paused on stale data | | |
| 4 | Check TwelveData status page | Status checked and documented | | |
| 5 | Check API rate limits | Rate limits verified | | |
| 6 | Assess duration estimate | Estimate documented | | |
| 7 | If >1 hour: activate kill switch | Kill switch activated (if applicable) | | |
| 8 | Monitor for recovery | Continuous monitoring | | |
| 9 | Data source recovers | Fresh data flowing | | |
| 10 | Verify feature quality | Quality check passes | | |
| 11 | Resume trading | Trading resumed | | |

### Success Criteria

| Criterion | Target | Actual | Pass/Fail |
|-----------|--------|--------|-----------|
| Auto-pause on stale data | Yes | | |
| Clear messaging about data status | Yes | | |
| No trading with stale features | Yes | | |
| Recovery detection automated | Yes | | |

### Metrics to Record

| Metric | Value |
|--------|-------|
| Time to stale data detection | |
| Time to auto-pause | |
| Duration of outage | |
| Time to resume after recovery | |

### Notes

```
{Space for facilitator notes}
```

---

## Scenario 4: Database Failover

**Trigger**: Primary PostgreSQL database becomes unavailable (container stopped)

**Time Allotted**: 30 minutes

### Steps

| Step | Action | Expected Outcome | Actual Outcome | Time |
|------|--------|------------------|----------------|------|
| 1 | Database alert triggered | Alert within 30 seconds | | |
| 2 | On-call acknowledges | Acknowledgment recorded | | |
| 3 | Verify automatic failover | Failover to replica (if configured) | | |
| 4 | If no auto-failover, manual intervention | Manual failover executed | | |
| 5 | Check trading API responsive | API health check passes | | |
| 6 | Verify write operations work | Test write succeeds | | |
| 7 | Verify read operations work | Test read succeeds | | |
| 8 | Document recovery time | RTO documented | | |
| 9 | Verify no data loss | RPO validated | | |
| 10 | Restore primary (after exercise) | Primary restored | | |

### Success Criteria

| Criterion | Target | Actual | Pass/Fail |
|-----------|--------|--------|-----------|
| Failover < 30 seconds | < 30s | | |
| No data loss (RPO=0) | Yes | | |
| Trading continues or gracefully pauses | Yes | | |
| Write operations work post-failover | Yes | | |

### Metrics to Record

| Metric | Value |
|--------|-------|
| Time to detection | |
| Time to failover complete | |
| Data loss (if any) | |
| Service interruption duration | |

### Notes

```
{Space for facilitator notes}
```

---

## Scenario 5: Security Incident (Optional Advanced)

**Trigger**: Simulated unauthorized API access detected

**Time Allotted**: 45 minutes

### Steps

| Step | Action | Expected Outcome | Actual Outcome | Time |
|------|--------|------------------|----------------|------|
| 1 | Security alert triggered | Alert in security channel | | |
| 2 | On-call acknowledges and escalates | Security team notified | | |
| 3 | Assess scope of access | Audit logs reviewed | | |
| 4 | Rotate compromised credentials | API keys rotated | | |
| 5 | Block suspicious IP (if applicable) | WAF rules updated | | |
| 6 | Verify no unauthorized trades | Trade audit complete | | |
| 7 | Document incident | Full timeline recorded | | |
| 8 | Notify stakeholders | Communications sent | | |

### Success Criteria

| Criterion | Target | Actual | Pass/Fail |
|-----------|--------|--------|-----------|
| Detection within 5 minutes | < 5 min | | |
| Credential rotation completed | Yes | | |
| No financial impact | Yes | | |
| Audit trail complete | Yes | | |

---

## Post-Game Day Review

### Immediate Debrief (30 minutes after scenarios)

#### What Went Well

- [ ] All scenarios completed
- [ ] Response times met targets
- [ ] Communication was clear
- [ ] Documentation was followed

```
{List specific positives}
1.
2.
3.
```

#### What Could Be Improved

```
{List specific improvements}
1.
2.
3.
```

#### Runbook Updates Needed

| Runbook Section | Update Required | Owner | Due Date |
|-----------------|-----------------|-------|----------|
| | | | |
| | | | |

### Action Items

| Issue Found | Action Required | Owner | Priority | Due Date | Status |
|-------------|-----------------|-------|----------|----------|--------|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

### Metrics Summary

| Scenario | Target Time | Actual Time | Pass/Fail |
|----------|-------------|-------------|-----------|
| Kill Switch | < 5 min | | |
| Model Rollback | < 5 min | | |
| Data Outage | N/A (observe) | | |
| DB Failover | < 2 min | | |

### Scores

| Area | Score (1-5) | Notes |
|------|-------------|-------|
| Detection Speed | | |
| Response Time | | |
| Communication | | |
| Documentation Use | | |
| Tool Proficiency | | |
| Decision Making | | |
| **Overall** | | |

---

## Sign-Off

### Participants

| Role | Name | Signature | Date |
|------|------|-----------|------|
| On-Call Engineer | | | |
| Trading Lead | | | |
| ML Engineer | | | |
| Facilitator | | | |

### Approvals

| Role | Name | Approved | Date |
|------|------|----------|------|
| Engineering Lead | | Yes / No | |
| Trading Lead | | Yes / No | |

---

## Follow-Up Schedule

| Activity | Due Date | Owner | Status |
|----------|----------|-------|--------|
| Action items assigned | Game Day + 1 day | Facilitator | |
| Runbook updates complete | Game Day + 1 week | Assigned owners | |
| Action items review | Game Day + 2 weeks | Engineering Lead | |
| Next Game Day scheduled | Game Day + 1 month | Trading Ops | |

---

## Game Day History

| Date | Scenarios Run | Overall Score | Key Findings |
|------|---------------|---------------|--------------|
| {Date} | 1,2,3,4 | {X}/5 | {Summary} |
| {Date} | 1,2,3,4 | {X}/5 | {Summary} |

---

## Appendix: Scenario Injection Scripts

### A. Kill Switch Scenario

```bash
# Inject high drawdown simulation
curl -X POST "http://staging-api:8000/api/v1/test/inject-drawdown" \
  -H "Content-Type: application/json" \
  -d '{"drawdown_pct": 0.10, "duration_minutes": 5}'
```

### B. Model Degradation Scenario

```bash
# Inject low Sharpe ratio
curl -X POST "http://staging-api:8000/api/v1/test/inject-metrics" \
  -H "Content-Type: application/json" \
  -d '{"sharpe_ratio": 0.25, "win_rate": 0.35}'
```

### C. Data Source Outage Scenario

```bash
# Mock TwelveData failure
docker exec twelvedata-mock bash -c "echo 'FAIL_MODE=503' > /tmp/mock_config"
```

### D. Database Failover Scenario

```bash
# Stop primary database
docker stop postgres-primary
# Watch for failover (if configured)
```

---

**Previous Game Day**: {DATE}
**Next Game Day**: {DATE}
**Document Owner**: Trading Operations Team

---

*This checklist is a living document and should be updated after each Game Day based on lessons learned. All participants should be familiar with the incident response playbook before participating.*
