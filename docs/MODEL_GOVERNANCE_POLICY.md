# MODEL GOVERNANCE POLICY
## USD/COP RL Trading System

**Version**: 1.0.0
**Effective Date**: 2026-01-17
**Owner**: Trading Operations Team
**Review Frequency**: Quarterly
**Document Status**: APPROVED

---

## Table of Contents

1. [Model Lifecycle Stages](#1-model-lifecycle-stages)
2. [Promotion Requirements](#2-promotion-requirements)
3. [Model Ownership](#3-model-ownership)
4. [Monitoring and Alerting](#4-monitoring-and-alerting)
5. [Retraining Policy](#5-retraining-policy)
6. [Retirement Policy](#6-retirement-policy)
7. [Audit and Compliance](#7-audit-and-compliance)
8. [Exceptions](#8-exceptions)
9. [Appendix A: Model Card Template](#appendix-a-model-card-template)

---

## 1. MODEL LIFECYCLE STAGES

### 1.1 Stage Definitions

| Stage | Description | Duration | Exit Criteria |
|-------|-------------|----------|---------------|
| **Development** | Training and initial validation | Variable | Backtest passes thresholds |
| **Registered** | Backtest validated, pending review | <= 7 days | Manual promotion to Staging |
| **Staging** | Shadow mode validation | >= 7 days | Shadow metrics pass |
| **Production** | Live trading | Indefinite | Degradation or replacement |
| **Archived** | Retired from active use | Indefinite | None |

### 1.2 Stage Transitions

```
Development -> Registered: Automatic (backtest passes)
Registered  -> Staging:    Manual approval required
Staging     -> Production: Manual approval + 7-day minimum
Production  -> Archived:   On replacement or manual retirement
Any Stage   -> Archived:   Emergency retirement
```

### 1.3 Stage Transition Diagram

```
                    +--------------+
                    | Development  |
                    +--------------+
                           |
                    (Backtest Pass)
                           v
                    +--------------+
                    |  Registered  |
                    +--------------+
                           |
                   (Manual Approval)
                           v
                    +--------------+
                    |   Staging    |<---+
                    +--------------+    |
                           |            |
                  (7+ days + metrics)   |
                           v            |
                    +--------------+    |
                    | Production   |----+ (Rollback)
                    +--------------+
                           |
               (Retirement/Replacement)
                           v
                    +--------------+
                    |   Archived   |
                    +--------------+
```

---

## 2. PROMOTION REQUIREMENTS

### 2.1 Registered -> Staging

**Quantitative Requirements:**

| Metric | Threshold | Validation Method |
|--------|-----------|-------------------|
| Sharpe Ratio | >= 0.5 | Backtest report |
| Win Rate | >= 45% | Backtest report |
| Max Drawdown | <= 15% | Backtest report |
| Total Trades | >= 50 | Backtest report |
| Out-of-Sample R^2 | >= 0.1 | Validation split |

**Qualitative Requirements:**
- [ ] Backtest report reviewed by model owner
- [ ] Training artifacts logged in MLflow
- [ ] norm_stats.json hash verified
- [ ] Dataset version tracked in DVC
- [ ] Feature order matches contract CTR-FEAT-001

**Approvers:**
- Model Owner (required)

### 2.2 Staging -> Production

**Quantitative Requirements:**

| Metric | Threshold | Validation Method |
|--------|-----------|-------------------|
| Sharpe Ratio | >= 1.0 | Staging live metrics |
| Win Rate | >= 50% | Staging live metrics |
| Max Drawdown | <= 10% | Staging live metrics |
| Total Trades | >= 100 | Staging trade count |
| Staging Duration | >= 7 days | System timestamp |
| Shadow Agreement Rate | >= 85% | Shadow mode comparison |

**Qualitative Requirements:**
- [ ] Staging performance report reviewed
- [ ] No critical alerts during staging period
- [ ] Team notified of promotion intent (48h advance)
- [ ] Rollback plan documented
- [ ] Model card generated and reviewed
- [ ] On-call engineer briefed

**Approvers:**
- Model Owner (required)
- Engineering Lead (required for first production deployment)

### 2.3 Emergency Promotion

In exceptional circumstances, requirements may be waived by:

1. **CTO approval** (documented in writing via email or Slack)
2. **Post-promotion review** within 48 hours
3. **Enhanced monitoring** for 14 days post-promotion
4. **Automatic rollback threshold** lowered by 50%

Emergency promotions must be documented in the incident log with:
- Justification for emergency
- Risk assessment
- Mitigation measures
- Review timeline

---

## 3. MODEL OWNERSHIP

### 3.1 Roles and Responsibilities

#### Model Owner
**Primary accountability for model performance**

| Responsibility | Frequency |
|----------------|-----------|
| Review model metrics | Daily |
| Approve promotions | As needed |
| Respond to degradation alerts | Within SLA |
| Document model decisions | Ongoing |
| Conduct quarterly reviews | Quarterly |
| Update model card | On changes |

#### Backup Owner
**Secondary accountability when owner unavailable**

| Responsibility | Frequency |
|----------------|-----------|
| Cover owner absences | As needed |
| Receive alert escalations | As configured |
| Emergency decision authority | When owner unavailable |

#### Engineering Lead
**Technical oversight and approval authority**

| Responsibility | Frequency |
|----------------|-----------|
| Approve first production deployments | As needed |
| Review promotion process compliance | Quarterly |
| Escalation point for issues | As needed |
| Policy updates | As needed |

#### On-Call Engineer
**Operational response during off-hours**

| Responsibility | Frequency |
|----------------|-----------|
| Monitor model health | 24/7 |
| Execute emergency procedures | As needed |
| Document incidents | Per incident |
| Escalate per playbook | Per incident |

### 3.2 Assignment Requirements

Every model in **Production** or **Staging** MUST have:

| Field | Requirement | Storage |
|-------|-------------|---------|
| Model Owner | Individual (not team) | model_registry.owner_id |
| Backup Owner | Individual (not team) | model_registry.backup_owner_id |
| On-Call Schedule | Linked to PagerDuty | External system |

### 3.3 Ownership Transfer

When transferring model ownership:

1. Current owner documents transfer in model audit log
2. New owner reviews model card and performance history
3. New owner confirms acceptance in writing
4. Database updated within 24 hours
5. Notification sent to team

---

## 4. MONITORING AND ALERTING

### 4.1 Required Metrics

All Production models MUST have these metrics tracked:

| Metric | Granularity | Retention |
|--------|-------------|-----------|
| Sharpe Ratio (rolling) | 30-day window | 2 years |
| Win Rate (rolling) | 100 trades | 2 years |
| Daily P&L | Daily | 2 years |
| Maximum Drawdown | Daily | 2 years |
| Inference Latency (P50, P95, P99) | Per request | 90 days |
| Error Rate | Per 5-minute window | 90 days |
| Feature Drift (PSI) | Hourly | 1 year |
| Prediction Distribution | Daily | 1 year |

### 4.2 Alert Thresholds

| Metric | Warning Threshold | Critical Threshold | Action |
|--------|-------------------|-------------------|--------|
| Sharpe (30d) | < 0.8 | < 0.5 | Review model, consider rollback |
| Win Rate (100 trades) | < 48% | < 45% | Investigate, prepare rollback |
| Daily Drawdown | > 3% | > 5% | Auto-pause trading |
| Consecutive Losses | >= 5 | >= 10 | Alert team / Auto-rollback |
| Error Rate (5m) | > 1% | > 5% | Auto-rollback triggered |
| Latency P99 | > 1s | > 2s | Auto-rollback triggered |
| Feature PSI | > 0.1 | > 0.2 | Initiate retraining evaluation |

### 4.3 Drift Detection

**Feature Drift Monitoring:**
- Checked hourly for all 15 input features
- PSI (Population Stability Index) calculated against training baseline
- Threshold: PSI > 0.2 triggers warning

**Concept Drift Monitoring:**
- Checked daily using prediction distribution analysis
- Threshold: KL divergence > 0.1 from baseline

**Drift Response Process:**
1. Alert triggered when threshold exceeded
2. Model owner notified within 15 minutes
3. Decision required within 24 hours: continue, retrain, or rollback
4. If 3 consecutive days of drift: automatic retraining initiated

### 4.4 Alert Routing

| Severity | Channel | Response Time |
|----------|---------|---------------|
| P0 (Critical) | #trading-alerts-p0, PagerDuty | Immediate |
| P1 (High) | #trading-alerts, Email | 15 minutes |
| P2 (Medium) | #trading-alerts | 1 hour |
| P3 (Low) | #trading-info | Next business day |

---

## 5. RETRAINING POLICY

### 5.1 Automatic Retraining Triggers

| Trigger Condition | Action |
|-------------------|--------|
| Feature drift PSI > 0.2 for 3 consecutive days | Queue retraining |
| Sharpe ratio < 0.5 for 7 consecutive days | Queue retraining |
| Model age > 90 days | Queue retraining |
| Win rate < 45% for 50 consecutive trades | Queue retraining |

### 5.2 Manual Retraining Triggers

| Trigger | Authorization Required |
|---------|----------------------|
| Market regime change | Model Owner |
| New data sources available | Model Owner + Eng Lead |
| Performance degradation investigation | Model Owner |
| Strategy modification | Model Owner + Eng Lead + CTO |

### 5.3 Retraining Process

```
1. Trigger Detected (auto or manual)
           |
           v
2. Data Preparation
   - Last 6 months of data (minimum)
   - Macro indicators updated
   - Feature pipeline validated
           |
           v
3. Model Training
   - L3 Training DAG executed
   - MLflow run created with full tracking
   - Artifacts logged with hashes
           |
           v
4. Automatic Backtest Validation
   - Out-of-sample period: 20% of data
   - Must meet Registered thresholds
           |
           v
5. Performance Comparison
   - Compare to current production model
   - Document delta in each metric
           |
           v
6. Decision Point
   |
   +-- If BETTER: Promote to Staging automatically
   |
   +-- If WORSE: Discard, alert team with analysis
   |
   +-- If MARGINAL: Human review required
```

### 5.4 Retraining Constraints

| Constraint | Value | Rationale |
|------------|-------|-----------|
| Max automatic retrainings per week | 1 | Prevent thrashing |
| Cooldown between manual retrainings | 48 hours | Allow metrics to stabilize |
| Minimum data for retraining | 6 months | Statistical significance |
| Production model during retraining | Continues operating | Zero downtime |
| Parallel training limit | 2 models | Resource constraint |

---

## 6. RETIREMENT POLICY

### 6.1 Criteria for Retirement

A model SHOULD be retired when:

| Condition | Required Action |
|-----------|-----------------|
| Replaced by better-performing model | Archive immediately |
| Fundamental strategy change | Archive after new model stable |
| Regulatory requirement | Immediate retirement |
| Persistent underperformance (>30 days below threshold) | Retirement review |
| Security vulnerability identified | Emergency retirement |

### 6.2 Retirement Process

```
1. Model Owner documents retirement reason
           |
           v
2. Model transitioned to Archived status
   - Database: status = 'archived'
   - No new predictions allowed
           |
           v
3. Final model card update
   - Retirement date added
   - Final performance metrics documented
   - Reason for retirement documented
           |
           v
4. Notification sent to team
   - Slack notification
   - Email to stakeholders
           |
           v
5. Artifacts retained per policy
   - Model files: 2 years minimum
   - Training data pointer: 7 years
   - Predictions: 7 years
```

### 6.3 Retention Requirements

| Artifact Type | Retention Period | Storage Location | Rationale |
|---------------|------------------|------------------|-----------|
| Model artifacts | 2 years | S3/MinIO | Audit/rollback |
| Training data | 7 years | DVC + S3 | Regulatory |
| Trade history | 7 years | PostgreSQL + backup | Regulatory |
| Audit logs | 7 years | PostgreSQL + backup | Regulatory |
| Model cards | Indefinite | Git repository | Documentation |
| MLflow runs | 2 years | MLflow backend | Reproducibility |

---

## 7. AUDIT AND COMPLIANCE

### 7.1 Audit Trail

All model actions MUST be logged with:

| Field | Description | Example |
|-------|-------------|---------|
| timestamp | ISO 8601 format | 2026-01-17T10:30:00Z |
| action | Action type | promotion, rollback, update |
| from_model | Previous model ID (if applicable) | ppo_v19 |
| to_model | New model ID (if applicable) | ppo_v20 |
| user_id | Who performed action | trading_ops |
| reason | Documented justification | "Improved Sharpe ratio" |
| metadata | Additional context (JSON) | {"metrics": {...}} |

**Logged Actions:**
- Model promotions
- Model rollbacks
- Parameter changes
- Configuration updates
- Retraining events
- Retirement events
- Kill switch activations
- Alert acknowledgments

### 7.2 Quarterly Review

Every quarter, the following review MUST be conducted:

| Activity | Owner | Due |
|----------|-------|-----|
| Review all Production model performance | Model Owners | Q+15 days |
| Validate monitoring effectiveness | Eng Lead | Q+15 days |
| Update governance policy if needed | Trading Ops | Q+30 days |
| Document review in meeting notes | Trading Ops | Q+7 days |
| Audit trail sampling (10% of actions) | Compliance | Q+30 days |

### 7.3 Annual Audit

Annually, the following comprehensive audit MUST be conducted:

| Activity | Owner | Due |
|----------|-------|-----|
| Full model inventory review | Trading Ops | Annual |
| Compliance check against this policy | Compliance | Annual |
| Policy update if required | Trading Ops + Legal | Annual |
| Training for new team members | Trading Ops | Annual |
| Penetration testing of model APIs | Security | Annual |
| Disaster recovery drill | All | Annual |

### 7.4 Audit Report Template

```markdown
## Model Governance Quarterly Audit Report

**Period**: Q[X] 2026
**Auditor**: [Name]
**Date**: [Date]

### Summary
- Models in Production: [count]
- Models in Staging: [count]
- Models Archived this quarter: [count]
- Incidents related to governance: [count]

### Findings
1. [Finding description]
   - Severity: [High/Medium/Low]
   - Action Required: [Yes/No]
   - Remediation: [Description]

### Recommendations
1. [Recommendation]

### Sign-off
- Trading Lead: ________________ Date: ________
- Engineering Lead: ________________ Date: ________
```

---

## 8. EXCEPTIONS

### 8.1 Exception Request Process

Exceptions to this policy require:

1. **Written justification** including:
   - Policy section being excepted
   - Business reason for exception
   - Risk assessment
   - Proposed mitigation measures
   - Requested duration

2. **Approval chain**:
   - Model Owner (required)
   - Engineering Lead (required)
   - CTO (required for P0 risk exceptions)

3. **Time limits**:
   - Maximum exception duration: 30 days
   - Extension requires new request

4. **Documentation**:
   - Exception logged in audit trail
   - Post-exception review within 7 days of expiry

### 8.2 Exception Categories

| Category | Max Duration | Approval Level |
|----------|--------------|----------------|
| Skip staging period | 7 days | CTO |
| Lower promotion thresholds | 30 days | Eng Lead |
| Extend model age limit | 30 days | Model Owner |
| Bypass automated checks | 7 days | CTO |

---

## APPENDIX A: Model Card Template

```yaml
---
# Model Card: {MODEL_ID}
# Generated: {DATE}
---

model_id: ppo_v20_20260115
version: 20
owner: trading_team
backup_owner: ml_team
created_date: 2026-01-15
promoted_to_production: 2026-01-22
current_stage: production

training:
  mlflow_run_id: run_xyz123
  mlflow_experiment: usdcop_ppo_training
  dataset_hash: abc123def456...
  dataset_period:
    start: 2024-01-01
    end: 2025-12-31
  norm_stats_hash: def456ghi789...
  feature_order_hash: ghi789jkl012...
  training_duration_hours: 4.5
  total_timesteps: 2000000

artifacts:
  model_zip_hash: sha256:abc123...
  model_onnx_hash: sha256:def456...
  norm_stats_hash: sha256:ghi789...

performance:
  backtest:
    sharpe_ratio: 1.85
    win_rate: 0.54
    max_drawdown: -0.08
    profit_factor: 1.65
    total_trades: 1247
    total_return: 0.32
  staging:
    sharpe_ratio: 1.62
    win_rate: 0.52
    agreement_rate: 0.89
    days_in_staging: 10
  production:
    sharpe_ratio: null  # Updated after 30 days
    win_rate: null
    total_trades: null

hyperparameters:
  learning_rate: 0.0001
  n_steps: 2048
  batch_size: 128
  n_epochs: 10
  gamma: 0.90
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.05

risks:
  - description: High volatility regime sensitivity
    severity: medium
    mitigation: Circuit breaker at 5% daily drawdown
  - description: Dependent on DXY correlation
    severity: low
    mitigation: Daily macro indicator monitoring
  - description: Performance degradation in low liquidity
    severity: medium
    mitigation: Trading hours restriction (8-16 COT)

known_limitations:
  - Not suitable for timeframes other than 5-minute
  - Requires 14-bar warmup before first prediction
  - May exhibit stuck behavior in very low volatility
  - Training data from 2020-2025 may overweight COVID recovery

change_history:
  - date: 2026-01-15
    change: Initial training complete
    author: ml_team
  - date: 2026-01-17
    change: Promoted to staging
    author: trading_ops
  - date: 2026-01-22
    change: Promoted to production
    author: trading_ops

monitoring:
  grafana_dashboard: https://grafana.internal/d/model-ppo-v20
  alerting_policy: trading-alerts-ppo-v20
  drift_check_schedule: "0 * * * *"  # Hourly

next_review_date: 2026-04-15
```

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-17 | Trading Operations | Initial release |

**Document Owner**: Trading Operations Team
**Last Review**: 2026-01-17
**Next Review**: 2026-04-17
**Approval**: [Engineering Lead] / [CTO]

---

*This document is the authoritative source for model governance policies in the USD/COP RL Trading System. All team members are expected to comply with these policies. Questions or clarification requests should be directed to the Trading Operations Team.*
