# SDD Spec: Observability & Monitoring Stack

> **Responsibility**: Authoritative source for the USDCOP trading system's monitoring,
> alerting, logging, and tracing infrastructure. Covers all 7 observability services,
> 4 Prometheus alert rule files (37+ rules), AlertManager routing, Loki log aggregation,
> Promtail log shipping, and 4 Grafana dashboards.
>
> Contract: CTR-OBS-001
> Version: 1.0.0
> Date: 2026-03-14

---

## Architecture Overview

```
                    ┌─────────────────────────────────────────────────┐
                    │              Docker Containers                   │
                    │  trading-api, inference-api, airflow, postgres,  │
                    │  redis, mlflow, feast-server, analytics-api      │
                    └───────────┬────────────────────┬────────────────┘
                                │                    │
                    ┌───────────▼──────┐  ┌──────────▼──────────┐
                    │   Prometheus     │  │     Promtail        │
                    │   :9090          │  │  (Docker socket)    │
                    │   Scrape metrics │  │  Ship logs to Loki  │
                    └───────┬──────────┘  └──────────┬──────────┘
                            │                        │
               ┌────────────▼────────┐   ┌───────────▼──────────┐
               │   AlertManager      │   │       Loki           │
               │   :9093             │   │       :3100          │
               │   Route + notify    │   │   Log aggregation    │
               └────────┬───────────┘   └───────────┬──────────┘
                        │                            │
                        │    ┌───────────────────────┘
                        │    │
               ┌────────▼────▼───────┐       ┌──────────────┐
               │     Grafana         │       │    Jaeger     │
               │     :3002           │       │    :16686     │
               │  4 dashboards       │       │  Tracing      │
               │  4 datasources      │       │  (in-memory)  │
               └─────────────────────┘       └──────────────┘
```

---

## Operational Status Matrix (2026-04-16)

Services below are **infrastructure-ready** but have activation gaps. Treat them accordingly.

| Service | Infra Status | Activation Gap | Recommended Narrative |
|---------|--------------|----------------|----------------------|
| Prometheus | ✅ Active | 53 rules evaluating, metrics being scraped from 9 targets | Fully operational |
| Grafana | ✅ Active | 4 dashboards provisioned, 4 datasources connected | Fully operational |
| Loki + Promtail | ✅ Active | Log aggregation working, queries via Grafana Explore | Fully operational |
| AlertManager | ⚠️ Infra ready | `SLACK_WEBHOOK_URL` placeholder empty in `.env`, PagerDuty service key not set | "Rules cargadas, activación pendiente de secrets. UI estática muestra 53 alert rules." |
| Jaeger | ⚠️ Infra ready | 0 services instrumented with OpenTelemetry | Omitir del spec hasta instrumentación de servicios (roadmap) |
| MinIO | ⚠️ Partially used | 11 buckets operativos; MLflow artifacts OK; DAGs no usan como backup destino | "Bucket storage disponible; modelos en filesystem. Roadmap: migración artefactos." |
| MLflow | ⚠️ Partially used | Server + DB + S3 configurados; scripts ad-hoc loguean; DAGs L3 (H1/H5) NO loguean runs | "Tracking server desplegado; integración automática en DAGs en roadmap." |

**What "infra ready" means**: The service is running, healthy, and accepts requests — but downstream
callers or secrets are not yet wired. Don't present as broken; present as "ready for activation."

---

## Service Inventory (7 Services)

| Service | Image | Container | Port | Purpose | Health Check |
|---------|-------|-----------|------|---------|-------------|
| Prometheus | `prom/prometheus:latest` | `usdcop-prometheus` | 9090 | Metrics collection + alert evaluation | `GET /-/healthy` |
| Grafana | `grafana/grafana:latest` | `usdcop-grafana` | 3002 | Dashboard visualization | `GET /api/health` |
| AlertManager | `prom/alertmanager:v0.26.0` | `usdcop-alertmanager` | 9093 | Alert routing + notification | `GET /-/healthy` |
| Loki | `grafana/loki:2.9.0` | `usdcop-loki` | 3100 | Log aggregation | `GET /ready` |
| Promtail | `grafana/promtail:2.9.0` | `usdcop-promtail` | -- | Log shipping (Docker socket) | -- |
| Jaeger | `jaegertracing/all-in-one:1.53` | `usdcop-jaeger` | 16686 | Distributed tracing | `GET :14269/` |
| TimescaleDB | (shared) | `usdcop-postgres` | 5432 | Grafana datasource (trading data) | (see L0 spec) |

### Service Endpoints

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | `http://localhost:3002` | `${GRAFANA_USER}` / `${GRAFANA_PASSWORD}` from `.env` |
| Prometheus | `http://localhost:9090` | None |
| AlertManager | `http://localhost:9093` | None |
| Loki | `http://localhost:3100` | None |
| Jaeger UI | `http://localhost:16686` | None |

### Docker Volumes

| Volume | Purpose |
|--------|---------|
| `prometheus_data` | Prometheus TSDB storage |
| `grafana_data` | Grafana config, plugins, state |
| `loki_data` | Log chunks + indexes |
| `alertmanager_data` | Alert state + silences |

---

## Prometheus Configuration

### Global Settings

**File**: `prometheus/prometheus.yml`

| Setting | Value |
|---------|-------|
| `scrape_interval` | 15s |
| `evaluation_interval` | 15s |
| `external_labels.monitor` | `usdcop-trading` |
| `external_labels.environment` | `production` |

### Scrape Targets (9 Jobs)

| Job | Target | Metrics Path | Interval | Purpose |
|-----|--------|-------------|----------|---------|
| `prometheus` | `localhost:9090` | `/metrics` | 15s | Self-monitoring |
| `trading-api` | `trading-api:8000` | `/metrics` | 15s | Market data service |
| `analytics-api` | `analytics-api:8001` | `/metrics` | 15s | Analytics service |
| `multi-model-api` | `multi-model-api:8006` | `/metrics` | 15s | Multi-model inference |
| `inference-api` | `inference-api:8000` | `/metrics` | 15s | PPO model predictions |
| `postgres` | `postgres-exporter:9187` | `/metrics` | 15s | Database metrics |
| `redis` | `redis:6379` | `/metrics` | 15s | Cache metrics |
| `airflow` | `airflow-webserver:8080` | `/metrics` | 15s | DAG scheduler metrics |
| `feast-server` | `feast-server:6566` | `/metrics` | 30s | Feature store (legacy) |
| `mlflow` | `mlflow:5000` | `/metrics` | 30s | Experiment tracking |

### Alert Rule Files

Prometheus loads rules from `config/prometheus/rules/*.yml` (mapped to `/etc/prometheus/rules/` in container):

| File | Groups | Rules | Focus |
|------|--------|-------|-------|
| `model_alerts.yml` | 4 | 16 | Shadow mode, model health, prediction distribution, reloads |
| `trading_alerts.yml` | 6 | 21 | Service health, trading ops, data quality, infra, pipelines |
| `drift_alerts.yml` | 3 | 7 | Feature drift (KS test), drift monitoring, drift recovery |
| `alerts/latency.yml` | 3 | 9 | Inference latency SLA, feature quality, model health |

**Total**: 16 groups, 53 alert rules.

---

## Alert Rules (53 Rules Across 4 Files)

### model_alerts.yml (16 Rules)

**Shadow Mode** (group: `shadow_mode`, interval: 1m):

| Alert | Expression | For | Severity | PagerDuty | Description |
|-------|-----------|-----|----------|-----------|-------------|
| `ShadowModelHighDivergence` | `shadow_divergence_rate > 0.3` | 15m | warning | no | Champion vs shadow divergence >30% |
| `ShadowModelVeryHighDivergence` | `shadow_divergence_rate > 0.5` | 10m | critical | yes | Divergence >50%, investigate |
| `ShadowModelHighAgreement` | `shadow_divergence_rate < 0.05` | 1h | info | no | Models nearly identical |
| `ShadowModelNotLoaded` | `absent(model_loaded{staging})` | 5m | warning | no | No shadow model running |

**Model Health** (group: `model_health`, interval: 30s):

| Alert | Expression | For | Severity | PagerDuty | Description |
|-------|-----------|-----|----------|-----------|-------------|
| `ChampionModelNotLoaded` | `absent(model_loaded{production})` | 1m | critical | yes | No production model loaded |
| `ModelPredictionLatencyHigh` | `p95(prediction_latency) > 50ms` | 5m | warning | no | Elevated inference latency |
| `ModelPredictionLatencyCritical` | `p99(prediction_latency) > 100ms` | 2m | critical | yes | SLA breach |
| `ChampionModelStale` | `model_age > 7 days` | 1h | info | no | Model older than 7 days |

**Prediction Distribution** (group: `prediction_distribution`, interval: 5m):

| Alert | Expression | For | Severity | PagerDuty | Description |
|-------|-----------|-----|----------|-----------|-------------|
| `PredictionDistributionSkewed` | `>70% one direction in 1h` | 30m | warning | no | Heavily skewed predictions |
| `ModelStuckBehavior` | `>95% HOLD in 1h` | 1h | critical | yes | Model appears stuck |
| `NoPredictionsGenerated` | `0 predictions in 10m` | 15m | warning | no | Inference not producing signals |

**Model Reload** (group: `model_reload`, interval: 1m):

| Alert | Expression | For | Severity | PagerDuty | Description |
|-------|-----------|-----|----------|-----------|-------------|
| `ModelReloadTriggered` | `increase(reload_total) > 0` | 0m | info | no | Reload initiated |
| `ModelReloadFailed` | `increase(reload_failures) > 0` | 0m | critical | yes | Reload failed |

### trading_alerts.yml (21 Rules)

**Service Health** (group: `service_health`, interval: 30s):

| Alert | Expression | For | Severity | PagerDuty |
|-------|-----------|-----|----------|-----------|
| `ServiceDown` | `up == 0` | 2m | critical | yes |
| `TradingAPIDown` | `absent(up{trading-api})` | 1m | critical | yes |
| `InferenceAPIDown` | `absent(up{inference-api})` | 1m | critical | yes |
| `DatabaseConnectionPoolExhausted` | `active/max > 80%` | 5m | warning | no |
| `RedisConnectionsFailing` | `rejected_connections > 0` | 5m | warning | no |
| `HighRestartRate` | `restarts > 3 in 1h` | 0s | warning | no |

**Trading Operations** (group: `trading_operations`, interval: 30s):

| Alert | Expression | For | Severity | PagerDuty |
|-------|-----------|-----|----------|-----------|
| `NoTradingSignals` | `0 signals in 15m (market hours)` | 10m | warning | no |
| `TradingSignalAnomalyHigh` | `>80% one direction in 1h` | 30m | warning | no |
| `LowConfidenceSignals` | `avg confidence < 0.5 in 30m` | 30m | warning | no |
| `ConsecutiveLossesExceeded` | `consecutive_losses >= 7` | 0s | critical | yes |
| `DailyLossLimitApproaching` | `daily_pnl < -1.5%` | 0s | warning | no |
| `DailyLossLimitBreached` | `daily_pnl < -2%` | 0s | critical | yes |

**Data Quality** (group: `data_quality`, interval: 30s):

| Alert | Expression | For | Severity | PagerDuty |
|-------|-----------|-----|----------|-----------|
| `MarketDataStale` | `freshness > 5m` | 5m | warning | no |
| `MarketDataVeryStaleCritical` | `freshness > 15m` | 2m | critical | yes |
| `OHLCVDataGap` | `no data 10m (market hours)` | 5m | warning | no |
| `HighNullRatioInFeatures` | `null_ratio > 10%` | 10m | warning | no |
| `FeatureValueOutOfRange` | `|z-score| > 5` | 5m | warning | no |
| `MacroDataMissing` | `age > 48h` | 1h | warning | no |

**Model Health** (group: `model_health`, interval: 60s):

| Alert | Expression | For | Severity | PagerDuty |
|-------|-----------|-----|----------|-----------|
| `NoModelsLoaded` | `active_models == 0` | 1m | critical | yes |
| `ModelInferenceErrors` | `error_rate > 0.1/s` | 5m | warning | no |
| `ModelDriftDetected` | `drift_score > 0.3` | 1h | warning | no |
| `ModelAccuracyDegraded` | `7d rolling < 45%` | 0s | warning | no |
| `FeatureCalculationFailures` | `calc_errors > 0` | 5m | warning | no |

**Infrastructure** (group: `infrastructure`, interval: 30s):

| Alert | Expression | For | Severity | PagerDuty |
|-------|-----------|-----|----------|-----------|
| `HighMemoryUsage` | `container memory > 85%` | 5m | warning | no |
| `HighCPUUsage` | `container CPU > 80%` | 10m | warning | no |
| `DiskSpaceLow` | `disk available < 15%` | 5m | warning | no |
| `DiskSpaceCritical` | `disk available < 5%` | 1m | critical | yes |
| `PostgresHighConnections` | `connections > 80` | 5m | warning | no |
| `RedisMemoryHigh` | `memory > 80%` | 5m | warning | no |

**Pipeline Health** (group: `pipeline_health`, interval: 60s):

| Alert | Expression | For | Severity | PagerDuty |
|-------|-----------|-----|----------|-----------|
| `AirflowDAGFailed` | `failed_runs > 0 in 1h` | 0s | warning | no |
| `AirflowSchedulerUnhealthy` | `no heartbeat 5m` | 5m | critical | yes |
| `L0PipelineDelayed` | `no success 2h` | 10m | warning | no |

**Watchdog** (group: `watchdog`):

| Alert | Expression | For | Severity | Description |
|-------|-----------|-----|----------|-------------|
| `Watchdog` | `vector(1)` | always | none | Heartbeat (proves alerting works) |

### drift_alerts.yml (7 Rules)

**Feature Drift** (group: `feature_drift`, interval: 5m):

| Alert | Expression | For | Severity | PagerDuty | Description |
|-------|-----------|-----|----------|-----------|-------------|
| `FeatureDriftDetected` | `drift_alert == 1` | 15m | warning | no | KS test flagged single feature |
| `ModerateFeatureDrift` | `drift_score > 0.2` | 10m | warning | no | KS statistic >0.2 |
| `MultipleFeaturesDrifted` | `count(drift_alert) > 3` | 10m | critical | yes | 3+ features drifted simultaneously |
| `HighFeatureDrift` | `drift_score > 0.3` | 5m | critical | yes | Severe distribution shift |
| `OverallDriftScoreHigh` | `avg(drift_score) > 0.15` | 15m | critical | no | System-wide drift |

**Drift Monitoring** (group: `drift_monitoring`, interval: 1m):

| Alert | Expression | For | Severity | Description |
|-------|-----------|-----|----------|-------------|
| `DriftChecksStalled` | `0 checks in 10m` | 15m | warning | Drift detector not running |
| `FeatureWindowsLow` | `<10 features have data` | 30m | info | System warming up |

**Drift Recovery** (group: `drift_recovery`, interval: 5m):

| Alert | Expression | For | Severity | Description |
|-------|-----------|-----|----------|-------------|
| `FeatureDriftResolved` | `alert 0 now, was 1 at -10m` | 0m | info | Feature returned to normal |

### latency.yml (9 Rules)

**Latency Alerts** (group: `latency_alerts`, interval: 30s):

| Alert | Expression | For | Severity | PagerDuty | Description |
|-------|-----------|-----|----------|-----------|-------------|
| `InferenceLatencyP95Warning` | `p95 > 75ms` | 5m | warning | no | Approaching SLA |
| `InferenceLatencyP99Warning` | `p99 > 150ms` | 5m | warning | no | P99 elevated |
| `FeatureCalculationSlow` | `p95 > 40ms` | 5m | warning | no | Feature calc slow |
| `DatabaseQuerySlow` | `p95 > 30ms` | 5m | warning | no | DB queries slow |
| `InferenceLatencyP99Critical` | `p99 > 200ms` | 2m | critical | yes | SLA breach |
| `InferenceLatencyP50Critical` | `p50 > 50ms` | 2m | critical | yes | Systemic issue |
| `InferenceThroughputLow` | `rate < 0.1/s` | 10m | warning | no | Low request volume |
| `InferenceErrorRateHigh` | `error_rate > 1%` | 2m | critical | yes | SLA breach |

**Feature Quality** (group: `feature_quality_alerts`, interval: 30s):

| Alert | Expression | For | Severity | PagerDuty | Description |
|-------|-----------|-----|----------|-----------|-------------|
| `FeatureCircuitBreakerActivated` | `circuit_breaker == 1` | 0s | critical | yes | Trading paused |
| `FeatureNaNRatioHigh` | `nan_ratio > 15%` | 5m | warning | no | Approaching CB threshold |
| `DataFreshnessStale` | `freshness > 10m` | 5m | warning | no | Market data lagging |

**Model Health** (group: `model_health_alerts`, interval: 60s):

| Alert | Expression | For | Severity | PagerDuty | Description |
|-------|-----------|-----|----------|-----------|-------------|
| `ModelNotLoaded` | `active_models == 0` | 1m | critical | yes | No models loaded |
| `ModelDriftHigh` | `feature_drift > 3.0` | 15m | warning | no | Distribution shift |
| `ConsecutiveLossesHigh` | `consecutive_losses >= 5` | 0s | warning | no | Risk warning |

### Latency SLA Targets

| Percentile | Target | Warning | Critical |
|------------|--------|---------|----------|
| p50 | < 20ms | -- | > 50ms |
| p95 | < 50ms | > 75ms | -- |
| p99 | < 100ms | > 150ms | > 200ms |

---

## AlertManager Configuration

**File**: `config/alertmanager/alertmanager.yml`

### Global Settings

| Setting | Value |
|---------|-------|
| `resolve_timeout` | 5m |
| `slack_api_url` | (empty -- set via `SLACK_WEBHOOK_URL` env var) |

### Routing Tree

```
Default: slack-trading (group_wait=30s, group_interval=5m, repeat=4h)
    |
    ├── severity=critical → slack-critical (repeat=15m, continue=true)
    │       └── pagerduty=true → pagerduty-critical (repeat=5m)
    |
    ├── team=trading → slack-trading
    │       ├── FeatureCircuitBreakerActivated → slack-critical (repeat=5m)
    │       ├── Model.* → slack-trading (group_by: alertname+model_id)
    │       └── .*Latency.* → slack-trading (group_interval=2m)
    |
    ├── team=infrastructure → slack-infrastructure
    |
    └── alertname=Watchdog → null (repeat=24h, suppressed)
```

### Grouping

- Group by: `alertname`, `severity`, `service`
- Group wait: 30s (initial notification delay)
- Group interval: 5m (update frequency)
- Repeat interval: 4h (re-notification of unresolved alerts)

### Receivers (5)

| Receiver | Channel | Features |
|----------|---------|----------|
| `slack-trading` | `#trading-alerts` | Buttons: Runbook + Silence links |
| `slack-critical` | `#trading-critical` | `@channel` mention, danger color |
| `slack-infrastructure` | `#infrastructure-alerts` | Warning/good color toggle |
| `pagerduty-critical` | (service key placeholder) | Not configured (placeholder) |
| `null` | (drop) | Used for Watchdog suppression |

### Inhibition Rules (3)

| Source | Suppresses | Match On |
|--------|-----------|----------|
| `severity=critical` | `severity=warning` (same alertname) | `alertname`, `service` |
| `FeatureCircuitBreakerActivated` | `Inference.*`, `Feature.*` | `service` |
| `PostgresDown` | `.*QuerySlow.*`, `.*DataStale.*` | (global) |

---

## Loki Configuration

**File**: `config/loki/loki-config.yml`

### Core Settings

| Setting | Value |
|---------|-------|
| HTTP port | 3100 |
| gRPC port | 9096 |
| Auth | Disabled |
| Storage backend | Filesystem |
| Replication factor | 1 (single node) |
| Ring KV store | In-memory |

### Schema

| Field | Value |
|-------|-------|
| Schema version | v13 |
| Store | tsdb |
| Object store | filesystem |
| Index prefix | `index_` |
| Index period | 24h |
| Effective since | 2024-01-01 |

### Ingester

| Setting | Value |
|---------|-------|
| WAL enabled | true |
| Chunk idle period | 1h |
| Max chunk age | 1h |
| Chunk target size | 1 MB |
| Chunk retain period | 30s |

### Retention & Limits

| Setting | Value |
|---------|-------|
| Retention period | 744h (31 days) |
| Max old samples age | 168h (7 days) |
| Ingestion rate | 10 MB/s |
| Ingestion burst | 20 MB/s |
| Max streams per user | 10,000 |
| Max line size | 256 KB |
| Max entries per query | 5,000 |
| Query parallelism | 32 |

### Compaction

| Setting | Value |
|---------|-------|
| Interval | 10m |
| Retention enabled | true |
| Delete delay | 2h |
| Delete workers | 150 |

### Ruler

AlertManager integration at `http://alertmanager:9093` for Loki-based alerting rules.
Rules stored locally at `/loki/rules`.

---

## Promtail Configuration

**File**: `config/promtail/promtail-config.yml`

### Client Settings

| Setting | Value |
|---------|-------|
| Loki URL | `http://loki:3100/loki/api/v1/push` |
| Tenant ID | `usdcop-trading` |
| Batch wait | 1s |
| Batch size | 1 MB |
| Timeout | 10s |
| Max retries | 10 |

### Scrape Configs (5 Jobs)

| Job | Source | Filter | Labels Added |
|-----|--------|--------|-------------|
| `docker_containers` | Docker socket | `com.docker.compose.project=usdcop-rl-models` | container, service, compose_service, environment=production, team=trading |
| `trading_signals` | Docker socket | Container name matches `usdcop-(inference\|backtest\|mlops)` | action, model_id, symbol |
| `airflow_logs` | Docker socket | Container name matches `usdcop-airflow.*` | level, dag_id, service=airflow |
| `postgres_logs` | Docker socket | Container name matches `usdcop-postgres.*` | level, service=postgres |
| `redis_logs` | Docker socket | Container name matches `usdcop-redis.*` | service=redis |

### JSON Log Parsing

For structured Python logs (detected by `{ "level": ` prefix), extracts:

| Field | Label | Purpose |
|-------|-------|---------|
| `level` | Yes | Log level routing |
| `message` | No | Log body |
| `timestamp` | No | Event time |
| `logger` | Yes | Module name |
| `service` | Yes | Service identifier |
| `request_id` | Yes | Request correlation |
| `duration_ms` | No | Latency extraction |
| `model_id` | No | Model identification |
| `action` | No | Trading action |
| `confidence` | No | Signal confidence |

### Airflow Log Parsing

Regex: `^\[(?P<timestamp>[^\]]+)\]\s+\{(?P<dag_id>[^}]+)\}\s+(?P<level>[A-Z]+)\s+-\s+(?P<message>.*)$`

Extracts `timestamp`, `dag_id`, `level`, `message` from Airflow's structured log format.

### PostgreSQL Log Parsing

Regex: `^(?P<timestamp>...)\s+\[(?P<pid>\d+)\]\s+(?P<level>[A-Z]+):\s+(?P<message>.*)$`

Extracts `timestamp`, `pid`, `level`, `message` from PostgreSQL log format.

### Metrics Extraction from Logs

| Metric | Type | Source | Description |
|--------|------|--------|-------------|
| `log_lines_total` | Counter | `level` | Total log lines by level (all logs) |
| `inference_duration_seconds` | Histogram | `duration_ms` | Inference duration from structured logs |
| `trading_signals_total` | Counter | `action` | Trading signals by action type |

Histogram buckets for `inference_duration_seconds`: `[10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s]`

### Error Detection

- Logs with `level=ERROR` matching `exception|error|failed|failure` get `error_type=runtime_error` label
- Logs with `level=CRITICAL` get `alert=critical` label

---

## Grafana Configuration

### Settings

| Setting | Value |
|---------|-------|
| Admin user | `${GRAFANA_USER}` (from `.env`) |
| Admin password | `${GRAFANA_PASSWORD}` (from `.env`) |
| Timezone | `America/Bogota` |
| Anonymous access | Disabled |
| External port | 3002 (mapped from internal 3000) |
| Dashboard auto-update | Every 30s (from provisioning files) |
| UI edits | Allowed (`allowUiUpdates: true`) |

### Datasources (4, Auto-Provisioned)

**File**: `config/grafana/provisioning/datasources/datasources.yml`

| Datasource | Type | URL | Default | Features |
|------------|------|-----|---------|----------|
| Prometheus | `prometheus` | `http://prometheus:9090` | Yes | Manage alerts, exemplar trace linking to Jaeger |
| Loki | `loki` | `http://loki:3100` | No | Derived fields: TraceID linking to Jaeger |
| TimescaleDB | `postgres` | `postgres:5432` | No | TimescaleDB extensions, max 10 connections |
| Jaeger | `jaeger` | `http://jaeger:16686` | No | Traces-to-logs (Loki), traces-to-metrics (Prometheus), node graph |

### Cross-Datasource Linking

```
Prometheus ←→ Jaeger     (exemplar trace IDs)
Loki ←→ Jaeger           (derived field: traceId regex)
Jaeger → Loki            (trace-to-log correlation, +/-1h window)
Jaeger → Prometheus      (trace-to-metric: request rate, p95 latency)
TimescaleDB              (standalone, trading data queries)
```

### Dashboards (4, Auto-Provisioned)

**File**: `config/grafana/provisioning/dashboards/dashboards.yml`
**Folder**: `USDCOP Trading` (uid: `usdcop-trading`)
**Source**: `config/grafana/dashboards/*.json`

| Dashboard | File | Focus Area |
|-----------|------|------------|
| Trading Performance | `trading-performance.json` | Real-time P&L, win rate, Sharpe, positions, equity curve |
| MLOps Monitoring | `mlops-monitoring.json` | Model health, training status, inference latency, drift |
| System Health | `system-health.json` | Container resources, DB connections, disk, Redis |
| Macro Ingestion | `macro-ingestion.json` | Economic data freshness, completeness, source health |

---

## Jaeger Configuration

### Deployment

| Setting | Value |
|---------|-------|
| Image | `jaegertracing/all-in-one:1.53` |
| Storage | In-memory (`SPAN_STORAGE_TYPE=memory`) |
| OTLP | Enabled (gRPC: 4317, HTTP: 4318) |
| Zipkin | Port 9411 |
| Log level | info |

### Exposed Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 5775 | UDP | Compact Thrift (legacy) |
| 6831 | UDP | Compact Thrift (agent) |
| 6832 | UDP | Binary Thrift (agent) |
| 5778 | TCP | Agent config HTTP |
| 16686 | TCP | Jaeger UI |
| 14250 | TCP | gRPC collector |
| 14268 | TCP | HTTP collector |
| 14269 | TCP | Health check |
| 4317 | TCP | OTLP gRPC |
| 4318 | TCP | OTLP HTTP |

### Status

Jaeger is deployed and healthy but currently has **no instrumented services** sending traces.
The Grafana datasource configuration is pre-wired for trace-to-log and trace-to-metric
correlation. Instrumentation is a future enhancement.

---

## Configuration File Layout

```
prometheus/
└── prometheus.yml                            <- Scrape config + alerting + rule file paths

config/prometheus/
├── rules/
│   ├── model_alerts.yml                      <- 16 rules: shadow mode, model health, predictions
│   ├── trading_alerts.yml                    <- 21 rules: services, trading, data, infra, pipelines
│   └── drift_alerts.yml                      <- 7 rules: feature drift (KS test), recovery
└── alerts/
    └── latency.yml                           <- 9 rules: inference SLA, feature quality

config/alertmanager/
└── alertmanager.yml                          <- Routing tree, receivers, inhibitions

config/loki/
└── loki-config.yml                           <- Schema, retention, limits, compaction

config/promtail/
└── promtail-config.yml                       <- 5 scrape jobs, JSON parsing, metrics extraction

config/grafana/
├── provisioning/
│   ├── datasources/datasources.yml           <- 4 datasources (Prometheus, Loki, TimescaleDB, Jaeger)
│   └── dashboards/dashboards.yml             <- Dashboard provider config
└── dashboards/
    ├── trading-performance.json              <- Trading P&L dashboard
    ├── mlops-monitoring.json                 <- MLOps model health dashboard
    ├── system-health.json                    <- Infrastructure dashboard
    └── macro-ingestion.json                  <- Macro data freshness dashboard
```

---

## Alert Severity Guide

| Severity | Action | Notification | Repeat |
|----------|--------|-------------|--------|
| `info` | Informational, no action | Slack #trading-alerts | 4h |
| `warning` | May require attention | Slack #trading-alerts | 4h |
| `critical` | Requires immediate attention | Slack #trading-critical | 15m |
| `critical` + `pagerduty=true` | Impacts trading, urgent | PagerDuty (if configured) | 5m |
| `none` | System heartbeat (Watchdog) | Suppressed (null receiver) | 24h |

### Alert Escalation Path

```
1. Prometheus evaluates rule → fires alert
2. AlertManager receives alert
3. Group by (alertname, severity, service), wait 30s
4. Route to receiver:
   - warning → #trading-alerts or #infrastructure-alerts
   - critical → #trading-critical (+ PagerDuty if pagerduty=true)
   - Watchdog → null (suppressed)
5. Inhibition check:
   - Critical suppresses warning for same alertname+service
   - CircuitBreaker suppresses inference/feature alerts
   - PostgresDown suppresses query/data stale alerts
```

---

## Key Metrics (Expected from Services)

### Trading Metrics

| Metric | Type | Source | Description |
|--------|------|--------|-------------|
| `usdcop_trading_signals_total` | Counter | trading-api | Signals by action (BUY/SELL) |
| `usdcop_signal_confidence` | Gauge | trading-api | Current signal confidence |
| `usdcop_consecutive_losses` | Gauge | trading-api | Consecutive losing trades |
| `usdcop_daily_pnl` | Gauge | trading-api | Daily P&L percentage |
| `usdcop_data_freshness_seconds` | Gauge | trading-api | Market data age |
| `usdcop_last_ohlcv_timestamp` | Gauge | trading-api | Last OHLCV bar timestamp |
| `usdcop_macro_data_age_hours` | Gauge | trading-api | Macro data age |

### Model Metrics

| Metric | Type | Source | Description |
|--------|------|--------|-------------|
| `model_loaded` | Gauge | inference-api | Model loaded flag (by model_type) |
| `model_loaded_timestamp` | Gauge | inference-api | When model was loaded |
| `model_predictions_total` | Counter | inference-api | Predictions by action |
| `prediction_latency_seconds` | Histogram | inference-api | Prediction latency |
| `model_reload_total` | Counter | inference-api | Reload attempts |
| `model_reload_failures_total` | Counter | inference-api | Failed reloads |
| `shadow_divergence_rate` | Gauge | inference-api | Champion vs shadow divergence |
| `usdcop_active_models` | Gauge | inference-api | Active model count |
| `usdcop_model_drift_score` | Gauge | inference-api | Model drift score |
| `usdcop_model_accuracy_rolling_7d` | Gauge | inference-api | 7-day rolling accuracy |

### Feature Metrics

| Metric | Type | Source | Description |
|--------|------|--------|-------------|
| `usdcop_feature_null_ratio` | Gauge | inference-api | Null ratio in features |
| `usdcop_feature_zscore` | Gauge | inference-api | Feature z-score (by feature) |
| `usdcop_feature_nan_ratio` | Gauge | inference-api | NaN ratio |
| `usdcop_circuit_breaker_state` | Gauge | inference-api | Circuit breaker state |
| `feature_drift_alert` | Gauge | inference-api | Drift alert flag (by feature) |
| `feature_drift_score` | Gauge | inference-api | KS statistic (by feature) |
| `feature_drift_pvalue` | Gauge | inference-api | KS p-value (by feature) |
| `feature_drift_checks_total` | Counter | inference-api | Total drift checks |

### Infrastructure Metrics

| Metric | Type | Source | Description |
|--------|------|--------|-------------|
| `container_memory_usage_bytes` | Gauge | Docker | Container memory |
| `container_cpu_usage_seconds_total` | Counter | Docker | Container CPU |
| `container_restart_count` | Counter | Docker | Container restarts |
| `pg_stat_activity_count` | Gauge | postgres-exporter | Active DB connections |
| `pg_settings_max_connections` | Gauge | postgres-exporter | Max DB connections |
| `redis_memory_used_bytes` | Gauge | redis | Redis memory usage |
| `redis_memory_max_bytes` | Gauge | redis | Redis max memory |
| `redis_rejected_connections_total` | Counter | redis | Rejected connections |
| `node_filesystem_avail_bytes` | Gauge | node-exporter | Disk space available |

### Airflow Metrics

| Metric | Type | Source | Description |
|--------|------|--------|-------------|
| `airflow_dag_run_failed_total` | Counter | airflow | Failed DAG runs |
| `airflow_scheduler_heartbeat` | Gauge | airflow | Scheduler alive |
| `usdcop_l0_pipeline_last_success_timestamp` | Gauge | custom | Last L0 success |

### Inference Latency Metrics

| Metric | Type | Source | Description |
|--------|------|--------|-------------|
| `usdcop_inference_latency_seconds` | Histogram | inference-api | End-to-end inference |
| `usdcop_inference_requests_total` | Counter | inference-api | Total requests (by status) |
| `usdcop_feature_calculation_seconds` | Histogram | inference-api | Feature calc time (by set) |
| `usdcop_db_query_seconds` | Histogram | inference-api | DB query time |
| `usdcop_model_inference_errors_total` | Counter | inference-api | Inference errors |
| `usdcop_feature_calculation_errors_total` | Counter | inference-api | Feature calc errors |
| `usdcop_feature_drift` | Gauge | inference-api | Feature drift z-score |

---

## Operational Procedures

### Silencing Alerts

During planned maintenance (e.g., DB migration, model retraining):

```bash
# Via AlertManager UI
open http://localhost:9093/#/silences/new

# Silence by alertname for 2 hours
# Filter: alertname="MarketDataStale"
# Duration: 2h
# Comment: "Planned OHLCV backfill"
```

### Checking Alert Status

```bash
# Active alerts
curl -s http://localhost:9090/api/v1/alerts | python -m json.tool

# Firing alerts only
curl -s http://localhost:9093/api/v2/alerts?filter=active | python -m json.tool

# Watchdog check (should always be firing)
curl -s http://localhost:9090/api/v1/alerts | python -c "
import json,sys
alerts = json.load(sys.stdin)['data']['alerts']
watchdog = [a for a in alerts if a['labels']['alertname']=='Watchdog']
print('Watchdog:', 'OK' if watchdog else 'MISSING')
"
```

### Querying Logs

```bash
# Via Loki API (last 1h of errors)
curl -G http://localhost:3100/loki/api/v1/query_range \
  --data-urlencode 'query={level="ERROR"}' \
  --data-urlencode 'start='$(date -d '1 hour ago' +%s)000000000 \
  --data-urlencode 'end='$(date +%s)000000000

# Via Grafana Explore UI
open http://localhost:3002/explore
# Select Loki datasource, enter LogQL query:
# {service="airflow"} |= "ERROR"
# {container=~"usdcop-.*"} | json | level="ERROR"
```

### Reloading Prometheus Config

```bash
# Hot-reload after rule file changes (--web.enable-lifecycle flag is set)
curl -X POST http://localhost:9090/-/reload
```

---

## Integration with Trading System

### How Alerts Map to Trading Pipelines

| Pipeline | Relevant Alerts | Action on Fire |
|----------|----------------|----------------|
| L0 OHLCV | `MarketDataStale`, `OHLCVDataGap`, `L0PipelineDelayed` | Trigger backfill DAG |
| L0 Macro | `MacroDataMissing` | Trigger macro backfill DAG |
| H1/H5 Training | `AirflowDAGFailed`, `AirflowSchedulerUnhealthy` | Check L3 DAG logs |
| H1 Inference | `NoModelsLoaded`, `ModelInferenceErrors`, `ChampionModelNotLoaded` | Check L5 DAG, reload model |
| H5 Execution | `ConsecutiveLossesExceeded`, `DailyLossLimitBreached` | Circuit breaker activates |
| Feature Store | `FeatureCircuitBreakerActivated`, `HighNullRatioInFeatures` | Pause trading, fix data |
| Model Health | `ModelDriftDetected`, `ModelAccuracyDegraded`, `MultipleFeaturesDrifted` | Consider retraining |

### Relationship to Data Freshness (CTR-DQ-OPS-001)

The observability stack complements the Airflow-based data freshness gates:

| Concern | Airflow Gate | Prometheus Alert |
|---------|-------------|-----------------|
| OHLCV freshness | `validate_training_data_freshness()` (BLOCKING) | `MarketDataStale` (WARNING) |
| Macro freshness | `validate_training_data_freshness()` (BLOCKING) | `MacroDataMissing` (WARNING) |
| Model freshness | `check_model_freshness()` (WARNING) | `ChampionModelStale` (INFO) |
| Feature quality | -- | `FeatureCircuitBreakerActivated` (CRITICAL) |

Airflow gates prevent bad training. Prometheus alerts provide real-time visibility.

---

## Related Specs

- `data-freshness-enforcement.md` -- Data freshness thresholds and automated recovery
- `backup-recovery-protocol.md` -- Backup architecture and disaster recovery
- `elite-operations.md` -- Operational rulebook, schedule coordination, monitoring checklist
- `l0-data-governance.md` -- L0 OHLCV + macro pipeline (data that alerts monitor)
- `l1-l5-inference-pipeline.md` -- Inference pipeline (model alerts)

---

## DO NOT

- Do NOT disable the Watchdog alert -- it proves the alerting pipeline is functional
- Do NOT remove inhibition rules -- they prevent alert storms during cascading failures
- Do NOT set `resolve_timeout` below 5m -- transient flaps will flood channels
- Do NOT add PagerDuty to non-critical alerts -- PagerDuty is reserved for trading-impacting events
- Do NOT modify Prometheus rule files without running `promtool check rules` first
- Do NOT expose Prometheus, AlertManager, or Loki ports externally without authentication
- Do NOT increase Loki retention beyond 31 days without checking disk capacity
- Do NOT skip the Promtail Docker socket mount -- it is required for container log discovery
- Do NOT configure Jaeger with persistent storage until trace instrumentation is implemented
- Do NOT use Grafana anonymous access in production -- keep `GF_AUTH_ANONYMOUS_ENABLED=false`
- Do NOT hardcode Slack webhook URLs in alertmanager.yml -- use `${SLACK_WEBHOOK_URL}` env var
- Do NOT delete Grafana dashboard JSON files -- they are auto-provisioned on startup
