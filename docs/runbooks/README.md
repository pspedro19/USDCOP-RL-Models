# Operational Runbooks

Quick-reference playbooks for the top 5 failure scenarios in the USDCOP trading system.

---

## 1. Prometheus Target DOWN

**Symptom**: Grafana dashboards show "No Data". Prometheus targets page shows `DOWN`.

**Diagnosis**:
```bash
# Check which targets are down
curl -s http://localhost:9090/api/v1/targets | python -c "
import sys, json
targets = json.load(sys.stdin)['data']['activeTargets']
for t in targets:
    print(f\"{t['labels']['job']:25s} {t['health']:10s} {t.get('lastError','')[:60]}\")
"

# Check if the service container is running
docker ps --filter "name=usdcop-" --format "table {{.Names}}\t{{.Status}}"

# Check service logs
docker logs usdcop-trading-api --tail 20
```

**Fix**:
```bash
# Restart the affected service
docker compose restart trading-api

# If /metrics endpoint missing (404), rebuild with observability wiring
docker compose up -d --build trading-api
```

**Verification**:
```bash
# Confirm target is UP
curl -s http://localhost:9090/api/v1/targets | python -c "
import sys, json
[print(t['labels']['job'], t['health']) for t in json.load(sys.stdin)['data']['activeTargets']]
"
```

**Prevention**: All FastAPI services must call `setup_prometheus_metrics(app, "service-name")` after `app = FastAPI(...)`.

---

## 2. OHLCV Data Stale (Training Blocked)

**Symptom**: H1-L3 or H5-L3 training DAG fails with `OHLCV freshness: age=Xd, threshold=3d`. Alert: `MarketDataStale`.

**Diagnosis**:
```bash
# Check latest OHLCV timestamp
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading \
  -c "SELECT symbol, MAX(time) as latest, COUNT(*) FROM usdcop_m5_ohlcv GROUP BY symbol;"

# Check realtime DAG status
docker exec usdcop-airflow-scheduler airflow dags list-runs -d core_l0_02_ohlcv_realtime --limit 5

# Check if it's a weekend/holiday (no data expected)
date +%u  # 6=Sat, 7=Sun
```

**Fix**:
```bash
# Trigger backfill to fill the gap
airflow dags trigger core_l0_01_ohlcv_backfill

# If macro is also stale
airflow dags trigger core_l0_03_macro_backfill
```

**Verification**:
```bash
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading \
  -c "SELECT symbol, MAX(time) as latest FROM usdcop_m5_ohlcv GROUP BY symbol;"
# latest should be within 3 days of now (excluding weekends)
```

**Prevention**: `core_l0_02_ohlcv_realtime` runs every 5 min Mon-Fri during market hours. Check Airflow scheduler health if it stops.

---

## 3. Model Freshness >10 Days

**Symptom**: H1-L5 inference logs `MODEL FRESHNESS WARNING: models are X days old`. Alert: `ChampionModelStale`.

**Diagnosis**:
```bash
# Check model file timestamps
ls -la /opt/airflow/outputs/forecasting/h1_daily_models/latest/*.pkl 2>/dev/null
ls -la /opt/airflow/outputs/forecasting/h5_weekly_models/latest/*.pkl 2>/dev/null

# Check why L3 training hasn't run
docker exec usdcop-airflow-scheduler airflow dags list-runs -d forecast_h1_l3_weekly_training --limit 3
docker exec usdcop-airflow-scheduler airflow dags list-runs -d forecast_h5_l3_weekly_training --limit 3
```

**Fix**:
```bash
# If data is fresh, re-trigger training
airflow dags trigger forecast_h1_l3_weekly_training
airflow dags trigger forecast_h5_l3_weekly_training

# If data is stale, fix data first (see Runbook #2), then retrain
```

**Verification**:
```bash
ls -la /opt/airflow/outputs/forecasting/h1_daily_models/latest/*.pkl
# Modified date should be within last 10 days
```

**Prevention**: L3 training runs every Sunday at 01:00-01:30 COT. Models retrain weekly on expanding window.

---

## 4. Kill Switch Activated

**Symptom**: All trading halted. SignalBridge returns `KILLED` mode. Alert: `KillSwitchActivated`.

**Diagnosis**:
```bash
# Check kill switch status via API
curl -s http://localhost:8085/api/signal-bridge/status | python -m json.tool

# Check audit trail in DB
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading \
  -c "SELECT * FROM audit.get_kill_switch_status();"

# Check what triggered it
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading \
  -c "SELECT action, reason, triggered_by, created_at FROM audit.kill_switch_audit ORDER BY created_at DESC LIMIT 5;"
```

**Fix**:
```bash
# Only reset after confirming the triggering condition is resolved
curl -X POST http://localhost:8085/api/signal-bridge/kill-switch \
  -H "Content-Type: application/json" \
  -d '{"activate": false, "reason": "Condition resolved - [describe]", "confirm": true}'
```

**Verification**:
```bash
curl -s http://localhost:8085/api/signal-bridge/status | python -c "
import sys, json; d = json.load(sys.stdin); print('Mode:', d.get('trading_mode'), 'Kill:', d.get('kill_switch_active'))
"
# Expected: Mode: PAPER, Kill: False
```

**Prevention**: Kill switch triggers on drawdown >= 15%. Monitor daily P&L and reduce position sizing if approaching threshold.

---

## 5. Container Restart Loop

**Symptom**: `docker ps` shows container restarting. Alert: `HighRestartRate`.

**Diagnosis**:
```bash
# Find which container is restarting
docker ps --filter "name=usdcop-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check logs for the failing container
docker logs usdcop-<name> --tail 50

# Common causes:
# - Port conflict (another process using the port)
# - Missing dependency (DB not ready)
# - OOM (check memory limits in docker-compose.yml)
# - Missing config/secrets files
```

**Fix**:
```bash
# Port conflict: find what's using the port
netstat -ano | findstr :8000

# Dependency not ready: restart with proper ordering
docker compose up -d

# OOM: increase memory limit in docker-compose.yml deploy.resources.limits.memory

# Missing secrets: regenerate
python scripts/generate_secrets.py
```

**Verification**:
```bash
# Container should be running without restarts
docker ps --filter "name=usdcop-<name>" --format "{{.Status}}"
# Expected: "Up X minutes" (not "Restarting")

# Health check passes
docker inspect --format='{{.State.Health.Status}}' usdcop-<name>
# Expected: "healthy"
```

**Prevention**: Always check `docker compose up -d` output for errors. Use `docker compose logs -f <service>` to monitor startup.
