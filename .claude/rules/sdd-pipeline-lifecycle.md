# SDD Spec: Pipeline Lifecycle (Quick Reference)

> Condensed reference for the 8-stage lifecycle, CLI conventions, and DAG schedules.
>
> **Responsibility**: Quick-reference lookup (stage table, CLI commands, DAG schedules).
> For the full operator's guide with checklists and narratives, see `sdd-mlops-lifecycle.md`.

---

## 8-Stage Lifecycle

```
Stage 0: BOOTSTRAP   ──→  Docker + DB + seed restore (one-time)
Stage 1: DATA ALIGN  ──→  L0 backfill DAGs bring OHLCV + macro current
Stage 2: TRAIN       ──→  Train model(s) on historical data (expanding window)
Stage 3: FORECAST    ──→  Dashboard /forecasting activates (model zoo)
Stage 4: BACKTEST    ──→  OOS walk-forward validation on /dashboard
Stage 5: APPROVE     ──→  Human reviews gates, clicks Approve/Reject (Vote 2/2)
Stage 6: DEPLOY      ──→  Retrain on full data, generate production trades
Stage 7: MONITOR     ──→  Airflow weekly cycle with guardrails
```

### Stage Details

| Stage | Actor | Inputs | Outputs | CLI / Trigger |
|-------|-------|--------|---------|---------------|
| BOOTSTRAP | Operator | `docker-compose.yml`, seed parquets | Running DB with historical data | `docker-compose up -d` |
| DATA ALIGN | Airflow / Operator | Empty or stale DB | Current OHLCV + macro data | `airflow dags trigger core_l0_01_ohlcv_backfill` |
| TRAIN | Python script | OHLCV + macro data, config YAML | Models, CSV, PNGs, JSONs | `python scripts/generate_weekly_forecasts.py` |
| FORECAST | Automatic | Generated CSV + PNGs | `/forecasting` page active | (none — dashboard reads files) |
| BACKTEST | Dashboard | `summary_2025.json`, trades, gates | Human-reviewed evidence | Navigate to `/dashboard` |
| APPROVE | Human | Gate results, KPIs, trades | `approval_state.json` updated | Click Approve/Reject on `/dashboard` |
| DEPLOY | Python script | Approved model, full data | `summary.json`, production trades | `--phase production` |
| MONITOR | Airflow DAGs | Live trades, gate thresholds | Alerts, circuit breaker actions | Automated (weekly/daily schedule) |

---

## CLI Convention

Each strategy has a pipeline script with `--phase` argument:

```bash
# Stages 2-4 (Train + Backtest export → Stage 3 activates automatically):
python scripts/{strategy}_pipeline.py --phase backtest

# Stage 6 (Deploy — retrain on full data, generate production trades):
python scripts/{strategy}_pipeline.py --phase production

# Stages 2-4 + 6 (Full pipeline):
python scripts/{strategy}_pipeline.py --phase both

# Reset approval to PENDING_APPROVAL (re-evaluate after changes):
python scripts/{strategy}_pipeline.py --reset-approval

# Skip PNG generation (faster iteration):
python scripts/{strategy}_pipeline.py --phase both --no-png

# Seed DB tables with production results (used by deploy API):
python scripts/{strategy}_pipeline.py --phase production --seed-db
```

### Phase Mapping

| `--phase` | Stages | Notes |
|-----------|--------|-------|
| `backtest` | 2 → 3 (auto) → 4 | Train, backtest export with gates + embed `deploy_manifest` in approval_state.json |
| `production` | 6 | Retrain on full data, production export. Add `--seed-db` for DB seeding |
| `both` | 2 → 3 → 4 → 6 | All of the above |

### Deploy Manifest

Each `--phase backtest` embeds a `deploy_manifest` in `approval_state.json` that tells the
deploy API which script and args to run. The deploy API is strategy-agnostic — it reads the
manifest and spawns the correct pipeline. On approve, deploy fires automatically.

> Stage 3 (Forecasting Dashboard Activation) is automatic — the dashboard reads files
> generated during Stage 2. No separate CLI action required.

### Current Implementations

| Strategy | Script | Config YAML |
|----------|--------|-------------|
| Smart Simple v1.1 | `scripts/train_and_export_smart_simple.py` | `config/execution/smart_simple_v1.yaml` |
| H1 Forecast + VT + Trail | `scripts/generate_weekly_forecasts.py` | `config/execution/smart_executor_v1.yaml` |

---

## Operator Checklists

For detailed per-stage checklists, see `sdd-mlops-lifecycle.md` (each stage has a checklist section).

---

## Approval Integration

> **IMPORTANT**: Vote 2 (human review) happens on `/dashboard`, NOT on `/production`.

For the full two-vote approval system, gate definitions, approval_state.json schema,
and dashboard components, see `sdd-approval-spec.md`.

---

## Monitoring (Stage 7)

### Active Monitor DAGs

| Strategy | Monitor DAG | Schedule |
|----------|-------------|----------|
| Smart Simple v1.1 (H5) | `forecast_h5_l6_weekly_monitor.py` | Fri 14:30 COT |
| Forecast VT+Trail (H1) | `forecast_h1_l6_paper_monitor.py` | Mon-Fri 19:00 COT |

### Full Weekly DAG Schedule

**H5 Weekly Pipeline** (6 DAGs):

| Day | Time (COT) | DAG | File | Action |
|-----|-----------|-----|------|--------|
| Sun | 01:30 | H5-L3 Training | `forecast_h5_l3_weekly_training.py` | Retrain Ridge+BR (expanding window) |
| Mon | Manual | H5-L4 Backtest | `forecast_h5_l4_backtest_promotion.py` | OOS backtest + dashboard export |
| Mon | 08:15 | H5-L5 Signal | `forecast_h5_l5_weekly_signal.py` | Ensemble + confidence scoring |
| Mon | 08:45 | H5-L5 Vol-Target | `forecast_h5_l5_vol_targeting.py` | Adaptive stops + sizing |
| Mon-Fri | */30 9-13 | H5-L7 Executor | `forecast_h5_l7_multiday_executor.py` | TP/HS monitor + Friday close |
| Fri | 14:30 | H5-L6 Monitor | `forecast_h5_l6_weekly_monitor.py` | Weekly evaluation + guardrails |

**H1 Daily Pipeline** (5 DAGs):

| Day | Time (COT) | DAG | File | Action |
|-----|-----------|-----|------|--------|
| Sun | 01:00 | H1-L3 Training | `forecast_h1_l3_weekly_training.py` | Retrain 9 models |
| Mon-Fri | 13:00 | H1-L5 Inference | `forecast_h1_l5_daily_inference.py` | Ensemble top_3 signal |
| Mon-Fri | 13:30 | H1-L5 Vol-Target | `forecast_h1_l5_vol_targeting.py` | Position sizing + trailing params |
| Mon-Fri | 13:35 | H1-L7 Executor | `forecast_h1_l7_smart_executor.py` | Place SHORT + trail |
| Mon-Fri | 19:00 | H1-L6 Monitor | `forecast_h1_l6_paper_monitor.py` | Paper trading results |

### Guardrails

| Guardrail | Trigger | Action |
|-----------|---------|--------|
| Circuit breaker | 5 consecutive losses OR 12% drawdown | Pause trading + alert |
| Long insistence | > 60% LONGs in 8-week window | Alert only |
| Rolling DA (SHORT) | SHORT DA < 55% in 16-week window | Pause SHORTs |
| Rolling DA (LONG) | LONG DA < 45% in 16-week window | Pause LONGs |

---

## Related Specs

- `sdd-mlops-lifecycle.md` — **Master lifecycle document** (Stages 0-7)
- `sdd-strategy-spec.md` — Universal strategy schemas + StrategySelector contract
- `sdd-approval-spec.md` — Approval gates and lifecycle
- `sdd-dashboard-integration.md` — JSON/PNG file schemas, page data flows
- `h5-smart-simple-pipeline.md` — H5 weekly pipeline architecture
