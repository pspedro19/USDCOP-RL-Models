---
kind: as-built
status: IMPLEMENTED
version: 1.0.0
last_verified: 2026-09-10
supersedes: []
code_anchors:
  - config/execution/smart_simple_v1.yaml
  - database/migrations/043_forecast_h5_tables.sql
  - airflow/dags/forecast_h5_l3_weekly_training.py
  - airflow/dags/forecast_h5_l4_backtest_promotion.py
  - airflow/dags/forecast_h5_l4b_production_deploy.py
  - airflow/dags/forecast_h5_l5_weekly_signal.py
  - airflow/dags/forecast_h5_l5_vol_targeting.py
  - airflow/dags/forecast_h5_l7_multiday_executor.py
  - airflow/dags/forecast_h5_l6_weekly_monitor.py
---
# Rule: H5 Weekly Smart Simple Pipeline

> Governs the H5 (5-day horizon) weekly forecasting pipeline with Smart Simple v2.0 execution.
> This is the PRIMARY production track. H1 daily pipeline is PAUSED.
> Created: 2026-02-16 | Updated: 2026-04-06 (v2.0 — regime gate + effective HS + dynamic leverage)
> | 2026-09-10 (KPIs re-anchored to the official 2026-07-21 bundle; XGBoost off; L4b deploy DAG)

---

## v2.0 Changes (2026-03-18 audit → 2026-04-06 deployment)

A 10-agent audit revealed:
- Ridge/BR model has R² < 0 in both 2025 and 2026 (worse than predicting the mean)
- Model alpha vs Always SHORT is NEGATIVE (-4.33 pp even in 2025)
- Alpha comes from regime gate (knows when NOT to trade) + TP/HS mechanics
- Q1 2026: Hurst = 0.28 (mean-reverting). Gate correctly blocked 13 of 14 weeks.

v2.0 additions: Regime Gate (Hurst R/S), Effective HS (portfolio cap), Dynamic Leverage,
vol_regime_ratio + trend_slope_60d features, weekly retraining restored. XGBoost = offline
experiment only, never promoted (`use_xgboost: false`, `smart_simple_v1.yaml:206`): the live
ensemble is Ridge + BayesianRidge.

---

## Architecture

```
Sunday 01:30 COT
    |
    v
H5-L3: Weekly Training (forecast_h5_l3_weekly_training.py)
    |  Train Ridge + BayesianRidge on expanding window (2020 -> last Friday)
    |  (XGBoost = offline experiment, use_xgboost: false — not in the live ensemble)
    |  23 features (21 base + vol_regime_ratio + trend_slope_60d)
    |  target = ln(close[t+5]/close[t])
    |  Write to forecast_h5_predictions
    |
Monday 08:15 COT
    |
    v
H5-L5: Weekly Signal (forecast_h5_l5_weekly_signal.py)
    |  Generate ensemble prediction (mean of Ridge + BR; XGB off, use_xgboost: false)
    |  Score confidence (3-tier: HIGH/MEDIUM/LOW)
    |  Skip LOW-confidence LONGs
    |  Write to forecast_h5_signals
    |
Monday 08:45 COT
    |
    v
H5-L5: Vol-Targeting + Regime Gate (forecast_h5_l5_vol_targeting.py) [v2.0]
    |  1. Compute realized vol (21d lookback)
    |  2. Base leverage from vol-targeting (tv=0.15)
    |  3. Apply asymmetric sizing + confidence multiplier
    |  4. **REGIME GATE**: Compute Hurst(60d), classify regime
    |     - TRENDING (H>0.52): sizing × 1.0 (full)
    |     - INDETERMINATE (0.42-0.52): sizing × 0.40
    |     - MEAN-REVERTING (H<0.42): skip_trade=True (DO NOT TRADE)
    |  5. **DYNAMIC LEVERAGE**: Scale by rolling WR + drawdown [0.25, 1.0]
    |  6. **EFFECTIVE HS**: min(HS_base, 3.5% / leverage) — caps portfolio loss
    |  Write adjusted_leverage, hard_stop_pct, take_profit_pct, regime, hurst to forecast_h5_signals
    |
Monday 09:00 COT
    |
    v
H5-L7: Entry (forecast_h5_l7_multiday_executor.py)
    |  Read signal + stops from DB
    |  Place limit entry order (0% maker fee)
    |  Write to forecast_h5_executions + forecast_h5_subtrades
    |
Mon-Fri */30 08:00-12:55 COT  (cron `*/30 13-17 * * 1-5`)
    |
    v
H5-L7: Monitor TP/HS (same DAG, monitor_position task)
    |  Check hard_stop: direction=SHORT -> bar_high >= entry * (1 + HS%)
    |  Check take_profit: direction=SHORT -> bar_low <= entry * (1 - TP%)
    |  If hit: close subtrade, update execution status
    |  No trailing stop, no re-entry
    |
Friday 12:50 COT
    |
    v
H5-L7: Friday Close (same DAG, close_week task)
    |  Market order to close remaining position
    |  Update execution status + subtrade
    |
Friday 14:30 COT
    |
    v
H5-L6: Weekly Monitor (forecast_h5_l6_weekly_monitor.py)
    |  Calculate DA, Sharpe, MaxDD
    |  Check guardrails (circuit breaker, long insistence, rolling DA)
    |  Write to forecast_h5_paper_trading
    |  Evaluate promotion gates (week 15)
```

---

## Smart Simple v2.0 Config

**File**: `config/execution/smart_simple_v1.yaml` (file name is historical; `version: 2.0.0`,
`use_xgboost: false`)

### Key Parameters (DO NOT change without backtest evidence)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `vol_multiplier` | **2.0** | Wider stops eliminate hard stops (v1.0 had 2, v1.1 has 0) |
| `tp_ratio` | 0.5 | TP = HS * 0.5 (sweet spot from sensitivity analysis) |
| `hard_stop_min_pct` | 1% | Floor for calm markets |
| `hard_stop_max_pct` | 3% | Cap for volatile markets |
| SHORT sizing | **flat 1.5x** | Scorer doesn't discriminate (LOW WR 79% > MEDIUM 60%, N=24) |
| LONG LOW sizing | **0.0 (SKIP)** | Net if taken = -0.75%, correct to skip |
| LONG HIGH/MED sizing | 1.0x / 0.5x | Exploratory, small size |
| `target_vol` | 0.15 | 15% annualized target |
| `max_leverage` | 2.0 | Absolute ceiling |
| `min_leverage` | 0.5 | Absolute floor |

### Confidence Scoring

Inputs: Ridge prediction, BayesianRidge prediction, ensemble direction.

| Tier | Condition | SHORT mult | LONG mult |
|------|-----------|------------|-----------|
| HIGH | Tight agreement + high magnitude | 1.5x | 1.0x |
| MEDIUM | Loose agreement OR medium magnitude | 1.5x | 0.5x |
| LOW | Neither | 1.5x | **SKIP** |

Agreement = `|ridge_pred - br_pred|`. Tight < 0.1%, Loose < 0.5%.
Magnitude = `|ensemble_mean|`. High > 1.0%, Medium > 0.5%.

### Adaptive Stops Formula

```python
vol_weekly = realized_vol_annualized * sqrt(5/252)
hard_stop_pct = clamp(vol_weekly * vol_multiplier, min_pct, max_pct)
take_profit_pct = hard_stop_pct * tp_ratio
```

Examples (vol_multiplier=2.0):
- vol_ann=10% -> HS=2.81%, TP=1.41%
- vol_ann=15% -> HS=3.00%, TP=1.50% (capped)
- vol_ann= 5% -> HS=1.41%, TP=0.70%

---

## Database Tables

**Migration**: `database/migrations/043_forecast_h5_tables.sql` + `044_smart_simple_columns.sql` + `049_regime_gate_columns.sql` + `054_h5_subtrades_unique.sql` (**required for `--seed-db`**: adds `UNIQUE(execution_id, subtrade_index)` so the subtrades `ON CONFLICT` upsert works — without it the whole seed transaction rolls back and all 5 tables stay empty)

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `forecast_h5_predictions` | Model predictions per week | week, model_name, pred_return, pred_direction |
| `forecast_h5_signals` | Ensemble signal + confidence + stops | week, ensemble_return, direction, confidence_tier, sizing_multiplier, skip_trade, hard_stop_pct, take_profit_pct, adjusted_leverage |
| `forecast_h5_executions` | Weekly execution tracking | week, direction, entry_price, exit_price, exit_reason, pnl_pct, confidence_tier |
| `forecast_h5_subtrades` | Individual subtrade records | execution_id, entry_bar, exit_bar, entry_price, exit_price |
| `forecast_h5_paper_trading` | Weekly paper trading evaluation | week, da_pct, sharpe, max_dd, gate_status |

**Views**:
- `v_h5_performance_summary` — Aggregated performance metrics
- `v_h5_collapse_monitor` — Direction collapse detection

---

## DAG Files

| DAG | File | Schedule (COT) |
|-----|------|-----------------|
| H5-L3 Training | `airflow/dags/forecast_h5_l3_weekly_training.py` | Sun 01:30 |
| H5-L4 Backtest Promotion | `airflow/dags/forecast_h5_l4_backtest_promotion.py` | Manual / event-driven (Vote 1) |
| H5-L4b Production Deploy | `airflow/dags/forecast_h5_l4b_production_deploy.py` | Event-driven, post-Vote-2 (dashboard → Airflow REST; `guard_approved` re-checks `status == APPROVED` server-side) |
| H5-L5 Signal | `airflow/dags/forecast_h5_l5_weekly_signal.py` | Mon 08:15 |
| H5-L5 Vol-Target | `airflow/dags/forecast_h5_l5_vol_targeting.py` | Mon 08:45 |
| H5-L7 Executor | `airflow/dags/forecast_h5_l7_multiday_executor.py` | Mon-Fri */30 08:00-12:55 (`*/30 13-17 * * 1-5`) |
| H5-L6 Monitor | `airflow/dags/forecast_h5_l6_weekly_monitor.py` | Fri 14:30 |

---

## Guardrails

| Guardrail | Config Key | Threshold | Action |
|-----------|-----------|-----------|--------|
| Long insistence | `guardrails.long_insistence_alarm` | >60% LONGs in 8-week window | Alert only |
| Rolling DA (SHORT) | `guardrails.rolling_da_monitor` | SHORT DA < 55% in 16 weeks | Pause SHORTs |
| Rolling DA (LONG) | `guardrails.rolling_da_monitor` | LONG DA < 45% in 16 weeks | Pause LONGs |
| Circuit breaker | `guardrails.circuit_breaker` | 5 consecutive losses OR 12% DD | Pause + alert |

---

## Promotion Gates (Week 15)

| Gate | Threshold | Action if PASS | Action if FAIL |
|------|-----------|----------------|----------------|
| DA overall > 55% AND DA SHORT > 60% | Promote | Check keep conditions |
| DA overall < 50% | -- | Discard |
| SHORT DA > 60% but LONG DA < 45% | -- | Switch to SHORT-only |

---

## Backtest Results (v2.0, 2025 OOS — official bundle)

> **Source of truth**: `usdcop-trading-dashboard/public/data/production/summary_2025.json`
> (generated 2026-07-21) = [HYPOTHESIS-REGISTRY.md](../assets/usdcop/HYPOTHESIS-REGISTRY.md)
> § "RE-MEDICIÓN #3". Honesty cascade: +26.05 → +13.05 (data fix) → +7.66 (purge) → **+7.35**
> (open-aware HS fills). The earlier **+25.63% / Sharpe 3.35 / p=0.006 / 34 trades / $12,563**
> and **+20.03% / p=0.0097 / 24 trades** figures are **SUPERSEDED** — do not quote them.

| Metric | v2.0 (official, 2026-07-21) |
|--------|-----------------------------|
| Return | **+7.35%** |
| Sharpe | 0.942 |
| p-value | 0.2277 — **NOT statistically significant** |
| MaxDD | 7.84% |
| Trades | 32 (2L / 30S) |
| WR | 71.9% |
| Exits | TP 16 / week_end 11 / HS 5 |
| $10K -> | **$10,734.62** |

Trial-aware DSR = 0.50-0.92 < 0.95 in every scenario (`approval_state.json` gate value 0.0587),
so the 2025 backtest cannot prove edge after selection: **v11 is FROZEN and the 2026 forward is
the only clean judge** (`../../rules/quant-constitution.md` §2, `WITHDRAWAL-PROTOCOL.md`).

### 2026 (v2.0) — two honest series, both must be labeled

| Series | Source | Return | Trades | $10K -> |
|--------|--------|--------|--------|---------|
| (a) Corrected replay of 2026, purged method | `public/data/production/summary.json` (2026-07-21) | +3.36% | 11 (`insufficient_trades: true`) | ≈$10,336 |
| (b) Forward paper ledger, as actually run by the DAGs | `public/data/production/paper/candidates_ledger_2026.json` (2026-08-28) | **+0.66% YTD** | 12 through ISO 2026-W33 (last trade 2026-08-10) | **$10,066** |

Both series have **N < 20 trades**: per `quant-constitution.md` §6 only count and PnL are
reported — **no Sharpe, no p-value**. The regime gate is still what kept 2026 small: Hurst
0.28-0.49 (mean-reverting → transitioning) blocked most weeks. The older "+0.61%, 1 trade,
13 of 14 weeks blocked" reading (2026-04-06) was a point-in-time snapshot superseded by (b).

---

## Dashboard Integration (SDD)

The H5 pipeline exports to the dashboard via the **SDD (Spec-Driven Development)** universal contract.
See these specs for the full data flow:

- `.claude/rules/strategy-contract.md` — Universal strategy schemas (trades, metrics, exit reasons)
- `.claude/specs/platform/dashboard-integration.md` — JSON file schemas, PNG conventions
- `.claude/rules/approval-gates.md` — Approval lifecycle and gates
- `.claude/specs/platform/mlops-lifecycle.md` — Master lifecycle (bootstrap to production, 8 stages)

### Export Script

```bash
python scripts/pipeline/train_and_export_smart_simple.py --phase both                  # Full pipeline
python scripts/pipeline/train_and_export_smart_simple.py --phase backtest              # 2025 OOS only
python scripts/pipeline/train_and_export_smart_simple.py --phase production            # 2026 only
python scripts/pipeline/train_and_export_smart_simple.py --phase production --seed-db  # 2026 + seed DB tables
python scripts/pipeline/train_and_export_smart_simple.py --reset-approval              # Reset to PENDING
```

> `--phase backtest` embeds a `deploy_manifest` in `approval_state.json`. On Approve,
> the deploy API reads the manifest and spawns the correct script with `--seed-db`.
> `--seed-db` UPSERTs the 5 `forecast_h5_*` tables; it resolves the DB connection from
> `DATABASE_URL`, or builds it from `POSTGRES_*` env when `DATABASE_URL` is unset (so it also works
> from the airflow scheduler / data-seeder), and registers numpy→postgres adapters
> (`np.bool_`/`int`/`float`) to avoid "can't adapt type 'numpy.bool_'". Requires migration 054.

### Key Contracts
- TypeScript: `lib/contracts/strategy.contract.ts` (universal types)
- TypeScript: `lib/contracts/production-approval.contract.ts` (DeployManifest, ApprovalState)
- Python: `src/contracts/strategy_schema.py` (mirror types + safe JSON serializer)
- YAML: `config/strategy_registry.yaml` (strategy-to-pipeline mapping reference)

### DB Tables (NOT yet connected to dashboard)
The H5 pipeline also writes to 5 DB tables for production monitoring.
These are read by the Airflow DAGs but NOT by the dashboard frontend:
1. API routes: `/api/h5/signals`, `/api/h5/performance`, `/api/h5/executions`
2. Dashboard page: `app/h5/page.tsx` or section in `/production`
3. Update `/api/production/monitor` to include H5 data

---

## DO NOT

- Do NOT add confidence tiers for SHORT sizing (flat 1.5x is correct, N too small)
- Do NOT reduce HS multiplier below 2.0 (eliminates hard stops)
- Do NOT enable trailing stop in Smart Simple (simplicity is the edge)
- Do NOT take LOW-confidence LONGs (net effect = -0.75%)
- Do NOT hardcode params in scripts (read from smart_simple_v1.yaml)
- Do NOT modify smart_simple_v1.yaml stops without running full comparative backtest
- Do NOT compare H5 vs H1 returns directly (different horizons, different trade frequency)
- Do NOT disable the regime gate — it is the primary alpha source (prevented -5.17% → +0.61%)
- Do NOT use monthly retraining — DEPRECATED since v1.1.1 (methodology gap, never backtested)
- Do NOT use fallback/simulated data in dashboard — always use DATABASE_URL for real prices
- Do NOT trust model predictions alone — R² < 0 in both years, gate + stops provide the real edge
- Do NOT operate without effective HS — portfolio cap at 3.5% prevents catastrophic single-trade losses

---

## Reconciliation (audit 2026-07)

> See [AUDIT-2026-07-remediation.md](../audit/AUDIT-2026-07-remediation.md) §A3.

- **Feature parity fixed (A3-02).** The then-headline **+25.63% / 34-trade** backtest was produced on the **23-feature** v2.0 model (base 21 + `vol_regime_ratio` + `trend_slope_60d`). As of 2026-07 the live weekly **L3 training and L5 signal** also build the identical 23-feature set via the shared `src/forecasting/enhance_v2.py`; L3 persists the trained feature list to `feature_cols_h5.json` and L5 reads it — so live models match the backtest. The A3-02 re-run reproduced that then-headline; it was itself **superseded by re-measurement #3 (2026-07-21)** — data fix, purge and open-aware HS fills brought the official 2025 OOS to **+7.35% / 32 trades** (see [HYPOTHESIS-REGISTRY.md](../assets/usdcop/HYPOTHESIS-REGISTRY.md) § "RE-MEDICIÓN #3" and § Backtest Results above).
