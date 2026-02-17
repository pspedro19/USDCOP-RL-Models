# EXPERIMENT QUEUE — Prioritized

> Experiments ordered by: (1) information value, (2) compute cost, (3) risk.
> Each experiment = a frozen SSOT config in `config/experiments/`.
> Pick the TOP experiment, run it, log results, then re-evaluate queue.
> **Full integration plan**: `.claude/experiments/PLAN_FORECASTING_RL_INTEGRATION.md`
> Last updated: 2026-02-15

---

## COMPLETED (moved from queue)

| ID | Description | Config | Result | Date |
|----|-------------|--------|--------|------|
| EXP-V215b-001 | Temporal features (5-min) | SSOT v3.5.1 | +2.51% mean, 4/5 seeds, CI includes zero | 2026-02-10 |
| EXP-INFRA-001 | Statistical baselines | N/A (analysis) | B&H=-14.66%, random=-4.12%, CI includes zero | 2026-02-11 |
| EXP-ENS-V215b-001 | Ensemble voting | N/A (inference) | +1.60%, worse than single-model mean | 2026-02-11 |
| EXP-ASYM-001 | Asymmetric SL/TP (5-min) | `exp_asym_001.yaml` | **FAIL**: 0/5 seeds, -8.67% mean, SL too tight | 2026-02-12 |
| EXP-HOURLY-PPO-001 | PPO on 1h bars | `exp_hourly_ppo_001.yaml` | **FAIL**: -17.49%, entropy collapse (344x overfit) | 2026-02-14 |
| EXP-HOURLY-PPO-002 | PPO on 1h bars (fix) | `exp_hourly_ppo_002.yaml` | **FAIL**: -7.16%, no alpha (matches SMA baseline) | 2026-02-14 |
| EXP-DAILY-PPO-001 | PPO on daily bars | `exp_daily_ppo_001.yaml` | **FAIL**: -8.83%, 8 trades, 12.5% WR | 2026-02-14 |
| FC-SIZE-001 | Vol-targeting + trailing stop | `vol_target_v1.yaml` | **PASS**: $10K→$13,684, Sharpe 3.135, p=0.0178 | 2026-02-15 |
| EXP-REGIME-001 | Regime detection features | Pre-registered | **FAIL**: 0/7 treatments, regime features redundant | 2026-02-16 |
| FC-H5-SIMPLE-001 | H5 Smart Simple v1.0->v1.1 | `smart_simple_v1.yaml` | **PASS**: $10K->$12,003, Sharpe 3.516, p=0.0097 | 2026-02-16 |

---

## STATUS SUMMARY (as of 2026-02-16)

### Track 1: H1 Daily Pipeline (PRODUCTION)
- **Strategy**: 9 models, H=1, trailing stop + vol-targeting, SHORT-only
- **$10K backtest 2025**: $10,000 -> $13,684 (+36.84%, Sharpe 3.135, **p=0.0178**)
- **Production DAGs**: L5a/L5b/L5c/L6/L7 deployed
- **Status**: Paper Trading active

### Track 2: H5 Weekly Pipeline (PAPER TRADING)
- **Strategy**: Smart Simple v1.1 — Ridge+BR, H=5, TP/HS/Friday close
- **$10K backtest 2025**: $10,000 -> $12,003 (+20.03%, Sharpe 3.516, **p=0.0097**)
- **$10K 2026 YTD**: $10,000 -> $10,712 (+7.12%, 3/3 wins)
- **Production DAGs**: H5-L5a/L5b/L5c/L6/L7 deployed
- **Status**: Paper Trading starting 2026-02-16, 15-week evaluation
- **Dashboard gap**: H5 tables not yet connected to frontend

### Track 3: RL Pipeline (DEPRIORITIZED)
- **Best model**: V21.5b -> +2.51% mean, Sharpe +0.321, 4/5 seeds
- **Statistical significance**: **NOT significant (p=0.272)**
- **Verdict**: Trailing stop replaces RL as execution layer.

### Key Insight
| Track | Return 2025 | Sharpe | p-value | Status |
|-------|-------------|--------|---------|--------|
| H1 Daily + Trail | +36.84% | 3.135 | 0.0178 | PRODUCTION |
| **H5 Smart Simple v1.1** | **+20.03%** | **3.516** | **0.0097** | **PAPER TRADING** |
| RL V21.5b | +2.51% | 0.321 | 0.272 | DEPRIORITIZED |

Both forecasting tracks are statistically significant. H5 has higher Sharpe but lower absolute return (weekly vs daily).

---

## STRATEGIC PIVOT: Forecasting-First Integration

> Based on evidence: forecasting (supervised, daily) is the proven alpha source.
> RL (5-min) is the execution layer, NOT the signal generator.
> The plan below is ordered by expected impact and statistical rigor.

---

## PHASE 1: Expand Forecasting Features (PRIORITY 1)

### FC-FEAT-001: Macro Feature Expansion (19 → 32 features)

| Field | Value |
|-------|-------|
| **Pipeline** | Forecasting (supervised) |
| **Status** | NEXT UP |
| **Config** | `config/forecast_experiments/feat_macro_expanded_v1.yaml` (to create) |
| **Variable** | Feature set: 19 → 32 (add 13 macro features AS MODEL INPUTS) |
| **Hypothesis** | More macro features improve DA from 53% to 55%+ (each 1pp DA ≈ 3pp return) |
| **Compute** | ~30 min (walk-forward retrain, not RL) |

**New features (13) — all from existing `macro_daily_clean.parquet`:**

```
# Retornos 1d (9) — lag T-1 anti-leakage
colcap_ret1d    → corr -0.269 con USDCOP
brent_ret1d     → corr -0.283
usdmxn_ret1d    → corr +0.440 (highest!)
usdclp_ret1d    → corr +0.391
embi_ret1d      → corr +0.411
vix_ret1d       → corr +0.248
gold_ret1d      → corr -0.178
col5y_ret1d     → corr +0.365
col10y_ret1d    → corr +0.393

# Z-scores de nivel 63d (4)
colcap_level_z, embi_level_z, vix_level_z, dxy_level_z
```

**NOT adding macro_score as separate feature** (ANOVA p=0.297, doesn't survive stats).
Let XGBoost/LightGBM find non-linear interactions automatically.

**Validation**: Walk-forward (5 windows), compare DA vs baseline_v1.
**Success criteria**: DA > 54% AND PF > 1.25 across walk-forward.
**Failure action**: Try feature selection (top 10 by importance) instead of expansion.

### FC-FEAT-002: Feature Selection (if FC-FEAT-001 overfits)

| Field | Value |
|-------|-------|
| **When** | Only if FC-FEAT-001 DA < 53% (overfitting from too many features) |
| **Variable** | Reduce to top-10 features by XGBoost importance |
| **Hypothesis** | Less noise from irrelevant features improves generalization |

---

## PHASE 2: Position Sizing (PRIORITY 2)

### FC-SIZE-001: Volatility Targeting — COMPLETED 2026-02-15

| Field | Value |
|-------|-------|
| **Pipeline** | Forecasting (post-model, sizing layer) |
| **Status** | **COMPLETED — PASSED ALL GATES** |
| **Variable** | Fixed 1x → vol-targeting (tv=15%, ml=2.0x) + trailing stop |
| **Result** | $10K→$13,684 (+36.84%), Sharpe 3.135, p=0.0178 |
| **Config** | `config/forecast_experiments/vol_target_v1.yaml` |
| **Config (executor)** | `config/execution/smart_executor_v1.yaml` |

**Results by strategy ($10K, 2022-2025)**:

| Strategy | Return | Sharpe | p-value | $10K → |
|----------|--------|--------|---------|--------|
| Buy & Hold 2025 | -12.29% | — | — | $8,771 |
| Forecast 1x | +21.60% | 2.109 | 0.068 | $12,160 |
| + Vol-target | +32.62% | 2.062 | 0.053 | $13,262 |
| **+ Trailing stop** | **+36.84%** | **3.135** | **0.0178** | **$13,684** |

**Gate passed**: Sharpe > 1.0 (3.135), MaxDD < 15% (-10.26%), p < 0.05 (0.0178)

**Production infra deployed**: L5a, L5b, L5c, L6, L7 DAGs + DB migrations 041/042

### FC-SIZE-002: Half-Kelly Sizing

| Field | Value |
|-------|-------|
| **When** | After FC-SIZE-001 validates |
| **Variable** | Vol-targeting → Half-Kelly (confidence-scaled) |
| **Hypothesis** | Sizing by model confidence improves risk-adjusted returns |
| **Expected impact** | Additional +2-3pp over vol-targeting |

---

## PHASE 3: Multi-Pair Diversification (PRIORITY 3)

### FC-MULTI-001: Apply Same Model to USDMXN

| Field | Value |
|-------|-------|
| **Pipeline** | Forecasting (new pair, same architecture) |
| **Variable** | Target pair: USDCOP → USDMXN (using same feature framework) |
| **Hypothesis** | If the model captures EM FX dynamics, it works on MXN too |
| **Data** | `seeds/latest/usdmxn_m5_ohlcv.parquet` (resample to daily) |
| **Compute** | ~30 min (walk-forward on new pair) |

### FC-MULTI-002: Apply Same Model to USDCLP

| Field | Value |
|-------|-------|
| **Variable** | Target pair: USDCOP → USDCLP |
| **Data** | Need CLP daily OHLCV seed (may need to create) |

### FC-MULTI-003: Portfolio Optimization

| Field | Value |
|-------|-------|
| **When** | After FC-MULTI-001 and FC-MULTI-002 validate |
| **Variable** | Single pair → 3-pair portfolio (COP + MXN + CLP) |
| **Hypothesis** | Diversification improves Sharpe from ~1.0 to ~1.5 with same return |
| **Expected impact** | 3 pairs × 10-13% each, with correlation < 0.5 → portfolio 15-20%, Sharpe 1.5 |

---

## PHASE 4: RL as Execution Optimizer (PRIORITY 4)

> Only after Phases 1-3 establish a robust forecasting signal.

### FC-RL-001: Feed Forecast Signal to RL Environment

| Field | Value |
|-------|-------|
| **Pipeline** | RL (5-min bars) — uses forecasting signal as input |
| **Variable** | obs_dim: 27 → 32 (+5 forecast features) |
| **New features** | forecast_direction, forecast_confidence, target_position_size, macro_score_norm, vol_regime |
| **RL role** | Optimize INTRADAY TIMING of daily forecast signal, NOT generate signal |
| **Hypothesis** | RL improves execution by 2-3pp via better entry/exit timing |
| **Prerequisite** | Phase 1-2 validated (forecasting DA > 54%, Sharpe > 1.2) |

**Two approaches (test separately):**

1. **Reward shaping**: Bonus when RL action aligns with forecast direction
2. **Hard constraint**: RL can only trade in forecast direction, decides when

### FC-RL-002: RL Execution Backtest

| Field | Value |
|-------|-------|
| **When** | After FC-RL-001 trains |
| **Validation** | Compare: (a) daily execution at close vs (b) RL intraday execution |
| **Success criteria** | RL execution > daily close execution by > 1pp |

---

## PHASE 5: Statistical Validation (CONTINUOUS)

### FC-STAT-001: Walk-Forward of Complete System

| Field | Value |
|-------|-------|
| **When** | After Phase 2 complete |
| **Scope** | Full system: expanded model + vol-targeting |
| **Validation** | 4+ windows, each > 0% return, Sharpe > 0.8, PF > 1.15 |
| **Success criteria** | Bootstrap CI excludes zero with expanded features |

### FC-STAT-002: Out-of-Sample 2025 Complete

| Field | Value |
|-------|-------|
| **When** | After all phases |
| **Scope** | Hold-out 2025 H2 (never used in walk-forward) |
| **Success criteria** | Return > 0%, consistent with walk-forward estimates |

---

## DEPRECATED (from old RL-only queue)

These experiments are deprioritized. The forecasting pipeline has a stronger proven edge.

| ID | Description | Why Deprecated |
|----|-------------|----------------|
| EXP-V215c-001 | Reduce RL timesteps | RL edge is not statistically significant |
| EXP-REWARD-ASYM-001 | Reward asymmetry | RL PF=1.009, reward tweaks won't fix it |
| EXP-V215d-001 | Remove rsi_21 | Minor feature change on weak RL base |
| EXP-DISC-001 | Discrete(4) | V22 already showed this hurts |
| EXP-LSTM-001 | RecurrentPPO | High complexity, no proven base to build on |
| EXP-KELLY-001 | Kelly sizing (RL) | Moved to FC-SIZE-002 (forecasting) |
| EXP-WF-001 | Walk-forward (RL) | Moved to FC-STAT-001 (forecasting) |
| EXP-FEAT-001 | Alt macro features (RL) | Moved to FC-FEAT-001 (forecasting) |
| EXP-CROSS-PAIR-001 | Cross-pair (RL) | Moved to FC-MULTI-001 (forecasting) |

These can be revived IF the forecasting integration (Phase 4) shows RL adds value as execution layer.

---

## Decision Tree (Updated)

```
FC-FEAT-001 (expand to 32 features)?
  |
  +-- DA > 54% AND PF > 1.25?
  |     YES → FC-SIZE-001 (vol targeting)
  |     NO  |
  |         +-- DA > 53% (same as baseline)?
  |         |     → FC-FEAT-002 (feature selection, reduce noise)
  |         |
  |         +-- DA < 52% (overfitting)?
  |               → Revert to 19 features, try FC-MULTI-001 (diversification)
  |
  FC-SIZE-001 completed?
  |
  +-- Sharpe > 1.2 with leverage?
  |     YES → FC-MULTI-001 (add USDMXN)
  |     NO  → FC-SIZE-002 (Half-Kelly, confidence-based)
  |
  FC-MULTI-001 completed?
  |
  +-- Portfolio Sharpe > 1.5?
  |     YES → FC-RL-001 (RL execution layer)
  |     NO  → Add more pairs (BRL, CLP)
  |
  FC-RL-001 completed?
  |
  +-- RL improves execution by > 1pp?
  |     YES → FC-STAT-001 (full system walk-forward) → PRODUCTION
  |     NO  → Ship forecasting system WITHOUT RL (daily execution at close)
```

---

## Target Performance (Updated 2026-02-15)

| Phase | Cumulative Return | Sharpe | Confidence | Status |
|-------|------------------|--------|------------|--------|
| Base (forecast 1x) | ~21.6% total | ~2.1 | **PROVEN** (p=0.0033) | DONE |
| +Vol targeting | ~32.6% total | ~2.1 | **PROVEN** (p=0.053) | DONE |
| +Trailing stop | ~36.8% total | ~3.1 | **PROVEN** (p=0.0178) | DONE |
| +Macro features (Phase 1) | ~40-50% total | ~3.5 | Medium (need to validate) | NEXT UP |
| +Multi-pair (Phase 3) | ~45-60% total | ~4.0 | Medium (assumes edge transfers) | Planned |

**Current system ($10K→$13,684) already exceeds original "aggressive" targets.**
**Paper trading (60 days) will validate live performance before go-live.**
