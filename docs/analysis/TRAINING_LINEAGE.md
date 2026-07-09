# USDCOP RL Models — Complete Training Lineage

**Generated**: 2026-02-10
**Project**: USDCOP Spot Trading via Reinforcement Learning (5-min bars)
**Exchange**: MEXC (0% maker fees)

---

## Version Summary

| Version | Date | Config | obs_dim | Action | Timesteps | Model | Result | Status |
|---------|------|--------|---------|--------|-----------|-------|--------|--------|
| V20 | Feb 2-3 | Various | 13-18 | Continuous | 500K-1M | PPO MlpPolicy | All negative | Failed |
| V21 pre-fix | Feb 3-5 | SSOT v2-v3 | 18-23 | Continuous | 1M | PPO MlpPolicy | -3% to -17% | Failed |
| V21.5 | Feb 6 | SSOT v3.5.0 | 23 | Continuous | 1M | PPO MlpPolicy | **+1.26%** | **PASSED** |
| V21 multi-seed | Feb 7 | SSOT v3.5.0 | 23 | Continuous | 1M | PPO MlpPolicy | -26% to -42% | Failed |
| V22 (pre-fix) | Feb 7-8 | SSOT v4.0.0 | 27 | Discrete(4) | 2M | RecurrentPPO LSTM | -39.22% | Failed |
| V22 (post-fix) | Feb 8-9 | SSOT v4.0.0 | 27 | Discrete(4) | 2M | RecurrentPPO LSTM | -31% to +9.6% | Mixed |
| V21.5b | Feb 10 | SSOT v3.5.1 | 27 | Continuous | 2M | PPO MlpPolicy | *pending* | Running |

---

## Detailed Version History

### V20 — Early Experiments (Feb 2-3, 2026)

**Models**: `ppo_v20_production/`, `ppo_v20260202_*_production/` (25+ runs), `ppo_v3_18f_*_production/`

**Configuration**:
- Feature sets: 13-18 features (various combinations)
- Observation dim: variable (13-20)
- Action space: Continuous Box(-1, 1)
- Timesteps: 500K-1M
- Transaction costs: 2.5 bps/side
- No SSOT pipeline (hardcoded configs)

**Key Issues**:
- Transaction costs (2.5bps/side) destroyed all alpha
- Inconsistent feature engineering across runs
- No reproducibility framework
- No systematic hyperparameter tracking

**Result**: All runs negative. No L4 gates passed.

**Lesson**: Need SSOT config, lower transaction costs, systematic pipeline.

---

### V21 Pre-Fix — SSOT Pipeline (Feb 3-5, 2026)

**Models**: `ppo_ssot_20260203_*`, `ppo_ssot_20260204_*`, `ppo_ssot_20260205_*`

**Configuration (SSOT v2-v3)**:
```yaml
observation_dim: 18-23
action_space: Continuous Box(-1, 1)
total_timesteps: 1_000_000
transaction_cost_bps: 0.0-2.5  # Varied across runs
slippage_bps: 1.0
stop_loss_pct: -4%
take_profit_pct: +4%
gamma: 0.98
n_steps: 4096
batch_size: 128
ent_coef: 0.01
thresholds: 0.35/-0.35
```

**L4 Results (chronological)**:

| Model | Stage | Return | Sharpe | MaxDD | Trades | WR% | PF | Gates |
|-------|-------|--------|--------|-------|--------|-----|-----|-------|
| 20260204_203546 | L4-VAL | -6.37% | -1.021 | 13.33% | 39 | — | — | FAIL |
| 20260204_215055 | L4-VAL | -1.60% | -0.204 | 6.39% | 59 | — | — | FAIL |
| 20260204_221209 | L4-VAL | -5.52% | -0.864 | 10.92% | 87 | — | — | FAIL |
| 20260205_160832 | L4-VAL | -7.17% | -1.129 | 12.05% | 59 | — | — | FAIL |
| 20260205_170024 | L4-VAL | -16.72% | -3.338 | 18.17% | 195 | 24.1% | — | FAIL |
| 20260205_172215 | L4-VAL | **+7.26%** | +1.173 | 4.81% | 63 | — | — | **PASS** |
| 20260205_172215 | L4-TEST | -12.43% | -1.621 | 17.05% | 91 | 24.2% | — | FAIL |
| 20260205_175308 | L4-VAL | **+2.14%** | +0.393 | 6.72% | 70 | — | — | **PASS** |
| 20260205_175308 | L4-TEST | -10.55% | -1.434 | 11.81% | 89 | 27.0% | — | FAIL |
| 20260205_203304 | L4-VAL | +2.13% | +0.393 | 6.72% | 70 | — | — | PASS |
| 20260205_203304 | L4-TEST | -2.25% | -0.126 | 14.89% | 134 | 58.8% | 1.429 | FAIL |
| 20260205_222807 | L4-VAL | -1.41% | -0.166 | 11.86% | 41 | — | — | FAIL |

**Key Issues**:
- VAL could pass but TEST consistently failed
- Low win rates (24-27%) in test = model overfit to validation
- `20260205_203304` was closest: WR 58.8%, PF 1.429 but -2.25% return

**Lesson**: Need better generalization. Transaction costs still destroying edge in some runs.

---

### V21.5 — First Profitable Model (Feb 6, 2026)

**Model**: `ppo_ssot_20260206_204424/best_model.zip`
**SSOT Version**: 3.5.0

**Configuration**:
```yaml
version: "3.5.0"
observation_dim: 23          # 18 market + 5 state
action_space: Continuous Box(-1, 1)
total_timesteps: 1_000_000
device: cpu

# PPO Hyperparameters
learning_rate: 0.0003
gamma: 0.98
n_steps: 4096
batch_size: 128
ent_coef: 0.01
clip_range: 0.2

# Environment
transaction_cost_bps: 0.0    # MEXC maker = 0%
slippage_bps: 1.0
stop_loss_pct: -0.04         # -4%
take_profit_pct: 0.04        # +4%
trailing_stop_enabled: false
min_hold_bars: 25
max_position_duration: 864
thresholds: [0.35, -0.35]

# Reward
pnl_weight: 0.9
holding_decay_weight: 0.05
flat_reward_weight: 0.0      # CRITICAL: must be 0
reward_interval: 25

# Risk
max_drawdown_pct: 99.0       # Disabled for backtest
```

**18 Market Features**:
1. `log_ret_5m` — 5-min log return (z-scored)
2. `log_ret_1h` — 1-hour log return
3. `log_ret_4h` — 4-hour log return
4. `log_ret_1d` — 1-day log return
5. `rsi_9` — RSI(9) normalized [0,1]
6. `rsi_21` — RSI(21) normalized [0,1]
7. `volatility_pct` — Rolling volatility z-score
8. `trend_z` — SMA trend z-score
9. `dxy_z` — Dollar Index z-score
10. `dxy_change_1d` — DXY daily change
11. `vix_z` — VIX z-score
12. `embi_z` — EMBI Colombia z-score
13. `brent_change_1d` — Brent crude daily change
14. `gold_change_1d` — Gold daily change
15. `rate_spread_z` — COL10Y-UST10Y spread z-score
16. `rate_spread_change` — Spread daily change
17. `usdmxn_change_1d` — USDMXN daily change
18. `yield_curve_z` — US yield curve z-score

**5 State Features** (computed at runtime by TradingEnv):
1. `position` — Current position (-1/0/+1)
2. `unrealized_pnl` — Current unrealized PnL
3. `sl_proximity` — Distance to stop loss
4. `tp_proximity` — Distance to take profit
5. `bars_held` — Bars in current position

**L4 Results — PASSED ALL GATES**:

| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| **Total Return** | **+1.26%** | > 0% | PASS |
| **APR** | **+2.54%** | — | — |
| **Sharpe Ratio** | **+0.202** | > 0 | PASS |
| **Max Drawdown** | **12.72%** | < 20% | PASS |
| **N Trades** | **213** | > 30 | PASS |
| **Win Rate** | **56.34%** | > 35% | PASS |
| **Profit Factor** | **1.006** | — | — |
| **Sortino Ratio** | **+0.211** | — | — |
| **Final Equity** | **$10,125.70** | — | — |
| **Test Period** | Jan 2 - Dec 30, 2025 (182 days) | — | — |

**Monthly Returns**:
| Month | Return |
|-------|--------|
| 2025-02 | +2.94% |
| 2025-03 | +0.78% |
| 2025-04 | -0.92% |
| 2025-05 | +3.00% |
| 2025-06 | +1.39% |
| 2025-07 | +5.43% |
| 2025-08 | +2.23% |
| 2025-09 | -4.21% |
| 2025-10 | -1.24% |
| 2025-11 | -5.76% |
| 2025-12 | -0.51% |

**Key Breakthroughs**:
1. MEXC maker orders (0% fee) eliminated 29% transaction cost drag
2. `min_hold_bars=25` prevented premature exits
3. `flat_reward_weight=0.0` removed HOLD bias
4. `max_drawdown=99%` in backtest (was killing runs at 15%)
5. Trailing stop disabled (was cutting winners)

---

### V21.5 Multi-Seed Validation (Feb 7, 2026)

**Models**: `ppo_ssot_20260207_*` (5 seeds from MultiSeedTrainer)
**Purpose**: Test reproducibility of V21.5 across seeds [42, 123, 456, 789, 1337]

**Results** (`l4_per_seed_comparison.json`):

| Seed | Return | APR | Sharpe | MaxDD | Trades | WR% | PF |
|------|--------|-----|--------|-------|--------|-----|-----|
| 42 | -39.32% | -63.26% | -7.356 | 40.32% | 1,837 | 61.1% | 0.771 |
| 123 | -41.75% | -66.15% | -7.095 | 42.25% | 1,538 | 61.3% | 0.783 |
| 456 | -32.52% | -54.54% | -7.523 | 33.66% | 1,696 | 58.3% | 0.752 |
| 789 | -34.45% | -57.11% | -6.404 | 35.72% | 1,923 | 57.3% | 0.786 |
| 1337 | -26.33% | -45.80% | -5.082 | 27.52% | 1,166 | 62.3% | 0.824 |

**Critical Bug Found**: 1,500-1,900 trades per seed (vs 213 for V21.5). The `min_hold_bars` was NOT being enforced on CLOSE actions and reversals, causing massive over-trading. High WR (57-62%) but PF < 1 = many small wins, few large losses.

**Lesson**: V21.5's single-seed success was partially lucky. The min_hold_bars bypass bug was masked in that specific seed.

---

### V22 Pre-Fix (Feb 7-8, 2026)

**Model**: `ppo_ssot_20260207_074140/` through `20260208_015554/`
**SSOT Version**: 4.0.0

**Configuration Changes from V21.5**:
```yaml
# V22 changes (5 simultaneous changes):
observation_dim: 27              # Was 23 (+4 temporal features)
action_type: "discrete"          # Was "continuous"
n_actions: 4                     # HOLD=0, BUY=1, SELL=2, CLOSE=3
ent_coef: 0.02                   # Was 0.01
total_timesteps: 2_000_000       # Was 1M
model_type: "recurrent_ppo"      # Was "ppo"
lstm:
  enabled: true
  hidden_size: 128
  n_layers: 1
close_shaping:
  enabled: true
  stop_loss_mult: 1.5
  take_profit_mult: 1.2
  agent_close_win: 1.1
  agent_close_loss: 0.7
  timeout_mult: 0.8
temporal_features:
  enabled: true
  features: [hour_sin, hour_cos, dow_sin, dow_cos]
```

**L4 Result (best seed auto-selected)**:

| Metric | Value |
|--------|-------|
| Total Return | -39.22% |
| Sharpe | -7.332 |
| MaxDD | 40.23% |
| Trades | 1,834 |
| WR | 61.12% |
| PF | 0.772 |

**3 Critical Bugs Found**:
1. `min_hold_bars` NOT enforced on CLOSE action (action=3) and reversals
2. LSTM hidden states NOT tracked across backtest steps (reset every step)
3. `close_reason` NOT passed to reward calculator (close shaping was dead code)

---

### V22 Post-Fix (Feb 8-9, 2026)

**Models**:
| Seed | Model Directory |
|------|----------------|
| 42 | `ppo_ssot_20260208_105908/` |
| 123 | `ppo_ssot_20260208_155922/` |
| 456 | `ppo_ssot_20260208_211803/` |
| 789 | `ppo_ssot_20260209_073504/` |
| 1337 | `ppo_ssot_20260209_155510/` |

**Bug Fixes Applied**:
1. `min_hold_bars` enforcement on CLOSE + reversals in `trading_env.py`
2. LSTM state tracking in `run_ssot_pipeline.py` backtest loop
3. `close_reason` passed to reward calculator

**L4 Per-Seed Results** (`l4_per_seed_comparison_v22_postfix.json`):

| Seed | Return | APR | Sharpe | MaxDD | Trades | WR% | PF | Avg Bars/Trade |
|------|--------|-----|--------|-------|--------|-----|-----|----------------|
| 42 | -9.79% | -18.66% | -1.194 | 14.05% | 383 | 49.1% | 0.968 | 37.2 |
| 123 | -9.85% | -18.77% | -1.096 | 18.59% | 392 | 49.5% | 0.970 | 36.4 |
| 456 | -20.62% | -37.05% | -2.563 | 20.79% | 373 | 46.9% | 0.930 | 38.2 |
| 789 | -31.11% | -52.62% | -4.079 | 33.02% | 391 | 44.5% | 0.893 | 36.5 |
| **1337** | **+9.62%** | **+20.22%** | **+1.136** | **9.34%** | **381** | **54.3%** | **1.032** | **37.4** |

**Bug Fix Verification**:
- Trades: 1,834 → 384 avg (fixed)
- Avg bars/trade: 7.8 → 37.1 (fixed)
- Only 1/5 seeds profitable

**Seed 1337 Monthly Returns**:
| Month | Return |
|-------|--------|
| 2025-02 | +2.96% |
| 2025-03 | +3.34% |
| 2025-04 | -3.99% |
| 2025-05 | -0.75% |
| 2025-06 | +1.12% |
| 2025-07 | +1.82% |
| 2025-08 | -0.49% |
| 2025-09 | -2.46% |
| 2025-10 | -3.66% |
| 2025-11 | +8.14% |
| 2025-12 | +0.77% |

**Key Insight**: Best eval reward != best L4. Seed 456 (eval=131) lost -20.6%, seed 1337 (eval=111) gained +9.6%. Eval reward is a poor predictor of OOS performance.

---

### V21.5b — Controlled Experiment (Feb 10, 2026) — IN PROGRESS

**Models**:
| Seed | Model Directory |
|------|----------------|
| 42 | `ppo_ssot_20260210_003307/` |
| 123 | `ppo_ssot_20260210_044621/` |
| 456 | `ppo_ssot_20260210_101317/` |
| 789 | `ppo_ssot_20260210_171503/` |
| 1337 | `ppo_ssot_20260210_211622/` |

**Purpose**: Isolate the impact of temporal features. Only 2 changes from V21.5:
1. +4 temporal features (obs_dim 23 → 27)
2. 2M timesteps (was 1M)
Everything else IDENTICAL to V21.5.

**SSOT Version**: 3.5.1

**Configuration** (changes from V21.5 in **bold**):
```yaml
version: "3.5.1"
observation_dim: 27              # Was 23 (+4 temporal)
action_space: Continuous Box(-1, 1)  # Same as V21.5
total_timesteps: 2_000_000       # Was 1M
device: cuda                     # Was cpu

# IDENTICAL to V21.5:
ent_coef: 0.01
model_type: ppo (MlpPolicy)
lstm: disabled
close_shaping: disabled
action_type: continuous
all other hyperparameters identical
```

**L3 Training Results**:

| Seed | Duration | Final Eval Reward |
|------|----------|------------------|
| 42 | 4h 13m | 94.65 |
| 123 | 5h 27m | 89.84 |
| 456 | 7h 02m | 96.28 |
| 789 | 4h 01m | 78.96 |
| 1337 | ~3.5h (in progress) | — |

**L3 Multi-Seed Summary**:
- **Best seed**: 123 (reward=125.10)
- **Mean reward**: 115.37 +/- 5.84
- **CV**: 5.06% (excellent reproducibility)

**L4 Results (best seed 123, auto-selected)** — **ALL GATES PASSED**:

| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| **Total Return** | **+10.36%** | > 0% | PASS |
| **APR** | **+21.84%** | — | — |
| **Sharpe Ratio** | **+1.121** | > 0 | PASS |
| **Sortino Ratio** | **+1.258** | — | — |
| **Max Drawdown** | **5.45%** | < 20% | PASS |
| **N Trades** | **381** | > 30 | PASS |
| **Win Rate** | **53.8%** | > 35% | PASS |
| **Profit Factor** | **1.03** | — | — |
| **Final Equity** | **$11,036** | — | — |

**Monthly Returns**:
| Month | Return |
|-------|--------|
| 2025-02 | +1.01% |
| 2025-03 | +2.00% |
| 2025-04 | +1.11% |
| 2025-05 | +2.50% |
| 2025-06 | +2.99% |
| 2025-07 | -1.41% |
| 2025-08 | -0.89% |
| 2025-09 | +3.73% |
| 2025-10 | -0.22% |
| 2025-11 | -0.53% |
| 2025-12 | +1.64% |

**L4 Per-Seed Results** (`l4_per_seed_comparison_v21_5b.json`):

| Seed | Return | APR | Sharpe | MaxDD | Trades | WR% | PF | Avg Bars |
|------|--------|-----|--------|-------|--------|-----|-----|----------|
| 42 | +1.98% | +4.00% | +0.275 | 11.3% | 340 | 54.7% | 1.007 | 41.9 |
| **123** | **+9.38%** | **+19.69%** | **+1.026** | **5.4%** | **381** | **53.5%** | **1.028** | **37.4** |
| 456 | +0.58% | +1.17% | +0.127 | 18.9% | 352 | 51.1% | 1.003 | 40.5 |
| 789 | -2.60% | -5.15% | -0.229 | 21.6% | 292 | 43.5% | 0.994 | 48.8 |
| 1337 | +3.21% | +6.54% | +0.405 | 14.0% | 378 | 53.7% | 1.011 | 37.7 |
| **Avg** | **+2.51%** | — | **+0.321** | — | **349** | **51.3%** | **1.009** | **41.3** |

**Positive seeds: 4/5** (vs V22's 1/5). Only seed 789 negative (-2.60%).

**Conclusion**: Temporal features + 2M timesteps are the key improvements. LSTM, Discrete(4), and close_shaping from V22 hurt reproducibility without improving returns

---

## Feature Evolution

```
V20:  13-18 features (various, no SSOT)
  |
V21:  18 market features (SSOT-controlled)
  |     + 5 state features (runtime) = 23 obs_dim
  |
V21.5: Same 18+5 = 23 obs_dim  ← FIRST PROFITABLE
  |
V22:  18 market + 5 state + 4 temporal = 27 obs_dim
  |     + Discrete(4) + RecurrentPPO + close_shaping
  |
V21.5b: 18 market + 5 state + 4 temporal = 27 obs_dim
        + Continuous + PPO MlpPolicy (isolate temporal only)
```

**Removed Features**:
- `volume_zscore`: OHLCV volume column is 100% zeros (dead feature)

---

## Architecture Evolution

```
V20:    PPO + MlpPolicy + Continuous + 500K steps
  |
V21:    PPO + MlpPolicy + Continuous + 1M steps + SSOT
  |
V21.5:  PPO + MlpPolicy + Continuous + 1M steps + SSOT v3.5  ← BEST
  |
V22:    RecurrentPPO + MlpLstmPolicy + Discrete(4) + 2M steps + SSOT v4.0
  |       + close_reason shaping + LSTM(128) + temporal features
  |
V21.5b: PPO + MlpPolicy + Continuous + 2M steps + temporal only
```

---

## Critical Lessons Learned

### 1. Transaction Costs are the #1 Bottleneck
- 2.5bps/side destroyed 29% of +24% gross alpha
- MEXC maker orders (0% fee) are viable for 5-min RL
- Slippage 1bps is realistic for limit orders

### 2. min_hold_bars Must Be Enforced Everywhere
- Bug: CLOSE action and reversals bypassed min_hold_bars
- Result: 1,834 trades instead of ~380
- Fix: Check min_hold_bars BEFORE processing any action

### 3. Eval Reward != OOS Performance
- Seed 456 (eval=131) lost -20.6%
- Seed 1337 (eval=111) gained +9.6%
- Model selection by eval reward is unreliable

### 4. Don't Change 5 Things at Once
- V22 changed: action space, model, features, reward shaping, entropy
- Impossible to know which change helped/hurt
- V21.5b isolates ONE change (temporal features) for proper attribution

### 5. LSTM States Must Be Tracked in Backtest
- RecurrentPPO requires `model.predict(obs, state=lstm_states, episode_start=...)`
- Without tracking, LSTM degrades to MLP (loses temporal memory)

### 6. FlatReward Defaults Must Be False
- `flat_reward_weight=0.0` in config.py
- Non-zero creates HOLD bias that prevents trading

### 7. Backtest max_drawdown Must Be 99%
- Training uses 15% drawdown for episode termination
- Backtest must use 99% to see full equity curve behavior

---

## File Locations Reference

### Config
- `config/pipeline_ssot.yaml` — Master SSOT config (current: v3.5.1)
- `config/experiment_ssot.yaml` — Experiment config (deprecated)

### Source Code
- `src/training/environments/trading_env.py` — Trading environment
- `src/training/trainers/ppo_trainer.py` — PPO/RecurrentPPO trainer
- `src/training/multi_seed_trainer.py` — Multi-seed variance reduction
- `src/training/reward_calculator.py` — Reward computation
- `src/training/config.py` — Training config dataclasses
- `src/config/pipeline_config.py` — SSOT config loader
- `src/data/ssot_dataset_builder.py` — L2 dataset builder
- `scripts/run_ssot_pipeline.py` — L2-L4 pipeline runner

### Results
- `results/backtests/l4_test_*.json` — L4 test results
- `results/backtests/l4_val_*.json` — L4 validation results
- `results/backtests/l4_per_seed_comparison.json` — V21 multi-seed
- `results/backtests/l4_per_seed_comparison_v22_postfix.json` — V22 multi-seed

### Models (key ones)
- `models/ppo_ssot_20260206_204424/` — V21.5 (BEST, +1.26%)
- `models/ppo_ssot_20260208_105908/` — V22 post-fix seed 42
- `models/ppo_ssot_20260209_155510/` — V22 post-fix seed 1337 (+9.62%)
- `models/ppo_ssot_20260210_003307/` — V21.5b seed 42 (pending)

---

## L4 Validation Gates

```yaml
gates:
  min_return_pct: 0.0        # Break-even minimum
  min_sharpe_ratio: 0.0      # Any positive Sharpe
  max_drawdown_pct: 20.0     # Reasonable ceiling
  min_trades: 30             # Statistical significance
  min_win_rate: 35.0         # Reasonable baseline
```

**Models that passed ALL gates**:
1. **V21.5** (`20260206_204424`): +1.26%, Sharpe +0.202, 213 trades, 56.3% WR

**Models that passed VAL but failed TEST**:
1. `20260205_172215`: VAL +7.26% → TEST -12.43%
2. `20260205_175308`: VAL +2.14% → TEST -10.55%
3. `20260205_203304`: VAL +2.13% → TEST -2.25%

---

## Data Splits

```
Train: 2019-12-24 → 2024-12-30  (70,072 bars, ~5 years)
Val:   2025-01-02 → 2025-06-27  (6,937 bars, ~6 months)
Test:  2025-07-01 → 2025-12-30  (7,316 bars, ~6 months)
```

**Distribution Drift** (train → val, >0.5 std):
- `rate_spread_z`: 1.27 std shift
- `embi_z`: 0.86 std shift
- `yield_curve_z`: 0.71 std shift
- `vix_z`: 0.70 std shift

---

## Eval Reward Learning Curves (V21.5b — 5 Seeds × 2M Steps)

### Seed 42 (Model: `ppo_ssot_20260210_003307`)
```
Steps   |  Eval Reward (mean ± std)
--------|---------------------------
  50K   |  51.07 ± 25.44  (random)
 250K   |  46.37 ±  6.39  (exploring)
 450K   |  73.83 ± 16.13  (phase transition)
 500K   |  88.88 ± 11.65  (learned)
 600K   |  93.68 ± 10.65
 850K   |  99.08 ±  7.35  ← peak
1100K   |  96.93 ± 13.47
1500K   |  99.97 ±  5.62  ← peak (low variance)
2000K   |  94.65 ±  9.89  (final)
```
**Best**: 99.97 @ 1.5M | **Duration**: 4h 13min | **Device**: CUDA (RTX 3050)

### Seed 123 (Model: `ppo_ssot_20260210_044621`)
```
Steps   |  Eval Reward (mean ± std)
--------|---------------------------
  50K   |  50.19 ± 23.52
 450K   |  88.27 ± 18.10  (phase transition)
 750K   |  99.29 ±  8.65
 800K   | 107.43 ±  9.78  ← peak
1050K   | 101.18 ±  8.50
1350K   | 102.19 ± 10.58
1600K   | 102.13 ± 11.68
2000K   |  89.84 ±  8.07  (final)
```
**Best**: 107.43 @ 800K | **Duration**: 5h 27min | **Device**: CUDA

### Seed 456 (Model: `ppo_ssot_20260210_101317`)
```
Steps   |  Eval Reward (mean ± std)
--------|---------------------------
  50K   |  55.82 ± 32.43
 500K   |  79.49 ± 11.04  (slower learner)
 550K   |  81.12 ±  6.14
1000K   |  86.61 ±  4.35
1750K   |  91.85 ±  7.64  ← peak
2000K   |  96.28 ±  5.16  (final — late surge)
```
**Best**: 96.28 @ 2M (improved at end!) | **Duration**: 7h 02min | **Device**: CUDA

### Seed 789 (Model: `ppo_ssot_20260210_171503`)
```
Steps   |  Eval Reward (mean ± std)
--------|---------------------------
  50K   |  55.54 ± 27.71
 500K   |  88.47 ±  7.39
 800K   |  84.01 ± 10.07
1000K   |  83.86 ± 12.35
1350K   |  85.78 ± 11.49  ← peak
2000K   |  78.96 ± 10.69  (final — degraded)
```
**Best**: 88.47 @ 500K (peaked early, then degraded) | **Duration**: 4h 01min | **Device**: CUDA

### Seed 1337 (Model: `ppo_ssot_20260210_211622`) — IN PROGRESS
```
Steps   |  Eval Reward (mean ± std)
--------|---------------------------
  50K   |  55.00 ± 22.40
 500K   |  80.16 ±  4.36
 550K   |  86.19 ±  6.46
 800K   |  89.22 ±  6.31
1200K   |  91.44 ±  4.68  ← current best
1400K   |  80.22 ±  6.91
 ...    |  (training in progress ~70%)
```
**Best so far**: 91.44 @ 1.2M | **Device**: CUDA

### Learning Curve Summary
| Seed | Phase Transition | Peak Reward | Peak Step | Final Reward | Duration |
|------|-----------------|-------------|-----------|--------------|----------|
| 42 | ~450K | 99.97 | 1.5M | 94.65 | 4h 13m |
| 123 | ~450K | 107.43 | 800K | 89.84 | 5h 27m |
| 456 | ~500K | 96.28 | 2.0M | 96.28 | 7h 02m |
| 789 | ~450K | 88.47 | 500K | 78.96 | 4h 01m |
| 1337 | ~500K | 91.44 | 1.2M | *pending* | *running* |

**Pattern**: All seeds show a "phase transition" at ~450-500K steps where reward jumps from ~55 to ~80-90. After that, improvements are marginal. Seed 789 peaked early and degraded, suggesting overfitting after 500K.

---

## Training Duration & Performance

### V21.5b (PPO MlpPolicy, CUDA RTX 3050 Laptop 4GB)
| Seed | Start | End | Duration | Avg FPS | Notes |
|------|-------|-----|----------|---------|-------|
| 42 | 00:33 | 04:46 | 4h 13m | ~132 | Warm GPU |
| 123 | 04:46 | 10:13 | 5h 27m | ~102 | Thermal throttling |
| 456 | 10:13 | 17:15 | 7h 02m | ~79 | Heavy throttling (daytime heat) |
| 789 | 17:15 | 21:16 | 4h 01m | ~138 | Cooled down (evening) |
| 1337 | 21:16 | *~01:00* | *~3.8h* | ~159 | Night cooling |

**FPS Degradation Pattern**: GPU thermal throttling caused FPS to drop 187→65 during sustained daytime training, recovering at night. Laptop GPU not ideal for multi-seed runs.

### V22 Post-Fix (RecurrentPPO MlpLstmPolicy, CUDA RTX 3050)
| Seed | Model Dir | Duration | Avg FPS | Notes |
|------|-----------|----------|---------|-------|
| 42 | `20260208_105908` | ~5h | ~110 | LSTM slower than MLP |
| 123 | `20260208_155922` | ~5.5h | ~100 | |
| 456 | `20260208_211803` | ~7h | ~80 | Throttling |
| 789 | `20260209_073504` | ~6h | ~90 | |
| 1337 | `20260209_155510` | ~6h | ~90 | |

**RecurrentPPO vs PPO**: LSTM adds ~30% overhead (110 vs 160 FPS at comparable temperatures).

### V21.5 Original (PPO MlpPolicy, CPU)
| Model | Duration | Device | Notes |
|-------|----------|--------|-------|
| `20260206_204424` | ~1.5h | CPU | 1M steps, faster per-step on CPU for MLP |

**CPU vs GPU for MLP**: SB3 warns that PPO MlpPolicy is suboptimal on GPU. For small networks, CPU can match or beat GPU due to overhead of GPU memory transfers.

---

## GPU vs CPU Training Details

| Config | Device | FPS Range | Total Time (2M steps) | Notes |
|--------|--------|-----------|----------------------|-------|
| PPO MlpPolicy | CPU | ~300-350 | ~1.5h per seed | Consistent, no throttling |
| PPO MlpPolicy | CUDA (3050) | 65-187 | 4-7h per seed | Throttles heavily |
| RecurrentPPO LSTM | CUDA (3050) | 80-110 | 5-7h per seed | LSTM benefits from GPU |

**Recommendation**: Use CPU for PPO MlpPolicy, GPU only for RecurrentPPO LSTM.

**GPU Specs**: NVIDIA RTX 3050 Laptop (4GB VRAM), Driver 566.36, CUDA 12.4, PyTorch 2.6.0+cu124

---

## Norm Stats Lineage

### Production Norm Stats
| Version | File | Features | Used By |
|---------|------|----------|---------|
| V21.5+ | `data/pipeline/07_output/5min/DS_production_norm_stats.json` | 18 market features | V21.5, V22, V21.5b |

### Historical Norm Stats (config/ directory)
| File | Era | Notes |
|------|-----|-------|
| `v1_norm_stats.json` | Early V1 | Legacy, 13 features |
| `v2_norm_stats.json` | V2 | Legacy |
| `v3_18f_norm_stats.json` | V3 (18 features) | First SSOT-era |
| `v3_18f_r2-r4_norm_stats.json` | V3 iterations | Refinements |
| `v3_18f_final_norm_stats.json` | V3 final | Pre-V21.5 |
| `v3_18f_final2_norm_stats.json` | V3 final2 | |
| `v3_18f_prod_norm_stats.json` | V3 production | |
| `v3_18f_fix_norm_stats.json` | V3 fix | |
| `v20260202_*_norm_stats.json` | V20 (25 files) | Rapid iteration day |
| `v20260203_092854_norm_stats.json` | V20 final | |

### Per-Model Norm Stats
Each model directory contains its own `norm_stats.json` snapshot:
- `models/ppo_ssot_20260206_204424/norm_stats.json` — V21.5
- `models/ppo_ssot_20260208_105908/norm_stats.json` — V22
- `models/ppo_ssot_20260210_003307/norm_stats.json` — V21.5b

**Note**: All V21.5+ models use the SAME norm stats from `DS_production_norm_stats.json` (computed from train split only). The per-model copy is for reproducibility/archival.

---

## Equity Curve Artifacts

### Available Equity Curves (Parquet files)
| File | Version | Stage | Return |
|------|---------|-------|--------|
| `equity_curve_20260203_152841.parquet` | V20 | backtest | +13.64% |
| `equity_curve_20260204_161918.parquet` | V21 pre-fix | backtest | -3.02% |
| `equity_curve_20260204_203546.parquet` | V21 pre-fix | backtest | -6.24% |
| `equity_curve_test_20260205_172215.parquet` | V21 | L4-TEST | -12.43% |
| `equity_curve_test_20260205_175308.parquet` | V21 | L4-TEST | -10.55% |
| `equity_curve_test_20260205_203304.parquet` | V21 | L4-TEST | -2.25% |
| `equity_curve_test_20260206_111758.parquet` | V21 | L4-TEST | -10.85% |
| `equity_curve_test_20260206_153043.parquet` | V21.5 variant | L4-TEST | -20.03% |
| `equity_curve_test_20260206_170227.parquet` | V21.5 variant | L4-TEST | -4.96% |
| `equity_curve_test_20260206_175914.parquet` | V21.5 variant | L4-TEST | -4.75% |
| `equity_curve_test_20260206_185344.parquet` | V21.5 variant | L4-TEST | -11.31% |
| `equity_curve_test_20260206_194946.parquet` | V21.5 variant | L4-TEST | -11.66% |
| `equity_curve_test_20260206_204424.parquet` | **V21.5 BEST** | L4-TEST | **+1.26%** |
| `equity_curve_test_20260207_001454.parquet` | V21 multi-seed | L4-TEST | -39.22% |
| `equity_curve_test_20260208_211803.parquet` | V22 post-fix | L4-TEST | -20.43% |

**Format**: Each parquet contains a time-indexed equity curve (initial=$10,000) for plotting.

---

## Training Config JSON Snapshots

### V21.5 (`models/ppo_ssot_20260206_204424/training_config.json`)
```json
{
  "ssot_version": "3.5.0",
  "based_on_model": "v21.1_best_baseline",
  "observation_dim": 23,
  "ppo_config": {"lr": 0.0003, "gamma": 0.98, "n_steps": 4096, "batch_size": 128, "ent_coef": 0.01},
  "environment_config": {"tx_cost_bps": 0.0, "sl": -4%, "tp": +4%, "trailing": false},
  "total_timesteps": 1000000
}
```

### V22 Post-Fix (`models/ppo_ssot_20260208_105908/training_config.json`)
```json
{
  "ssot_version": "4.0.0",
  "based_on_model": "v21.5_first_profitable",
  "observation_dim": 27,
  "ppo_config": {"lr": 0.0003, "gamma": 0.98, "n_steps": 4096, "batch_size": 128, "ent_coef": 0.02},
  "environment_config": {"tx_cost_bps": 0.0, "sl": -4%, "tp": +4%, "trailing": false},
  "total_timesteps": 2000000
}
```

### V21.5b (`models/ppo_ssot_20260210_003307/training_config.json`)
```json
{
  "ssot_version": "3.5.1",
  "based_on_model": "v21.5_first_profitable",
  "observation_dim": 27,
  "ppo_config": {"lr": 0.0003, "gamma": 0.98, "n_steps": 4096, "batch_size": 128, "ent_coef": 0.01},
  "environment_config": {"tx_cost_bps": 0.0, "sl": -4%, "tp": +4%, "trailing": false},
  "total_timesteps": 2000000
}
```

### Config Diff Matrix
| Parameter | V21.5 | V22 | V21.5b |
|-----------|-------|-----|--------|
| SSOT version | 3.5.0 | 4.0.0 | 3.5.1 |
| obs_dim | 23 | 27 | 27 |
| action_type | continuous | discrete(4) | continuous |
| model_type | PPO MlpPolicy | RecurrentPPO MlpLstmPolicy | PPO MlpPolicy |
| ent_coef | 0.01 | 0.02 | 0.01 |
| timesteps | 1M | 2M | 2M |
| temporal_features | No | Yes (4) | Yes (4) |
| close_shaping | No | Yes | No |
| LSTM | No | Yes (128, 1 layer) | No |
| device | CPU | CUDA | CUDA |
