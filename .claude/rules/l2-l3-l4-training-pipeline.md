# Rule: L2→L3→L4 Training Pipeline

> Governs the full training pipeline: dataset build, model training, and backtest validation.
> Orchestrator: `scripts/run_ssot_pipeline.py` (Contract: CTR-PIPELINE-RUNNER-001)
> Created: 2026-02-12

---

## Architecture

```
pipeline_ssot.yaml (SSOT config)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ L2: Dataset Build (SSOTDatasetBuilder)                          │
│   OHLCV parquet + macro parquet                                 │
│   → calculator_registry (dynamic from SSOT)                     │
│   → train-only norm_stats (anti-leakage)                        │
│   → train.parquet, val.parquet, test.parquet, norm_stats.json   │
└─────────────────────┬───────────────────────────────────────────┘
                      │ L2 result dict
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ L3: Model Training (PPO / RecurrentPPO)                         │
│   Load train.parquet + val.parquet + norm_stats.json            │
│   → create_env_config(stage="training")                         │
│   → TradingEnvironment(df, norm_stats, config)                  │
│   → model.learn() with DeterministicEvalCallback                │
│   → best_model.zip, final_model.zip, norm_stats.json copy       │
└─────────────────────┬───────────────────────────────────────────┘
                      │ L3 result dict
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ L4: Backtest Validation                                         │
│   Load best_model.zip + val+test parquets + norm_stats.json     │
│   → create_env_config(stage="backtest")                         │
│   → validate_training_backtest_parity()                         │
│   → run_backtest_loop() → equity_curve                          │
│   → calculate_backtest_metrics() → validate_gates()             │
└─────────────────────────────────────────────────────────────────┘
```

### CLI Usage
```bash
# Full pipeline (L2→L3→L4)
python scripts/run_ssot_pipeline.py --stage all --seed 42

# Multi-seed (5 seeds, selects best by eval reward)
python scripts/run_ssot_pipeline.py --stage all --multi-seed

# Individual stages
python scripts/run_ssot_pipeline.py --stage l2
python scripts/run_ssot_pipeline.py --stage l3 --seed 42
python scripts/run_ssot_pipeline.py --stage l4
python scripts/run_ssot_pipeline.py --stage l4-val   # Validation only (gate)
python scripts/run_ssot_pipeline.py --stage l4-test   # Test only (final OOS)

# Specific experiment config
python scripts/run_ssot_pipeline.py --config config/experiments/exp_asym_001.yaml --multi-seed
```

---

## L2: Dataset Build

**File**: `src/data/ssot_dataset_builder.py` (Contract: CTR-L2-DATASET-BUILDER-001)

### What It Does
1. Loads OHLCV from `seeds/latest/usdcop_m5_ohlcv.parquet`
2. Loads macro from `data/pipeline/04_cleaning/output/macro_daily_clean.parquet`
3. Renames macro columns via `pipeline_ssot.yaml → paths.sources.macro_column_map`
4. Calculates features dynamically from SSOT feature definitions (calculator registry)
5. Merges macro → OHLCV via `merge_asof(direction='backward')` (T-1 anti-leakage)
6. Splits into train/val/test by fixed dates from SSOT `date_ranges`
7. Computes normalization stats **from training split ONLY** (anti-leakage)
8. Applies Z-score normalization, clips [-5, 5]
9. Saves parquets + norm_stats.json + lineage.json

### Outputs

**Files**:
```
{output_dir}/
├── DS_production_train.parquet       ← Training dataset (normalized)
├── DS_production_val.parquet         ← Validation dataset (normalized with train stats)
├── DS_production_test.parquet        ← Test dataset (normalized with train stats)
├── DS_production_norm_stats.json     ← Train-only normalization stats (CRITICAL)
├── DS_production_lineage.json        ← Feature order, hash, row counts
└── DS_production_descriptive_stats.json  ← QA statistics
```

**L2 Result Dict** (passed to L3):
```python
{
    "train_path": str,          # Path to train.parquet
    "val_path": str,            # Path to val.parquet
    "test_path": str,           # Path to test.parquet
    "norm_stats_path": str,     # Path to norm_stats.json
    "feature_columns": List[str],  # Market feature names (no state features)
    "observation_dim": int,     # len(market) only (state added at runtime)
    "lineage": dict,            # Feature order hash, row counts
}
```

**Dataset Schema** (each parquet):
- **Market features**: `log_ret_5m`, `rsi_9`, `dxy_z`, etc. (Z-score normalized)
- **Auxiliary columns**: `raw_log_ret_5m` (raw, for PnL calculation), `close` (raw price)
- **Index**: datetime (tz-naive, America/Bogota stripped)

### Anti-Leakage Mechanisms
1. **Train-only norm_stats**: `mean`, `std` computed exclusively on training split
2. **Macro T-1 shift**: `merge_asof(direction='backward')` ensures macro data is from previous day
3. **Fixed date splits**: No random shuffling, chronological splits from SSOT

### Feature Calculator Registry
- Location: `src/features/calculator_registry.py`
- Calculators: `calculate_log_returns`, `calculate_rsi_wilders`, `calculate_volatility_pct`, `calculate_trend_z`, `calculate_macro_zscore`, `calculate_spread_zscore`, `calculate_pct_change`
- RSI uses **Wilder's EMA** (NOT pandas `ewm`)
- Each feature in SSOT has `calculator` + `params` → registry dispatches dynamically

---

## L3: Model Training

**Files**:
- Orchestrator: `scripts/run_ssot_pipeline.py` → `run_l3_training()`
- Multi-seed: `scripts/run_ssot_pipeline.py` → `run_l3_multi_seed()`
- Engine (alternative): `src/training/engine.py` → `TrainingEngine`
- PPO trainer: `src/training/trainers/ppo_trainer.py`

### What It Does
1. Loads train/val parquets + norm_stats.json from L2 paths
2. Creates `TradingEnvConfig` via shared factory `create_env_config(stage="training")`
3. Creates `TradingEnvironment(df, norm_stats, config)` for train and eval
4. Configures PPO or RecurrentPPO from SSOT (`lstm.enabled`)
5. Sets reproducible seeds **before** model creation
6. Trains with `model.learn()` using `DeterministicEvalCallback`
7. Saves `best_model.zip` (best eval reward) + `final_model.zip` + config snapshot

### Reproducibility (CRITICAL)

```python
set_reproducible_seeds(seed)  # BEFORE model creation
```

Must propagate to:
1. `random.seed(seed)`
2. `np.random.seed(seed)`
3. `torch.manual_seed(seed)` + CUDA variants
4. `torch.backends.cudnn.deterministic = True`
5. `PPO(..., seed=seed)` — SB3 internal seeding

**DeterministicEvalCallback**: Reseeds eval env's `np_random` before each evaluation round → same eval episodes every time → reproducible `best_model` selection.

### Model Selection
- `best_model.zip` = best mean eval reward during training (saved by `DeterministicEvalCallback`)
- `final_model.zip` = model at end of training
- **Always prefer `best_model.zip`** — but eval reward is a noisy predictor of OOS (Rule 6)

### Outputs

**Files**:
```
models/ppo_ssot_{timestamp}/
├── best_model.zip            ← Best by eval reward (prefer this)
├── final_model.zip           ← Final trained model
├── norm_stats.json           ← Copy of L2 norm_stats (CRITICAL for L4/L5)
├── training_config.json      ← Config snapshot
├── logs/evaluations.npz      ← Eval callback results
└── checkpoints/              ← Periodic checkpoints
```

**L3 Result Dict** (passed to L4):
```python
{
    "model_path": str,          # Path to final_model.zip
    "model_dir": str,           # Directory containing all artifacts
    "norm_stats_path": str,     # Path to norm_stats.json (copy of L2's)
    "training_config": dict,    # Config snapshot
    "timestamp": str,           # Training timestamp
    "use_lstm": bool,           # True if RecurrentPPO
}
```

### Multi-Seed Training
- Seeds: `[42, 123, 456, 789, 1337]` (mandatory 5)
- Trains each seed sequentially via `run_l3_training()`
- Selects best by max eval reward (from `evaluations.npz`)
- Reports CV (coefficient of variation) — CV > 30% triggers warning
- Returns all results + best model path

### PPO vs RecurrentPPO
- `pipeline_ssot.yaml → lstm.enabled: false` → PPO + MlpPolicy (use CPU)
- `pipeline_ssot.yaml → lstm.enabled: true` → RecurrentPPO + MlpLstmPolicy (GPU OK)
- Policy layers from `training.network.policy_layers` (default: [256, 256])
- Device: auto-detect (`cuda` if available, else `cpu`)

---

## L4: Backtest Validation

**File**: `scripts/run_ssot_pipeline.py` → `run_l4_backtest()`, `run_l4_validation()`

### What It Does
1. Loads `best_model.zip` (falls back to `final_model.zip`)
2. Loads val + test parquets, concatenates for full 2025 coverage
3. Validates training↔backtest parity (`validate_training_backtest_parity()`)
4. Creates env with `create_env_config(stage="backtest")`
5. Runs `run_backtest_loop()` → equity curve
6. Calculates metrics → validates gates

### Model Loading
```python
# PPO vs RecurrentPPO based on l3_result["use_lstm"]
if is_lstm:
    model = RecurrentPPO.load(path, device="cpu")
else:
    model = PPO.load(path, device="cpu")
```

### LSTM State Tracking in Backtest (CRITICAL)
RecurrentPPO requires explicit state management:
```python
if is_recurrent:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    episode_starts = np.array([False])  # Only True on first step
```
Without this, LSTM resets to zeros every step, defeating temporal memory.

### L4 Variants
- **`l4` / `l4-test`**: Combined val+test backtest (default, full 2025 coverage)
- **`l4-val`**: Validation-only backtest (gate for promotion, must pass before test)

### Validation Gates (from `pipeline_ssot.yaml → backtest`)

| Gate | Default | Description |
|------|---------|-------------|
| `min_return_pct` | 0.0% | Must be break-even or positive |
| `min_sharpe_ratio` | 0.0 | Must have positive risk-adjusted return |
| `max_drawdown_pct` | 20.0% | Cannot exceed max drawdown |
| `min_trades` | 30 | Must trade actively |
| `min_win_rate` | 35% | Cannot be worse than random |

### Metrics (SSOT Constants)
- `bars_per_year`: From `pipeline_ssot.yaml → trading_schedule.bars_per_year` (19,656)
- `bars_per_day`: From `pipeline_ssot.yaml → trading_schedule.bars_per_day` (78)
- Sharpe annualized with `√bars_per_year`
- Trade count = `portfolio.winning_trades + portfolio.losing_trades` (closed trades only)
- Profit factor = PnL-based (sum of positive returns / abs sum of negative returns)

---

## Data Contracts Between Layers

### L2 → L3 Contract

| What | Details |
|------|---------|
| Datasets | Parquets with market features (normalized) + auxiliary columns (`raw_log_ret_5m`, `close`) |
| Norm stats | JSON with `{feature_name: {mean, std, min, max}}` from training split only |
| Feature order | List of feature names in canonical order |
| Feature hash | SHA256 of feature name list → stored in lineage |
| State features | NOT in dataset — added at runtime by `TradingEnvironment` |

**Critical**: L3 MUST use L2's `norm_stats.json`. If L3 recomputes normalization, features will be double-normalized → garbage model.

### L3 → L4 Contract

| What | Details |
|------|---------|
| Model | `best_model.zip` (PPO or RecurrentPPO) |
| Norm stats | Copy of L2's `norm_stats.json` (saved alongside model) |
| LSTM flag | `use_lstm: bool` in result dict (controls model loading class) |
| Config snapshot | `training_config.json` for audit |

### Environment Parity (Training ↔ Backtest)

Shared factory `create_env_config()` ensures consistency. Differences:

| Parameter | Training | Backtest | Why |
|-----------|----------|----------|-----|
| `reward_interval` | 25 (from SSOT) | **1** | Backtest evaluates every bar |
| `max_drawdown_pct` | 15% (from SSOT) | **99%** | No early termination in backtest |
| `initial_balance` | From `environment` | From `backtest` | May differ |

Everything else (costs, thresholds, stops, min_hold_bars) MUST be identical.

Validated by `config.validate_training_backtest_parity()` before L4 runs.

---

## Configuration (SSOT)

**Primary**: `config/pipeline_ssot.yaml`
**Loader**: `src/config/pipeline_config.py` → `load_pipeline_config()`

### Key Sections

```yaml
features:
  market_features:          # Feature definitions (name, calculator, params, order)
  state_features:           # State features added at runtime (position, unrealized_pnl, etc.)

training:
  ppo:                      # PPO hyperparameters (lr, gamma, ent_coef, n_steps, batch_size)
  environment:              # Training env config (costs, stops, thresholds, episode_length)
  schedule:                 # total_timesteps, eval_freq, n_eval_episodes
  network:                  # policy_layers (default: [256, 256])
  lstm:                     # enabled, hidden_size, n_layers

backtest:                   # Backtest config (costs, stops, gates, initial_capital)

date_ranges:                # Fixed train/val/test splits
  use_fixed_dates: true
  train_start/end, val_start/end, test_start/end

paths:
  sources:                  # OHLCV and macro paths + macro_column_map
  l2_output:                # L2 output directory
  l3_output:                # L3 models directory

trading_schedule:           # bars_per_day (78), bars_per_year (19656)
```

### Lazy Loading Pattern
```python
from src.config.pipeline_config import load_pipeline_config
config = load_pipeline_config()  # Cached after first call

features = config.get_market_features()       # List[FeatureDefinition]
obs_dim = config.get_observation_dim()         # int
schedule = config.get_training_schedule()      # dict
state_features = config.get_state_feature_names()  # List[str]
```

---

## Feature Contract

**File**: `src/core/contracts/feature_contract.py` (Contract: CTR-FEATURE-001)

### Resolution Priority
1. `experiment_ssot.yaml` (via `experiment_loader`) — preferred
2. `feature_contracts_registry` — legacy fallback
3. Hardcoded fallback (20 features: 18 market + 2 state)

### Key Exports
- `FEATURE_ORDER: Tuple[str, ...]` — canonical feature name order
- `OBSERVATION_DIM: int` — total observation size
- `FEATURE_ORDER_HASH: str` — SHA256 hash for validation
- `FEATURE_CONTRACT: FeatureContract` — validates observations

### Validation
```python
from src.core.contracts.feature_contract import validate_feature_vector
is_valid = validate_feature_vector(obs_array, strict=True)  # Raises on error
```

---

## DO NOT

- Do NOT compute norm_stats in L3 — always use L2's `norm_stats.json` (double normalization risk)
- Do NOT skip `set_reproducible_seeds()` before model creation or any stochastic operation
- Do NOT select best model by eval reward alone — eval reward is a poor predictor of OOS (Rule 6)
- Do NOT change env config between training and backtest except `reward_interval` (=1) and `max_drawdown_pct` (=99%)
- Do NOT use GPU for PPO MlpPolicy — CPU is faster on RTX 3050 laptop (~300 FPS vs 65-187 GPU)
- Do NOT use `RecurrentPPO` without LSTM state tracking in backtest loop
- Do NOT load `RecurrentPPO` with `PPO.load()` — use `RecurrentPPO.load()`
- Do NOT count open trades — use `portfolio.winning_trades + portfolio.losing_trades` (closed only)
- Do NOT use `reward_interval > 1` in backtest — must be 1 for accurate equity curve
- Do NOT set `max_drawdown_pct < 99%` in backtest — early termination prevents full evaluation
