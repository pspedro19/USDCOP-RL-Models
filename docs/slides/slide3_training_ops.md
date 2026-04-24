# Slide 3/7 — L3+L4 TRAINING OPS: Model Training & Validation

> 8 DAGs | 1 ACTIVE (H5-L3) + 7 PAUSED | 9 models x 7 horizons = 63 variants
> "Sunday: retrain everything. L4: backtest OOS + 5 automatic gates + human approval."

```mermaid
flowchart TB
    subgraph INPUTS["INPUT DATA (from L0)"]
        OHLCV_SEED["seeds/latest/<br/>usdcop_daily_ohlcv.parquet<br/>~3K rows, 2020-2026"]
        MACRO_CLEAN["MACRO_DAILY_CLEAN.parquet<br/>17 macro cols cleaned"]
        M5_SEED["usdcop_m5_ohlcv.parquet<br/>~90K rows, 5-min bars<br/>(RL only)"]
    end

    subgraph L3["L3 MODEL TRAINING — Sunday"]
        direction TB
        H5L3["<b>forecast_h5_l3_weekly_training</b><br/>Sun 01:30 COT | ACTIVE<br/><br/>1. validate_data_freshness<br/>   OHLCV less than 3d, Macro less than 7d<br/>2. load_and_build_features (23 features)<br/>3. train Ridge + BayesianRidge<br/>   target: ln close_t+5 / close_t<br/>   window: 2020-01 to last Friday<br/>   EXPANDING every week<br/>4. validate_models (collapse detect)<br/>5. persist_models to .pkl<br/>6. MLflow log_artifact + log_metrics"]
        H1L3["<b>forecast_h1_l3_weekly_training</b><br/>Sun 01:00 COT | PAUSED<br/><br/>Same pipeline but:<br/>9 models x 7 horizons = 63 variants<br/>target: ln close_t+H / close_t<br/>H = 1, 5, 10, 15, 20, 25, 30 days"]
        MISC["<b>forecast_l3_01_model_training</b><br/>MANUAL | PAUSED<br/>Generic forecasting trainer"]
    end

    subgraph MODELS_63["9 MODELS x 7 HORIZONS = 63 Variants"]
        direction TB
        LINEAR["LINEAR<br/>Ridge | Bayesian Ridge | ARD<br/>requires_scaling = True"]
        BOOST["BOOSTING<br/>XGBoost | LightGBM | CatBoost<br/>requires_scaling = False"]
        HYBRID["HYBRID (Linear+Boosting)<br/>Hybrid XGB | Hybrid LGB | Hybrid CB<br/>Stacked ensemble"]
        HORIZ["HORIZONS<br/>H=1: tomorrow<br/>H=5: next week<br/>H=10: 2 weeks<br/>H=15: 3 weeks<br/>H=20: 1 month<br/>H=25: 5 weeks<br/>H=30: 6 weeks"]
    end

    subgraph L4["L4 VALIDATION & APPROVAL — Manual"]
        direction TB
        H5L4["<b>forecast_h5_l4_backtest_promotion</b><br/>MANUAL | PAUSED<br/><br/>1. Run OOS backtest (2025 data)<br/>2. Walk-forward expanding window<br/>3. Calculate metrics<br/>4. Evaluate 5 GATES (Vote 1/2 auto)"]
        H1L4["<b>forecast_h1_l4_backtest_promotion</b><br/>MANUAL | PAUSED"]
        RL_EXP["<b>rl_l4_01_experiment_runner</b><br/>MANUAL | PAUSED<br/>Hyperparameter sweep, 5 seeds"]
        RL_BT["<b>rl_l4_02_backtest_validation</b><br/>MANUAL | PAUSED<br/>Test set final OOS"]
        RL_SCH["<b>rl_l4_03_scheduled_retraining</b><br/>MANUAL | PAUSED"]
        RL_PROM["<b>rl_l4_04_backtest_promotion</b><br/>MANUAL | PAUSED<br/>Gates + model promotion"]
    end

    subgraph GATES["5 AUTOMATIC GATES (Vote 1/2)"]
        G1["min_return_pct > -15%"]
        G2["min_sharpe > 0.0"]
        G3["max_drawdown < 20%"]
        G4["min_trades >= 10"]
        G5["p-value < 0.05"]
    end

    subgraph APPROVAL["HUMAN APPROVAL (Vote 2/2)"]
        DASH["Dashboard /dashboard<br/>Review KPIs + trades + gates<br/>Click Aprobar or Rechazar"]
    end

    subgraph OUTPUT["OUTPUT"]
        MODELS_OUT["h5_weekly_models/latest/<br/>ridge.pkl + br.pkl<br/><br/>h1_daily_models/latest/<br/>9 x .pkl files per horizon"]
        JSON_OUT["public/data/production/<br/>summary_2025.json<br/>approval_state.json<br/>trades/*_2025.json"]
        MLFLOW["MLflow s3://mlflow/<br/>Params + metrics + artifacts"]
    end

    OHLCV_SEED --> H5L3 & H1L3
    MACRO_CLEAN --> H5L3 & H1L3
    M5_SEED --> RL_EXP & RL_BT

    H5L3 --> MODELS_OUT
    H5L3 --> MLFLOW
    H1L3 --> MODELS_OUT
    H1L3 -.-> MODELS_63

    H5L3 -->|"models"| H5L4
    H5L4 --> GATES --> APPROVAL
    H5L4 --> JSON_OUT
    APPROVAL -->|"APPROVED"| DEPLOY["Stage 6: --phase production"]

    style L3 fill:#064e3b,stroke:#10b981,color:#d1fae5
    style L4 fill:#4a1d96,stroke:#8b5cf6,color:#ede9fe
    style GATES fill:#78350f,stroke:#f59e0b,color:#fef3c7
    style APPROVAL fill:#7c2d12,stroke:#ef4444,color:#fecaca
    style MODELS_63 fill:#1e3a5f,stroke:#3b82f6,color:#dbeafe
```

## Walk-Forward Validation (How 63 Variants are Tested)

```
For each model x horizon combination:

    EXPANDING WINDOW (grows each iteration):
    ┌────────────────────────────────────────────────────────────┐
    │ iter 1: Train [2020───2023]  Predict [2023-W01]  actual?  │
    │ iter 2: Train [2020────2023+1w]  Predict [next]  actual?  │
    │ iter 3: Train [2020─────2023+2w]  Predict [next]  actual? │
    │ ...                                                        │
    │ iter N: Train [2020──────────2025]  Predict [last] actual? │
    └────────────────────────────────────────────────────────────┘
    
    Metrics from all iterations:
    DA (Direction Accuracy), RMSE, MAE, R2,
    Sharpe, Profit Factor, MaxDD, Total Return
    
    THEN train FINAL model on ALL data → predict FUTURE
```

## The 63 Variants Matrix

| Model | H=1 | H=5 | H=10 | H=15 | H=20 | H=25 | H=30 |
|-------|-----|-----|------|------|------|------|------|
| Ridge | DA% | DA% | DA% | DA% | DA% | DA% | DA% |
| Bayesian Ridge | DA% | DA% | DA% | DA% | DA% | DA% | DA% |
| ARD | DA% | DA% | DA% | DA% | DA% | DA% | DA% |
| XGBoost | DA% | DA% | DA% | DA% | DA% | DA% | DA% |
| LightGBM | DA% | DA% | DA% | DA% | DA% | DA% | DA% |
| CatBoost | DA% | DA% | DA% | DA% | DA% | DA% | DA% |
| Hybrid XGB | DA% | DA% | DA% | DA% | DA% | DA% | DA% |
| Hybrid LGB | DA% | DA% | DA% | DA% | DA% | DA% | DA% |
| Hybrid CB | DA% | DA% | DA% | DA% | DA% | DA% | DA% |

> 63 backtest rows + 63 forward_forecast rows = 126 rows per week in CSV.
> Only Ridge + BR at H=5 go to PRODUCTION (Smart Simple v2.0).
> The other 61 variants feed the /forecasting DASHBOARD for model comparison.

## RL Experiment Protocol (L4)

| Rule | Enforcement |
|------|-------------|
| ONE variable per experiment | Never change action space + model + features simultaneously |
| 5 seeds minimum | [42, 123, 456, 789, 1337], no exceptions |
| Statistical validation | p < 0.05, DA > 55%, Sharpe > 1.0, CI excludes zero |
| Compare vs baselines | Buy-and-hold (-12.29%), random agent (-4.12%) |
| Frozen SSOT config | Each experiment = complete pipeline_ssot.yaml |
