# Slide 2/7 — L1+L2 FEATURE OPS: Feature Engineering & Datasets

> 4 DAGs | ALL PAUSED (RL track only) | H1/H5 load data directly in L3
> "Prepare normalized observations for RL models. Anti-leakage by design."

```mermaid
flowchart TB
    subgraph L0["L0 DATA (upstream)"]
        OHLCV[("usdcop_m5_ohlcv<br/>5-min bars")]
        MACRO[("macro_indicators_daily<br/>40 variables")]
    end

    subgraph L1["L1 FEATURE COMPUTATION — every 5 min"]
        direction TB
        FR["<b>rl_l1_01_feature_refresh</b><br/>every 5 min 8-12 COT<br/>STATUS: PAUSED<br/><br/>CanonicalFeatureBuilder computes:<br/>log_ret_5m, rsi_9, atr_14, adx_14<br/>dxy_z, vix_z, wti_z, embi_z<br/>vol_5m, trend_z, spread_z...<br/>(18 market features)<br/><br/>Wilder EMA for RSI/ATR/ADX<br/>NOT pandas ewm<br/><br/>Z-score normalize with<br/>TRAINING-ONLY norm_stats<br/>clip to -5, +5"]
        MP["<b>rl_l1_03_model_promotion</b><br/>MANUAL trigger on model approval<br/>STATUS: PAUSED<br/><br/>1. TRUNCATE inference_ready_nrt<br/>2. Recompute ALL historical features<br/>   with NEW norm_stats from approved model<br/>3. Batch insert historical<br/>4. Copy norm_stats to config/"]
    end

    subgraph NRT["inference_ready_nrt Table"]
        NRTTBL[("inference_ready_nrt<br/>FLOAT[] 18 features<br/>+ price + feature_order_hash<br/>+ norm_stats_hash<br/>+ pg_notify trigger")]
    end

    subgraph L2["L2 DATASET BUILD — Manual, pre-training"]
        direction TB
        DS["<b>rl_l2_01_dataset_build</b><br/>MANUAL trigger<br/>STATUS: PAUSED<br/><br/>1. Load OHLCV + Macro<br/>2. Calculator registry (dynamic features)<br/>3. merge_asof backward = T-1 anti-leakage<br/>4. Split by FIXED dates:<br/>   Train: 2019-12 to 2024-12 (70K bars)<br/>   Val:   2025-01 to 2025-06 (7K bars)<br/>   Test:  2025-07 to 2025-12 (7K bars)<br/>5. norm_stats FROM TRAIN ONLY<br/>6. Z-score, clip -5 to +5<br/>7. Save parquets + lineage"]
        DR["<b>rl_l2_02_drift_retrain</b><br/>ON DRIFT EVENT trigger<br/>STATUS: PAUSED<br/><br/>Detect drift via KS test<br/>Rebuild dataset filtered<br/>by drift-affected features"]
    end

    subgraph L2OUT["L2 Output Files"]
        TRAIN["DS_production_train.parquet<br/>DS_production_val.parquet<br/>DS_production_test.parquet<br/>DS_production_norm_stats.json<br/>DS_production_lineage.json"]
    end

    OHLCV --> FR
    MACRO --> FR
    FR -->|"FLOAT 18 + normalize"| NRTTBL
    FR -->|"pg_notify features_ready"| L5_SENSOR["L5 FeatureReadySensor"]
    MP -->|"bulk recompute"| NRTTBL

    OHLCV --> DS
    MACRO --> DS
    DS --> TRAIN
    DR --> TRAIN

    NRTTBL -.->|"consumed by"| RL_L5["RL L5 Inference"]
    TRAIN -.->|"consumed by"| RL_L3["RL L3 Training (PPO)"]

    style L1 fill:#1e3a5f,stroke:#3b82f6,color:#dbeafe
    style L2 fill:#312e81,stroke:#6366f1,color:#e0e7ff
    style NRT fill:#7c2d12,stroke:#ea580c,color:#fed7aa
```

## Anti-Leakage Mechanisms

| Mechanism | Where | How |
|-----------|-------|-----|
| Train-only norm_stats | L2 step 5 | mean/std computed ONLY on training split |
| Macro T-1 shift | L2 step 3 | `merge_asof(direction='backward')` ensures yesterday's macro |
| Fixed date splits | L2 step 4 | Chronological, no random shuffle, dates from SSOT YAML |
| Feature hash validation | L1 write | SHA256 of FEATURE_ORDER stored alongside features |

## Key Contract: Feature Order

```python
FEATURE_ORDER = (
    'log_ret_5m', 'rsi_9', 'atr_14', 'adx_14',
    'vol_5m', 'trend_z', 'spread_z', 'dxy_z',
    'vix_z', 'wti_z', 'embi_z', 'ust10y_z',
    'ust2y_z', 'ibr_z', 'pct_from_open', 'bar_position',
    'dow_encoded', 'hour_sin'
)
FEATURE_ORDER_HASH = SHA256(FEATURE_ORDER)[:16]
```

> If hash mismatches between L1 and L5 at inference time, predictions are garbage.

## Why H1/H5 Skip L1+L2

H1 and H5 forecasting tracks load data DIRECTLY in their L3 DAGs from seed parquets.
They do NOT use `inference_ready_nrt` or the L2 dataset builder.
L1+L2 exist ONLY for the RL track (PPO on 5-min bars, deprioritized).
