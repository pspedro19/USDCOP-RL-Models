# DEPRECATED DAGS - V2 Architecture

## Status: DEPRECATED as of 2025-12-16

These DAGs have been replaced by the V3 consolidated architecture.

## Migration Path

### Old (14 DAGs) → New (5 DAGs)

| Old DAG | Status | Replaced By | Action |
|---------|--------|-------------|--------|
| `usdcop_m5__01_l0_intelligent_acquire.py` | DEPRECATED | `v3/l0_ohlcv_realtime.py` | Delete after v3 validation |
| `usdcop_m5__00b_l0_macro_scraping.py` | DEPRECATED | `v3/l0_macro_daily.py` | Delete after v3 validation |
| `usdcop_m5__01b_l0_macro_acquire.py` | DEPRECATED | `v3/l0_macro_daily.py` | Delete after v3 validation |
| `usdcop_m5__02_l1_standardize.py` | REDUNDANT | N/A | Delete immediately |
| `usdcop_m5__03_l2_prepare.py` | REDUNDANT | N/A | Delete immediately |
| `usdcop_m5__04_l3_feature.py` | REDUNDANT | `v3/l1_feature_refresh.py` | Delete after v3 validation |
| `usdcop_m5__04b_l3_llm_features.py` | EXPERIMENTAL | N/A | Archive |
| `usdcop_m5__05_l4_rlready.py` | REDUNDANT | `v3/l1_feature_refresh.py` | Delete after v3 validation |
| `usdcop_m5__06_l5_realtime_inference.py` | BROKEN | `v3/l5_realtime_inference.py` | DELETE (19 vs 15 features!) |
| `usdcop_m5__99_alert_monitor.py` | OK | `v3/alert_monitor.py` | Replace with v3 |
| `l2_causal_deseasonalization.py` | EXPERIMENTAL | N/A | Archive |

## Why V3?

### Problems with V2 (14 DAGs):
1. **Hardcoded features** - No SSOT, config scattered
2. **Dimension mismatch** - Inference DAG expects 19 features, model wants 15
3. **Redundant layers** - L1-L4 unnecessary (data already in DB)
4. **Duplicated code** - Feature calculations in 7+ places
5. **No config reading** - feature_config.json exists but not used

### V3 Architecture (5 DAGs):
1. **l0_ohlcv_realtime.py** - OHLCV acquisition every 5min
2. **l0_macro_daily.py** - Macro scraping 3x/day
3. **l1_feature_refresh.py** - Feature refresh (SQL + Python)
4. **l5_realtime_inference.py** - RL inference (READS CONFIG!)
5. **alert_monitor.py** - System monitoring

### Key Improvements:
- ✅ Reads from `feature_config.json` (SSOT)
- ✅ Correct 15-dim observation (13 features + position + time_normalized)
- ✅ Uses `FeatureBuilder` service (consolidated from 7 locations)
- ✅ time_normalized = (bar-1)/60 → [0, 0.983] ✓
- ✅ No hardcoded features or norm_stats

## Validation Checklist

Before deleting V2 DAGs:

- [ ] V3 DAGs deployed to Airflow
- [ ] l0_ohlcv_realtime.py running successfully for 1 week
- [ ] l0_macro_daily.py running successfully for 1 week
- [ ] l1_feature_refresh.py running successfully for 1 week
- [ ] l5_realtime_inference.py running successfully for 1 week
- [ ] alert_monitor.py running successfully for 1 week
- [ ] All v3 tables exist (inference_features_5m, python_features_5m)
- [ ] No regressions in model performance
- [ ] Equity curve shows expected behavior

## Rollback Plan

If V3 fails:

```bash
# Pause V3 DAGs
airflow dags pause l0_ohlcv_realtime
airflow dags pause l0_macro_daily
airflow dags pause l1_feature_refresh
airflow dags pause l5_realtime_inference
airflow dags pause alert_monitor

# Re-enable V2 DAGs (from this deprecated folder)
cp deprecated/usdcop_m5__01_l0_intelligent_acquire.py ../
cp deprecated/usdcop_m5__00b_l0_macro_scraping.py ../
# ... etc

# Unpause V2 DAGs
airflow dags unpause usdcop_m5__01_l0_intelligent_acquire
# ... etc
```

## Timeline

- **2025-12-16**: V3 DAGs created
- **2025-12-23**: 1 week validation period
- **2025-12-30**: Delete V2 DAGs (if validation successful)

## Questions?

Contact: Pedro @ Lean Tech Solutions

---

**DO NOT use these DAGs in production!**

Use v3/ DAGs instead.
