# L2 Pipeline - Quick Reference Guide

## What is L2?

**L2 Prepare** takes L1 standardized data and produces ML-ready features by:
1. Removing extreme returns (winsorization)
2. Removing time-of-day patterns (deseasonalization)
3. Creating normalized features
4. Quality gating (STRICT vs FLEX variants)

**Input**: L1 standardized data (55K rows, 929 episodes)
**Output**: 
- STRICT: 245-250 complete episodes (60 bars each)
- FLEX: 250-255 episodes (with 1 padded bar for 59/60 episodes)

---

## Key Metrics Overview

### 1. Winsorization Rate (Most Important)
- **What**: % of returns clipped to remove outliers
- **Pass Threshold**: <= 1.0%
- **Why**: Keeps extreme outliers from dominating RL training
- **How**: 4-sigma clipping using MAD-based robust scale
- **Display**: Large green/yellow/red metric card

### 2. Deseasonalization Coverage
- **What**: % of bars with HOD pattern removed
- **Pass Threshold**: >= 99%
- **Why**: Makes returns stationary for modeling
- **How**: (ret - hourly_median) / hourly_scale
- **Display**: Coverage gauge

### 3. Data Quality Flags
- **PERFECT**: 60/60 bars, 0 stale
- **OK**: 60/60 bars, stale <= 1%, winsor <= 0.5%
- **WARN**: 60/60 bars, stale <= 2%, winsor <= 1%
- **MARGINAL**: Everything else
- **Display**: Pie chart or stacked bar

### 4. STRICT vs FLEX Completeness
- **STRICT**: Count of 60/60 episodes only (for RL)
- **FLEX**: Count of all 60-bar episodes (includes padded)
- **Padding**: Max 1 missing bar per episode
- **Display**: Side-by-side counts

---

## Quality Gates (GO/NO-GO)

| Gate | Metric | Pass | Fail | Impact |
|------|--------|------|------|--------|
| Winsorization | `winsor_rate_pct` | <= 1.0% | > 1.0% | HARD STOP |
| HOD Median | `hod_median_abs` | <= 0.05 | > 0.05 | HARD STOP |
| HOD Scale | `hod_mad_range` | [0.8, 1.2] | else | HARD STOP |
| Deseason Var | `ret_deseason_std` | [0.8, 1.2] | else | HARD STOP |
| NaN Rate | `nan_rate_pct` | <= 0.5% | > 0.5% | WARNING |
| OHLC Violations | Count | == 0 | > 0 | WARNING |
| Stale Rate | `is_stale %` | <= 2% | > 2% | WARNING |

---

## Per-Episode Quality Flags

```
STRICT_PERFECT  → 60 bars, 0 stale, 0 winsor
STRICT_OK       → 60 bars, stale <= 1%, winsor <= 0.5%
STRICT_WARN     → 60 bars, stale <= 2%, winsor <= 1%
STRICT_MARGINAL → Anything worse than WARN
```

**Display**: Distribution chart (should be 80%+ PERFECT+OK)

---

## Frontend Display Checklist

### Must Display (Above Fold)
- [ ] Winsorization rate (0-1.0% range)
- [ ] Deseasonalization coverage (95-100% range)
- [ ] STRICT episodes count
- [ ] FLEX episodes count
- [ ] Quality status badge (GO/NO-GO)

### Should Display (Below Fold)
- [ ] Stale rate %
- [ ] Range outlier rate %
- [ ] NaN rate %
- [ ] OHLC violations count
- [ ] Quality flag distribution (pie/bar)

### Can Display (Tabs/Expandable)
- [ ] Per-hour winsorization rates
- [ ] HOD baseline table
- [ ] Return distribution (pre/post deseason)
- [ ] Range distribution
- [ ] Outlier list (top 20)
- [ ] Episode quality detail table
- [ ] Coverage lineage (L1 → L2 → L3)

---

## Formulas (for Frontend Tooltips)

### Winsorization
```
For each hour of day:
  threshold = median ± 4 × robust_scale
  where robust_scale = max(
    1.4826 × MAD,
    (Q75-Q25) / 1.349,
    floor (8.5-10.0 bps)
  )
  
clipped_return = clip(raw_return, threshold_lo, threshold_hi)
```

### Deseasonalization
```
deseasonalized_return = (clipped_return - hod_median) / hod_scale
  
Target: mean ≈ 0, std ≈ 1.0 (unit variance)
```

### Range Normalization
```
range_bps = (high - low) / close × 10000
range_normalized = range_bps / p95_range_per_hour
  
Typical values: 0-2 (below 95th percentile = 0-1)
```

---

## Column Groups

### Tier 1: Required for L3
- `episode_id`, `t_in_episode`
- `open`, `high`, `low`, `close`
- `ret_deseason` (normalized return)
- `range_norm` (normalized range)
- `atr_14_norm` (volatility)

### Tier 2: Quality Flags
- `ret_winsor_flag` (was return clipped?)
- `is_outlier_range` (range > p95?)
- `is_stale` (repeated OHLC?)
- `is_missing_bar` (FLEX only)

### Tier 3: Reference/Audit
- `hod_median`, `hod_mad` (hour-of-day baselines)
- `ret_log_1` (raw return)
- `ret_log_1_winsor` (clipped return)
- `range_bps` (raw range)

---

## Sample API Response

```json
{
  "layer": "l2",
  "run_id": "2025-10-23-abc123",
  "quality_metrics": {
    "winsorization_rate_pct": 0.65,
    "hod_median_abs": 0.002,
    "hod_mad_mean": 0.95,
    "nan_rate_pct": 0.1,
    "indicator_count": 67
  },
  "datasets": {
    "strict": {
      "episodes": 245,
      "rows": 14700,
      "missing_bars": 0,
      "pass": true
    },
    "flex": {
      "episodes": 250,
      "rows": 15000,
      "missing_bars": 5,
      "pass": true
    }
  },
  "quality_gates": {
    "winsor_rate_pct": { "pass": true, "value": 0.65, "threshold": 1.0 },
    "hod_mad_range": { "pass": true, "value": 0.95, "threshold": "[0.8,1.2]" },
    "hod_median_abs": { "pass": true, "value": 0.002, "threshold": 0.05 },
    "nan_rate_pct": { "pass": true, "value": 0.1, "threshold": 0.5 }
  },
  "pass": true
}
```

---

## Debugging Checklist

**If winsor_rate > 1%**:
- Check specific hours with highest rates
- May indicate market volatility spike
- Check if price outliers are legitimate gaps

**If deseason coverage < 99%**:
- Check for NaN hours in HOD baseline
- May indicate insufficient L1 data for some hours
- Check if data loading from L1 failed

**If hod_mad_range out of [0.8, 1.2]**:
- Scale mismatch from L1
- May indicate frozen recipe not applied correctly
- Check L1 baseline calculation

**If episode quality < 80% OK+PERFECT**:
- Too many stale bars from L1
- May need to investigate source data quality
- Check market conditions during that period

---

## L2 Files Generated

| File | Format | Size | Purpose |
|------|--------|------|---------|
| `data_premium_strict.parquet` | Parquet | ~100MB | STRICT episodes only |
| `data_premium_flex.parquet` | Parquet | ~105MB | STRICT + padded 59/60 |
| `quality_metrics.json` | JSON | ~50KB | Summary stats |
| `dq_counts_daily.csv` | CSV | ~20KB | Per-episode metrics |
| `outlier_report.csv` | CSV | ~50KB | Outlier details |
| `coverage_report.json` | JSON | ~10KB | Lineage/coverage |
| `episode_quality.csv` | CSV | ~50KB | Quality flags per episode |
| `hod_baseline.csv` | CSV | ~5KB | HOD reference stats |

---

## Integration with L3

L3 Feature Engineering expects:
- STRICT dataset: 100+ episodes minimum
- Columns: `ret_deseason`, `range_norm`, `atr_14_norm`, `hour_cot`
- No NaNs in data (except first 14 bars for ATR)
- Episode boundaries clearly marked

L2 guarantees:
- All required columns present
- Data contract validated
- Quality gates passed
- Ready for causal feature engineering

---

## Performance Characteristics

- **Processing time**: ~2-5 minutes per 1000 episodes
- **Throughput**: ~300-500 bars/second
- **Memory**: ~500MB for typical day
- **Storage**: ~100-110MB per dataset variant
- **Compression ratio**: 8-10x (parquet vs CSV)

---

## Related Documentation

- [L1 Standardization](./L1_STANDARDIZED_DATA.md) - Input source
- [L3 Feature Engineering](./L3_FEATURE_ENGINEERING.md) - Downstream consumer
- [Storage Registry](../config/storage.yaml) - Configuration
- [Quality Gates Config](../config/storage.yaml#quality_gates) - Thresholds
