# L2 Pipeline: Key Outputs, Contracts, and Metrics

## Executive Summary

The L2 (Prepare) layer transforms L1 standardized data into ML-ready features through:
- **Winsorization**: Robust clipping of return outliers (4-sigma, MAD-based)
- **Deseasonalization**: HOD (Hour-of-Day) baseline removal using robust z-score
- **Quality Gating**: Dual dataset variants (STRICT: 60/60 only, FLEX: 59/60 padded)
- **60+ Technical Indicators**: Calculated on premium hours (8 AM - 1 PM COT)

---

## 1. KEY METRICS L2 PRODUCES

### 1.1 Winsorization Metrics
**Purpose**: Remove extreme return values while preserving market structure

| Metric | Source | Definition | Display Priority |
|--------|--------|-----------|-------------------|
| `winsor_rate_pct` | Window during day | % of returns clipped | **HIGH** |
| `thr_lo`, `thr_hi` | Per-hour HOD | Lower/upper clipping bounds | **MEDIUM** |
| `ret_log_1_winsor` | Clipped returns | Actual winsorized return series | **HIGH** |
| `ret_winsor_flag` | Boolean flag | Which bars were winsorized | **MEDIUM** |

**Winsorization Recipe (Frozen)**:
```
For each hour:
  MAD_scaled = 1.4826 × median_absolute_deviation
  IQR_scaled = (Q75 - Q25) / 1.349
  floor_bps = 10.0 bps (hour 8) or 8.5 bps (other hours)
  scale_h = max(MAD_scaled, IQR_scaled, floor_bps)
  
  threshold_lo = median - 4.0 × scale_h
  threshold_hi = median + 4.0 × scale_h
  
  clipped_value = clip(original, threshold_lo, threshold_hi)
```

**Quality Gate**: `winsor_rate_pct <= 1.0%`

---

### 1.2 Deseasonalization Outputs
**Purpose**: Remove systematic HOD patterns to make returns stationary

| Metric | Source | Definition | Display Priority |
|--------|--------|-----------|-------------------|
| `hod_median` | L1 baseline | Median return per hour | **HIGH** |
| `hod_mad` | L1 baseline | Median absolute deviation per hour | **HIGH** |
| `ret_deseason` | Calculated | Deseasonalized returns (z-score normalized) | **HIGH** |
| `ret_deseason_std` | Distribution stat | Std dev of deseasonalized returns | **HIGH** |

**Deseasonalization Formula**:
```
ret_deseason = (ret_log_1_winsor - hod_median) / (scale_h + epsilon)

Where:
  scale_h = max(MAD_scaled, IQR_scaled, floor_bps)
  epsilon = 1e-10 (prevent division by zero)
```

**Quality Gates**:
- `hod_median_abs <= 0.05`: Median near zero
- `hod_mad_range ∈ [0.8, 1.2]`: Scale stable
- `ret_deseason_std ∈ [0.8, 1.2]`: Normalized returns unit variance

**Coverage**: `>= 99%` of bars deseasonalized

---

### 1.3 Range Normalization Metrics
**Purpose**: Normalize intrabar volatility by hour-of-day patterns

| Metric | Source | Definition | Display Priority |
|--------|--------|-----------|-------------------|
| `range_bps` | Raw OHLC | High - Low as basis points | **MEDIUM** |
| `range_norm` | Normalized | range_bps / p95_range_bps | **MEDIUM** |
| `p95_range_bps` | L1 baseline | 95th percentile range per hour | **MEDIUM** |

**Formula**:
```
range_bps = (high - low) / close × 10000
range_norm = range_bps / p95_range_bps

Note: Uses frozen L1 p95 baselines (not recalculated in L2)
```

---

### 1.4 ATR (Average True Range)
**Purpose**: Measure volatility across episode boundaries

| Metric | Source | Definition | Display Priority |
|--------|--------|-----------|-------------------|
| `atr_14` | EMA calculation | 14-period ATR in price units | **LOW** |
| `atr_14_norm` | Normalized | ATR / close × 10000 (basis points) | **MEDIUM** |

**Calculation**:
```
For each episode:
  TR_i = max(H-L, |H-Close_{i-1}|, |L-Close_{i-1}|)
  ATR = EMA(TR, period=14)
  
Note: Reset at episode boundaries, first 14 bars = NaN
```

---

## 2. L2 CONTRACTS & ASSERTIONS

### 2.1 Required Columns (Data Contract)
```python
REQUIRED_COLUMNS = [
    # Primary identifiers
    'episode_id', 't_in_episode', 'time_utc', 'time_cot',
    
    # OHLCV (from L1, unchanged)
    'open', 'high', 'low', 'close', 'volume',
    
    # Base features (calculated in L2 Step 1)
    'ohlc4',           # (O+H+L+C)/4
    'ret_log_1',       # log return over 1 bar
    'range_bps',       # (H-L)/C × 10000
    
    # Winsorization (Step 3)
    'ret_log_1_winsor',     # clipped return
    'ret_winsor_flag',      # boolean flag
    'is_outlier_ret',       # alias for winsor_flag
    'is_outlier_range',     # range > p95
    
    # Deseasonalization (Step 4)
    'ret_deseason',         # (ret_winsor - hod_median) / scale
    'range_norm',           # range / p95
    'hod_median',           # reference for audit
    'hod_mad',              # reference for audit
    
    # ATR (Step 4)
    'atr_14',               # 14-period ATR
    'atr_14_norm',          # ATR / close × 10000
    'tr',                   # true range
    
    # Metadata from L1
    'hour_cot', 'minute_cot', 'is_stale', 'is_valid_bar',
    'ohlc_valid', 'is_ohlc_violation', 'quality_flag'
]
```

### 2.2 Quality Assertions (from storage.yaml)
```yaml
l2:
  winsor_rate_pct: "<= 1.0"          # Max 1% of returns clipped
  hod_mad_range: "[0.8, 1.2]"        # HOD scale stable
  hod_median_abs: "<= 0.05"          # Median returns near zero
  nan_rate_pct: "<= 0.5"             # Max 0.5% missing values
```

### 2.3 Dataset Variant Contracts

**STRICT Variant** (`data_premium_strict.parquet`)
- Episodes: 60/60 bars only
- Rows: 60 × num_complete_episodes
- Missing bars: 0
- Quality flag: `STRICT_PERFECT`, `STRICT_OK`, `STRICT_WARN`, `STRICT_MARGINAL`
- Use case: RL training, feature engineering requiring complete episodes

**FLEX Variant** (`data_premium_flex.parquet`)
- Episodes: All (60/60 + padded 59/60)
- Rows: 60 × (num_complete + num_59bar_episodes)
- Missing bars: Up to 1 per episode
- Padding: Forward-filled or marked with `is_missing_bar=True`
- Use case: Analysis requiring complete date coverage

### 2.4 Episode Quality Flags
```python
quality_flag_l2 = {
    'STRICT_PERFECT':  n_bars==60 & n_valid==60 & n_stale==0,
    'STRICT_OK':       n_stale/n_bars <= 1% & n_winsor/n_bars <= 0.5%,
    'STRICT_WARN':     n_stale/n_bars <= 2% & n_winsor/n_bars <= 1%,
    'STRICT_MARGINAL': anything else,
}
```

---

## 3. L2 REPORTS & STATISTICS

### 3.1 Generated Reports
The DAG produces 5 critical reports:

1. **quality_metrics.json**
   - Aggregated quality stats across entire dataset
   - Per-hour statistics (stale rate, winsor rate, etc.)
   - Return distribution (mean, std, min, max)
   - Range distribution (p50, p95)

2. **dq_counts_daily.csv**
   - One row per episode (day)
   - Columns: `bars_day`, `stale_count`, `winsor_count`, `valid_bars`, `status_day`
   - Used for daily monitoring

3. **outlier_report.csv**
   - Rows flagged as winsorized or range outliers
   - Columns: `episode_id`, `t_in_episode`, `ret_log_1`, `ret_log_1_winsor`, `outlier_type`
   - Used for audit and anomaly investigation

4. **coverage_report.json**
   - Input/output row counts (lineage tracking)
   - Deseasonalization coverage %
   - Winsorization coverage %

5. **episode_quality.csv** (from gating step)
   - Per-episode quality flag
   - Stale/winsor rates per episode
   - Used to identify problematic trading days

### 3.2 Sample Quality Metrics Output
```json
{
  "timestamp": "2025-10-23T14:00:00Z",
  "execution_date": "2025-10-23",
  "episodes": {
    "strict": 245,
    "flex": 250,
    "rejected": 5
  },
  "bar_level": {
    "stale_rate_pct": 0.8,
    "ohlc_violations": 0,
    "winsor_ret_pct": 0.65,
    "range_outlier_pct": 0.5
  },
  "statistics": {
    "ret_log_1": {
      "mean": 0.0002,
      "std": 0.0045,
      "min": -0.0234,
      "max": 0.0198
    },
    "ret_deseason": {
      "mean": -0.0001,
      "std": 0.95,
      "min": -3.2,
      "max": 2.8
    }
  }
}
```

---

## 4. WHAT DOWNSTREAM LAYERS (L3) EXPECT

### 4.1 L3 Input Contracts
From `usdcop_m5__04_l3_feature.py`:

```python
L3_EXPECTS = {
    'strict': {
        'min_episodes': 100,
        'min_bars': 6000,
        'required_columns': [
            'episode_id', 't_in_episode', 'time_utc', 'close',
            'ret_log_1', 'ret_deseason', 'range_bps', 'range_norm',
            'atr_14', 'atr_14_norm', 'hour_cot'
        ],
        'no_nans': True,  # STRICT = 0% NaN allowed
        'no_missing_bars': True,  # All 60 bars present
    },
    'flex': {
        'min_episodes': 100,
        'min_bars': 6000,
        'allows_missing_bar_flag': 'is_missing_bar',
        'allows_padding': True,
    }
}
```

### 4.2 Feature Engineering Inputs
L3 uses L2 outputs as:
- **Causal rolling features** (shift(5) to avoid leakage)
- **HOD residual features** (volatility surprise vs baseline)
- **Base features**: ret_deseason, range_norm as normalized targets

---

## 5. PRIORITY DISPLAY ORDER FOR FRONTEND

### Tier 1: Critical Quality Gates (Display First)
1. **Winsorization Rate** - Main quality indicator
   - Pass: `<= 1.0%`
   - Display: Percentage + count
   
2. **Deseasonalization Coverage** - Completeness
   - Pass: `>= 99%`
   - Display: Percentage + count
   
3. **Dataset Variants** - STRICT vs FLEX
   - STRICT: Count + % of complete episodes
   - FLEX: Count + missing bars count

4. **Quality Gate Summary** - GO/NO-GO
   - Overall PASS/FAIL status
   - Which gates failed (if any)

### Tier 2: Quality Metrics (Display Second)
5. **Stale Rate** (`is_stale` percentage)
6. **Range Outlier Rate** (p95 exceedance)
7. **OHLC Violations** (should be 0)
8. **NaN Rate** (should be < 0.5%)

### Tier 3: Statistical Summaries (Display Third)
9. **Return Statistics**:
   - Pre-deseason: mean, std, min, max
   - Post-deseason: mean, std, min, max
   - Deseason effectiveness (std reduction)

10. **Range Statistics**:
    - mean, p50, p95 of range_bps
    - range_norm distribution

11. **HOD Baselines** (reference):
    - Per-hour median returns
    - Per-hour MAD/scale
    - Verification that frozen recipe applied

### Tier 4: Advanced Metrics (Display Conditionally)
12. **Per-Hour Breakdown** (if user requests)
    - Hour-specific winsor rates
    - Hour-specific coverage
    - Hour-specific statistics

13. **Episode Quality Distribution** (bar chart)
    - Count by quality flag
    - Stale/winsor rate distribution

14. **Outlier Details** (expandable)
    - Total outliers found
    - Breakdown by type (return vs range)
    - Top outliers table

---

## 6. PROPOSED FRONTEND LAYOUT

### Tab 1: Overview (Summary Cards)
- **Winsorization Rate**: Large card, green/yellow/red based on % range
- **Deseasonalization**: Coverage %, target 99%+
- **Dataset Completeness**: STRICT episodes, FLEX episodes, rejected count
- **Quality Status**: GO/NO-GO badge + gate details

### Tab 2: Quality Breakdown
- Bar charts: Distribution by quality flag
- Stale vs winsorized rate trend
- OHLC violations counter
- NaN rate gauge

### Tab 3: Statistics
- Return distribution (before/after deseason)
- Range distribution
- HOD baseline table (hour, median, MAD, coverage)
- ATR statistics

### Tab 4: Audit Trail
- Winsorization per-hour breakdown
- Coverage report (row counts)
- Outlier list (top 20)
- Episode quality detail (searchable table)

---

## 7. KEY METRICS FOR APIS

### GET /api/pipeline/l2/prepared
Returns:
```json
{
  "status": "OK",
  "layer": "l2",
  "run_id": "uuid",
  "quality_metrics": {
    "winsorization_rate_pct": 0.65,
    "hod_median_abs": 0.002,
    "hod_mad_mean": 0.95,
    "nan_rate_pct": 0.1,
    "indicator_count": 60
  },
  "data_shape": {
    "rows": 14700,
    "columns": 67
  },
  "quality_gates": { ... },
  "pass": true
}
```

### GET /api/pipeline/l2/contract
Returns: L2 schema, variants, required columns, technical indicator list

### GET /api/pipeline/l2/indicators
Returns: Sample data with all 60+ indicators (for exploration)

---

## 8. COLUMN DEFINITIONS

### Winsorization Columns
- `ret_log_1`: Raw log return
- `ret_log_1_winsor`: Clipped return (main output)
- `ret_winsor_flag`: Boolean, True if value was clipped
- `is_outlier_ret`: Alias for ret_winsor_flag

### Deseasonalization Columns
- `ret_deseason`: (ret_winsor - hod_median) / scale (unit variance, mean ≈ 0)
- `hod_median`: Reference median return for this hour
- `hod_mad`: Reference MAD for this hour (used to calc scale)

### Range/Volatility
- `range_bps`: (high - low) / close × 10000
- `range_norm`: range_bps / p95_range_bps (normalized 0-2 typical)
- `is_outlier_range`: True if range > p95

### ATR
- `atr_14`: 14-period EMA of true range (in price units)
- `atr_14_norm`: atr_14 / close × 10000 (basis points)
- `tr`: True range = max(H-L, |H-Close_prev|, |L-Close_prev|)

---

## 9. CRITICAL NOTES FOR DISPLAY

1. **Never display raw hod_median/hod_mad** - These are metadata for audit only. Display aggregated HOD statistics instead.

2. **Winsorization is "bad" semantically** - Show as "Data Quality Issue", not "Data Corruption". Users want to know % of returns needed clipping.

3. **Deseasonalization must be near unit variance** - If std < 0.8 or > 1.2, flag as quality issue.

4. **STRICT vs FLEX is critical distinction**:
   - STRICT: For RL training only
   - FLEX: For analysis/monitoring
   - Always show both counts

5. **NaN rate should be invisible** - If <= 0.5%, don't show. If > 0.5%, highlight in red.

6. **Episode quality flags** - Four tiers of quality, users need summary pie chart:
   - PERFECT (use for critical models)
   - OK (good for training)
   - WARN (use with caution)
   - MARGINAL (flag issues)

---

## 10. METRICS THAT SHOULD BE SAVED TO DATABASE

For dashboarding/monitoring, recommend saving to PostgreSQL:
- `l2_quality_metrics` table (one row per execution date)
- `l2_episode_quality` table (one row per episode)
- Both indexed by execution_date and episode_id

This enables historical trending and alerting.

