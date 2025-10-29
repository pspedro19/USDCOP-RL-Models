# L2 Frontend Display Guide

## Priority Pyramid

```
        ┌─────────────────────────┐
        │  GO/NO-GO Badge         │ ← User Decision
        │  Overall Quality Status │
        └────────────┬────────────┘
                     │
         ┌───────────┴───────────┐
         │  Critical Metrics     │ ← Above Fold
         │  • Winsor Rate        │
         │  • Deseason Coverage  │
         │  • Episode Counts     │
         └───────────┬───────────┘
                     │
         ┌───────────┴───────────┐
         │ Quality Metrics       │ ← Below Fold
         │ • Stale Rate          │
         │ • Range Outliers      │
         │ • NaN Rate            │
         │ • OHLC Violations     │
         └───────────┬───────────┘
                     │
         ┌───────────┴───────────┐
         │ Statistics & Audit    │ ← Tabs/Expandable
         │ • Return Distribution │
         │ • Range Distribution  │
         │ • HOD Baselines       │
         │ • Outlier Details     │
         └───────────────────────┘
```

---

## Tier 1: Critical Quality Gates (ABOVE FOLD)

### Card 1: Overall Status Badge
```
┌─────────────────────────────────────┐
│  L2 QUALITY STATUS                  │
├─────────────────────────────────────┤
│                                     │
│         🟢 PASS (GO)                │
│                                     │
│  All quality gates satisfied        │
│  Ready for L3 feature engineering   │
│                                     │
│  Run ID: 2025-10-23-abc123          │
│  Execution: 14:00 UTC               │
└─────────────────────────────────────┘
```

**Alternative States**:
- 🟡 WARNING: Some gates borderline but pass
- 🔴 FAIL: One or more gates failed, STOP

---

### Card 2: Winsorization Rate
```
┌─────────────────────────────────────┐
│  WINSORIZATION RATE                 │
├─────────────────────────────────────┤
│                                     │
│           0.65%                     │
│         ▓░░░░░░░░░░░░░░░░░░░░░░    │
│         0%        Pass: <=1.0%      │
│                                     │
│  Returns clipped: 96 of 14,700      │
│  Status: ✓ PASS                     │
│                                     │
│  Meaning: Extreme outliers removed  │
│  without damaging market structure  │
└─────────────────────────────────────┘
```

**Color Coding**:
- Green: 0.0% - 0.8% (excellent)
- Yellow: 0.8% - 1.0% (good)
- Red: > 1.0% (FAIL)

---

### Card 3: Deseasonalization Coverage
```
┌─────────────────────────────────────┐
│  DESEASONALIZATION COVERAGE         │
├─────────────────────────────────────┤
│                                     │
│          99.7%                      │
│      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░          │
│      0%         Target: >=99%       │
│                                     │
│  Bars deseasonalized: 14,605        │
│  Status: ✓ PASS                     │
│                                     │
│  Meaning: HOD patterns successfully │
│  removed from return series         │
└─────────────────────────────────────┘
```

**Color Coding**:
- Green: 99%+ (excellent)
- Yellow: 95-99% (acceptable)
- Red: < 95% (FAIL)

---

### Card 4: Dataset Completeness
```
┌──────────────┐  ┌──────────────┐
│   STRICT     │  │    FLEX      │
├──────────────┤  ├──────────────┤
│              │  │              │
│    245       │  │     250      │
│  EPISODES    │  │   EPISODES   │
│              │  │              │
│  14,700 rows │  │  15,000 rows │
│   0 missing  │  │   5 padded   │
│   bars       │  │   bars       │
│              │  │              │
│ Ready for RL │  │Ready for all │
│  training    │  │   analysis   │
│              │  │              │
└──────────────┘  └──────────────┘
```

---

### Card 5: Quality Gate Summary Table
```
┌────────────────────┬───────┬──────────┐
│ Gate               │ Pass? │ Value    │
├────────────────────┼───────┼──────────┤
│ Winsor Rate        │ ✓     │ 0.65%    │
│ HOD Median (abs)   │ ✓     │ 0.002    │
│ HOD Scale (MAD)    │ ✓     │ 0.95     │
│ Deseason Variance  │ ✓     │ 0.98     │
│ NaN Rate           │ ✓     │ 0.1%     │
│ OHLC Violations    │ ✓     │ 0        │
│ Stale Rate         │ ✓     │ 0.8%     │
└────────────────────┴───────┴──────────┘
```

---

## Tier 2: Quality Metrics (BELOW FOLD)

### Card 6: Stale Data Rate
```
┌─────────────────────────────────────┐
│  STALE DATA RATE                    │
├─────────────────────────────────────┤
│                                     │
│           0.8%                      │
│         ▓░░░░░░░░░░░░░░░░░░░░░░    │
│         0%        Max: 2.0%         │
│                                     │
│  Stale bars (repeated OHLC): 118    │
│  Status: ✓ PASS                     │
│                                     │
│  Meaning: Very few bars with zero   │
│  price movement (good data quality) │
└─────────────────────────────────────┘
```

---

### Card 7: Range Outlier Rate
```
┌─────────────────────────────────────┐
│  RANGE OUTLIER RATE                 │
├─────────────────────────────────────┤
│                                     │
│           0.5%                      │
│         ▓░░░░░░░░░░░░░░░░░░░░░░    │
│         0%        Max: 1.0%         │
│                                     │
│  Bars exceeding p95: 74             │
│  Status: ✓ PASS                     │
│                                     │
│  Meaning: Very few bars with        │
│  unusually high intraday range      │
└─────────────────────────────────────┘
```

---

### Card 8: Episode Quality Distribution
```
┌─────────────────────────────────────┐
│  EPISODE QUALITY DISTRIBUTION       │
├─────────────────────────────────────┤
│                                     │
│     PERFECT   OK   WARN MARGINAL    │
│      ███░░░   ███  ░░░   ░░░        │
│      69%     20%  8%   3%           │
│                                     │
│  PERFECT:  169 episodes             │
│  OK:        49 episodes             │
│  WARN:      20 episodes             │
│  MARGINAL:   7 episodes             │
│                                     │
│  → 89% of episodes are PERFECT+OK   │
└─────────────────────────────────────┘
```

---

### Card 9: Basic Quality Checks
```
┌─────────────────────────────────────┐
│  DATA QUALITY CHECKS                │
├─────────────────────────────────────┤
│                                     │
│  ✓ OHLC Violations:        0        │
│  ✓ Missing Timestamps:     0        │
│  ✓ Duplicate Times:        0        │
│  ✓ NaN Rate:          0.1% (pass)   │
│                                     │
│  Status: All checks PASSED          │
└─────────────────────────────────────┘
```

---

## Tier 3: Statistics & Audit (TABS/EXPANDABLE)

### Tab: Return Statistics
```
┌────────────────────┬──────────┬──────────┐
│ Statistic          │ Raw Ret  │ Deseason │
├────────────────────┼──────────┼──────────┤
│ Mean               │ 0.00020  │ -0.0001  │
│ Std Dev            │ 0.00450  │ 0.98     │
│ Skewness           │ -0.15    │ -0.08    │
│ Kurtosis           │ 3.2      │ 2.9      │
│ Min                │ -0.0234  │ -3.2     │
│ 25%ile             │ -0.0020  │ -0.674   │
│ 50%ile (median)    │ 0.0001   │ 0.012    │
│ 75%ile             │ 0.0022   │ 0.681    │
│ Max                │ 0.0198   │ 2.8      │
│                    │          │          │
│ Deseason Effect    │          │ ✓ Success│
│ (Std reduction)    │          │ 0.45→0.98│
└────────────────────┴──────────┴──────────┘

Interpretation:
• Raw returns show typical 5m bar volatility
• Deseasonalized returns have unit variance
• Small remaining skew/kurtosis is normal
```

---

### Tab: Range Statistics
```
┌────────────────────┬──────────────┐
│ Statistic          │ Value        │
├────────────────────┼──────────────┤
│ Mean Range (bps)   │ 3.2          │
│ Median Range (bps) │ 2.8          │
│ P95 Range (bps)    │ 7.1          │
│ Min Range (bps)    │ 0.1          │
│ Max Range (bps)    │ 45.2         │
│                    │              │
│ Range Normalized   │              │
│  Mean (0-2 scale)  │ 0.45         │
│  P95 (0-1 scale)   │ 1.0          │
└────────────────────┴──────────────┘

Interpretation:
• Typical intrabar range: 2.8-3.2 bps
• Extreme outliers exist (45 bps)
• Normalization works well
```

---

### Tab: HOD Baselines
```
┌────────┬────────────┬─────┬──────────┐
│ Hour   │ Median Ret │ MAD │ Coverage │
│ (COT)  │ (x10000)   │     │ %        │
├────────┼────────────┼─────┼──────────┤
│ 8      │ 0.15       │0.95 │ 100%     │
│ 9      │ 0.08       │0.92 │ 100%     │
│ 10     │ 0.10       │0.98 │ 100%     │
│ 11     │ 0.05       │0.96 │ 100%     │
│ 12     │ -0.02      │1.01 │ 100%     │
│ 13     │ 0.03       │0.97 │ 100%     │
│ ─      │ ─          │─    │ ─        │
│ MEAN   │ 0.07       │0.97 │ 100%     │
└────────┴────────────┴─────┴──────────┘

Meaning:
• Median returns shift slightly across day
• Scale (MAD) is stable (~0.95-1.01)
• All hours have 100% coverage
• Frozen recipe: MAD × 1.4826 ≈ 1.3-1.4 bps floor
```

---

### Tab: Per-Hour Winsorization Rates
```
┌────────┬────────┬──────────┬──────────┐
│ Hour   │ Count  │ Winsorized│ Rate %   │
│ (COT)  │        │          │          │
├────────┼────────┼──────────┼──────────┤
│ 8      │ 2450   │ 18       │ 0.73%    │
│ 9      │ 2400   │ 14       │ 0.58%    │
│ 10     │ 2410   │ 16       │ 0.66%    │
│ 11     │ 2380   │ 12       │ 0.50%    │
│ 12     │ 2390   │ 18       │ 0.75%    │
│ 13     │ 2370   │ 18       │ 0.76%    │
│ ─      │ ─      │ ─        │ ─        │
│ TOTAL  │ 14000  │ 96       │ 0.69%    │
└────────┴────────┴──────────┴──────────┘

Findings:
• Hour 8 has slightly higher rate (busy session)
• Hour 11 has lowest rate (calm period)
• All hours well below 1% threshold
```

---

### Tab: Outlier Details
```
┌────────┬────────┬──────────────┬──────────┐
│Episode │ T_In_E │ Return Type  │ Value    │
│ ID     │        │              │          │
├────────┼────────┼──────────────┼──────────┤
│2025-10-│ 23     │ RETURN       │ 0.0187   │
│ 12     │        │ (clipped)    │ (3.8σ)   │
├────────┼────────┼──────────────┼──────────┤
│2025-10-│ 45     │ RANGE        │ 18.2 bps │
│ 10     │        │ (p95 exceed) │ vs 7.1   │
├────────┼────────┼──────────────┼──────────┤
│2025-10-│ 7      │ RETURN       │ -0.0156  │
│ 18     │        │ (clipped)    │ (-3.5σ)  │
└────────┴────────┴──────────────┴──────────┘

Top 20 outliers shown (full list in CSV)
```

---

### Tab: Episode Quality Details (Searchable Table)
```
┌─────────┬──────────┬───────────┬───────────┬─────────┐
│ Episode │ Quality  │ Bars      │ Stale %   │ Winsor %│
│ ID      │ Flag     │ Valid/60  │           │         │
├─────────┼──────────┼───────────┼───────────┼─────────┤
│2025-10-1│ PERFECT  │ 60/60     │ 0.0%      │ 0.0%    │
│2025-10-2│ PERFECT  │ 60/60     │ 0.0%      │ 0.0%    │
│2025-10-3│ OK       │ 60/60     │ 0.0%      │ 1.7%    │
│2025-10-4│ PERFECT  │ 60/60     │ 0.0%      │ 0.0%    │
│2025-10-5│ WARN     │ 60/60     │ 1.7%      │ 1.7%    │
│ ...     │ ...      │ ...       │ ...       │ ...     │
└─────────┴──────────┴───────────┴───────────┴─────────┘

Filters:
  Quality: [All] PERFECT OK WARN MARGINAL
  Date Range: [2025-10-01] to [2025-10-23]
  Stale %: [0-2%]
  Winsor %: [0-2%]
```

---

### Tab: Coverage & Lineage
```
┌─────────────────────────────────────┐
│  DATA LINEAGE REPORT                │
├─────────────────────────────────────┤
│                                     │
│ INPUT (from L1):                    │
│  Episodes:     929 total            │
│  Rows:       55,740 bars            │
│  Quality:     245 perfect           │
│               250 acceptable        │
│               434 rejected          │
│                                     │
│ OUTPUT (L2 STRICT):                 │
│  Episodes:     245 complete         │
│  Rows:       14,700 bars            │
│  Coverage:    26.3% of L1 input     │
│                                     │
│ OUTPUT (L2 FLEX):                   │
│  Episodes:     250 with padding     │
│  Rows:       15,000 bars            │
│  Coverage:    26.9% of L1 input     │
│                                     │
│ TRANSFORMATIONS:                    │
│  Winsorization: 96 returns clipped  │
│  Deseasonalization: 14,605 bars     │
│  Missing bars padded: 5 episodes    │
│                                     │
│ READY FOR L3:    ✓ YES              │
│ (100+ episodes available)           │
│                                     │
└─────────────────────────────────────┘
```

---

## Layout Template (Responsive)

### Desktop (1400px+)
```
┌─────────────────────────────────────────────────────┐
│ Header: L2 Prepared Data Dashboard                  │
├──────────────────────┬────────────────────────────────┤
│ GO/NO-GO Badge       │ Run ID / Timestamp            │
│ (Large, prominent)   │ Refresh button                │
├──────────────────────┴────────────────────────────────┤
│                                                     │
│ Row 1 (Critical Metrics):                          │
│ ┌────────────┐ ┌────────────┐ ┌────────────────┐  │
│ │ Winsor     │ │ Deseason   │ │ Episodes       │  │
│ │ Rate       │ │ Coverage   │ │ STRICT/FLEX    │  │
│ └────────────┘ └────────────┘ └────────────────┘  │
│                                                     │
│ Row 2 (Quality Metrics):                           │
│ ┌────────────┐ ┌────────────┐ ┌────────────────┐  │
│ │ Stale Rate │ │ Range Out. │ │ Quality Dist.  │  │
│ └────────────┘ └────────────┘ └────────────────┘  │
│                                                     │
│ Row 3 (Tabs):                                      │
│ ┌─────────────────────────────────────────────────┐│
│ │ [Statistics] [HOD] [Hourly] [Quality] [Audit]   ││
│ │ ─────────────────────────────────────────────── ││
│ │ [Content for selected tab]                      ││
│ └─────────────────────────────────────────────────┘│
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Mobile (< 768px)
```
┌──────────────────────────┐
│ GO/NO-GO Badge           │
├──────────────────────────┤
│ Winsor Rate      0.65%   │
│ Status:          ✓ PASS  │
├──────────────────────────┤
│ Deseason Coverage  99.7% │
│ Status:          ✓ PASS  │
├──────────────────────────┤
│ STRICT Episodes    245   │
│ FLEX Episodes      250   │
├──────────────────────────┤
│ [Collapse/Expand]        │
│  ▼ Quality Metrics       │
│    Stale:      0.8%      │
│    Range Out:  0.5%      │
│ [Collapse/Expand]        │
│  ▼ More Details          │
└──────────────────────────┘
```

---

## Color & Icon System

| Status | Color | Icon | Meaning |
|--------|-------|------|---------|
| PASS | Green | ✓ | Gate passed, no action needed |
| WARNING | Yellow | ⚠ | Gate borderline, monitor |
| FAIL | Red | ✗ | Gate failed, stop pipeline |
| INFO | Blue | ℹ | Informational, no gate |
| N/A | Gray | — | Not applicable |

---

## Interaction Patterns

### Hover Tooltips
```
On "Winsorization Rate" card:
  Shows: "4-sigma robust clipping removes extreme returns
          while preserving market structure"
  Plus: Formula details expandable
  Plus: Link to "How Winsorization Works"

On "HOD Median" in table:
  Shows: Hour-specific baseline value
  Plus: Distribution histogram
  Plus: How it's used in deseasonalization
```

### Expandable Sections
```
[+] Per-Hour Breakdown          → Expands to show table
[+] Return Distribution         → Expands to show chart
[+] Outlier Analysis            → Expands to show list
[+] Quality Gate Details        → Expands to show thresholds
```

### Refresh Button
- Auto-refresh every 60 seconds (if streaming)
- Manual refresh available
- Shows last refresh timestamp
- Loader animation during fetch

---

## Success Criteria

Display should help users answer:
1. Is L2 ready for L3? (GO/NO-GO badge)
2. What's the data quality? (Metrics overview)
3. Why did we reject episodes? (Rejection breakdown)
4. What transformations happened? (Stats comparison)
5. Are there anomalies? (Outlier details)
6. How much data is usable? (Episode breakdown)

All within 2-3 seconds load time.

