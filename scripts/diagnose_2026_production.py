"""
Root cause analysis: Why 2026 production underperforms.
Uses the EXACT same load_data() as train_and_export_smart_simple.py.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse load_data from the export script
from scripts.train_and_export_smart_simple import load_data

df, feature_cols = load_data()

print("=" * 70)
print("  ROOT CAUSE ANALYSIS: 2026 Production Performance")
print("=" * 70)

# 1. USDCOP price direction 2026
d26 = df[df["date"] >= "2026-01-01"]
d25 = df[(df["date"] >= "2025-01-01") & (df["date"] < "2026-01-01")]
if len(d26) > 1:
    p0, p1 = d26["close"].iloc[0], d26["close"].iloc[-1]
    chg = (p1 / p0 - 1) * 100
    print(f"\n1. USDCOP 2026 YTD: {p0:.2f} -> {p1:.2f} ({chg:+.2f}%)")
    print(f"   {'COP strengthened => SHORT correct' if chg < 0 else 'COP weakened => LONG correct'}")

# 2. Weekly actual returns (what the model should predict)
print(f"\n2. ACTUAL WEEKLY RETURNS (Mon close -> Fri close):")
mondays_2026 = sorted(d26[d26["date"].dt.dayofweek == 0]["date"].values)
for mon in mondays_2026:
    mon_ts = pd.Timestamp(mon)
    fri_ts = mon_ts + pd.offsets.BDay(4)
    mon_row = df[df["date"] == mon_ts]
    fri_rows = df[(df["date"] >= fri_ts) & (df["date"] <= fri_ts + pd.Timedelta(days=2))]
    if len(mon_row) > 0 and len(fri_rows) > 0:
        mc = float(mon_row["close"].iloc[0])
        fc = float(fri_rows["close"].iloc[0])
        ret = (fc / mc - 1) * 100
        correct_dir = "SHORT" if ret < 0 else "LONG"
        print(f"   {mon_ts.strftime('%Y-%m-%d')}: {mc:.2f} -> {fc:.2f} = {ret:+.3f}% => {correct_dir} was correct")

# 3. Model predictions comparison (single vs monthly)
print(f"\n3. MODEL PREDICTIONS (Single-train vs Monthly-retrain):")
print(f"   {'Monday':<12} {'Single%':>9} {'Monthly%':>9} {'S-Dir':>6} {'M-Dir':>6} {'Actual%':>9} {'Correct?':>10}")
print(f"   {'-'*12:<12} {'-'*9:>9} {'-'*9:>9} {'-'*6:>6} {'-'*6:>6} {'-'*9:>9} {'-'*10:>10}")

# Single model (trained on ALL 2020-2025)
te_s = pd.Timestamp("2025-12-31")
dts = df[(df["date"] <= te_s) & df["target_return_5d"].notna()].copy()
dts = dts[dts[feature_cols].notna().all(axis=1)]
Xs = dts[feature_cols].values.astype(np.float64)
ys = dts["target_return_5d"].values.astype(np.float64)
scs = StandardScaler().fit(Xs)
rs = Ridge(alpha=1.0).fit(scs.transform(Xs), ys)
bs = BayesianRidge(max_iter=300).fit(scs.transform(Xs), ys)

for mon in mondays_2026:
    mon_ts = pd.Timestamp(mon)
    prev = df[(df["date"] < mon_ts)]
    prev = prev[prev[feature_cols].notna().all(axis=1)]
    if prev.empty:
        continue
    feat = prev[feature_cols].iloc[-1:].values.astype(np.float64)

    # Single prediction
    lat_s = scs.transform(feat)
    ens_s = (float(rs.predict(lat_s)[0]) + float(bs.predict(lat_s)[0])) / 2.0
    dir_s = "SHORT" if ens_s < 0 else "LONG"

    # Monthly prediction
    m_num = mon_ts.month
    te_m = pd.Timestamp("2025-12-31") if m_num == 1 else pd.Timestamp(f"2026-{m_num:02d}-01") - pd.Timedelta(days=1)
    dtm = df[(df["date"] <= te_m) & df["target_return_5d"].notna()].copy()
    dtm = dtm[dtm[feature_cols].notna().all(axis=1)]
    Xm = dtm[feature_cols].values.astype(np.float64)
    ym = dtm["target_return_5d"].values.astype(np.float64)
    scm = StandardScaler().fit(Xm)
    rm = Ridge(alpha=1.0).fit(scm.transform(Xm), ym)
    bm = BayesianRidge(max_iter=300).fit(scm.transform(Xm), ym)
    lat_m = scm.transform(feat)
    ens_m = (float(rm.predict(lat_m)[0]) + float(bm.predict(lat_m)[0])) / 2.0
    dir_m = "SHORT" if ens_m < 0 else "LONG"

    # Actual
    fri_ts = mon_ts + pd.offsets.BDay(4)
    mr = df[df["date"] == mon_ts]
    fr = df[(df["date"] >= fri_ts) & (df["date"] <= fri_ts + pd.Timedelta(days=2))]
    if len(mr) > 0 and len(fr) > 0:
        actual = (float(fr["close"].iloc[0]) / float(mr["close"].iloc[0]) - 1) * 100
        correct_dir = "SHORT" if actual < 0 else "LONG"
        cs = "YES" if dir_s == correct_dir else "no"
        cm = "YES" if dir_m == correct_dir else "no"
        print(f"   {mon_ts.strftime('%Y-%m-%d'):<12} {ens_s*100:>+8.4f}% {ens_m*100:>+8.4f}% {dir_s:>6} {dir_m:>6} {actual:>+8.3f}% S={cs}/M={cm}")
    else:
        print(f"   {mon_ts.strftime('%Y-%m-%d'):<12} {ens_s*100:>+8.4f}% {ens_m*100:>+8.4f}% {dir_s:>6} {dir_m:>6}      N/A")

# 4. Prediction magnitudes comparison
print(f"\n4. PREDICTION MAGNITUDES (conviction level):")
mags_2026 = []
for mon in mondays_2026:
    mon_ts = pd.Timestamp(mon)
    prev = df[(df["date"] < mon_ts)]
    prev = prev[prev[feature_cols].notna().all(axis=1)]
    if prev.empty:
        continue
    feat = prev[feature_cols].iloc[-1:].values.astype(np.float64)
    lat = scs.transform(feat)
    ens = (float(rs.predict(lat)[0]) + float(bs.predict(lat)[0])) / 2.0
    mags_2026.append(abs(ens * 100))

# 2025 magnitudes
te_24 = pd.Timestamp("2024-12-31")
dt24 = df[(df["date"] <= te_24) & df["target_return_5d"].notna()].copy()
dt24 = dt24[dt24[feature_cols].notna().all(axis=1)]
X24 = dt24[feature_cols].values.astype(np.float64)
y24 = dt24["target_return_5d"].values.astype(np.float64)
sc24 = StandardScaler().fit(X24)
r24 = Ridge(alpha=1.0).fit(sc24.transform(X24), y24)
b24 = BayesianRidge(max_iter=300).fit(sc24.transform(X24), y24)

mondays_2025 = sorted(d25[d25["date"].dt.dayofweek == 0]["date"].values)
mags_2025 = []
for mon in mondays_2025:
    mon_ts = pd.Timestamp(mon)
    prev = df[(df["date"] < mon_ts)]
    prev = prev[prev[feature_cols].notna().all(axis=1)]
    if prev.empty:
        continue
    feat = prev[feature_cols].iloc[-1:].values.astype(np.float64)
    lat = sc24.transform(feat)
    ens = (float(r24.predict(lat)[0]) + float(b24.predict(lat)[0])) / 2.0
    mags_2025.append(abs(ens * 100))

if mags_2025 and mags_2026:
    print(f"   2025 avg magnitude: {np.mean(mags_2025):.4f}% (N={len(mags_2025)})")
    print(f"   2026 avg magnitude: {np.mean(mags_2026):.4f}% (N={len(mags_2026)})")
    print(f"   Ratio: {np.mean(mags_2026)/np.mean(mags_2025):.2f}x")

# 5. Feature coefficients — what drives the model
print(f"\n5. TOP RIDGE COEFFICIENTS (standardized, model trained 2020-2025):")
coefs = list(zip(feature_cols, rs.coef_))
coefs.sort(key=lambda x: abs(x[1]), reverse=True)
for name, coef in coefs[:10]:
    print(f"   {name:<25} {coef:+.6f}")

# 6. Key feature regime shift
print(f"\n6. FEATURE REGIME SHIFT (2025 avg vs 2026 start):")
d26_start = df[df["date"] >= "2025-12-29"].head(5)
key_feats = ["close", "return_5d", "return_20d", "rsi_14d", "ma_ratio_20d",
             "ma_ratio_50d", "volatility_20d", "dxy_close_lag1", "oil_close_lag1",
             "vix_close_lag1", "embi_close_lag1"]
print(f"   {'Feature':<25} {'2025 avg':>10} {'2026 Jan':>10} {'Shift':>10}")
for col in key_feats:
    if col in df.columns:
        avg25 = d25[col].mean()
        jan26 = d26_start[col].mean()
        shift = jan26 - avg25
        print(f"   {col:<25} {avg25:>10.4f} {jan26:>10.4f} {shift:>+10.4f}")

# 7. Macro data freshness
print(f"\n7. MACRO DATA FRESHNESS (last available dates):")
macro_feats = ["dxy_close_lag1", "oil_close_lag1", "vix_close_lag1", "embi_close_lag1"]
for col in macro_feats:
    last_valid = df[df[col].notna()]["date"].max()
    last_val = df.loc[df["date"] == last_valid, col].iloc[0] if last_valid else None
    n_2026 = d26[col].notna().sum()
    print(f"   {col:<25} last={last_valid.strftime('%Y-%m-%d') if last_valid else 'N/A'} val={last_val:.2f} 2026_valid={n_2026}/{len(d26)}")

# 8. Direction accuracy analysis
print(f"\n8. DIRECTION ACCURACY COMPARISON:")
correct_s_count = 0
correct_m_count = 0
total = 0
for mon in mondays_2026:
    mon_ts = pd.Timestamp(mon)
    fri_ts = mon_ts + pd.offsets.BDay(4)
    prev = df[(df["date"] < mon_ts)]
    prev = prev[prev[feature_cols].notna().all(axis=1)]
    mr = df[df["date"] == mon_ts]
    fr = df[(df["date"] >= fri_ts) & (df["date"] <= fri_ts + pd.Timedelta(days=2))]
    if prev.empty or len(mr) == 0 or len(fr) == 0:
        continue
    feat = prev[feature_cols].iloc[-1:].values.astype(np.float64)
    ens_s = (float(rs.predict(scs.transform(feat))[0]) + float(bs.predict(scs.transform(feat))[0])) / 2.0
    actual = float(fr["close"].iloc[0]) / float(mr["close"].iloc[0]) - 1
    if (ens_s > 0 and actual > 0) or (ens_s < 0 and actual < 0):
        correct_s_count += 1
    total += 1
if total > 0:
    print(f"   Single model DA: {correct_s_count}/{total} = {correct_s_count/total*100:.1f}%")
    print(f"   (2025 backtest DA was 62.5% with walk-forward)")

print(f"\n{'=' * 70}")
print(f"  SUMMARY")
print(f"{'=' * 70}")
