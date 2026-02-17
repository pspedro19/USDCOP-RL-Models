"""Quick LONG vs SHORT breakdown for H=5 2025 OOS trades."""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.data_contracts import FEATURE_COLUMNS
from src.forecasting.models.factory import ModelFactory
from src.forecasting.contracts import MODEL_IDS, get_horizon_config
from src.forecasting.vol_targeting import VolTargetConfig, compute_vol_target_signal, compute_realized_vol
from src.execution.trailing_stop import TrailingStopConfig, TrailingStopTracker, TrailingState

# Load data
ohlcv = pd.read_parquet(PROJECT_ROOT / "seeds/latest/usdcop_daily_ohlcv.parquet").reset_index()
ohlcv.rename(columns={"time": "date"}, inplace=True)
ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.tz_localize(None).dt.normalize()
ohlcv = ohlcv[["date", "open", "high", "low", "close"]].sort_values("date").reset_index(drop=True)

macro = pd.read_parquet(PROJECT_ROOT / "data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet").reset_index()
macro.rename(columns={macro.columns[0]: "date"}, inplace=True)
macro["date"] = pd.to_datetime(macro["date"]).dt.tz_localize(None).dt.normalize()
mc = {
    "FXRT_INDEX_DXY_USA_D_DXY": "dxy_close_lag1",
    "COMM_OIL_WTI_GLB_D_WTI": "oil_close_lag1",
    "VOLT_VIX_USA_D_VIX": "vix_close_lag1",
    "CRSK_SPREAD_EMBI_COL_D_EMBI": "embi_close_lag1",
}
ms = macro[["date"] + list(mc.keys())].copy()
ms.rename(columns=mc, inplace=True)
ms = ms.sort_values("date")
for c in mc.values():
    ms[c] = ms[c].shift(1)

df = pd.merge_asof(ohlcv, ms, on="date", direction="backward")

# Features
df["return_1d"] = df["close"].pct_change(1)
df["return_5d"] = df["close"].pct_change(5)
df["return_10d"] = df["close"].pct_change(10)
df["return_20d"] = df["close"].pct_change(20)
df["volatility_5d"] = df["return_1d"].rolling(5).std()
df["volatility_10d"] = df["return_1d"].rolling(10).std()
df["volatility_20d"] = df["return_1d"].rolling(20).std()
delta = df["close"].diff()
gain = delta.where(delta > 0, 0.0)
loss_s = -delta.where(delta < 0, 0.0)
avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
avg_loss = loss_s.ewm(alpha=1/14, min_periods=14).mean()
rs = avg_gain / avg_loss.replace(0, np.nan)
df["rsi_14d"] = 100 - (100 / (1 + rs))
df["ma_ratio_20d"] = df["close"] / df["close"].rolling(20).mean()
df["ma_ratio_50d"] = df["close"] / df["close"].rolling(50).mean()
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
for c in mc.values():
    df[c] = df[c].ffill()

df["target_5d"] = np.log(df["close"].shift(-5) / df["close"])
df["target_1d"] = np.log(df["close"].shift(-1) / df["close"])
feat_cols = list(FEATURE_COLUMNS)
df = df[df[feat_cols].notna().all(axis=1)].reset_index(drop=True)

# Train 2020-2024
train = df[(df["date"] >= pd.Timestamp("2020-01-01")) & (df["date"] <= pd.Timestamp("2024-12-31"))]
test_2025 = df[(df["date"] >= pd.Timestamp("2025-01-01")) & (df["date"] <= pd.Timestamp("2025-12-31"))]
test_2026 = df[df["date"] >= pd.Timestamp("2026-01-01")]

hconfig = get_horizon_config(5)
models_list = list(MODEL_IDS)
try:
    ModelFactory.create("ard")
except Exception:
    models_list = [m for m in models_list if m != "ard"]

X_tr = train[feat_cols].values.astype(np.float64)
y_tr = train["target_5d"].values.astype(np.float64)
scaler = StandardScaler()
X_sc = scaler.fit_transform(X_tr)

trained = {}
for mid in models_list:
    try:
        if mid in {"ridge", "bayesian_ridge", "ard"}:
            params = None
        elif mid == "catboost_pure":
            params = {"iterations": hconfig.get("n_estimators", 50), "depth": hconfig.get("max_depth", 3),
                      "learning_rate": hconfig.get("learning_rate", 0.05), "verbose": False, "allow_writing_files": False}
        elif "hybrid" in mid and "catboost" in mid:
            params = {"iterations": hconfig.get("n_estimators", 50), "depth": hconfig.get("max_depth", 3),
                      "learning_rate": hconfig.get("learning_rate", 0.05), "verbose": False, "allow_writing_files": False}
        elif "hybrid" in mid:
            params = hconfig
        else:
            params = hconfig
        m = ModelFactory.create(mid, params=params, horizon=5)
        if m.requires_scaling:
            m.fit(X_sc, y_tr)
        else:
            m.fit(X_tr, y_tr)
        trained[mid] = m
    except Exception:
        pass

print(f"Trained {len(trained)} models: {list(trained.keys())}")


def predict_all(df_period, target_col="target_5d"):
    preds = []
    for _, row in df_period.iterrows():
        if pd.isna(row[target_col]):
            continue
        X = row[feat_cols].values.astype(np.float64).reshape(1, -1)
        Xs = scaler.transform(X)
        ps = {}
        for mid, m in trained.items():
            try:
                p = m.predict(Xs)[0] if m.requires_scaling else m.predict(X)[0]
                ps[mid] = p
            except Exception:
                pass
        if len(ps) < 3:
            continue
        top3 = sorted(ps.keys(), key=lambda x: abs(ps[x]), reverse=True)[:3]
        ens = np.mean([ps[x] for x in top3])
        n_long_models = sum(1 for v in ps.values() if v > 0)
        n_short_models = sum(1 for v in ps.values() if v < 0)
        preds.append({
            "date": row["date"], "close": row["close"],
            "actual_return": row[target_col],
            "ensemble_pred": ens,
            "direction": 1 if ens > 0 else -1,
            "n_models_long": n_long_models,
            "n_models_short": n_short_models,
            "top3_models": top3,
            "per_model_preds": ps,
        })
    return pd.DataFrame(preds)


def select_weekly(pdf, h=5):
    pdf = pdf.sort_values("date").reset_index(drop=True)
    sel = [0]
    last = 0
    for i in range(1, len(pdf)):
        if i - last >= h:
            sel.append(i)
            last = i
    return pdf.iloc[sel].copy().reset_index(drop=True)


# Load 5-min data
m5 = pd.read_parquet(PROJECT_ROOT / "seeds/latest/usdcop_m5_ohlcv.parquet")
if "symbol" in m5.columns:
    m5 = m5[m5["symbol"] == "USD/COP"]
m5 = m5.reset_index()
if "time" in m5.columns:
    m5.rename(columns={"time": "timestamp"}, inplace=True)
m5["timestamp"] = pd.to_datetime(m5["timestamp"])
if m5["timestamp"].dt.tz is not None:
    m5["timestamp"] = m5["timestamp"].dt.tz_localize(None)
m5["date"] = m5["timestamp"].dt.normalize()
m5 = m5[m5["date"] >= pd.Timestamp("2024-12-01")].sort_values("timestamp").reset_index(drop=True)
m5 = m5[["timestamp", "date", "open", "high", "low", "close"]]
m5_dates = sorted(m5["date"].unique())


def trailing_detail(trade, config):
    """Run trailing stop and return detailed info."""
    sig_date = trade["date"]
    direction = int(trade["direction"])
    future = [d for d in m5_dates if d > sig_date]
    if len(future) == 0:
        return {"total_pnl": None, "sub_trades": [], "n_subtrades": 0}

    holding = future[:5]
    slip = 1.0 / 10000
    sub_trades = []
    need_entry = True
    day_idx = 0
    total_pnl = 0.0

    while day_idx < len(holding):
        day = holding[day_idx]
        bars = m5[m5["date"] == day].sort_values("timestamp")
        if len(bars) == 0:
            day_idx += 1
            continue

        if need_entry:
            entry_price = float(bars.iloc[0]["open"])
            if direction == 1:
                slipped_entry = entry_price * (1 + slip)
            else:
                slipped_entry = entry_price * (1 - slip)
            tracker = TrailingStopTracker(entry_price=slipped_entry, direction=direction, config=config)
            need_entry = False
            sub_entry_day = day_idx

        triggered = False
        max_fav = 0
        max_adv = 0
        for bi, (_, bar) in enumerate(bars.iterrows()):
            mid = float(bar["close"])
            unreal = direction * (mid - slipped_entry) / slipped_entry
            max_fav = max(max_fav, unreal)
            max_adv = min(max_adv, unreal)
            state = tracker.update(float(bar["high"]), float(bar["low"]), float(bar["close"]), bi)
            if state == TrailingState.TRIGGERED:
                triggered = True
                break

        if triggered:
            ep = tracker.exit_price
            if direction == 1:
                ep *= (1 - slip)
            else:
                ep *= (1 + slip)
            sub_pnl = direction * (ep - slipped_entry) / slipped_entry
            total_pnl += sub_pnl
            reason = str(tracker.exit_reason)
            sub_trades.append({
                "entry": round(slipped_entry, 2), "exit": round(ep, 2),
                "pnl_pct": round(sub_pnl * 100, 4), "reason": reason,
                "entry_day": sub_entry_day + 1, "exit_day": day_idx + 1,
                "mfe_pct": round(max_fav * 100, 4), "mae_pct": round(max_adv * 100, 4),
            })
            if day_idx < len(holding) - 1:
                need_entry = True
                day_idx += 1
                continue
            else:
                break
        else:
            if day_idx == len(holding) - 1:
                lc = float(bars.iloc[-1]["close"])
                if direction == 1:
                    ep = lc * (1 - slip)
                else:
                    ep = lc * (1 + slip)
                sub_pnl = direction * (ep - slipped_entry) / slipped_entry
                total_pnl += sub_pnl
                sub_trades.append({
                    "entry": round(slipped_entry, 2), "exit": round(ep, 2),
                    "pnl_pct": round(sub_pnl * 100, 4), "reason": "week_end",
                    "entry_day": sub_entry_day + 1, "exit_day": 5,
                    "mfe_pct": round(max_fav * 100, 4), "mae_pct": round(max_adv * 100, 4),
                })
        day_idx += 1

    return {"total_pnl": round(total_pnl * 100, 4), "sub_trades": sub_trades, "n_subtrades": len(sub_trades)}


config = TrailingStopConfig(activation_pct=0.004, trail_pct=0.003, hard_stop_pct=0.04)

# === 2025 Analysis ===
print("\n" + "=" * 80)
print("H=5 LONG vs SHORT BREAKDOWN - 2025 OOS")
print("=" * 80)

preds_2025 = predict_all(test_2025)
weekly_2025 = select_weekly(preds_2025)
print(f"\nTotal weekly trades 2025: {len(weekly_2025)}")

long_2025 = weekly_2025[weekly_2025["direction"] == 1].copy()
short_2025 = weekly_2025[weekly_2025["direction"] == -1].copy()
print(f"LONG: {len(long_2025)}, SHORT: {len(short_2025)}")

# LONG breakdown
print(f"\n{'='*60}")
print(f"LONG TRADES ({len(long_2025)} trades)")
print(f"{'='*60}")

long_win = 0
long_details = []
for idx, (_, trade) in enumerate(long_2025.iterrows()):
    actual_pct = (np.exp(trade["actual_return"]) - 1) * 100
    correct = trade["actual_return"] > 0
    if correct:
        long_win += 1

    trail = trailing_detail(trade, config)

    model_agreement = f"{trade['n_models_long']}L/{trade['n_models_short']}S"

    print(f"\n  Trade {idx+1}: {trade['date'].strftime('%Y-%m-%d')}")
    print(f"    Pred: LONG (ensemble={trade['ensemble_pred']:.6f}, models={model_agreement})")
    print(f"    Actual 5d return: {actual_pct:+.3f}% [{'CORRECT' if correct else 'WRONG'}]")
    print(f"    Trail total PnL:  {trail['total_pnl']:+.4f}%")
    print(f"    Sub-trades: {trail['n_subtrades']}")
    for st in trail["sub_trades"]:
        print(f"      entry={st['entry']:.2f} exit={st['exit']:.2f} pnl={st['pnl_pct']:+.4f}% "
              f"reason={st['reason']} days={st['entry_day']}-{st['exit_day']} "
              f"MFE={st['mfe_pct']:+.4f}% MAE={st['mae_pct']:+.4f}%")

    long_details.append({
        "date": trade["date"].strftime("%Y-%m-%d"),
        "correct": correct,
        "actual_pct": actual_pct,
        "trail_pnl": trail["total_pnl"],
        "n_subtrades": trail["n_subtrades"],
        "exit_reasons": [st["reason"] for st in trail["sub_trades"]],
    })

long_da = long_win / len(long_2025) * 100 if len(long_2025) > 0 else 0
print(f"\n--- LONG SUMMARY ---")
print(f"  DA: {long_win}/{len(long_2025)} = {long_da:.1f}%")
long_trail_pnls = [d["trail_pnl"] for d in long_details if d["trail_pnl"] is not None]
long_trail_wins = sum(1 for p in long_trail_pnls if p > 0)
print(f"  Trail WR: {long_trail_wins}/{len(long_trail_pnls)} = {long_trail_wins/len(long_trail_pnls)*100:.1f}%")
print(f"  Trail total: {sum(long_trail_pnls):.4f}%")
print(f"  Trail avg:   {np.mean(long_trail_pnls):.4f}%")

# Exit reasons for LONGs
all_reasons = []
for d in long_details:
    all_reasons.extend(d["exit_reasons"])
from collections import Counter
reason_counts = Counter(all_reasons)
print(f"\n  LONG exit reasons:")
for r, c in reason_counts.most_common():
    print(f"    {r}: {c}")

# SHORT breakdown
print(f"\n{'='*60}")
print(f"SHORT TRADES ({len(short_2025)} trades)")
print(f"{'='*60}")

short_win = 0
short_details = []
for idx, (_, trade) in enumerate(short_2025.iterrows()):
    actual_pct = (np.exp(trade["actual_return"]) - 1) * 100
    correct = trade["actual_return"] < 0
    if correct:
        short_win += 1

    trail = trailing_detail(trade, config)
    short_details.append({
        "date": trade["date"].strftime("%Y-%m-%d"),
        "correct": correct,
        "actual_pct": actual_pct,
        "trail_pnl": trail["total_pnl"],
        "n_subtrades": trail["n_subtrades"],
        "exit_reasons": [st["reason"] for st in trail["sub_trades"]],
    })

short_da = short_win / len(short_2025) * 100 if len(short_2025) > 0 else 0
print(f"  DA: {short_win}/{len(short_2025)} = {short_da:.1f}%")
short_trail_pnls = [d["trail_pnl"] for d in short_details if d["trail_pnl"] is not None]
short_trail_wins = sum(1 for p in short_trail_pnls if p > 0)
print(f"  Trail WR: {short_trail_wins}/{len(short_trail_pnls)} = {short_trail_wins/len(short_trail_pnls)*100:.1f}%")
print(f"  Trail total: {sum(short_trail_pnls):.4f}%")
print(f"  Trail avg:   {np.mean(short_trail_pnls):.4f}%")

# Exit reasons for SHORTs
all_reasons_short = []
for d in short_details:
    all_reasons_short.extend(d["exit_reasons"])
reason_counts_short = Counter(all_reasons_short)
print(f"\n  SHORT exit reasons:")
for r, c in reason_counts_short.most_common():
    print(f"    {r}: {c}")

# === 2026 ===
print(f"\n{'='*80}")
print(f"H=5 LONG vs SHORT BREAKDOWN - 2026 OOS")
print(f"{'='*80}")

preds_2026 = predict_all(test_2026)
if len(preds_2026) > 0:
    weekly_2026 = select_weekly(preds_2026)
    print(f"Total weekly trades 2026: {len(weekly_2026)}")

    long_2026 = weekly_2026[weekly_2026["direction"] == 1]
    short_2026 = weekly_2026[weekly_2026["direction"] == -1]
    print(f"LONG: {len(long_2026)}, SHORT: {len(short_2026)}")

    for _, trade in weekly_2026.iterrows():
        actual_pct = (np.exp(trade["actual_return"]) - 1) * 100
        correct = (trade["direction"] == 1 and trade["actual_return"] > 0) or (trade["direction"] == -1 and trade["actual_return"] < 0)
        trail = trailing_detail(trade, config)
        dir_str = "LONG" if trade["direction"] == 1 else "SHORT"
        model_agr = f"{trade['n_models_long']}L/{trade['n_models_short']}S"
        print(f"  {trade['date'].strftime('%Y-%m-%d')}: {dir_str} ({model_agr}) actual={actual_pct:+.3f}% "
              f"trail={trail['total_pnl']:+.4f}% [{'OK' if correct else 'WRONG'}]")
        for st in trail["sub_trades"]:
            print(f"    sub: pnl={st['pnl_pct']:+.4f}% reason={st['reason']} MFE={st['mfe_pct']:+.4f}% MAE={st['mae_pct']:+.4f}%")
else:
    print("No 2026 predictions (target NaN)")

# === OVERALL SUMMARY ===
print(f"\n{'='*80}")
print(f"OVERALL SUMMARY")
print(f"{'='*80}")
print(f"2025 LONG:  DA={long_da:.1f}%, Trail WR={long_trail_wins}/{len(long_trail_pnls)}, Trail total={sum(long_trail_pnls):.4f}%")
print(f"2025 SHORT: DA={short_da:.1f}%, Trail WR={short_trail_wins}/{len(short_trail_pnls)}, Trail total={sum(short_trail_pnls):.4f}%")
print(f"\nKey question: Is LONG generating real alpha or bleeding?")
if long_da >= 55:
    print(">>> LONG DA >= 55%: Model has bidirectional capability, DO NOT filter")
elif long_da >= 40:
    print(">>> LONG DA 40-55%: Ambiguous, need more data before filtering")
else:
    print(">>> LONG DA < 40%: Asymmetry confirmed, consider SHORT-only filter")

print(f"\nKey question: Are LONGs dying to hard stops?")
hard_pct = reason_counts.get("hard_stop", 0) / sum(reason_counts.values()) * 100 if reason_counts else 0
trail_pct = sum(v for k, v in reason_counts.items() if "trail" in str(k).lower()) / sum(reason_counts.values()) * 100 if reason_counts else 0
print(f"  LONG hard stops: {hard_pct:.1f}%")
print(f"  LONG trail stops: {trail_pct:.1f}%")
if hard_pct > 50:
    print(">>> LONGs mostly hit hard stop -> problem is timing, not direction")
elif trail_pct > 50:
    print(">>> LONGs mostly hit trail stop -> trail captures LONG moves normally")
else:
    print(">>> Mixed exit reasons -> no clear pattern")
