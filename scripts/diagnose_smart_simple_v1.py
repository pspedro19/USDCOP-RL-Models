"""
Root Cause Analysis: Smart Simple v1.0
=======================================

Deep dive into:
1. Where is PnL being left on the table?
2. Losing trade anatomy — are they preventable?
3. Skipped LONGs — was the filter correct?
4. TP/HS calibration — are stops too tight/loose?
5. Confidence scoring accuracy — does tier predict outcome?
6. Weekly return distribution — is alpha decaying?
7. Feature importance — what drives the signal?
8. Alternative horizons — is H=5 optimal?
9. Sensitivity: TP ratio, HS multiplier, confidence thresholds
"""

import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.confidence_scorer import ConfidenceConfig, score_confidence
from src.forecasting.adaptive_stops import (
    AdaptiveStopsConfig, compute_adaptive_stops,
    check_hard_stop, check_take_profit, get_exit_price,
)

# ── Configs ──────────────────────────────────────────────────────────
CONF_CONFIG = ConfidenceConfig()
STOPS_CONFIG = AdaptiveStopsConfig()
VT_TV, VT_MAX, VT_MIN, VT_FLOOR = 0.15, 2.0, 0.5, 0.05
SLIPPAGE = 0.0001


def load_data():
    ohlcv_path = PROJECT_ROOT / "seeds" / "latest" / "usdcop_daily_ohlcv.parquet"
    macro_path = PROJECT_ROOT / "data" / "pipeline" / "04_cleaning" / "output" / "MACRO_DAILY_CLEAN.parquet"

    df_ohlcv = pd.read_parquet(ohlcv_path).reset_index()
    df_ohlcv.rename(columns={"time": "date"}, inplace=True)
    df_ohlcv["date"] = pd.to_datetime(df_ohlcv["date"]).dt.tz_localize(None).dt.normalize()
    df_ohlcv = df_ohlcv[["date", "open", "high", "low", "close"]].sort_values("date").reset_index(drop=True)

    df_macro = pd.read_parquet(macro_path).reset_index()
    df_macro.rename(columns={df_macro.columns[0]: "date"}, inplace=True)
    df_macro["date"] = pd.to_datetime(df_macro["date"]).dt.tz_localize(None).dt.normalize()

    macro_cols = {
        "FXRT_INDEX_DXY_USA_D_DXY": "dxy_close_lag1",
        "COMM_OIL_WTI_GLB_D_WTI": "oil_close_lag1",
        "VOLT_VIX_USA_D_VIX": "vix_close_lag1",
        "CRSK_SPREAD_EMBI_COL_D_EMBI": "embi_close_lag1",
    }
    df_macro_sub = df_macro[["date"] + list(macro_cols.keys())].copy()
    df_macro_sub.rename(columns=macro_cols, inplace=True)
    df_macro_sub = df_macro_sub.sort_values("date")
    for col in macro_cols.values():
        df_macro_sub[col] = df_macro_sub[col].shift(1)

    df = pd.merge_asof(df_ohlcv.sort_values("date"), df_macro_sub.sort_values("date"),
                        on="date", direction="backward")

    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)
    df["return_20d"] = df["close"].pct_change(20)
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14d"] = 100 - (100 / (1 + rs))
    df["ma_ratio_20d"] = df["close"] / df["close"].rolling(20).mean()
    df["ma_ratio_50d"] = df["close"] / df["close"].rolling(50).mean()
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["is_month_end"] = pd.to_datetime(df["date"]).dt.is_month_end.astype(int)

    for col in ["dxy_close_lag1", "oil_close_lag1", "vix_close_lag1", "embi_close_lag1"]:
        df[col] = df[col].ffill()

    feature_cols = [
        "close", "open", "high", "low",
        "return_1d", "return_5d", "return_10d", "return_20d",
        "volatility_5d", "volatility_10d", "volatility_20d",
        "rsi_14d", "ma_ratio_20d", "ma_ratio_50d",
        "day_of_week", "month", "is_month_end",
        "dxy_close_lag1", "oil_close_lag1", "vix_close_lag1", "embi_close_lag1",
    ]

    # Multiple targets
    for h in [1, 3, 5, 10]:
        df[f"target_{h}d"] = np.log(df["close"].shift(-h) / df["close"])

    return df, feature_cols


def simulate_week(direction, entry, bars, hs_pct, tp_pct):
    for i, bar in enumerate(bars):
        if check_hard_stop(direction, entry, bar["high"], bar["low"], hs_pct):
            ep = get_exit_price(direction, entry, "hard_stop", hs_pct, tp_pct, bar["close"])
            return ep, "hard_stop", i
        if check_take_profit(direction, entry, bar["high"], bar["low"], tp_pct):
            ep = get_exit_price(direction, entry, "take_profit", hs_pct, tp_pct, bar["close"])
            return ep, "take_profit", i
    return bars[-1]["close"], "week_end", len(bars) - 1


def run_full_analysis(df, feature_cols, year):
    """Run backtest collecting ALL trade data including skipped LONGs."""
    from sklearn.linear_model import Ridge, BayesianRidge
    from sklearn.preprocessing import StandardScaler

    test_start = pd.Timestamp(f"{year}-01-01")
    test_end = pd.Timestamp(f"{year}-12-31")
    test_data = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    mondays = test_data[test_data["date"].dt.dayofweek == 0]["date"].unique()

    all_trades = []

    for monday in mondays:
        monday_ts = pd.Timestamp(monday)
        friday_ts = monday_ts + pd.offsets.BDay(4)

        train_end = monday_ts - timedelta(days=1)
        df_train = df[(df["date"] <= train_end) & df["target_5d"].notna()].copy()
        mask = df_train[feature_cols].notna().all(axis=1)
        df_train = df_train[mask]
        if len(df_train) < 100:
            continue

        X = df_train[feature_cols].values.astype(np.float64)
        y = df_train["target_5d"].values.astype(np.float64)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        ridge = Ridge(alpha=1.0).fit(Xs, y)
        br = BayesianRidge(max_iter=300).fit(Xs, y)

        latest = scaler.transform(df_train[feature_cols].iloc[-1:].values.astype(np.float64))
        pred_ridge = float(ridge.predict(latest)[0])
        pred_br = float(br.predict(latest)[0])
        ensemble = (pred_ridge + pred_br) / 2.0
        direction = 1 if ensemble > 0 else -1

        # Confidence
        conf = score_confidence(pred_ridge, pred_br, direction, CONF_CONFIG)

        # Vol
        rets = df_train["return_1d"].dropna().values
        rv_daily = np.std(rets[-21:]) if len(rets) >= 21 else 0.0
        rv_ann = rv_daily * np.sqrt(252) if rv_daily > 0 else VT_TV
        safe_vol = max(rv_ann, VT_FLOOR)
        base_lev = np.clip(VT_TV / safe_vol, VT_MIN, VT_MAX)

        # Feature importances from ridge
        ridge_coefs = np.abs(ridge.coef_)

        # Stops
        stops = compute_adaptive_stops(rv_ann, STOPS_CONFIG)

        # Entry
        monday_row = df[df["date"] == monday_ts]
        if monday_row.empty:
            m2 = df["date"] >= monday_ts
            if m2.any():
                monday_row = df[m2].iloc[:1]
            else:
                continue
        entry = float(monday_row["close"].iloc[0])

        # Week bars
        week_dates = pd.bdate_range(monday_ts + timedelta(days=1), friday_ts)
        bars = []
        for day in week_dates:
            r = df[df["date"] == day]
            if not r.empty:
                bars.append({
                    "high": float(r["high"].iloc[0]),
                    "low": float(r["low"].iloc[0]),
                    "close": float(r["close"].iloc[0]),
                })
        if not bars:
            continue

        # Actual move
        actual_friday = bars[-1]["close"]
        actual_return = direction * (actual_friday - entry) / entry

        # Simulate with CURRENT stops
        exit_p, reason, exit_day = simulate_week(
            direction, entry, bars, stops.hard_stop_pct, stops.take_profit_pct)

        # Also simulate the OPPOSITE direction (counterfactual)
        opp_dir = -direction
        opp_exit_p, opp_reason, opp_exit_day = simulate_week(
            opp_dir, entry, bars, stops.hard_stop_pct, stops.take_profit_pct)

        # PnL for smart strategy
        final_lev = np.clip(base_lev * conf.sizing_multiplier, VT_MIN, VT_MAX)
        raw_pnl = direction * (exit_p - entry) / entry * final_lev
        cost = SLIPPAGE * final_lev if reason == "week_end" else 0.0
        net_pnl = raw_pnl - cost

        # PnL if we had taken this trade at 1x (no confidence filter)
        lev_1x = base_lev
        raw_pnl_1x = direction * (exit_p - entry) / entry * lev_1x
        cost_1x = SLIPPAGE * lev_1x if reason == "week_end" else 0.0
        net_pnl_1x = raw_pnl_1x - cost_1x

        # Max favorable / adverse excursion
        mfe = 0.0  # max favorable
        mae = 0.0  # max adverse
        for bar in bars:
            if direction == 1:
                fav = (bar["high"] - entry) / entry
                adv = (entry - bar["low"]) / entry
            else:
                fav = (entry - bar["low"]) / entry
                adv = (bar["high"] - entry) / entry
            mfe = max(mfe, fav)
            mae = max(mae, adv)

        all_trades.append({
            "week": str(monday_ts.date()),
            "direction": direction,
            "dir_str": "LONG" if direction == 1 else "SHORT",
            "pred_ridge": pred_ridge,
            "pred_br": pred_br,
            "ensemble": ensemble,
            "conf_tier": conf.tier.value,
            "conf_agreement": conf.agreement,
            "conf_magnitude": conf.magnitude,
            "sizing_mult": conf.sizing_multiplier,
            "skip_trade": conf.skip_trade,
            "rv_ann": rv_ann,
            "base_lev": base_lev,
            "final_lev": final_lev,
            "hs_pct": stops.hard_stop_pct,
            "tp_pct": stops.take_profit_pct,
            "entry": entry,
            "exit": exit_p,
            "reason": reason,
            "exit_day": exit_day,
            "actual_friday": actual_friday,
            "actual_return": actual_return,
            "mfe": mfe,
            "mae": mae,
            "net_pnl": net_pnl,
            "net_pnl_1x": net_pnl_1x,
            "opp_reason": opp_reason,
            "opp_pnl_raw": opp_dir * (opp_exit_p - entry) / entry,
            "ridge_top3_feats": np.argsort(ridge_coefs)[-3:][::-1].tolist(),
        })

    return pd.DataFrame(all_trades)


def print_section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def main():
    df, feature_cols = load_data()

    for year in [2025]:
        trades = run_full_analysis(df, feature_cols, year)
        if trades.empty:
            continue

        shorts = trades[trades["direction"] == -1]
        longs = trades[trades["direction"] == 1]
        executed = trades[~trades["skip_trade"]]
        skipped = trades[trades["skip_trade"]]

        # ─── 1. OVERVIEW ────────────────────────────────────────────
        print_section(f"1. OVERVIEW — {year}")
        print(f"  Total signals:  {len(trades)} ({len(shorts)}S / {len(longs)}L)")
        print(f"  Executed:       {len(executed)} ({len(executed[executed['direction']==-1])}S / {len(executed[executed['direction']==1])}L)")
        print(f"  Skipped:        {len(skipped)} (all LOW-confidence LONGs)")

        eq = 10000.0
        for _, t in executed.iterrows():
            eq *= (1 + t["net_pnl"])
        print(f"  Equity:         $10,000 -> ${eq:,.2f} ({(eq/10000-1)*100:+.2f}%)")

        # ─── 2. LOSING TRADE ANATOMY ────────────────────────────────
        print_section("2. LOSING TRADE ANATOMY")
        losers = executed[executed["net_pnl"] < 0]
        winners = executed[executed["net_pnl"] > 0]
        print(f"  Winners: {len(winners)}/{len(executed)} ({len(winners)/len(executed)*100:.0f}%)")
        print(f"  Losers:  {len(losers)}/{len(executed)} ({len(losers)/len(executed)*100:.0f}%)")
        print(f"  Avg win:  {winners['net_pnl'].mean()*100:+.3f}%")
        print(f"  Avg loss: {losers['net_pnl'].mean()*100:+.3f}%")
        print(f"  Win/Loss ratio: {abs(winners['net_pnl'].mean() / losers['net_pnl'].mean()):.2f}x")
        print()

        print(f"  {'Week':<12} {'Dir':<6} {'Conf':<8} {'Entry':>8} {'Exit':>8} {'Reason':<12} {'PnL%':>8} {'MFE%':>6} {'MAE%':>6} {'Problem'}")
        print(f"  {'-'*100}")
        for _, t in losers.iterrows():
            # Diagnose the problem
            if t["reason"] == "hard_stop":
                problem = "Adverse move hit HS"
            elif t["mae"] > t["hs_pct"] * 0.8:
                problem = "Near-miss HS"
            elif t["mfe"] > t["tp_pct"] * 0.8:
                problem = "TP almost hit, then reversed"
            elif t["actual_return"] > 0:
                problem = "Right direction, wrong timing"
            else:
                problem = "Wrong direction"

            print(f"  {t['week']:<12} {t['dir_str']:<6} {t['conf_tier']:<8} "
                  f"{t['entry']:>8.0f} {t['exit']:>8.0f} {t['reason']:<12} "
                  f"{t['net_pnl']*100:>+7.3f}% {t['mfe']*100:>5.2f}% {t['mae']*100:>5.2f}% "
                  f"{problem}")

        # ─── 3. SKIPPED LONGS — COUNTERFACTUAL ──────────────────────
        print_section("3. SKIPPED LONGS — Were we right to skip?")
        if len(skipped) > 0:
            skipped_would_profit = skipped[skipped["net_pnl_1x"] > 0]
            skipped_would_lose = skipped[skipped["net_pnl_1x"] <= 0]
            print(f"  Skipped LONGs: {len(skipped)}")
            print(f"  Would have won:  {len(skipped_would_profit)} ({skipped_would_profit['net_pnl_1x'].sum()*100:+.3f}% total)")
            print(f"  Would have lost: {len(skipped_would_lose)} ({skipped_would_lose['net_pnl_1x'].sum()*100:+.3f}% total)")
            print(f"  Net if taken:    {skipped['net_pnl_1x'].sum()*100:+.3f}%")
            print(f"  --> {'CORRECT to skip' if skipped['net_pnl_1x'].sum() < 0 else 'WRONG to skip (left money on table)'}")

            print(f"\n  {'Week':<12} {'Conf':<8} {'Ensemble':>10} {'Agree':>8} {'1x PnL%':>8} {'Actual':>8} {'Reason':<12} {'Verdict'}")
            print(f"  {'-'*85}")
            for _, t in skipped.iterrows():
                verdict = "Good skip" if t["net_pnl_1x"] <= 0 else "MISSED $$$"
                print(f"  {t['week']:<12} {t['conf_tier']:<8} {t['ensemble']:>+10.6f} "
                      f"{t['conf_agreement']:>8.6f} {t['net_pnl_1x']*100:>+7.3f}% "
                      f"{t['actual_return']*100:>+7.3f}% {t['reason']:<12} {verdict}")
        else:
            print("  No LONGs skipped")

        # ─── 4. TP/HS CALIBRATION ──────────────────────────────────
        print_section("4. TP/HS CALIBRATION — Are stops optimal?")
        tp_trades = executed[executed["reason"] == "take_profit"]
        hs_trades = executed[executed["reason"] == "hard_stop"]
        we_trades = executed[executed["reason"] == "week_end"]

        print(f"  Exits: {len(tp_trades)} TP, {len(we_trades)} week_end, {len(hs_trades)} HS")
        print(f"  TP trades avg PnL: {tp_trades['net_pnl'].mean()*100:+.3f}%")
        print(f"  HS trades avg PnL: {hs_trades['net_pnl'].mean()*100:+.3f}%")
        print(f"  WE trades avg PnL: {we_trades['net_pnl'].mean()*100:+.3f}%")

        # MFE analysis: how much further could we have gone?
        print(f"\n  MFE Analysis (Max Favorable Excursion):")
        print(f"  {'Exit':12} {'Avg MFE%':>10} {'Avg TP%':>10} {'MFE/TP':>10} {'Interpretation'}")
        for reason_name, group in [("take_profit", tp_trades), ("week_end", we_trades), ("hard_stop", hs_trades)]:
            if len(group) > 0:
                avg_mfe = group["mfe"].mean() * 100
                avg_tp = group["tp_pct"].mean() * 100
                ratio = avg_mfe / avg_tp if avg_tp > 0 else 0
                if ratio > 2.0:
                    interp = "TP too tight! Missing big moves"
                elif ratio > 1.5:
                    interp = "TP slightly tight"
                elif ratio < 0.5:
                    interp = "TP rarely reachable"
                else:
                    interp = "TP well calibrated"
                print(f"  {reason_name:12} {avg_mfe:>9.2f}% {avg_tp:>9.2f}% {ratio:>9.2f}x {interp}")

        # MAE analysis: how bad before exit?
        print(f"\n  MAE Analysis (Max Adverse Excursion):")
        for reason_name, group in [("take_profit", tp_trades), ("week_end", we_trades), ("hard_stop", hs_trades)]:
            if len(group) > 0:
                avg_mae = group["mae"].mean() * 100
                avg_hs = group["hs_pct"].mean() * 100
                ratio = avg_mae / avg_hs
                print(f"  {reason_name:12} avg MAE={avg_mae:.2f}%, avg HS={avg_hs:.2f}%, MAE/HS={ratio:.2f}x")

        # ─── 5. WEEK_END TRADES — UNREALIZED ALPHA ────────────────
        print_section("5. WEEK_END TRADES — Could we extract more?")
        if len(we_trades) > 0:
            we_positive = we_trades[we_trades["net_pnl"] > 0]
            we_negative = we_trades[we_trades["net_pnl"] <= 0]
            print(f"  week_end trades: {len(we_trades)} ({len(we_positive)} win, {len(we_negative)} lose)")
            print(f"  Avg MFE of week_end winners: {we_positive['mfe'].mean()*100:.2f}%")
            print(f"  Avg TP for those weeks:       {we_positive['tp_pct'].mean()*100:.2f}%")

            # How many week_end winners had MFE > TP/2?
            close_to_tp = we_positive[we_positive["mfe"] > we_positive["tp_pct"] * 0.7]
            print(f"  Winners with MFE > 70% of TP: {len(close_to_tp)}/{len(we_positive)}")
            print(f"    --> These nearly hit TP. Consider slightly tighter TP ratio or trailing")

            # How many week_end losers were small losses?
            small_loss = we_negative[we_negative["net_pnl"].abs() < 0.005]
            print(f"  Losers with |PnL| < 0.5%:     {len(small_loss)}/{len(we_negative)} (noise trades)")

        # ─── 6. CONFIDENCE TIER PERFORMANCE ─────────────────────────
        print_section("6. CONFIDENCE TIER PERFORMANCE")
        # Include ALL trades (even skipped, using 1x PnL)
        for tier in ["HIGH", "MEDIUM", "LOW"]:
            tier_trades = trades[trades["conf_tier"] == tier]
            if len(tier_trades) == 0:
                continue
            tier_shorts = tier_trades[tier_trades["direction"] == -1]
            tier_longs = tier_trades[tier_trades["direction"] == 1]

            print(f"\n  {tier} confidence:")
            print(f"    Total: {len(tier_trades)} ({len(tier_shorts)}S / {len(tier_longs)}L)")
            if len(tier_shorts) > 0:
                short_wr = (tier_shorts["net_pnl_1x"] > 0).mean() * 100
                short_avg = tier_shorts["net_pnl_1x"].mean() * 100
                print(f"    SHORT: WR={short_wr:.0f}%, avg PnL={short_avg:+.3f}%")
            if len(tier_longs) > 0:
                long_wr = (tier_longs["net_pnl_1x"] > 0).mean() * 100
                long_avg = tier_longs["net_pnl_1x"].mean() * 100
                print(f"    LONG:  WR={long_wr:.0f}%, avg PnL={long_avg:+.3f}% {'<-- skipped' if tier_longs['skip_trade'].all() else ''}")

        # ─── 7. MONTHLY SEASONALITY ─────────────────────────────────
        print_section("7. MONTHLY BREAKDOWN — Where is alpha?")
        executed["month"] = pd.to_datetime(executed["week"]).dt.month
        monthly = executed.groupby("month").agg(
            trades=("net_pnl", "count"),
            total_pnl=("net_pnl", lambda x: x.sum() * 100),
            avg_pnl=("net_pnl", lambda x: x.mean() * 100),
            wr=("net_pnl", lambda x: (x > 0).mean() * 100),
        ).round(2)
        print(f"  {'Month':>5} {'Trades':>7} {'Total PnL%':>11} {'Avg PnL%':>10} {'WR%':>6}")
        print(f"  {'-'*45}")
        for m, row in monthly.iterrows():
            print(f"  {m:>5} {row['trades']:>7.0f} {row['total_pnl']:>+10.2f}% {row['avg_pnl']:>+9.3f}% {row['wr']:>5.0f}%")

        # ─── 8. SENSITIVITY ANALYSIS ───────────────────────────────
        print_section("8. SENSITIVITY — TP ratio and HS multiplier")

        tp_ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
        hs_mults = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]

        print(f"\n  TP ratio sensitivity (HS mult = 1.5 fixed):")
        print(f"  {'TP ratio':>8} {'Return%':>9} {'Sharpe':>8} {'MaxDD%':>8} {'WR%':>6} {'#TP':>5} {'#WE':>5} {'#HS':>5}")
        print(f"  {'-'*58}")

        best_sharpe = -999
        best_config = ""

        for tp_r in tp_ratios:
            cfg = AdaptiveStopsConfig(vol_multiplier=1.5, tp_ratio=tp_r)
            eq = 10000.0
            rets_list = []
            n_tp, n_we, n_hs = 0, 0, 0
            for _, t in executed.iterrows():
                stops = compute_adaptive_stops(t["rv_ann"], cfg)
                exit_p, reason, _ = simulate_week(
                    t["direction"], t["entry"],
                    _get_bars(df, pd.Timestamp(t["week"])),
                    stops.hard_stop_pct, stops.take_profit_pct)
                lev = t["final_lev"]
                pnl = t["direction"] * (exit_p - t["entry"]) / t["entry"] * lev
                cost = SLIPPAGE * lev if reason == "week_end" else 0.0
                net = pnl - cost
                eq *= (1 + net)
                rets_list.append(net)
                if reason == "take_profit": n_tp += 1
                elif reason == "hard_stop": n_hs += 1
                else: n_we += 1

            ret = (eq / 10000 - 1) * 100
            wr = np.mean(np.array(rets_list) > 0) * 100
            sh = np.mean(rets_list) / np.std(rets_list) * np.sqrt(52) if np.std(rets_list) > 0 else 0
            peak = np.maximum.accumulate(np.cumprod(1 + np.array(rets_list)))
            dd = (np.cumprod(1 + np.array(rets_list)) - peak) / peak
            mdd = np.min(dd) * 100

            marker = " <-- current" if abs(tp_r - 0.5) < 0.01 else ""
            if sh > best_sharpe:
                best_sharpe = sh
                best_config = f"TP={tp_r}"
            print(f"  {tp_r:>8.1f} {ret:>+8.2f}% {sh:>8.3f} {mdd:>7.2f}% {wr:>5.0f}% {n_tp:>5} {n_we:>5} {n_hs:>5}{marker}")

        print(f"\n  HS multiplier sensitivity (TP ratio = 0.5 fixed):")
        print(f"  {'HS mult':>8} {'Return%':>9} {'Sharpe':>8} {'MaxDD%':>8} {'WR%':>6} {'#TP':>5} {'#WE':>5} {'#HS':>5}")
        print(f"  {'-'*58}")

        for hs_m in hs_mults:
            cfg = AdaptiveStopsConfig(vol_multiplier=hs_m, tp_ratio=0.5)
            eq = 10000.0
            rets_list = []
            n_tp, n_we, n_hs = 0, 0, 0
            for _, t in executed.iterrows():
                stops = compute_adaptive_stops(t["rv_ann"], cfg)
                exit_p, reason, _ = simulate_week(
                    t["direction"], t["entry"],
                    _get_bars(df, pd.Timestamp(t["week"])),
                    stops.hard_stop_pct, stops.take_profit_pct)
                lev = t["final_lev"]
                pnl = t["direction"] * (exit_p - t["entry"]) / t["entry"] * lev
                cost = SLIPPAGE * lev if reason == "week_end" else 0.0
                net = pnl - cost
                eq *= (1 + net)
                rets_list.append(net)
                if reason == "take_profit": n_tp += 1
                elif reason == "hard_stop": n_hs += 1
                else: n_we += 1

            ret = (eq / 10000 - 1) * 100
            wr = np.mean(np.array(rets_list) > 0) * 100
            sh = np.mean(rets_list) / np.std(rets_list) * np.sqrt(52) if np.std(rets_list) > 0 else 0
            peak = np.maximum.accumulate(np.cumprod(1 + np.array(rets_list)))
            dd = (np.cumprod(1 + np.array(rets_list)) - peak) / peak
            mdd = np.min(dd) * 100

            marker = " <-- current" if abs(hs_m - 1.5) < 0.01 else ""
            if sh > best_sharpe:
                best_sharpe = sh
                best_config = f"HS_mult={hs_m}"
            print(f"  {hs_m:>8.2f} {ret:>+8.2f}% {sh:>8.3f} {mdd:>7.2f}% {wr:>5.0f}% {n_tp:>5} {n_we:>5} {n_hs:>5}{marker}")

        # ─── 9. 2D GRID SEARCH ─────────────────────────────────────
        print_section("9. TOP-10 CONFIGURATIONS (2D Grid)")
        grid_results = []
        for tp_r in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
            for hs_m in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]:
                cfg = AdaptiveStopsConfig(vol_multiplier=hs_m, tp_ratio=tp_r)
                eq = 10000.0
                rets_list = []
                for _, t in executed.iterrows():
                    stops = compute_adaptive_stops(t["rv_ann"], cfg)
                    exit_p, reason, _ = simulate_week(
                        t["direction"], t["entry"],
                        _get_bars(df, pd.Timestamp(t["week"])),
                        stops.hard_stop_pct, stops.take_profit_pct)
                    lev = t["final_lev"]
                    pnl = t["direction"] * (exit_p - t["entry"]) / t["entry"] * lev
                    cost = SLIPPAGE * lev if reason == "week_end" else 0.0
                    eq *= (1 + pnl - cost)
                    rets_list.append(pnl - cost)

                ret = (eq / 10000 - 1) * 100
                sh = np.mean(rets_list) / np.std(rets_list) * np.sqrt(52) if np.std(rets_list) > 0 else 0
                peak = np.maximum.accumulate(np.cumprod(1 + np.array(rets_list)))
                dd = (np.cumprod(1 + np.array(rets_list)) - peak) / peak
                mdd = np.min(dd) * 100

                np.random.seed(42)
                boot = [np.mean(np.random.choice(rets_list, size=len(rets_list), replace=True)) for _ in range(5000)]
                p_val = np.mean(np.array(boot) <= 0)

                grid_results.append({
                    "tp_r": tp_r, "hs_m": hs_m, "ret": ret, "sharpe": sh,
                    "mdd": mdd, "p_val": p_val,
                    "current": (abs(tp_r-0.5) < 0.01 and abs(hs_m-1.5) < 0.01),
                })

        grid_df = pd.DataFrame(grid_results).sort_values("sharpe", ascending=False)
        print(f"  {'TP':>5} {'HS_m':>6} {'Return%':>9} {'Sharpe':>8} {'MaxDD%':>8} {'p-val':>7} {'Note'}")
        print(f"  {'-'*55}")
        for i, row in grid_df.head(10).iterrows():
            note = "<-- CURRENT" if row["current"] else ""
            print(f"  {row['tp_r']:>5.1f} {row['hs_m']:>6.2f} {row['ret']:>+8.2f}% {row['sharpe']:>8.3f} {row['mdd']:>7.2f}% {row['p_val']:>7.4f} {note}")

        # Where is current?
        curr_rank = grid_df.reset_index(drop=True)
        curr_idx = curr_rank[curr_rank["current"]].index[0]
        print(f"\n  Current config rank: #{curr_idx + 1} of {len(grid_df)}")

        # ─── 10. THEORETICAL CEILING ──────────────────────────────
        print_section("10. THEORETICAL CEILING — Perfect foresight")
        # If we knew the actual direction every week
        perfect_eq = 10000.0
        for _, t in trades.iterrows():
            bars = _get_bars(df, pd.Timestamp(t["week"]))
            if not bars:
                continue
            actual_dir = 1 if bars[-1]["close"] > t["entry"] else -1
            stops = compute_adaptive_stops(t["rv_ann"], STOPS_CONFIG)
            exit_p, reason, _ = simulate_week(
                actual_dir, t["entry"], bars, stops.hard_stop_pct, stops.take_profit_pct)
            lev = np.clip(t["base_lev"], VT_MIN, VT_MAX)
            pnl = actual_dir * (exit_p - t["entry"]) / t["entry"] * lev
            cost = SLIPPAGE * lev if reason == "week_end" else 0.0
            perfect_eq *= (1 + pnl - cost)

        print(f"  Perfect direction (DA=100%): $10K -> ${perfect_eq:,.2f} ({(perfect_eq/10000-1)*100:+.2f}%)")
        print(f"  Current strategy:            $10K -> ${eq:,.2f} ({(eq/10000-1)*100:+.2f}%)")
        print(f"  Capture ratio:               {((eq/10000-1) / (perfect_eq/10000-1) * 100):.1f}%")


def _get_bars(df, monday_ts):
    friday_ts = monday_ts + pd.offsets.BDay(4)
    week_dates = pd.bdate_range(monday_ts + timedelta(days=1), friday_ts)
    bars = []
    for day in week_dates:
        r = df[df["date"] == day]
        if not r.empty:
            bars.append({
                "high": float(r["high"].iloc[0]),
                "low": float(r["low"].iloc[0]),
                "close": float(r["close"].iloc[0]),
            })
    return bars


if __name__ == "__main__":
    main()
