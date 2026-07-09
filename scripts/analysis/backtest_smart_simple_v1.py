"""
Comparative Backtest: Smart Simple v1.x
========================================

Compares 3 strategy variants on OOS 2025 + 2026 data:

    A) SHORT-only:  Only SHORT trades, fixed 1.0x sizing
    B) Bidir naive: Both directions, fixed 1.0x sizing (no asymmetry)
    C) Bidir smart: Both directions, confidence-based asymmetric sizing

All params loaded from config/execution/smart_simple_v1.yaml.

Usage:
    python scripts/backtest_smart_simple_v1.py
"""

import sys
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.confidence_scorer import (
    ConfidenceConfig,
    ConfidenceTier,
    score_confidence,
)
from src.forecasting.adaptive_stops import (
    AdaptiveStopsConfig,
    compute_adaptive_stops,
    check_hard_stop,
    check_take_profit,
    get_exit_price,
)


def load_data():
    """Load OHLCV + macro, build 21 features, compute H=5 target."""
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

    df = pd.merge_asof(
        df_ohlcv.sort_values("date"),
        df_macro_sub.sort_values("date"),
        on="date",
        direction="backward",
    )

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
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14d"] = 100 - (100 / (1 + rs))
    df["ma_ratio_20d"] = df["close"] / df["close"].rolling(20).mean()
    df["ma_ratio_50d"] = df["close"] / df["close"].rolling(50).mean()
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["is_month_end"] = pd.to_datetime(df["date"]).dt.is_month_end.astype(int)

    for col in ["dxy_close_lag1", "oil_close_lag1", "vix_close_lag1", "embi_close_lag1"]:
        df[col] = df[col].ffill()

    df["target_return_5d"] = np.log(df["close"].shift(-5) / df["close"])

    feature_cols = [
        "close", "open", "high", "low",
        "return_1d", "return_5d", "return_10d", "return_20d",
        "volatility_5d", "volatility_10d", "volatility_20d",
        "rsi_14d", "ma_ratio_20d", "ma_ratio_50d",
        "day_of_week", "month", "is_month_end",
        "dxy_close_lag1", "oil_close_lag1",
        "vix_close_lag1", "embi_close_lag1",
    ]
    return df, feature_cols


# Load all params from YAML config
_cfg_path = PROJECT_ROOT / "config" / "execution" / "smart_simple_v1.yaml"
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

_vt = _cfg.get("vol_targeting", {})
VT_TV = _vt.get("target_vol", 0.15)
VT_MAX = _vt.get("max_leverage", 2.0)
VT_MIN = _vt.get("min_leverage", 0.5)
VT_FLOOR = _vt.get("vol_floor", 0.05)

_as = _cfg.get("adaptive_stops", {})
STOPS_CONFIG = AdaptiveStopsConfig(
    vol_multiplier=_as.get("vol_multiplier", 2.0),
    hard_stop_min_pct=_as.get("hard_stop_min_pct", 0.01),
    hard_stop_max_pct=_as.get("hard_stop_max_pct", 0.03),
    tp_ratio=_as.get("tp_ratio", 0.5),
)

_cc = _cfg.get("confidence", {})
CONF_CONFIG = ConfidenceConfig(
    agreement_tight=_cc.get("agreement_tight", 0.001),
    agreement_loose=_cc.get("agreement_loose", 0.005),
    magnitude_high=_cc.get("magnitude_high", 0.010),
    magnitude_medium=_cc.get("magnitude_medium", 0.005),
    short_high=_cc.get("short", {}).get("HIGH", 1.5),
    short_medium=_cc.get("short", {}).get("MEDIUM", 1.5),
    short_low=_cc.get("short", {}).get("LOW", 1.5),
    long_high=_cc.get("long", {}).get("HIGH", 1.0),
    long_medium=_cc.get("long", {}).get("MEDIUM", 0.5),
    long_low=_cc.get("long", {}).get("LOW", 0.0),
)

_costs = _cfg.get("execution", {}).get("costs", {})
MAKER_FEE = _costs.get("maker_fee_bps", 0.0) / 10000.0
TAKER_FEE = _costs.get("taker_fee_bps", 0.0) / 10000.0
SLIPPAGE = _costs.get("slippage_bps", 1.0) / 10000.0

_version = _cfg.get("executor", {}).get("version", "1.1.0")


def simulate_week(direction, entry, bars, hard_stop_pct, take_profit_pct):
    """
    Simulate a 5-day hold with TP/HS/Friday-close.

    Returns (exit_price, exit_reason, exit_bar_index).
    """
    for i, bar in enumerate(bars):
        h, l, c = bar["high"], bar["low"], bar["close"]

        # Check hard stop first (worst case)
        if check_hard_stop(direction, entry, h, l, hard_stop_pct):
            ep = get_exit_price(direction, entry, "hard_stop", hard_stop_pct, take_profit_pct, c)
            return ep, "hard_stop", i

        # Check take profit
        if check_take_profit(direction, entry, h, l, take_profit_pct):
            ep = get_exit_price(direction, entry, "take_profit", hard_stop_pct, take_profit_pct, c)
            return ep, "take_profit", i

    # Friday close (market order)
    return bars[-1]["close"], "week_end", len(bars) - 1


def compute_pnl(direction, entry, exit_price, leverage, exit_reason):
    """Compute PnL with costs."""
    raw_pnl = direction * (exit_price - entry) / entry * leverage

    # Costs: entry always limit (0%), exit depends on reason
    entry_cost = MAKER_FEE * leverage      # Limit order
    if exit_reason == "week_end":
        exit_cost = SLIPPAGE * leverage    # Market order, slippage only
    else:
        exit_cost = MAKER_FEE * leverage   # Limit order (TP/HS)

    return raw_pnl - entry_cost - exit_cost


def run_backtest(strategy_name, mode, year, df, feature_cols):
    """
    Run backtest for a specific strategy variant.

    Modes:
        "short_only": Only SHORT trades, 1.0x sizing
        "bidir_naive": Both directions, 1.0x sizing
        "bidir_smart": Both directions, confidence-based asymmetric sizing
    """
    from sklearn.linear_model import Ridge, BayesianRidge
    from sklearn.preprocessing import StandardScaler

    test_start = pd.Timestamp(f"{year}-01-01")
    test_end = pd.Timestamp(f"{year}-12-31")
    test_data = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    test_data["dow"] = test_data["date"].dt.dayofweek
    mondays = test_data[test_data["dow"] == 0]["date"].unique()

    if not len(mondays):
        return None

    equity = 10000.0
    trades = []

    for monday in mondays:
        monday_ts = pd.Timestamp(monday)
        friday_ts = monday_ts + pd.offsets.BDay(4)

        # Train on all data before Monday
        train_end = monday_ts - timedelta(days=1)
        df_train = df[(df["date"] <= train_end) & df["target_return_5d"].notna()].copy()
        mask = df_train[feature_cols].notna().all(axis=1) & df_train["target_return_5d"].notna()
        df_train = df_train[mask]
        if len(df_train) < 100:
            continue

        X = df_train[feature_cols].values.astype(np.float64)
        y = df_train["target_return_5d"].values.astype(np.float64)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        ridge = Ridge(alpha=1.0).fit(Xs, y)
        br = BayesianRidge(max_iter=300).fit(Xs, y)

        # Predict on latest training row
        latest = scaler.transform(df_train[feature_cols].iloc[-1:].values.astype(np.float64))
        pred_ridge = float(ridge.predict(latest)[0])
        pred_br = float(br.predict(latest)[0])
        ensemble = (pred_ridge + pred_br) / 2.0
        direction = 1 if ensemble > 0 else -1

        # --- STRATEGY FILTER ---

        if mode == "short_only":
            if direction == 1:
                continue  # Skip LONGs
            lev_mult = 1.0

        elif mode == "bidir_naive":
            lev_mult = 1.0

        elif mode == "bidir_smart":
            conf = score_confidence(pred_ridge, pred_br, direction, CONF_CONFIG)
            if conf.skip_trade:
                continue  # Skip LOW-confidence LONGs
            lev_mult = conf.sizing_multiplier

        # Vol-targeting (base leverage)
        rets = df_train["return_1d"].dropna().values
        rv_daily = np.std(rets[-21:]) if len(rets) >= 21 else 0.0
        rv_ann = rv_daily * np.sqrt(252) if rv_daily > 0 else VT_TV
        safe_vol = max(rv_ann, VT_FLOOR)
        base_lev = np.clip(VT_TV / safe_vol, VT_MIN, VT_MAX)

        # Apply confidence multiplier
        final_lev = base_lev * lev_mult
        final_lev = np.clip(final_lev, VT_MIN, VT_MAX)  # Re-clip after multiplier

        # Adaptive stops
        stops = compute_adaptive_stops(rv_ann, STOPS_CONFIG)

        # Entry price
        monday_row = df[df["date"] == monday_ts]
        if monday_row.empty:
            m2 = df["date"] >= monday_ts
            if m2.any():
                monday_row = df[m2].iloc[:1]
            else:
                continue
        entry = float(monday_row["close"].iloc[0])

        # Week bars (Tue-Fri)
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

        # Simulate
        exit_p, reason, _ = simulate_week(
            direction, entry, bars,
            stops.hard_stop_pct, stops.take_profit_pct,
        )

        pnl = compute_pnl(direction, entry, exit_p, final_lev, reason)
        equity *= (1 + pnl)

        # Direction accuracy
        actual_close = bars[-1]["close"]
        correct = (actual_close > entry) if direction == 1 else (actual_close < entry)

        # Confidence tier for reporting
        if mode == "bidir_smart":
            conf_tier = conf.tier.value
        else:
            conf_tier = "-"

        trades.append({
            "week": str(monday_ts.date()),
            "dir": "LONG" if direction == 1 else "SHORT",
            "conf": conf_tier,
            "lev": final_lev,
            "entry": entry,
            "exit": exit_p,
            "reason": reason,
            "hs_pct": stops.hard_stop_pct * 100,
            "tp_pct": stops.take_profit_pct * 100,
            "pnl": pnl * 100,
            "correct": correct,
        })

    if not trades:
        return None

    df_t = pd.DataFrame(trades)
    total_ret = (equity / 10000 - 1) * 100
    wr = np.mean(df_t["pnl"] > 0) * 100
    da = np.mean(df_t["correct"]) * 100
    weekly_rets = df_t["pnl"].values / 100
    sharpe = np.mean(weekly_rets) / np.std(weekly_rets) * np.sqrt(52) if np.std(weekly_rets) > 0 else 0

    # Max drawdown
    eq_c = [10000.0]
    for r in weekly_rets:
        eq_c.append(eq_c[-1] * (1 + r))
    peak_eq = np.maximum.accumulate(np.array(eq_c))
    dd = (np.array(eq_c) - peak_eq) / peak_eq
    max_dd = np.min(dd) * 100

    # Bootstrap p-value
    np.random.seed(42)
    boot = [np.mean(np.random.choice(weekly_rets, size=len(weekly_rets), replace=True)) for _ in range(10000)]
    p_val = np.mean(np.array(boot) <= 0)

    # Breakdown
    n_l = len(df_t[df_t["dir"] == "LONG"])
    n_s = len(df_t[df_t["dir"] == "SHORT"])
    n_tp = len(df_t[df_t["reason"] == "take_profit"])
    n_hs = len(df_t[df_t["reason"] == "hard_stop"])
    n_we = len(df_t[df_t["reason"] == "week_end"])

    # DA by direction
    longs = df_t[df_t["dir"] == "LONG"]
    shorts = df_t[df_t["dir"] == "SHORT"]
    da_long = np.mean(longs["correct"]) * 100 if len(longs) > 0 else 0
    da_short = np.mean(shorts["correct"]) * 100 if len(shorts) > 0 else 0
    wr_long = np.mean(longs["pnl"] > 0) * 100 if len(longs) > 0 else 0
    wr_short = np.mean(shorts["pnl"] > 0) * 100 if len(shorts) > 0 else 0

    return {
        "strategy": strategy_name,
        "year": year,
        "equity": equity,
        "total_ret": total_ret,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "wr": wr,
        "da": da,
        "p_val": p_val,
        "trades": len(df_t),
        "n_long": n_l,
        "n_short": n_s,
        "da_long": da_long,
        "da_short": da_short,
        "wr_long": wr_long,
        "wr_short": wr_short,
        "n_tp": n_tp,
        "n_hs": n_hs,
        "n_we": n_we,
        "df_trades": df_t,
    }


def main():
    df, feature_cols = load_data()
    print(f"Data: {len(df)} rows, {df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()}")

    strategies = [
        ("A) SHORT-only", "short_only"),
        ("B) Bidir naive", "bidir_naive"),
        ("C) Bidir smart", "bidir_smart"),
    ]

    all_results = []

    for year in [2025, 2026]:
        print(f"\n{'=' * 80}")
        print(f"  OOS {year} -- Smart Simple v{_version} Comparative Backtest")
        print(f"  Adaptive stops: HS=clamp(vol*sqrt(5)*{STOPS_CONFIG.vol_multiplier}, {STOPS_CONFIG.hard_stop_min_pct*100:.0f}%, {STOPS_CONFIG.hard_stop_max_pct*100:.0f}%), TP=HS*{STOPS_CONFIG.tp_ratio}")
        print(f"  Costs: 0% maker (limit), 1 bps slippage (market/Friday close)")
        print(f"{'=' * 80}")

        for name, mode in strategies:
            result = run_backtest(name, mode, year, df, feature_cols)
            if result is None:
                print(f"\n  {name}: No data for {year}")
                continue

            all_results.append(result)
            r = result

            print(f"\n  {name}")
            print(f"  {'-' * 50}")
            print(f"  $10K -> ${r['equity']:,.2f}  ({r['total_ret']:+.2f}%)")
            print(f"  Trades: {r['trades']} ({r['n_long']}L / {r['n_short']}S)")
            print(f"  WR: {r['wr']:.1f}%  |  DA: {r['da']:.1f}%")
            print(f"  DA_LONG: {r['da_long']:.1f}% (N={r['n_long']})  |  DA_SHORT: {r['da_short']:.1f}% (N={r['n_short']})")
            print(f"  WR_LONG: {r['wr_long']:.1f}%  |  WR_SHORT: {r['wr_short']:.1f}%")
            print(f"  Sharpe: {r['sharpe']:.3f}  |  MaxDD: {r['max_dd']:.2f}%  |  p-value: {r['p_val']:.4f}")
            print(f"  Exits: {r['n_tp']} TP, {r['n_we']} week_end, {r['n_hs']} hard_stop")

    # Summary comparison table
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY COMPARISON")
    print(f"{'=' * 80}")
    print(f"\n  {'Strategy':<20s} {'Year':>4s} {'Return':>8s} {'Sharpe':>7s} {'MaxDD':>7s} {'DA':>6s} {'WR':>6s} {'p-val':>7s} {'Trades':>7s} {'$10K':>10s}")
    print(f"  {'-' * 90}")
    for r in all_results:
        print(f"  {r['strategy']:<20s} {r['year']:>4d} {r['total_ret']:>+7.2f}% {r['sharpe']:>7.3f} {r['max_dd']:>6.2f}% {r['da']:>5.1f}% {r['wr']:>5.1f}% {r['p_val']:>7.4f} {r['trades']:>7d} ${r['equity']:>9,.2f}")

    # Print trade-by-trade for smart strategy
    for r in all_results:
        if r["strategy"] == "C) Bidir smart":
            print(f"\n  --- C) Bidir smart trades ({r['year']}) ---")
            for _, t in r["df_trades"].iterrows():
                print(f"    {t['week']}  {t['dir']:5s}  conf={t['conf']:6s}  lev={t['lev']:.3f}  "
                      f"entry={t['entry']:.0f}  exit={t['exit']:.0f}  "
                      f"HS={t['hs_pct']:.2f}%  TP={t['tp_pct']:.2f}%  "
                      f"pnl={t['pnl']:+.3f}%  [{t['reason']}]  "
                      f"{'OK' if t['correct'] else 'WRONG'}")


if __name__ == "__main__":
    main()
