"""
Train Smart Simple v1.1 and export to Dashboard format.
=======================================================

Two-phase approach matching the approval workflow:

  Phase 1 (Dashboard/Backtest):
    Walk-forward weekly retraining on 2020-2024, OOS 2025.
    Methodology IDENTICAL to backtest_smart_simple_v1.py (validated +20.03%).
    These are the approval metrics the user sees.

  Phase 2 (Production):
    After approval, retrain on 2020-2025 (since 2025 passed backtest).
    That model generates 2026 trades for the Production page.
    Monthly retraining from March.

Usage:
    python scripts/train_and_export_smart_simple.py                  # Both phases
    python scripts/train_and_export_smart_simple.py --phase backtest  # 2025 OOS only
    python scripts/train_and_export_smart_simple.py --phase production # 2026 only
    python scripts/train_and_export_smart_simple.py --reset-approval  # Reset to PENDING
    python scripts/train_and_export_smart_simple.py --no-png          # Skip PNG generation
    python scripts/train_and_export_smart_simple.py --seed-db         # Seed H5 DB tables

Output files (dashboard format):
    public/data/production/summary_2025.json
    public/data/production/summary.json
    public/data/production/trades/smart_simple_v11_2025.json
    public/data/production/trades/smart_simple_v11.json
    public/data/production/approval_state.json
"""

import sys
import os
import json
import math
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.confidence_scorer import ConfidenceConfig, score_confidence
from src.forecasting.adaptive_stops import (
    AdaptiveStopsConfig, compute_adaptive_stops,
    check_hard_stop, check_take_profit, get_exit_price,
)
from src.contracts.strategy_schema import safe_json_dump
from src.forecasting.ssot_config import ForecastingSSOTConfig
from src.forecasting.dataset_loader import ForecastingDatasetLoader

DASHBOARD_DIR = PROJECT_ROOT / "usdcop-trading-dashboard" / "public" / "data" / "production"
TRADES_DIR = DASHBOARD_DIR / "trades"
COT = timezone(timedelta(hours=-5))


# ---------------------------------------------------------------------------
# Config & Data
# ---------------------------------------------------------------------------

def load_config():
    cfg_path = PROJECT_ROOT / "config" / "execution" / "smart_simple_v1.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    _as = cfg.get("adaptive_stops", {})
    stops_config = AdaptiveStopsConfig(
        vol_multiplier=_as.get("vol_multiplier", 2.0),
        hard_stop_min_pct=_as.get("hard_stop_min_pct", 0.01),
        hard_stop_max_pct=_as.get("hard_stop_max_pct", 0.03),
        tp_ratio=_as.get("tp_ratio", 0.5),
    )

    _cc = cfg.get("confidence", {})
    conf_config = ConfidenceConfig(
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

    _vt = cfg.get("vol_targeting", {})
    _costs = cfg.get("execution", {}).get("costs", {})

    return {
        "raw": cfg,
        "stops": stops_config,
        "conf": conf_config,
        "vt_tv": _vt.get("target_vol", 0.15),
        "vt_max": _vt.get("max_leverage", 2.0),
        "vt_min": _vt.get("min_leverage", 0.5),
        "vt_floor": _vt.get("vol_floor", 0.05),
        "maker_fee": _costs.get("maker_fee_bps", 0.0) / 10000.0,
        "slippage": _costs.get("slippage_bps", 1.0) / 10000.0,
        "version": cfg.get("executor", {}).get("version", "1.1.0"),
    }


def load_data():
    """Load OHLCV + macro, build 21 features, compute H=5 target.

    Uses shared ForecastingDatasetLoader (DB-first with parquet fallback).
    """
    cfg = ForecastingSSOTConfig.load()
    loader = ForecastingDatasetLoader(cfg, project_root=PROJECT_ROOT)
    df, feature_cols = loader.load_dataset(target_horizon=5)
    return df, feature_cols


# ---------------------------------------------------------------------------
# Simulation helpers (identical to backtest_smart_simple_v1.py)
# ---------------------------------------------------------------------------

def simulate_week(direction, entry, bars, hard_stop_pct, take_profit_pct):
    """Simulate a 5-day hold with TP/HS/Friday-close. Bars = Tue-Fri."""
    for i, bar in enumerate(bars):
        h, l, c = bar["high"], bar["low"], bar["close"]
        if check_hard_stop(direction, entry, h, l, hard_stop_pct):
            ep = get_exit_price(direction, entry, "hard_stop", hard_stop_pct, take_profit_pct, c)
            return ep, "hard_stop", i
        if check_take_profit(direction, entry, h, l, take_profit_pct):
            ep = get_exit_price(direction, entry, "take_profit", hard_stop_pct, take_profit_pct, c)
            return ep, "take_profit", i
    return bars[-1]["close"], "week_end", len(bars) - 1


def compute_pnl(direction, entry, exit_price, leverage, exit_reason, maker_fee, slippage):
    """Compute PnL with costs — identical to backtest_smart_simple_v1.py."""
    raw_pnl = direction * (exit_price - entry) / entry * leverage
    entry_cost = maker_fee * leverage       # Limit order (0% on MEXC)
    if exit_reason == "week_end":
        exit_cost = slippage * leverage     # Market order, slippage only
    else:
        exit_cost = maker_fee * leverage    # Limit order (TP/HS)
    return raw_pnl - entry_cost - exit_cost


# ---------------------------------------------------------------------------
# Phase 1: Walk-forward backtest (Dashboard — OOS 2025)
# ---------------------------------------------------------------------------

def run_walkforward_backtest(df, feature_cols, cfg, year):
    """
    Walk-forward weekly retraining backtest.
    Methodology IDENTICAL to backtest_smart_simple_v1.py 'bidir_smart' mode.
    Each Monday: retrain on all data up to prior day, predict, trade.
    """
    test_start = pd.Timestamp(f"{year}-01-01")
    test_end = pd.Timestamp(f"{year}-12-31")
    test_data = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    test_data["dow"] = test_data["date"].dt.dayofweek
    mondays = test_data[test_data["dow"] == 0]["date"].unique()

    if not len(mondays):
        return {"trades": [], "equity": 10000.0, "metrics": {}}

    equity = 10000.0
    trades = []
    trade_id = 0

    for monday in mondays:
        monday_ts = pd.Timestamp(monday)
        friday_ts = monday_ts + pd.offsets.BDay(4)

        # Train on ALL data before Monday (expanding window)
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

        # Predict on latest training row (Friday before Monday)
        latest = scaler.transform(df_train[feature_cols].iloc[-1:].values.astype(np.float64))
        pred_ridge = float(ridge.predict(latest)[0])
        pred_br = float(br.predict(latest)[0])
        ensemble = (pred_ridge + pred_br) / 2.0
        direction = 1 if ensemble > 0 else -1

        # Confidence scoring
        conf = score_confidence(pred_ridge, pred_br, direction, cfg["conf"])
        if conf.skip_trade:
            continue

        lev_mult = conf.sizing_multiplier

        # Vol-targeting from training set
        rets = df_train["return_1d"].dropna().values
        rv_daily = np.std(rets[-21:]) if len(rets) >= 21 else 0.0
        rv_ann = rv_daily * np.sqrt(252) if rv_daily > 0 else cfg["vt_tv"]
        safe_vol = max(rv_ann, cfg["vt_floor"])
        base_lev = np.clip(cfg["vt_tv"] / safe_vol, cfg["vt_min"], cfg["vt_max"])

        # Apply confidence multiplier (no extra asymmetric layer)
        final_lev = np.clip(base_lev * lev_mult, cfg["vt_min"], cfg["vt_max"])

        # Adaptive stops
        stops = compute_adaptive_stops(rv_ann, cfg["stops"])

        # Entry = Monday close
        monday_row = df[df["date"] == monday_ts]
        if monday_row.empty:
            m2 = df["date"] >= monday_ts
            if m2.any():
                monday_row = df[m2].iloc[:1]
            else:
                continue
        entry = float(monday_row["close"].iloc[0])
        entry_date = pd.Timestamp(monday_row["date"].iloc[0])

        # Week bars = Tue-Fri (NOT Monday)
        week_dates = pd.bdate_range(monday_ts + timedelta(days=1), friday_ts)
        bars = []
        last_bar_date = entry_date
        for day in week_dates:
            r = df[df["date"] == day]
            if not r.empty:
                bars.append({
                    "high": float(r["high"].iloc[0]),
                    "low": float(r["low"].iloc[0]),
                    "close": float(r["close"].iloc[0]),
                })
                last_bar_date = pd.Timestamp(r["date"].iloc[0])
        if not bars:
            continue

        # Simulate
        exit_p, reason, exit_bar_idx = simulate_week(
            direction, entry, bars,
            stops.hard_stop_pct, stops.take_profit_pct,
        )

        pnl = compute_pnl(direction, entry, exit_p, final_lev, reason,
                           cfg["maker_fee"], cfg["slippage"])

        equity_at_entry = equity
        equity *= (1 + pnl)

        trade_id += 1

        # Compute exit date from bars
        exit_dates = [d for d in week_dates if not df[df["date"] == d].empty]
        exit_date = exit_dates[min(exit_bar_idx, len(exit_dates) - 1)] if exit_dates else last_bar_date

        trades.append({
            "trade_id": trade_id,
            "timestamp": datetime(entry_date.year, entry_date.month, entry_date.day,
                                  9, 0, 0, tzinfo=COT).isoformat(),
            "exit_timestamp": datetime(exit_date.year, exit_date.month, exit_date.day,
                                       12, 50, 0, tzinfo=COT).isoformat(),
            "side": "SHORT" if direction == -1 else "LONG",
            "entry_price": round(entry, 2),
            "exit_price": round(exit_p, 2),
            "pnl_usd": round(equity - equity_at_entry, 2),
            "pnl_pct": round(pnl * 100, 4),
            "exit_reason": reason,
            "equity_at_entry": round(equity_at_entry, 2),
            "equity_at_exit": round(equity, 2),
            "leverage": round(float(final_lev), 3),
            "confidence_tier": conf.tier.value,
            "hard_stop_pct": round(stops.hard_stop_pct * 100, 2),
            "take_profit_pct": round(stops.take_profit_pct * 100, 2),
        })

    return _compute_result_metrics(trades, equity, df, year)


# ---------------------------------------------------------------------------
# Phase 2: Production backtest (2026 with monthly retraining)
# ---------------------------------------------------------------------------

def run_production_backtest(df, feature_cols, cfg, year):
    """
    Production model: retrained MONTHLY, weekly forecasts.
    - Jan model: trained on 2020 -> Dec 31 prev year
    - Feb model: trained on 2020 -> Jan 31
    - Mar model: trained on 2020 -> Feb 28, etc.
    Same entry/exit/cost methodology as walk-forward.
    """
    test_start = pd.Timestamp(f"{year}-01-01")
    test_end = pd.Timestamp(f"{year}-12-31")
    test_data = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    test_data["dow"] = test_data["date"].dt.dayofweek
    mondays = sorted(test_data[test_data["dow"] == 0]["date"].unique())

    if not len(mondays):
        return {"trades": [], "equity": 10000.0, "metrics": {}, "skipped_weeks": []}

    # Group Mondays by month for monthly retraining
    mondays_by_month = {}
    for m in mondays:
        m_ts = pd.Timestamp(m)
        key = m_ts.month
        mondays_by_month.setdefault(key, []).append(m_ts)

    equity = 10000.0
    trades = []
    week_data = []  # Per-week intermediate data for DB seeding
    skipped_weeks = []
    trade_id = 0

    for month_num in sorted(mondays_by_month.keys()):
        # Monthly retraining: train on all data through end of previous month
        if month_num == 1:
            train_end = pd.Timestamp(f"{year - 1}-12-31")
        else:
            # End of previous month
            first_of_month = pd.Timestamp(f"{year}-{month_num:02d}-01")
            train_end = first_of_month - timedelta(days=1)

        df_train = df[(df["date"] <= train_end) & df["target_return_5d"].notna()].copy()
        mask = df_train[feature_cols].notna().all(axis=1)
        df_train = df_train[mask]

        if len(df_train) < 50:
            print(f"    Month {month_num}: insufficient training data ({len(df_train)} rows), skipping")
            continue

        X = df_train[feature_cols].values.astype(np.float64)
        y = df_train["target_return_5d"].values.astype(np.float64)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        ridge = Ridge(alpha=1.0).fit(Xs, y)
        br = BayesianRidge(max_iter=300).fit(Xs, y)

        month_name = pd.Timestamp(f"{year}-{month_num:02d}-01").strftime("%B")
        print(f"    {month_name} model: {len(df_train)} samples "
              f"(2020 -> {train_end.date()}), {len(mondays_by_month[month_num])} weeks")

        for monday_ts in mondays_by_month[month_num]:
            friday_ts = monday_ts + pd.offsets.BDay(4)

            # Predict using latest available features before Monday
            prev_data = df[(df["date"] < monday_ts)]
            prev_data = prev_data[prev_data[feature_cols].notna().all(axis=1)]
            if prev_data.empty:
                skipped_weeks.append({
                    "monday": monday_ts.strftime("%Y-%m-%d"),
                    "reason": "no_features",
                })
                continue

            latest = scaler.transform(prev_data[feature_cols].iloc[-1:].values.astype(np.float64))
            pred_ridge = float(ridge.predict(latest)[0])
            pred_br = float(br.predict(latest)[0])
            ensemble = (pred_ridge + pred_br) / 2.0
            direction = 1 if ensemble > 0 else -1

            # Confidence scoring
            conf = score_confidence(pred_ridge, pred_br, direction, cfg["conf"])
            if conf.skip_trade:
                skipped_weeks.append({
                    "monday": monday_ts.strftime("%Y-%m-%d"),
                    "reason": f"low_confidence_{('LONG' if direction == 1 else 'SHORT')}",
                    "prediction": round(ensemble * 100, 4),
                    "confidence": conf.tier,
                })
                continue

            lev_mult = conf.sizing_multiplier

        # Vol-targeting from data up to Monday
        rets = prev_data["return_1d"].dropna().values
        rv_daily = np.std(rets[-21:]) if len(rets) >= 21 else 0.0
        rv_ann = rv_daily * np.sqrt(252) if rv_daily > 0 else cfg["vt_tv"]
        safe_vol = max(rv_ann, cfg["vt_floor"])
        base_lev = np.clip(cfg["vt_tv"] / safe_vol, cfg["vt_min"], cfg["vt_max"])

        final_lev = np.clip(base_lev * lev_mult, cfg["vt_min"], cfg["vt_max"])
        stops = compute_adaptive_stops(rv_ann, cfg["stops"])

        # Entry = Monday close
        monday_row = df[df["date"] == monday_ts]
        if monday_row.empty:
            m2 = df["date"] >= monday_ts
            if m2.any():
                monday_row = df[m2].iloc[:1]
            else:
                continue
        entry = float(monday_row["close"].iloc[0])
        entry_date = pd.Timestamp(monday_row["date"].iloc[0])

        # Week bars = Tue-Fri
        week_dates = pd.bdate_range(monday_ts + timedelta(days=1), friday_ts)
        bars = []
        last_bar_date = entry_date
        for day in week_dates:
            r = df[df["date"] == day]
            if not r.empty:
                bars.append({
                    "high": float(r["high"].iloc[0]),
                    "low": float(r["low"].iloc[0]),
                    "close": float(r["close"].iloc[0]),
                })
                last_bar_date = pd.Timestamp(r["date"].iloc[0])
        if not bars:
            continue

        exit_p, reason, exit_bar_idx = simulate_week(
            direction, entry, bars,
            stops.hard_stop_pct, stops.take_profit_pct,
        )

        pnl = compute_pnl(direction, entry, exit_p, final_lev, reason,
                           cfg["maker_fee"], cfg["slippage"])

        equity_at_entry = equity
        equity *= (1 + pnl)

        trade_id += 1

        exit_dates = [d for d in week_dates if not df[df["date"] == d].empty]
        exit_date = exit_dates[min(exit_bar_idx, len(exit_dates) - 1)] if exit_dates else last_bar_date

        trade_dict = {
            "trade_id": trade_id,
            "timestamp": datetime(entry_date.year, entry_date.month, entry_date.day,
                                  9, 0, 0, tzinfo=COT).isoformat(),
            "exit_timestamp": datetime(exit_date.year, exit_date.month, exit_date.day,
                                       12, 50, 0, tzinfo=COT).isoformat(),
            "side": "SHORT" if direction == -1 else "LONG",
            "entry_price": round(entry, 2),
            "exit_price": round(exit_p, 2),
            "pnl_usd": round(equity - equity_at_entry, 2),
            "pnl_pct": round(pnl * 100, 4),
            "exit_reason": reason,
            "equity_at_entry": round(equity_at_entry, 2),
            "equity_at_exit": round(equity, 2),
            "leverage": round(float(final_lev), 3),
            "confidence_tier": conf.tier.value,
            "hard_stop_pct": round(stops.hard_stop_pct * 100, 2),
            "take_profit_pct": round(stops.take_profit_pct * 100, 2),
        }
        trades.append(trade_dict)

        # Collect per-week intermediate data for DB seeding
        week_data.append({
            "signal_date": monday_ts.date() if hasattr(monday_ts, 'date') else monday_ts,
            "direction": direction,
            "pred_ridge": pred_ridge,
            "pred_br": pred_br,
            "ensemble": ensemble,
            "rv_ann": float(rv_ann),
            "base_lev": float(base_lev),
            "final_lev": float(final_lev),
            "confidence_tier": conf.tier.value,
            "sizing_mult": float(lev_mult),
            "hard_stop_pct": stops.hard_stop_pct,
            "take_profit_pct": stops.take_profit_pct,
            "entry_price": round(entry, 2),
            "trade": trade_dict,
        })

    result = _compute_result_metrics(trades, equity, df, year)
    result["skipped_weeks"] = skipped_weeks
    result["week_data"] = week_data
    return result


# ---------------------------------------------------------------------------
# Shared metrics
# ---------------------------------------------------------------------------

def _compute_result_metrics(trades, equity, df, year):
    """Compute standard metrics from trade list."""
    # Buy and hold
    year_prices = df[df["date"].dt.year == year]["close"]
    bh_return = 0.0
    if len(year_prices) > 1:
        bh_return = (year_prices.iloc[-1] / year_prices.iloc[0] - 1) * 100

    n_trades = len(trades)
    if n_trades == 0:
        return {"trades": [], "equity": 10000.0, "metrics": {}}

    winners = [t for t in trades if t["pnl_usd"] > 0]
    losers = [t for t in trades if t["pnl_usd"] <= 0]
    total_return = (equity / 10000.0 - 1) * 100

    pnl_arr = np.array([t["pnl_pct"] for t in trades])
    weekly_rets = pnl_arr / 100.0

    sharpe = 0.0
    if len(weekly_rets) > 1 and np.std(weekly_rets) > 0:
        sharpe = float(np.mean(weekly_rets) / np.std(weekly_rets) * np.sqrt(52))

    gross_profit = sum(t["pnl_usd"] for t in winners)
    gross_loss = abs(sum(t["pnl_usd"] for t in losers))
    pf = gross_profit / gross_loss if gross_loss > 0 else None

    da = len(winners) / n_trades * 100

    # Max drawdown
    eq_c = [10000.0]
    for r in weekly_rets:
        eq_c.append(eq_c[-1] * (1 + r))
    peak_eq = np.maximum.accumulate(np.array(eq_c))
    dd = (np.array(eq_c) - peak_eq) / peak_eq
    max_dd = abs(float(np.min(dd)) * 100)

    # Bootstrap p-value
    np.random.seed(42)
    boot = [np.mean(np.random.choice(weekly_rets, size=len(weekly_rets), replace=True))
            for _ in range(10000)]
    p_value = float(np.mean(np.array(boot) <= 0))

    # Exit reason counts
    exit_counts = {}
    for t in trades:
        r = t["exit_reason"]
        exit_counts[r] = exit_counts.get(r, 0) + 1

    # Direction breakdown
    longs = [t for t in trades if t["side"] == "LONG"]
    shorts = [t for t in trades if t["side"] == "SHORT"]

    metrics = {
        "final_equity": round(equity, 2),
        "total_return_pct": round(total_return, 2),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_dd, 2),
        "win_rate_pct": round(len(winners) / n_trades * 100, 1),
        "profit_factor": round(pf, 3) if pf is not None else None,
        "direction_accuracy_pct": round(da, 1),
        "n_trades": n_trades,
        "n_winners": len(winners),
        "n_losers": len(losers),
        "n_long": len(longs),
        "n_short": len(shorts),
        "p_value": round(p_value, 4),
        "bh_return_pct": round(bh_return, 2),
        "exit_reasons": exit_counts,
    }

    return {"trades": trades, "equity": equity, "metrics": metrics}


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_summary(result, year, cfg):
    """Export summary.json in dashboard format."""
    m = result["metrics"]
    if not m:
        return {}

    summary = {
        "generated_at": datetime.now().isoformat(),
        "strategy_name": f"Smart Simple v{cfg['version']}",
        "strategy_id": "smart_simple_v11",
        "year": year,
        "initial_capital": 10000.0,
        "n_trading_days": m["n_trades"] * 5,
        "direction_accuracy_pct": m["direction_accuracy_pct"],
        "strategies": {
            "buy_and_hold": {
                "final_equity": round(10000 * (1 + m["bh_return_pct"] / 100), 2),
                "total_return_pct": m["bh_return_pct"],
            },
            "smart_simple_v11": {
                "final_equity": m["final_equity"],
                "total_return_pct": m["total_return_pct"],
                "sharpe": m["sharpe"],
                "max_dd_pct": m["max_dd_pct"],
                "win_rate_pct": m["win_rate_pct"],
                "profit_factor": m["profit_factor"],
                "trading_days": m["n_trades"] * 5,
                "exit_reasons": m["exit_reasons"],
                "n_long": m["n_long"],
                "n_short": m["n_short"],
            },
        },
        "statistical_tests": {
            "p_value": m["p_value"],
            "significant": bool(m["p_value"] < 0.05),
        },
    }

    # Monthly breakdown
    trades = result["trades"]
    if trades:
        trade_df = pd.DataFrame(trades)
        trade_df["month"] = pd.to_datetime(
            trade_df["timestamp"], utc=True
        ).dt.tz_convert(None).dt.to_period("M").astype(str)
        monthly = trade_df.groupby("month").agg(
            n_trades=("pnl_pct", "count"),
            total_pnl=("pnl_pct", "sum"),
        ).reset_index()
        summary["monthly"] = {
            "months": monthly["month"].tolist(),
            "trades": monthly["n_trades"].tolist(),
            "pnl_pct": [round(x, 2) for x in monthly["total_pnl"].tolist()],
        }

    return summary


def export_trades(result, year, cfg):
    """Export trades JSON in dashboard format."""
    trades = result["trades"]
    m = result["metrics"]
    if not m:
        return {}

    return {
        "strategy_name": f"Smart Simple v{cfg['version']}",
        "strategy_id": "smart_simple_v11",
        "initial_capital": 10000.0,
        "date_range": {
            "start": f"{year}-01-01",
            "end": trades[-1]["exit_timestamp"][:10] if trades else f"{year}-12-31",
        },
        "trades": trades,
        "summary": {
            "total_trades": m["n_trades"],
            "winning_trades": m["n_winners"],
            "losing_trades": m["n_losers"],
            "win_rate": m["win_rate_pct"],
            "total_pnl": round(m["final_equity"] - 10000, 2),
            "total_return_pct": m["total_return_pct"],
            "max_drawdown_pct": m["max_dd_pct"],
            "sharpe_ratio": m["sharpe"],
            "direction_accuracy_pct": m["direction_accuracy_pct"],
            "profit_factor": m["profit_factor"],
            "p_value": m["p_value"],
            "n_long": m["n_long"],
            "n_short": m["n_short"],
        },
    }


def export_approval_state(result_2025, cfg):
    """Export approval_state.json for 2-vote system."""
    m = result_2025["metrics"]
    if not m:
        return {}

    gates = [
        {"gate": "min_return_pct", "label": "Retorno Minimo",
         "passed": bool(m["total_return_pct"] > -15),
         "value": float(m["total_return_pct"]), "threshold": -15.0},
        {"gate": "min_sharpe_ratio", "label": "Sharpe Minimo",
         "passed": bool(m["sharpe"] > 0),
         "value": float(m["sharpe"]), "threshold": 0.0},
        {"gate": "max_drawdown_pct", "label": "Max Drawdown",
         "passed": bool(m["max_dd_pct"] < 20),
         "value": float(m["max_dd_pct"]), "threshold": 20.0},
        {"gate": "min_trades", "label": "Trades Minimos",
         "passed": bool(m["n_trades"] >= 10),
         "value": int(m["n_trades"]), "threshold": 10},
        {"gate": "statistical_significance", "label": "Significancia (p<0.05)",
         "passed": bool(m["p_value"] < 0.05),
         "value": float(m["p_value"]), "threshold": 0.05},
    ]
    n_passed = sum(1 for g in gates if g["passed"])

    return {
        "status": "PENDING_APPROVAL",
        "strategy": "smart_simple_v11",
        "strategy_name": f"Smart Simple v{cfg['version']}",
        "backtest_year": 2025,
        "backtest_recommendation": "PROMOTE" if n_passed == len(gates) else "REVIEW",
        "backtest_confidence": round(n_passed / len(gates), 2),
        "gates": gates,
        "backtest_metrics": {
            "return_pct": float(m["total_return_pct"]),
            "sharpe": float(m["sharpe"]),
            "max_dd_pct": float(m["max_dd_pct"]),
            "p_value": float(m["p_value"]),
            "trades": int(m["n_trades"]),
            "win_rate_pct": float(m["win_rate_pct"]),
        },
        "deploy_manifest": {
            "pipeline_type": "ml_forecasting",
            "script": "scripts/train_and_export_smart_simple.py",
            "args": ["--phase", "production", "--no-png", "--seed-db"],
            "config_path": "config/execution/smart_simple_v1.yaml",
            "db_tables": [
                "forecast_h5_predictions",
                "forecast_h5_signals",
                "forecast_h5_executions",
                "forecast_h5_subtrades",
                "forecast_h5_paper_trading",
            ],
        },
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# DB seeding (--seed-db flag, used by deploy API)
# ---------------------------------------------------------------------------

def seed_h5_db_tables(result_2026, cfg):
    """
    Seed the 5 H5 DB tables with production trades via UPSERT.

    Tables seeded:
      forecast_h5_predictions   — Ridge + BR predictions per week
      forecast_h5_signals       — Ensemble signal + confidence + stops
      forecast_h5_executions    — Weekly execution (entry/exit/PnL)
      forecast_h5_subtrades     — Single subtrade per execution
      forecast_h5_paper_trading — Running metrics for monitoring

    Uses DATABASE_URL env var for connection. All INSERTs use ON CONFLICT
    for idempotent re-runs and safe coexistence with Airflow L7 DAG.

    Current week (signal_date >= this Monday) is seeded as status='positioned'
    with no exit data so the L7 DAG picks it up for NRT monitoring.
    """
    try:
        import psycopg2
    except ImportError:
        print("    [WARN] psycopg2 not installed, skipping DB seeding")
        return

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("    [WARN] DATABASE_URL not set, skipping DB seeding")
        return

    trades = result_2026.get("trades", [])
    week_data = result_2026.get("week_data", [])
    if not trades and not week_data:
        print("    [WARN] No trades or week data to seed")
        return

    # Determine current Monday for "active position" detection
    today = datetime.now().date()
    days_since_monday = today.weekday()
    current_monday = today - timedelta(days=days_since_monday)

    conn = psycopg2.connect(db_url)
    try:
        cur = conn.cursor()

        # Running metrics accumulators
        cum_pnl = 0.0
        eq_curve = [10000.0]
        peak_eq = 10000.0
        max_dd = 0.0
        n_weeks_total = 0
        n_long_total = 0
        n_short_total = 0
        n_correct = 0
        n_correct_short = 0
        n_short_total_for_da = 0
        consec_losses = 0
        weekly_rets = []

        for wd in week_data:
            signal_date = wd["signal_date"]  # Monday as date
            is_current_week = signal_date >= current_monday

            inference_date = signal_date - timedelta(days=1)  # Sunday
            iso_cal = signal_date.isocalendar()
            iso_week = iso_cal[1]
            iso_year = iso_cal[0]

            direction = wd["direction"]
            dir_str = "SHORT" if direction == -1 else "LONG"

            # 1. forecast_h5_predictions (2 rows: ridge + br)
            for model_id, pred_val in [("ridge", wd["pred_ridge"]), ("bayesian_ridge", wd["pred_br"])]:
                pred_dir = "UP" if pred_val > 0 else "DOWN"
                base_price = wd["entry_price"]
                pred_price = base_price * math.exp(pred_val)
                cur.execute("""
                    INSERT INTO forecast_h5_predictions
                    (inference_date, inference_week, inference_year, target_date,
                     model_id, horizon_id, base_price, predicted_price,
                     predicted_return_pct, direction)
                    VALUES (%s, %s, %s, %s, %s, 5, %s, %s, %s, %s)
                    ON CONFLICT (inference_date, model_id, horizon_id) DO UPDATE SET
                        predicted_price = EXCLUDED.predicted_price,
                        predicted_return_pct = EXCLUDED.predicted_return_pct,
                        direction = EXCLUDED.direction,
                        base_price = EXCLUDED.base_price
                """, (inference_date, iso_week, iso_year,
                      signal_date + timedelta(days=4),
                      model_id, base_price, round(pred_price, 2),
                      round(pred_val * 100, 4), pred_dir))

            # 2. forecast_h5_signals
            cur.execute("""
                INSERT INTO forecast_h5_signals
                (signal_date, inference_date, inference_week, inference_year,
                 ensemble_return, direction, realized_vol_21d,
                 raw_leverage, clipped_leverage, adjusted_leverage,
                 confidence_tier, sizing_multiplier, skip_trade,
                 hard_stop_pct, take_profit_pct, config_version)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        'smart_simple_v1')
                ON CONFLICT (signal_date) DO UPDATE SET
                    ensemble_return = EXCLUDED.ensemble_return,
                    direction = EXCLUDED.direction,
                    realized_vol_21d = EXCLUDED.realized_vol_21d,
                    adjusted_leverage = EXCLUDED.adjusted_leverage,
                    confidence_tier = EXCLUDED.confidence_tier,
                    sizing_multiplier = EXCLUDED.sizing_multiplier,
                    hard_stop_pct = EXCLUDED.hard_stop_pct,
                    take_profit_pct = EXCLUDED.take_profit_pct
            """, (signal_date, inference_date, iso_week, iso_year,
                  wd["ensemble"], direction, wd["rv_ann"],
                  wd["base_lev"], wd["final_lev"], wd["final_lev"],
                  wd["confidence_tier"], wd["sizing_mult"],
                  False, wd["hard_stop_pct"], wd["take_profit_pct"]))

            # 3. forecast_h5_executions
            trade = wd.get("trade")
            if is_current_week and trade:
                # Current week: seed as 'positioned' for L7 to pick up
                cur.execute("""
                    INSERT INTO forecast_h5_executions
                    (signal_date, inference_week, inference_year, direction,
                     leverage, entry_price, entry_timestamp, status,
                     config_version, confidence_tier, hard_stop_pct, take_profit_pct)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'positioned',
                            'smart_simple_v1', %s, %s, %s)
                    ON CONFLICT (signal_date) DO UPDATE SET
                        entry_price = EXCLUDED.entry_price,
                        entry_timestamp = EXCLUDED.entry_timestamp,
                        status = EXCLUDED.status,
                        confidence_tier = EXCLUDED.confidence_tier,
                        hard_stop_pct = EXCLUDED.hard_stop_pct,
                        take_profit_pct = EXCLUDED.take_profit_pct,
                        updated_at = NOW()
                    RETURNING id
                """, (signal_date, iso_week, iso_year, direction,
                      wd["final_lev"], trade["entry_price"],
                      trade["timestamp"],
                      wd["confidence_tier"], wd["hard_stop_pct"],
                      wd["take_profit_pct"]))
            elif trade:
                # Completed week
                cur.execute("""
                    INSERT INTO forecast_h5_executions
                    (signal_date, inference_week, inference_year, direction,
                     leverage, entry_price, entry_timestamp,
                     exit_price, exit_timestamp, exit_reason,
                     week_pnl_pct, n_subtrades, status,
                     config_version, confidence_tier, hard_stop_pct, take_profit_pct)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 1, 'closed',
                            'smart_simple_v1', %s, %s, %s)
                    ON CONFLICT (signal_date) DO UPDATE SET
                        entry_price = EXCLUDED.entry_price,
                        exit_price = EXCLUDED.exit_price,
                        exit_timestamp = EXCLUDED.exit_timestamp,
                        exit_reason = EXCLUDED.exit_reason,
                        week_pnl_pct = EXCLUDED.week_pnl_pct,
                        status = EXCLUDED.status,
                        confidence_tier = EXCLUDED.confidence_tier,
                        hard_stop_pct = EXCLUDED.hard_stop_pct,
                        take_profit_pct = EXCLUDED.take_profit_pct,
                        updated_at = NOW()
                    RETURNING id
                """, (signal_date, iso_week, iso_year, direction,
                      wd["final_lev"], trade["entry_price"],
                      trade["timestamp"],
                      trade["exit_price"], trade["exit_timestamp"],
                      trade["exit_reason"], trade["pnl_pct"] / 100.0,
                      wd["confidence_tier"], wd["hard_stop_pct"],
                      wd["take_profit_pct"]))
            else:
                # Skipped week — no execution record
                continue

            exec_row = cur.fetchone()
            if not exec_row:
                continue
            exec_id = exec_row[0]

            # 4. forecast_h5_subtrades
            if is_current_week:
                cur.execute("""
                    INSERT INTO forecast_h5_subtrades
                    (execution_id, subtrade_index, direction, entry_price,
                     entry_timestamp, peak_price, trailing_state)
                    VALUES (%s, 0, %s, %s, %s, %s, 'active')
                    ON CONFLICT (execution_id, subtrade_index) DO UPDATE SET
                        entry_price = EXCLUDED.entry_price,
                        entry_timestamp = EXCLUDED.entry_timestamp,
                        updated_at = NOW()
                """, (exec_id, direction, trade["entry_price"],
                      trade["timestamp"], trade["entry_price"]))
            elif trade:
                raw_pnl = direction * (trade["exit_price"] - trade["entry_price"]) / trade["entry_price"]
                lev_pnl = trade["pnl_pct"] / 100.0
                cur.execute("""
                    INSERT INTO forecast_h5_subtrades
                    (execution_id, subtrade_index, direction, entry_price,
                     entry_timestamp, exit_price, exit_timestamp,
                     exit_reason, trailing_state,
                     pnl_pct, pnl_unleveraged_pct)
                    VALUES (%s, 0, %s, %s, %s, %s, %s, %s, 'triggered', %s, %s)
                    ON CONFLICT (execution_id, subtrade_index) DO UPDATE SET
                        exit_price = EXCLUDED.exit_price,
                        exit_timestamp = EXCLUDED.exit_timestamp,
                        exit_reason = EXCLUDED.exit_reason,
                        pnl_pct = EXCLUDED.pnl_pct,
                        pnl_unleveraged_pct = EXCLUDED.pnl_unleveraged_pct,
                        updated_at = NOW()
                """, (exec_id, direction, trade["entry_price"],
                      trade["timestamp"], trade["exit_price"],
                      trade["exit_timestamp"], trade["exit_reason"],
                      lev_pnl, raw_pnl))

            # 5. forecast_h5_paper_trading — progressive metrics
            if trade and not is_current_week:
                pnl_pct = trade["pnl_pct"] / 100.0
                cum_pnl += pnl_pct
                eq_curve.append(eq_curve[-1] * (1 + pnl_pct))
                peak_eq = max(peak_eq, eq_curve[-1])
                dd = (eq_curve[-1] - peak_eq) / peak_eq * 100
                max_dd = min(max_dd, dd)
                weekly_rets.append(pnl_pct)

                n_weeks_total += 1
                if direction == -1:
                    n_short_total += 1
                    n_short_total_for_da += 1
                    if pnl_pct > 0:
                        n_correct_short += 1
                else:
                    n_long_total += 1
                if pnl_pct > 0:
                    n_correct += 1
                    consec_losses = 0
                else:
                    consec_losses += 1

                running_da = n_correct / n_weeks_total * 100 if n_weeks_total > 0 else None
                running_da_short = (n_correct_short / n_short_total_for_da * 100
                                    if n_short_total_for_da > 0 else None)
                running_sharpe = None
                if len(weekly_rets) > 1 and np.std(weekly_rets) > 0:
                    running_sharpe = float(np.mean(weekly_rets) / np.std(weekly_rets) * np.sqrt(52))

                cb_triggered = consec_losses >= 5 or abs(max_dd) >= 12.0

                cur.execute("""
                    INSERT INTO forecast_h5_paper_trading
                    (signal_date, inference_week, inference_year, direction,
                     leverage, week_pnl_pct, n_subtrades, cumulative_pnl_pct,
                     running_da_pct, running_da_short_pct, running_sharpe,
                     running_max_dd_pct, n_weeks, n_long, n_short,
                     consecutive_losses, circuit_breaker)
                    VALUES (%s, %s, %s, %s, %s, %s, 1, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (signal_date) DO UPDATE SET
                        week_pnl_pct = EXCLUDED.week_pnl_pct,
                        cumulative_pnl_pct = EXCLUDED.cumulative_pnl_pct,
                        running_da_pct = EXCLUDED.running_da_pct,
                        running_da_short_pct = EXCLUDED.running_da_short_pct,
                        running_sharpe = EXCLUDED.running_sharpe,
                        running_max_dd_pct = EXCLUDED.running_max_dd_pct,
                        n_weeks = EXCLUDED.n_weeks,
                        consecutive_losses = EXCLUDED.consecutive_losses,
                        circuit_breaker = EXCLUDED.circuit_breaker
                """, (signal_date, iso_week, iso_year, direction,
                      wd["final_lev"], pnl_pct, cum_pnl * 100,
                      running_da, running_da_short, running_sharpe,
                      max_dd, n_weeks_total, n_long_total, n_short_total,
                      consec_losses, cb_triggered))

        # Ensure subtrade unique index exists for UPSERT
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
                uq_h5_subtrades_exec_idx ON forecast_h5_subtrades(execution_id, subtrade_index)
        """)

        conn.commit()
        n_seeded = len([wd for wd in week_data if wd.get("trade")])
        print(f"    [DB] Seeded {n_seeded} weeks into 5 H5 tables")

    except Exception as e:
        conn.rollback()
        print(f"    [ERROR] DB seeding failed: {e}")
        logging.exception("DB seeding error")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# PNG generation (optional, matplotlib)
# ---------------------------------------------------------------------------

def generate_pngs(result, df, year, output_dir):
    """Generate equity curve, monthly PnL, and trade distribution PNGs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("    [WARN] matplotlib not available, skipping PNGs")
        return

    trades = result["trades"]
    m = result["metrics"]
    if not trades or not m:
        return

    # Dark theme
    plt.style.use("dark_background")
    BG = "#0f172a"
    GRID = "#1e293b"
    GREEN = "#10b981"
    RED = "#ef4444"
    GRAY = "#64748b"
    WHITE = "#e2e8f0"

    # --- 1. Equity Curve ---
    try:
        weekly_rets = np.array([t["pnl_pct"] for t in trades]) / 100.0
        eq = [10000.0]
        for r in weekly_rets:
            eq.append(eq[-1] * (1 + r))
        dates = [pd.Timestamp(trades[0]["timestamp"]).tz_localize(None)]
        for t in trades:
            dates.append(pd.Timestamp(t["exit_timestamp"]).tz_localize(None))

        # Buy and hold
        year_data = df[(df["date"].dt.year == year) & df["close"].notna()]
        bh_dates = year_data["date"].values
        bh_eq = 10000.0 * year_data["close"].values / year_data["close"].iloc[0]

        fig, ax = plt.subplots(figsize=(12, 5), facecolor=BG)
        ax.set_facecolor(BG)
        ax.plot(dates, eq, color=GREEN, linewidth=2, label=f"Smart Simple v1.1")
        if len(bh_dates) > 0:
            ax.plot(bh_dates, bh_eq, color=GRAY, linewidth=1, linestyle="--", label="Buy & Hold", alpha=0.7)
        ax.axhline(y=10000, color=GRAY, linewidth=0.5, linestyle=":")
        ax.set_title(f"Equity Curve {year}", color=WHITE, fontsize=14, fontweight="bold")
        ax.set_ylabel("Equity ($)", color=WHITE)
        ax.legend(facecolor=BG, edgecolor=GRID, fontsize=9)
        ax.grid(True, color=GRID, alpha=0.3)
        ax.tick_params(colors=GRAY)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        fig.tight_layout()
        fig.savefig(output_dir / f"equity_curve_{year}.png", dpi=120, facecolor=BG)
        plt.close(fig)
        print(f"    -> equity_curve_{year}.png")
    except Exception as e:
        print(f"    [WARN] equity_curve PNG failed: {e}")

    # --- 2. Monthly PnL ---
    try:
        trade_df = pd.DataFrame(trades)
        trade_df["month"] = pd.to_datetime(
            trade_df["timestamp"], utc=True
        ).dt.tz_convert(None).dt.to_period("M").astype(str)
        monthly = trade_df.groupby("month")["pnl_pct"].sum().reset_index()

        fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
        ax.set_facecolor(BG)
        colors = [GREEN if v >= 0 else RED for v in monthly["pnl_pct"]]
        ax.bar(monthly["month"], monthly["pnl_pct"], color=colors, alpha=0.85, width=0.6)
        ax.axhline(y=0, color=GRAY, linewidth=0.5)
        ax.set_title(f"Monthly PnL {year} (%)", color=WHITE, fontsize=14, fontweight="bold")
        ax.set_ylabel("PnL %", color=WHITE)
        ax.grid(True, axis="y", color=GRID, alpha=0.3)
        ax.tick_params(colors=GRAY)
        plt.xticks(rotation=45)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        fig.tight_layout()
        fig.savefig(output_dir / f"monthly_pnl_{year}.png", dpi=120, facecolor=BG)
        plt.close(fig)
        print(f"    -> monthly_pnl_{year}.png")
    except Exception as e:
        print(f"    [WARN] monthly_pnl PNG failed: {e}")

    # --- 3. Trade Distribution ---
    try:
        pnl_pcts = [t["pnl_pct"] for t in trades]
        fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
        ax.set_facecolor(BG)
        n, bins, patches = ax.hist(pnl_pcts, bins=15, alpha=0.85, edgecolor=GRID)
        for patch, left_edge in zip(patches, bins):
            patch.set_facecolor(GREEN if left_edge >= 0 else RED)
        ax.axvline(x=0, color=GRAY, linewidth=1, linestyle="--")
        ax.set_title(f"Trade PnL Distribution {year}", color=WHITE, fontsize=14, fontweight="bold")
        ax.set_xlabel("PnL %", color=WHITE)
        ax.set_ylabel("Count", color=WHITE)
        ax.grid(True, axis="y", color=GRID, alpha=0.3)
        ax.tick_params(colors=GRAY)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        fig.tight_layout()
        fig.savefig(output_dir / f"trade_distribution_{year}.png", dpi=120, facecolor=BG)
        plt.close(fig)
        print(f"    -> trade_distribution_{year}.png")
    except Exception as e:
        print(f"    [WARN] trade_distribution PNG failed: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart Simple v1.1 — Train & Export to Dashboard"
    )
    parser.add_argument(
        "--phase", choices=["backtest", "production", "both"], default="both",
        help="Which phases to run: backtest (2025 OOS), production (2026), or both"
    )
    parser.add_argument(
        "--reset-approval", action="store_true",
        help="Reset approval_state.json to PENDING_APPROVAL"
    )
    parser.add_argument(
        "--no-png", action="store_true",
        help="Skip PNG generation (faster, JSON only)"
    )
    parser.add_argument(
        "--seed-db", action="store_true",
        help="Seed H5 DB tables with production trades (used by deploy API)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Handle --reset-approval
    if args.reset_approval:
        approval_path = DASHBOARD_DIR / "approval_state.json"
        if approval_path.exists():
            with open(approval_path) as f:
                state = json.load(f)
            state["status"] = "PENDING_APPROVAL"
            state["approved_by"] = None
            state["approved_at"] = None
            state["reviewer_notes"] = None
            state["rejected_by"] = None
            state["rejected_at"] = None
            state["rejection_reason"] = None
            state["last_updated"] = datetime.now().isoformat()
            with open(approval_path, "w") as f:
                safe_json_dump(state, f)
            print(f"Approval reset to PENDING_APPROVAL: {approval_path}")
        else:
            print(f"No approval_state.json found at {approval_path}")
        return

    print("=" * 70)
    print("  Smart Simple v1.1 -- Train & Export to Dashboard")
    print(f"  Phase: {args.phase}")
    print("=" * 70)

    cfg = load_config()
    print(f"\n[1] Config: Smart Simple v{cfg['version']}")
    print(f"    Stops: HS=clamp(vol*sqrt(5)*{cfg['stops'].vol_multiplier}, "
          f"{cfg['stops'].hard_stop_min_pct*100:.0f}%, {cfg['stops'].hard_stop_max_pct*100:.0f}%), "
          f"TP=HS*{cfg['stops'].tp_ratio}")

    df, feature_cols = load_data()
    print(f"[2] Data: {len(df)} rows, {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    TRADES_DIR.mkdir(parents=True, exist_ok=True)

    result_2025 = None
    result_2026 = None
    m25 = {}
    m26 = {}

    # -----------------------------------------------------------------------
    # PHASE 1: Walk-forward backtest 2025 (Dashboard — approval metrics)
    # -----------------------------------------------------------------------
    if args.phase in ("backtest", "both"):
        print(f"\n[3] PHASE 1: Walk-forward backtest 2025 OOS (weekly retraining)...")
        result_2025 = run_walkforward_backtest(df, feature_cols, cfg, 2025)
        m25 = result_2025["metrics"]
        if m25:
            print(f"    $10K -> ${m25['final_equity']:,.2f} ({m25['total_return_pct']:+.2f}%)")
            print(f"    Sharpe: {m25['sharpe']:.3f} | WR: {m25['win_rate_pct']:.1f}% | MaxDD: -{m25['max_dd_pct']:.2f}%")
            pf_str = f"{m25['profit_factor']:.3f}" if m25['profit_factor'] is not None else "N/A"
            print(f"    p-value: {m25['p_value']:.4f} | Trades: {m25['n_trades']} ({m25['n_long']}L/{m25['n_short']}S) | PF: {pf_str}")
            print(f"    Exits: {m25['exit_reasons']}")

        # Export backtest files
        print(f"\n    Exporting backtest files...")
        summary_2025 = export_summary(result_2025, 2025, cfg)
        with open(DASHBOARD_DIR / "summary_2025.json", "w") as f:
            safe_json_dump(summary_2025, f)
        print(f"    -> summary_2025.json (OOS walk-forward)")

        trades_2025 = export_trades(result_2025, 2025, cfg)
        with open(TRADES_DIR / "smart_simple_v11_2025.json", "w") as f:
            safe_json_dump(trades_2025, f)
        print(f"    -> trades/smart_simple_v11_2025.json ({m25.get('n_trades', 0)} trades)")

        approval = export_approval_state(result_2025, cfg)
        with open(DASHBOARD_DIR / "approval_state.json", "w") as f:
            safe_json_dump(approval, f)
        print(f"    -> approval_state.json (recommendation: {approval.get('backtest_recommendation', 'N/A')})")

        if not args.no_png:
            generate_pngs(result_2025, df, 2025, DASHBOARD_DIR)

    # -----------------------------------------------------------------------
    # PHASE 2: Production (monthly retraining, weekly forecasts, 2026)
    # -----------------------------------------------------------------------
    if args.phase in ("production", "both"):
        print(f"\n[4] PHASE 2: Production model (monthly retraining, weekly forecasts, 2026)...")
        result_2026 = run_production_backtest(df, feature_cols, cfg, 2026)
        m26 = result_2026["metrics"]
        if m26:
            print(f"    $10K -> ${m26['final_equity']:,.2f} ({m26['total_return_pct']:+.2f}%)")
            print(f"    Sharpe: {m26['sharpe']:.3f} | Trades: {m26['n_trades']} ({m26.get('n_long', 0)}L/{m26.get('n_short', 0)}S)")
            print(f"    Exits: {m26['exit_reasons']}")
        else:
            print(f"    No trades in 2026 YTD")
        # Report skipped weeks
        skipped = result_2026.get("skipped_weeks", [])
        if skipped:
            print(f"    Skipped weeks ({len(skipped)}):")
            for sw in skipped:
                print(f"      {sw['monday']}: {sw['reason']}"
                      f"{' (pred=' + str(sw.get('prediction', '')) + '%)' if 'prediction' in sw else ''}")

        # Export production files
        print(f"\n    Exporting production files...")
        summary_2026 = export_summary(result_2026, 2026, cfg)
        with open(DASHBOARD_DIR / "summary.json", "w") as f:
            safe_json_dump(summary_2026, f)
        print(f"    -> summary.json (production 2026)")

        trades_2026 = export_trades(result_2026, 2026, cfg)
        with open(TRADES_DIR / "smart_simple_v11.json", "w") as f:
            safe_json_dump(trades_2026, f)
        print(f"    -> trades/smart_simple_v11.json ({m26.get('n_trades', 0)} trades)")

        if not args.no_png:
            generate_pngs(result_2026, df, 2026, DASHBOARD_DIR)

        # DB seeding (--seed-db flag, typically set by deploy API)
        if args.seed_db:
            print(f"\n    Seeding DB tables...")
            print(f"    [seeding_db] Starting H5 table seeding...")
            seed_h5_db_tables(result_2026, cfg)
            print(f"    [seeding_db] Complete")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  DONE -- Smart Simple v{cfg['version']} (phase={args.phase})")
    print(f"{'=' * 70}")
    if m25:
        print(f"  Dashboard (2025 OOS walk-forward):")
        print(f"    ${m25['final_equity']:,.2f} ({m25['total_return_pct']:+.2f}%, p={m25['p_value']:.4f})")
        print(f"    {m25['n_trades']} trades ({m25['n_long']}L/{m25['n_short']}S), Sharpe {m25['sharpe']:.3f}")
    if m26:
        print(f"  Production (2026 with model trained on 2020-2025):")
        print(f"    ${m26['final_equity']:,.2f} ({m26['total_return_pct']:+.2f}%)")
        print(f"    {m26['n_trades']} trades ({m26.get('n_long', 0)}L/{m26.get('n_short', 0)}S)")
    print(f"  Output: {DASHBOARD_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
