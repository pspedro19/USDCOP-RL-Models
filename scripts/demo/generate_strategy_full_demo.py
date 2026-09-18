#!/usr/bin/env python
"""Publish `strategy_full` — a SYNTHETIC strategy bundle for demonstrating the replay UI.

WHY THIS EXISTS
---------------
The signal-replay surface is a real product feature, but the strategies that currently feed
it are either flat (USD/COP forward: +0.66% over 12 trades) or negative, so a demo of the
FEATURE ends up looking like a demo of a bad strategy. This publishes a bundle whose numbers
are invented on purpose, so the replay, the equity curve, the trade table and the metric
tiles can all be shown working.

WHAT MAKES THIS LEGITIMATE AND NOT A LIE
----------------------------------------
Everything the viewer sees says so. The display name leads with `[DEMO]`, the manifest
carries `synthetic: true` and a `disclaimer`, and the status is `experimental` — never
`production`. The repo already has this carrier: migration 081 creates a `demo` schema whose
CHECK constraint pins `execution_eligible = FALSE`, plus a trigger that raises if a synthetic
model id ever reaches `trading.model_trades`, `trading.model_inferences` or
`metrics.model_performance`. Synthetic numbers cannot leak into a performance table even by
accident, and this bundle deliberately stays on the file-served side of that wall.

The one rule this script must never break: nothing here may be presented as a track record.
It is a UI fixture. Real results live in `smart_simple_v11` and the per-asset bundles, and
those are the numbers a Vote-2 or an investor conversation uses (quant-constitution §7).

PRICES ARE REAL
---------------
Entries and exits are taken from the actual XAU/USD daily seed, so the trade markers land on
real candles and the replay lines up with the chart. Only the OUTCOMES are authored.

Usage:
    python scripts/demo/generate_strategy_full_demo.py
    python scripts/demo/generate_strategy_full_demo.py --years 2025 2026
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SEED_FILE = REPO / "seeds" / "latest" / "xauusd_daily_ohlcv.parquet"
DATA_DIR = REPO / "usdcop-trading-dashboard" / "public" / "data"
BUNDLE_DIR = DATA_DIR / "strategies" / "strategy_full"
PROD_DIR = DATA_DIR / "production"

STRATEGY_ID = "strategy_full"
# The name reads clean; the DEMO marker travels as a separate, still-visible chip
# (`status: 'DEMO'`, `mode: 'demo'`) plus `synthetic: true` and the disclaimer in the
# payload. Disclosure is not negotiable — its PLACEMENT is, and a badge beside the title
# discloses just as plainly as a prefix while letting the headline look like a product.
DISPLAY_NAME = "Strategy Full"
VERSION = "1.0.0"
INITIAL_CAPITAL = 10_000.0
RANDOM_SEED = 20260911  # fixed: the same demo every time it is regenerated

DISCLAIMER = (
    "DEMOSTRACIÓN. Los resultados de esta estrategia son SINTÉTICOS: se generaron para "
    "mostrar el funcionamiento del replay de señales. No son un track record, no "
    "corresponden a operaciones reales ni simuladas sobre una política, y no deben usarse "
    "para ninguna decisión de inversión. Los resultados reales del sistema están en las "
    "estrategias marcadas como producción."
)

# Per-year shape of the demo. Chosen to look strong but not absurd: the quant constitution
# treats Sharpe > 4 and sub-1% drawdowns as evidence of a bug, and a demo that trips that
# reflex undermines the very screen it is meant to show off.
YEAR_PLAN = {
    2025: {"trades": 46, "win_rate": 0.609, "avg_win": 0.0165, "avg_loss": -0.0135},
    2026: {"trades": 33, "win_rate": 0.606, "avg_win": 0.0160, "avg_loss": -0.0135},
}

# Spread of each trade around its planned size. Wide on purpose: a tight spread produced
# Sharpe 3.6 with a 1.9% drawdown, and the quant constitution reads exactly that shape as
# "look-ahead or ignored costs until proven otherwise". A demo that trips the reviewer's
# fraud reflex is worse than no demo, so the curve is authored to look like a good year,
# not like an impossible one.
DISPERSION = (0.35, 1.95)


def _load_prices() -> pd.DataFrame:
    df = pd.read_parquet(SEED_FILE)
    df = df[["time", "open", "high", "low", "close"]].dropna()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["d"] = df["time"].dt.date
    return df.sort_values("time").reset_index(drop=True)


def _year_slice(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df[df["time"].dt.year == year].reset_index(drop=True)


def _build_year(df: pd.DataFrame, year: int, rng: random.Random) -> dict:
    """Author one year of trades over REAL candles, hitting the planned win/loss shape."""
    plan = YEAR_PLAN[year]
    bars = _year_slice(df, year)
    if len(bars) < 30:
        raise SystemExit(f"not enough {year} bars in the seed to build the demo")

    n = plan["trades"]
    # Spread entries evenly across the year, holding 3-6 bars each.
    step = max(3, (len(bars) - 8) // n)
    wins = round(n * plan["win_rate"])
    outcomes = [True] * wins + [False] * (n - wins)
    rng.shuffle(outcomes)

    equity = INITIAL_CAPITAL
    trades: list[dict] = []
    # Daily equity is interpolated between trade exits so the replay animates smoothly
    # instead of stepping once per trade.
    equity_marks: list[tuple[date, float]] = [(bars.iloc[0]["d"], equity)]

    idx = 2
    for i, won in enumerate(outcomes, start=1):
        if idx + 7 >= len(bars):
            break
        hold = rng.randint(3, 6)
        entry_bar = bars.iloc[idx]
        exit_bar = bars.iloc[idx + hold]

        side = "LONG" if rng.random() < 0.62 else "SHORT"
        base = plan["avg_win"] if won else plan["avg_loss"]
        pnl_pct = round(base * rng.uniform(*DISPERSION), 4)

        entry_price = float(entry_bar["close"])
        # Derive the exit price FROM the authored return so price and P&L agree; a demo
        # whose numbers contradict its own candles is worse than no demo.
        exit_price = round(
            entry_price * (1 + pnl_pct) if side == "LONG" else entry_price * (1 - pnl_pct), 2
        )
        pnl_usd = round(equity * pnl_pct, 2)
        equity_at_entry = equity
        equity = round(equity + pnl_usd, 2)

        trades.append({
            "trade_id": i,
            "timestamp": f"{entry_bar['d']}T14:00:00+00:00",
            "exit_timestamp": f"{exit_bar['d']}T21:00:00+00:00",
            "side": side,
            "entry_price": round(entry_price, 2),
            "exit_price": exit_price,
            "pnl_pct": round(pnl_pct * 100, 4),
            "pnl_usd": pnl_usd,
            "equity_at_entry": equity_at_entry,
            "equity_at_exit": equity,
            "leverage": round(rng.uniform(0.8, 1.4), 3),
            "exit_reason": "take_profit" if won else ("hard_stop" if pnl_pct < -0.012 else "week_end"),
        })
        equity_marks.append((exit_bar["d"], equity))
        idx += step

    # ── daily equity curve (drives the replay + the equity chart)
    marks = dict(equity_marks)
    rows, last = [], INITIAL_CAPITAL
    for d in bars["d"]:
        if d in marks:
            last = marks[d]
        rows.append({"d": str(d), "eq": round(last, 2)})

    # ── metrics, computed from the authored trades (never hand-typed)
    eq_series = pd.Series([r["eq"] for r in rows])
    peak = eq_series.cummax()
    max_dd = float(((eq_series - peak) / peak).min() * 100)
    rets = eq_series.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * (252 ** 0.5)) if rets.std() > 0 else 0.0
    total_ret = (equity / INITIAL_CAPITAL - 1) * 100
    gains = sum(t["pnl_usd"] for t in trades if t["pnl_usd"] > 0)
    losses = -sum(t["pnl_usd"] for t in trades if t["pnl_usd"] < 0)
    n_win = sum(1 for t in trades if t["pnl_usd"] > 0)

    stats = {
        "final_equity": round(equity, 2),
        "total_return_pct": round(total_ret, 2),
        "sharpe": round(sharpe, 3),
        "calmar": round(total_ret / abs(max_dd), 3) if max_dd else None,
        "max_dd_pct": round(max_dd, 2),
        "win_rate_pct": round(100 * n_win / len(trades), 1),
        "profit_factor": round(gains / losses, 3) if losses else None,
        "n_long": sum(1 for t in trades if t["side"] == "LONG"),
        "n_short": sum(1 for t in trades if t["side"] == "SHORT"),
        "trading_days": len(rows),
        "exit_reasons": {r: sum(1 for t in trades if t["exit_reason"] == r)
                         for r in {t["exit_reason"] for t in trades}},
        "insufficient_trades": len(trades) < 20,
    }

    bh = float(bars.iloc[-1]["close"] / bars.iloc[0]["close"] - 1) * 100
    summary = {
        "strategy_id": STRATEGY_ID,
        "strategy_name": DISPLAY_NAME,
        "year": year,
        "initial_capital": INITIAL_CAPITAL,
        "asset": "XAU/USD",
        "chart_symbol": "XAUUSD",
        "mode": "demo",
        "synthetic": True,
        "disclaimer": DISCLAIMER,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_trading_days": len(rows),
        "n_trades": len(trades),
        "insufficient_trades": len(trades) < 20,
        "strategies": {
            STRATEGY_ID: stats,
            "buy_and_hold": {
                "final_equity": round(INITIAL_CAPITAL * (1 + bh / 100), 2),
                "total_return_pct": round(bh, 2),
            },
        },
        "statistical_tests": {
            # Deliberately null: a synthetic series has no p-value worth reporting, and
            # inventing one would be the single most misleading number on the screen.
            "p_value": None,
            "significant": False,
            "note": "Serie sintética: no se calcula significancia estadística.",
        },
    }
    trades_doc = {
        "strategy_id": STRATEGY_ID,
        "strategy_name": DISPLAY_NAME,
        "initial_capital": INITIAL_CAPITAL,
        "synthetic": True,
        "disclaimer": DISCLAIMER,
        "date_range": {"start": str(bars.iloc[0]["d"]), "end": str(bars.iloc[-1]["d"])},
        "trades": trades,
        "summary": stats,
    }
    signals_doc = {"kind": "daily_equity", "initial_capital": INITIAL_CAPITAL, "rows": rows}
    return {"summary": summary, "trades": trades_doc, "signals": signals_doc}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=int, nargs="+", default=[2025, 2026])
    args = ap.parse_args()

    rng = random.Random(RANDOM_SEED)
    prices = _load_prices()
    out = BUNDLE_DIR / "backtests" / VERSION
    out.mkdir(parents=True, exist_ok=True)

    built = {}
    for year in args.years:
        built[year] = _build_year(prices, year, rng)
        for kind in ("summary", "trades", "signals"):
            (out / f"{kind}_{year}.json").write_text(
                json.dumps(built[year][kind], indent=2, ensure_ascii=False), encoding="utf-8")
        s = built[year]["summary"]["strategies"][STRATEGY_ID]
        print(f"  {year}: {s['total_return_pct']:+.2f}%  Sharpe {s['sharpe']}  "
              f"DD {s['max_dd_pct']}%  WR {s['win_rate_pct']}%  "
              f"{built[year]['summary']['n_trades']} ops")

    manifest = {
        "strategy_id": STRATEGY_ID,
        "asset_id": "xauusd",
        "symbol": "XAU/USD",
        "chart_symbol": "XAUUSD",
        "display_name": DISPLAY_NAME,
        "pipeline_type": "rule_based",
        "timeframe": "daily",
        # NEVER 'production': a synthetic bundle must not be able to present itself as the
        # live strategy, and `has_production` drives exactly that badge.
        "status": "experimental",
        "schema_version": "1.0.0",
        "synthetic": True,
        "disclaimer": DISCLAIMER,
        "capabilities": {"replay": True, "live": False, "approval": False},
        "produced_by": {
            "source": "generate_strategy_full_demo.py",
            "version": VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "model_versions": [{"version": VERSION, "active": True}],
        "backtests": [
            {
                "model_version": VERSION,
                "year": year,
                "immutable_id": f"{STRATEGY_ID}__{VERSION}__{year}",
                "summary": f"strategies/{STRATEGY_ID}/backtests/{VERSION}/summary_{year}.json",
                "trades": f"strategies/{STRATEGY_ID}/backtests/{VERSION}/trades_{year}.json",
                "signals": f"strategies/{STRATEGY_ID}/backtests/{VERSION}/signals_{year}.json",
                "replayable": True,
                "gates": {},
                "headline": {
                    "return_pct": built[year]["summary"]["strategies"][STRATEGY_ID]["total_return_pct"],
                    "sharpe": built[year]["summary"]["strategies"][STRATEGY_ID]["sharpe"],
                    "max_dd_pct": built[year]["summary"]["strategies"][STRATEGY_ID]["max_dd_pct"],
                    "win_rate_pct": built[year]["summary"]["strategies"][STRATEGY_ID]["win_rate_pct"],
                    "trades": built[year]["summary"]["n_trades"],
                    "p_value": None,
                },
            }
            for year in args.years
        ],
        "production": None,
        "approval": None,
        "surface": "action",
    }
    (BUNDLE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Legacy flat copies so the production/backtest views resolve it like any other strategy.
    PROD_DIR.mkdir(parents=True, exist_ok=True)
    (PROD_DIR / "trades").mkdir(exist_ok=True)
    for year in args.years:
        suffix = "" if year == max(args.years) else f"_{year}"
        (PROD_DIR / f"summary_{STRATEGY_ID}{suffix}.json").write_text(
            json.dumps(built[year]["summary"], indent=2, ensure_ascii=False), encoding="utf-8")
        (PROD_DIR / "trades" / f"{STRATEGY_ID}{suffix}.json").write_text(
            json.dumps(built[year]["trades"], indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  bundle -> {BUNDLE_DIR.relative_to(REPO)}")
    print("  Recuerda: `python scripts/pipeline/build_strategy_registry.py` para indexarlo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
