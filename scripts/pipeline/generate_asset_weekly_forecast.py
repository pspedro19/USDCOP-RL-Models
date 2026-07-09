#!/usr/bin/env python
"""
Per-asset WEEKLY INFERENCE generator (Gold, BTC) → dashboard `/forecasting` JSON.
================================================================================

WHY
---
USD/COP `/forecasting` is a 9-model ML model-zoo (`generate_weekly_forecasts.py`). Gold and BTC are
**rule-based daily science stacks** (`src/gold_rl`, `src/btc_strategy`) — there is NO ML forecast to
show, and fabricating one would be dishonest. Their real "weekly inference" is the strategy's own
**causal weekly positioning**: for each ISO week, which direction/exposure the strategy holds, in which
regime, and what actually happened (strategy-realized vs buy&hold). This is a VIEW over the published
backtest — not a competing number and not a new model.

METHODOLOGY (all pairs; mirrors USD/COP smart_simple)
-----------------------------------------------------
    Train window : inception → Dec-2024   (rules are fixed constants; no fitting)
    Backtest     : 2025  (OOS — the DEFAULT view + metrics)
    Production   : 2026  (YTD)
Legitimate OOS because every feature is CAUSAL (rolling windows + `shift(1)`) and the rules are fixed —
a 2025 week only ever sees data ≤ that week, so computing features on the full series and slicing 2025
is identical to "trained up to that point" (no look-ahead). See `.claude/specs/assets/_strategy-science.md`
§6.1 and `.claude/specs/platform/dashboard-integration.md`.

HOW (reuse the real stack — do NOT reinvent signal logic)
---------------------------------------------------------
    build_daily_features → classify_regime(dwell) → build_positions(direction_fn) → compute_returns
    then group the causal daily `position`/`regime`/`strat_ret` series by time.dt.isocalendar() → ISO weeks.
Per week we emit:
    direction            sign of `position` at week start (LONG/SHORT/FLAT; Gold/BTC are long-only → LONG/FLAT)
    exposure / _raw      avg |position| over the week, normalized to the per-asset cap (0..1) + the raw value
    regime               regime label at week start (from classify_regime's hysteresis classifier)
    confidence           conviction proxy = fraction-of-week-in-market × mean regime_risk_mult
    expected_return_pct  TRANSPARENT EDGE PROXY = position_start × trailing-20d drift × 5  (NOT an ML prediction)
    realized_return_pct  strategy realized that week = prod(1+strat_ret)-1     (cost+swap net, from compute_returns)
    buyhold_return_pct   asset realized that week    = close_last/close_first-1
    hit                  was the week's positioning directionally right (long into up / flat into down)
Per-strategy `summary`: weeks_total/in_market/flat, hit_rate_pct, ytd_strategy_return_pct,
ytd_buyhold_return_pct, avg_exposure. All 3 strategies per asset are emitted (`is_primary` = PROMOTE one).

RESULTS SNAPSHOT (generated 2026-07; `--year all`) — the edge is RISK CONTROL, not raw upside
----------------------------------------------------------------------------------------------
    Gold  buy&hold 2025 +65.3% · 2026-YTD −3.7%
      gold_long_only_b1     2025 +50.8% (hit 63.5%, 52/52 wks) | 2026 −2.1%
      gold_regime_gated_v1  2025 +46.9% (hit 55.8%, 47/52)     | 2026 −0.5%
      gold_trend_b2 ★       2025 +26.9% (hit 46.2%, 26/52)     | 2026 +8.1%  (only one positive in the pullback)
    BTC   buy&hold 2025 −5.2% · 2026-YTD −27.1%
      btc_hodl_b1           2025 +5.4% (beats b&h in a down year) | 2026 −2.6%
      btc_trend_b2 ★        2025 −0.9% (in-market only 12/52)     | 2026 −2.6%
      btc_exposure_s3       2025 −2.3% (41/52)                    | 2026  0.0%  (regime gate SAT OUT the −27% crash)

OUTPUT (mirrors the per-asset `/analysis` namespace; consumed by WeeklyInferenceView.tsx)
-----------------------------------------------------------------------------------------
    usdcop-trading-dashboard/public/forecasting/<asset_id>/weekly_inference_<year>.json
    usdcop-trading-dashboard/public/forecasting/<asset_id>/index.json
Atomic writes; JSON-safe (Inf/NaN → null). Wired as the `l5_weekly_forecast` stage of the
`asset_<id>_pipeline_weekly` DAG (config/assets/pipelines.yaml).

Usage:
    python -m scripts.pipeline.generate_asset_weekly_forecast                     # all assets, all years
    python -m scripts.pipeline.generate_asset_weekly_forecast --asset btcusdt --year 2026
"""
from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = PROJECT_ROOT / "usdcop-trading-dashboard" / "public" / "forecasting"

# Per-asset science-stack wiring (SSOT for what differs; the logic is identical in shape).
ASSETS: dict[str, dict] = {
    "xauusd": {
        "pkg": "src.gold_rl", "display_name": "Oro (Gold)", "symbol": "XAU/USD",
        "chart_symbol": "XAUUSD", "asset_class": "commodity",
        "seed": "seeds/latest/xauusd_daily_ohlcv.parquet",
        "dwell": 4, "warmup": 252, "target_vol": 0.10, "cap": 1.5, "size_kw": "max_leverage",
        "dir_col": "direction", "primary": "gold_trend_b2",
    },
    "btcusdt": {
        "pkg": "src.btc_strategy", "display_name": "Bitcoin", "symbol": "BTC/USDT",
        "chart_symbol": "BTCUSDT", "asset_class": "crypto",
        "seed": "seeds/latest/btcusdt_daily_ohlcv.parquet",
        "dwell": 5, "warmup": 250, "target_vol": 0.30, "cap": 1.0, "size_kw": "max_exposure",
        "dir_col": "intent", "primary": "btc_trend_b2",
    },
}


def _safe(x, digits: int = 4):
    """JSON-safe float: NaN/Inf → None (never emit invalid JSON)."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return round(f, digits)


def _build_daily(cfg: dict) -> pd.DataFrame:
    """Run the real science stack once → causal daily features + regime (no per-strategy bits yet)."""
    ind = importlib.import_module(f"{cfg['pkg']}.indicators")
    seed = PROJECT_ROOT / cfg["seed"]
    if not seed.exists():
        raise FileNotFoundError(f"Missing daily OHLCV seed: {seed}")
    df = pd.read_parquet(seed).sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    feat = ind.build_daily_features(df)
    feat = ind.classify_regime(feat, dwell=cfg["dwell"])
    feat = feat.iloc[cfg["warmup"]:].reset_index(drop=True)  # match pipeline warmup drop
    return feat


def _weekly_records(daily: pd.DataFrame, cfg: dict, year: int) -> list[dict]:
    """Bucket the causal daily position/regime/returns into ISO weeks for one calendar year."""
    dir_col = cfg["dir_col"]
    cap = cfg["cap"]
    d = daily.copy()
    # trailing 20d mean daily return → transparent forward EDGE PROXY (not an ML forecast)
    d["drift20"] = d["ret"].rolling(20, min_periods=5).mean()
    iso = d["time"].dt.isocalendar()
    d["iso_year"] = iso["year"].astype(int)
    d["iso_week"] = iso["week"].astype(int)

    rows: list[dict] = []
    grp = d[d["iso_year"] == year].groupby(["iso_year", "iso_week"], sort=True)
    for (iy, iw), wk in grp:
        wk = wk.sort_values("time")
        first, last = wk.iloc[0], wk.iloc[-1]
        pos_start = float(first["position"])
        exposure = float(wk["position"].abs().mean())
        if pos_start > 1e-6:
            direction = "LONG"
        elif pos_start < -1e-6:
            direction = "SHORT"
        else:
            direction = "FLAT"
        # conviction proxy: fraction of the week in-market × regime favourability
        in_market = float((wk[dir_col].abs() > 1e-6).mean())
        regime_fav = float(wk.get("regime_risk_mult", pd.Series([1.0])).mean())
        confidence = max(0.0, min(1.0, in_market * regime_fav))
        strat_week = float((1.0 + wk["strat_ret"]).prod() - 1.0)
        bh_week = float(last["close"] / first["close"] - 1.0)
        expected = pos_start * float(first.get("drift20", 0.0) or 0.0) * 5.0  # ~1 trading week
        # was the positioning "right"? (long into up / flat into down / short into down)
        if direction == "LONG":
            hit = bh_week > 0
        elif direction == "SHORT":
            hit = bh_week < 0
        else:
            hit = bh_week <= 0

        rows.append({
            "iso_week": f"{iy}-W{iw:02d}",
            "week_start": first["time"].date().isoformat(),
            "week_end": last["time"].date().isoformat(),
            "direction": direction,
            "exposure": _safe(min(exposure / cap, 1.0), 3),        # normalized 0..1 for the bar
            "exposure_raw": _safe(exposure, 3),                     # true position magnitude
            "regime": str(first["regime"]),
            "confidence": _safe(confidence, 3),
            "expected_return_pct": _safe(expected * 100.0, 2),      # rule-based edge proxy
            "realized_return_pct": _safe(strat_week * 100.0, 2),    # strategy realized that week
            "buyhold_return_pct": _safe(bh_week * 100.0, 2),        # asset realized that week
            "entry_price": _safe(float(first["close"]), 2),
            "close_price": _safe(float(last["close"]), 2),
            "hit": bool(hit),
        })
    return rows


def _summary(rows: list[dict], daily: pd.DataFrame, year: int) -> dict:
    dy = daily[daily["time"].dt.isocalendar().year == year]
    strat_ytd = float((1.0 + dy["strat_ret"]).prod() - 1.0) if len(dy) else 0.0
    bh_ytd = float(dy["close"].iloc[-1] / dy["close"].iloc[0] - 1.0) if len(dy) else 0.0
    in_market = [r for r in rows if (r["exposure_raw"] or 0) > 0.01]
    hits = [r for r in rows if r["hit"]]
    return {
        "weeks_total": len(rows),
        "weeks_in_market": len(in_market),
        "weeks_flat": len(rows) - len(in_market),
        "hit_rate_pct": _safe(100.0 * len(hits) / len(rows), 1) if rows else None,
        "ytd_strategy_return_pct": _safe(strat_ytd * 100.0, 2),
        "ytd_buyhold_return_pct": _safe(bh_ytd * 100.0, 2),
        "avg_exposure": _safe(np.mean([r["exposure_raw"] or 0 for r in rows]), 3) if rows else None,
    }


def _atomic_write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def generate_asset(asset_id: str, years: list[int]) -> None:
    cfg = ASSETS[asset_id]
    strat_mod = importlib.import_module(f"{cfg['pkg']}.strategies")
    bt_mod = importlib.import_module(f"{cfg['pkg']}.backtest")
    daily_feat = _build_daily(cfg)

    # Precompute the causal position/returns frame per strategy (cheap, rule-based).
    strat_frames: dict[str, pd.DataFrame] = {}
    for sid, (name, fn, ptype) in strat_mod.STRATEGIES.items():
        pos = strat_mod.build_positions(daily_feat, fn, target_vol=cfg["target_vol"],
                                        **{cfg["size_kw"]: cfg["cap"]})
        strat_frames[sid] = bt_mod.compute_returns(pos)

    for year in years:
        strategies_out = []
        for sid, (name, fn, ptype) in strat_mod.STRATEGIES.items():
            d = strat_frames[sid]
            rows = _weekly_records(d, cfg, year)
            if not rows:
                continue
            strategies_out.append({
                "strategy_id": sid,
                "strategy_name": name,
                "strategy_type": ptype,
                "is_primary": sid == cfg["primary"],
                "weeks": rows,
                "summary": _summary(rows, d, year),
            })
        if not strategies_out:
            print(f"[weekly_forecast] {asset_id} {year}: no weeks, skipped")
            continue
        payload = {
            "asset_id": asset_id,
            "display_name": cfg["display_name"],
            "symbol": cfg["symbol"],
            "chart_symbol": cfg["chart_symbol"],
            "asset_class": cfg["asset_class"],
            "year": year,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "kind": "rule_based_weekly_inference",
            "strategies": strategies_out,
        }
        out = OUT_ROOT / asset_id / f"weekly_inference_{year}.json"
        _atomic_write(out, payload)
        n = strategies_out[0]["summary"]["weeks_total"]
        print(f"[weekly_forecast] {asset_id} {year}: {len(strategies_out)} strategies × {n} weeks "
              f"-> {out.relative_to(PROJECT_ROOT)}")

    # index.json: what the frontend uses to populate year/strategy selectors
    written_years = [y for y in years if (OUT_ROOT / asset_id / f"weekly_inference_{y}.json").exists()]
    idx = {
        "asset_id": asset_id,
        "display_name": cfg["display_name"],
        "chart_symbol": cfg["chart_symbol"],
        "years": sorted(written_years, reverse=True),
        "primary_strategy_id": cfg["primary"],
        "strategies": [{"strategy_id": sid, "strategy_name": nm, "strategy_type": pt}
                       for sid, (nm, fn, pt) in strat_mod.STRATEGIES.items()],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write(OUT_ROOT / asset_id / "index.json", idx)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", choices=[*sorted(ASSETS), "all"], default="all")
    ap.add_argument("--year", default="all", help="2025 | 2026 | all (default all)")
    args = ap.parse_args()

    assets = sorted(ASSETS) if args.asset == "all" else [args.asset]
    years = [2025, 2026] if args.year == "all" else [int(args.year)]
    for a in assets:
        generate_asset(a, years)


if __name__ == "__main__":
    main()
