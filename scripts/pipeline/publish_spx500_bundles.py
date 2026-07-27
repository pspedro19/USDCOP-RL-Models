"""Publica los bundles SPX500 con replay real 2025/2026 (plan SPX S3, 0 trials).

Publica DOS estrategias congeladas juntas (S3 del PLAN-RENTABILIDAD spx500):
- ``spx500_regime_gated_v1``  — candidata (MA200 x TSMOM 12-1 x techo de regimen)
- ``spx500_daily_ma200_v1``   — baseline operativo simple (MA200 causal, sin vol
  target ni gate), con el MISMO convenio corregido del ERRATUM 2026-07-21:
  lag exactamente 1 barra + costo |dW| x costo-unitario PROPIO.

Datos: índice S&P 500 OFICIAL de Investing (``load_real`` v2, seed 1995→,
price-return declarado; SPY retirado por directiva 2026-07-27). El motor es el
MISMO de ``profitability_evidence.py`` (adapter
``ADAPTERS['spx500']`` -> BacktestEngine); este script solo REBANA por anio y
publica via ``BundlePublisher`` (contrato StrategySummary, safe_json_dump,
inmutable por version/anio, registry refresh aditivo).

Honestidad (quant-constitution):
- N<20 segmentos de exposicion en el anio => sharpe/sortino/calmar/p-value = null
  (solo conteo y PnL).
- El veredicto full-window (DSR<0.95 => research_only) viaja en cada summary;
  ``backtest_recommendation`` nunca supera REVIEW mientras el DSR no pase el bar.
- profit_factor = null cuando no hay perdidas. Jamas Infinity/NaN.

Uso:
    python scripts/pipeline/publish_spx500_bundles.py [--version 1.0.0]
        [--years 2025 2026] [--no-publish]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PUBLIC_DATA = ROOT / "usdcop-trading-dashboard" / "public" / "data"
EVIDENCE_DIR = ROOT / ".claude" / "evidence" / "profitability"

GATED_ID = "spx500_regime_gated_v1"
MA200_ID = "spx500_daily_ma200_v1"
MIN_TRADES_FOR_STATS = 20
INITIAL_CAPITAL = 10_000.0


def _load_publisher():
    p = ROOT / "src" / "contracts" / "strategy_manifest.py"
    spec = importlib.util.spec_from_file_location("strategy_manifest", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["strategy_manifest"] = mod
    spec.loader.exec_module(mod)
    return mod


def _latest_evidence() -> dict:
    """Ultimo artefacto spx500.json de profitability_evidence (full-window)."""
    dirs = sorted(d for d in EVIDENCE_DIR.iterdir() if d.is_dir())
    for d in reversed(dirs):
        f = d / "spx500.json"
        if f.exists():
            doc = json.loads(f.read_text(encoding="utf-8"))
            doc["_evidence_dir"] = d.name
            return doc
    raise FileNotFoundError("no hay evidencia spx500.json en .claude/evidence/profitability/")


def _build_arms():
    """Sleeve real (motor identico a la evidencia) + brazo ma200 corregido."""
    from scripts.analysis.profitability_adapters import ADAPTERS
    sleeve = ADAPTERS["spx500"]()

    idx = pd.DatetimeIndex(pd.to_datetime(sleeve.index))
    gated_ret = np.asarray(sleeve.strat_ret, dtype=float)
    gated_pos = np.asarray(sleeve.position, dtype=float)
    asset_ret = np.asarray(sleeve.asset_ret, dtype=float)

    # ERRATUM 2026-07-21 (S0.3): baseline con lag 1 + costos por turnover PROPIO
    dumb_pos = np.roll(np.asarray(sleeve.dumb_position, dtype=float), 1)
    dumb_pos[0] = 0.0
    unit_cost = float(np.sum(sleeve.cost)) / max(
        float(np.sum(np.abs(np.diff(sleeve.position, prepend=0.0)))), 1e-12)
    dumb_cost = np.abs(np.diff(dumb_pos, prepend=0.0)) * unit_cost
    ma200_ret = dumb_pos * asset_ret - dumb_cost

    return {
        "index": idx,
        "asset_ret": asset_ret,
        GATED_ID: {"pos": gated_pos, "ret": gated_ret},
        MA200_ID: {"pos": dumb_pos, "ret": ma200_ret},
    }


def _load_prices() -> pd.DataFrame:
    pkg = ROOT / "src" / "strategies" / "spx500_regime_gated_v1"
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    from src.strategies.spx500_regime_gated_v1.load_real import load_real
    return load_real()


def _segments(pos: np.ndarray, idx: pd.DatetimeIndex, mask: np.ndarray):
    """Segmentos contiguos de exposicion (>1e-9) DENTRO del anio ('trades')."""
    active = (np.abs(pos) > 1e-9) & mask
    segs, start = [], None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(active) - 1))
    return segs


def _year_doc(sid: str, name: str, arms: dict, prices: pd.DataFrame,
              year: int, evidence: dict) -> tuple[dict, dict, dict, dict, dict]:
    idx: pd.DatetimeIndex = arms["index"]
    mask = np.asarray(idx.year == year)
    n_days = int(mask.sum())
    if n_days == 0:
        raise RuntimeError(f"{sid}: sin filas para {year} en el snapshot")

    ret = arms[sid]["ret"][mask]
    pos = arms[sid]["pos"]
    bh = arms["asset_ret"][mask]

    eq = INITIAL_CAPITAL * np.cumprod(1.0 + ret)
    eq_bh = INITIAL_CAPITAL * np.cumprod(1.0 + bh)
    peak = np.maximum.accumulate(eq)
    max_dd_pct = round(abs(float(np.min((eq - peak) / peak))) * 100, 2)
    total_ret_pct = round(float(eq[-1] / INITIAL_CAPITAL - 1.0) * 100, 2)
    bh_ret_pct = round(float(eq_bh[-1] / INITIAL_CAPITAL - 1.0) * 100, 2)

    # trades = segmentos de exposicion dentro del anio (estrategia de exposicion,
    # patron btc_exposure_s3). Precio de referencia = close TOTAL-RETURN del
    # snapshot (load_real no expone open crudo; el PnL viene del stream de
    # retornos open-to-open del motor, no de estos niveles).
    segs = _segments(pos, idx, np.asarray(mask))
    open_px = prices["close"].to_numpy(float)[: len(pos)]
    dates = idx
    trades, wins = [], 0
    eq_run = INITIAL_CAPITAL
    for k, (i0, i1) in enumerate(segs, start=1):
        seg_ret = arms[sid]["ret"][i0:i1 + 1]
        pnl_pct = float(np.prod(1.0 + seg_ret) - 1.0) * 100
        eq_entry = eq_run
        eq_run = eq_run * (1.0 + pnl_pct / 100)
        wins += int(pnl_pct > 0)
        trades.append({
            "trade_id": k,
            "timestamp": str(dates[i0])[:19],
            "exit_timestamp": str(dates[i1])[:19],
            "side": "LONG",
            "entry_price": round(float(open_px[i0]), 2),
            "exit_price": round(float(open_px[min(i1 + 1, len(open_px) - 1)]), 2),
            "pnl_pct": round(pnl_pct, 4),
            "pnl_usd": round(eq_run - eq_entry, 2),
            "equity_at_entry": round(eq_entry, 2),
            "equity_at_exit": round(eq_run, 2),
            "leverage": round(float(np.mean(np.abs(pos[i0:i1 + 1]))), 3),
            "exit_reason": "exposure_zero" if i1 + 1 < len(pos) and abs(pos[i1 + 1]) <= 1e-9
                           else "year_end_open",
        })
    n_trades = len(trades)
    stats_ok = n_trades >= MIN_TRADES_FOR_STATS

    # metricas anualizadas SOLO si hay N suficiente (constitucion §6)
    if stats_ok and np.std(ret) > 0:
        ann = np.sqrt(252.0)
        sharpe = round(float(np.mean(ret) / np.std(ret, ddof=1)) * ann, 3)
        downside = ret[ret < 0]
        sortino = (round(float(np.mean(ret) / np.std(downside, ddof=1)) * ann, 3)
                   if len(downside) > 1 and np.std(downside) > 0 else None)
        calmar = round((total_ret_pct / max_dd_pct), 3) if max_dd_pct > 0 else None
    else:
        sharpe = sortino = calmar = None

    losses = [t for t in trades if t["pnl_pct"] < 0]
    gains = [t for t in trades if t["pnl_pct"] > 0]
    profit_factor = (round(sum(t["pnl_pct"] for t in gains)
                           / abs(sum(t["pnl_pct"] for t in losses)), 3)
                     if losses else None)  # null, JAMAS Infinity

    dsr = ((evidence.get("deflated_sharpe") or {}).get("headline_dsr"))
    verdict = evidence.get("verdict")

    summary = {
        "strategy_id": sid,
        "strategy_name": name,
        "year": year,
        "initial_capital": INITIAL_CAPITAL,
        "asset": "SPX500",
        "n_trading_days": n_days,
        "n_trades": n_trades,
        "insufficient_trades": not stats_ok,
        "strategies": {
            sid: {
                "final_equity": round(float(eq[-1]), 2),
                "total_return_pct": total_ret_pct,
                "sharpe": sharpe,
                "sortino": sortino,
                "calmar": calmar,
                "max_dd_pct": max_dd_pct,
                "win_rate_pct": round(100.0 * wins / n_trades, 1) if n_trades else None,
                "profit_factor": profit_factor,
                "n_long": n_trades,
                "n_short": 0,
                "trading_days": n_days,
                "exit_reasons": {"exposure_zero":
                                 sum(1 for t in trades if t["exit_reason"] == "exposure_zero"),
                                 "year_end_open":
                                 sum(1 for t in trades if t["exit_reason"] == "year_end_open")},
            },
            "buy_and_hold": {
                "final_equity": round(float(eq_bh[-1]), 2),
                "total_return_pct": bh_ret_pct,
            },
        },
        "statistical_tests": {
            "p_value": None,
            "significant": False,
            "insufficient_trades": not stats_ok,
            "min_trades_for_stats": MIN_TRADES_FOR_STATS,
            "note": ("N<20 segmentos => solo conteo y PnL (quant-constitution §6)"
                     if not stats_ok else "stats sobre retornos diarios del anio"),
        },
        "backtest_recommendation": "REVIEW",
        "backtest_confidence": None,
        "gates": [
            {"gate": "min_return_pct", "label": "Retorno minimo", "value": total_ret_pct,
             "threshold": -15.0, "passed": bool(total_ret_pct > -15.0)},
            {"gate": "max_drawdown", "label": "MaxDD < 20%", "value": max_dd_pct,
             "threshold": 20.0, "passed": bool(max_dd_pct < 20.0)},
            {"gate": "min_trades", "label": "Trades >= 10", "value": n_trades,
             "threshold": 10, "passed": bool(n_trades >= 10)},
            {"gate": "dsr_trial_aware", "label": "DSR > 0.95 (full-window)",
             "value": dsr, "threshold": 0.95,
             "passed": bool(dsr is not None and dsr > 0.95)},
        ],
        "honest_gate": {
            "scope": "full-window (evidencia profitability, no el anio suelto)",
            "evidence_dir": evidence.get("_evidence_dir"),
            "verdict": verdict,
            "deflated_sharpe": dsr,
            "failed_gates": evidence.get("failed_gates"),
            "window": evidence.get("window"),
        },
        "data_source": {
            "series": "indice S&P 500 OFICIAL (Investing id 166), 1995-> via seed/DAG",
            "price_convention": "PRICE-RETURN declarado (sin dividendos, plan §1)",
            "nota": ("directiva operador 2026-07-27: SPY retirado; una sola serie por "
                     "corrida; research_only hasta DSR>0.95 + forward"),
        },
    }

    trades_doc = {
        "strategy_id": sid, "strategy_name": name, "year": year,
        "initial_capital": INITIAL_CAPITAL,
        "date_range": {"start": str(dates[np.asarray(mask)][0])[:10],
                       "end": str(dates[np.asarray(mask)][-1])[:10]},
        "trades": trades,
        "summary": {"n_trades": n_trades,
                    "total_return_pct": total_ret_pct,
                    "note_n": ("N<20 => solo conteo y PnL" if not stats_ok else None)},
    }

    signals = {"kind": "daily_equity", "initial_capital": INITIAL_CAPITAL,
               "rows": [{"d": str(d)[:10], "eq": round(float(e), 2)}
                        for d, e in zip(dates[np.asarray(mask)], eq)]}

    gates_meta = {"passed": sum(1 for g in summary["gates"] if g["passed"]),
                  "of": len(summary["gates"]), "recommendation": "REVIEW"}
    headline = {"total_return_pct": total_ret_pct, "max_dd_pct": max_dd_pct,
                "n_trades": n_trades, "year": year}
    return summary, trades_doc, signals, gates_meta, headline


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # v2.0.0 (2026-07-27): cambio de AUTORIDAD DE DATOS por directiva del operador —
    # snapshot SPY (Yahoo TR) RETIRADO; serie = indice S&P 500 OFICIAL de Investing
    # 1995->, price-return declarado. v1.0.0 (SPY, ventana 2020->) queda inmutable
    # como contexto historico.
    ap.add_argument("--version", default="2.0.0")
    ap.add_argument("--years", type=int, nargs="+", default=[2025, 2026])
    ap.add_argument("--no-publish", action="store_true")
    a = ap.parse_args(argv)

    evidence = _latest_evidence()
    print(f"[evidence] {evidence['_evidence_dir']} verdict={evidence.get('verdict')} "
          f"dsr={(evidence.get('deflated_sharpe') or {}).get('headline_dsr')}")

    arms = _build_arms()
    prices = _load_prices()
    print(f"[engine] {len(arms['index'])} filas "
          f"{arms['index'][0].date()} -> {arms['index'][-1].date()} (motor de la evidencia)")

    strategies = [
        (GATED_ID, "S&P 500 · Trend + régimen"),
        (MA200_ID, "S&P 500 · MA200 causal (baseline)"),
    ]

    sm = _load_publisher()
    pub = sm.BundlePublisher(PUBLIC_DATA,
                             generated_at=str(pd.Timestamp.utcnow().isoformat()))
    for sid, name in strategies:
        for year in a.years:
            summary, trades_doc, signals, gates_meta, headline = _year_doc(
                sid, name, arms, prices, year, evidence)
            s = summary["strategies"][sid]
            print(f"  {sid} {year}: ret={s['total_return_pct']}% dd={s['max_dd_pct']}% "
                  f"trades={summary['n_trades']} bh={summary['strategies']['buy_and_hold']['total_return_pct']}%")
            if a.no_publish:
                continue
            r = pub.publish(
                strategy_id=sid, asset_id="spx500", symbol="SPX500",
                display_name=name, pipeline_type="forecasting", timeframe="daily",
                version=a.version, year=year, summary=summary, trades=trades_doc,
                gates=gates_meta, headline=headline, signals=signals,
                status="experimental", refresh_registry=True,
            )
            print(f"    published v{a.version}/{year} wrote_new={r.get('wrote_new_files')} "
                  f"immutable_hit={r.get('immutable_hit')}")

    if not a.no_publish:
        print("\n[done] bundles SPX publicados (research_only; el forward decide).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
