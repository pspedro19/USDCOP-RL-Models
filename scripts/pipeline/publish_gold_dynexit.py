#!/usr/bin/env python3
"""Publica `gold_dynamic_exit` (campeona de Oro, §8.7) al registry dinámico y archiva las demás.

Directiva del operador 2026-07-07: "publica solamente las mejores estrategias que han sobrevivido,
las mejores de cada activo". Resultado: 3 visibles — smart_simple_v11 (COP, production),
gold_dynamic_exit (Oro, experimental-paper), btc_trend_b2 (BTC, experimental).

La estrategia es STATEFUL (Chandelier trailing) — no cabe en el loop direction_fn del pipeline
estándar; este script publica desde el simulador pre-registrado (scripts/analysis/gold_dynamic_exit,
prior M=3.0 congelado) con los mismos artefactos que BundlePublisher espera: summary (con OOS-2025,
honest_gate, DSR trial-aware del programa ≈74 trials), trades (StrategyTrade), serie diaria de
equity (replay dinámico) y gates. Idempotente (bundles inmutables).

Usage: python scripts/pipeline/publish_gold_dynexit.py [--version 1.0.0]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.gold_dynamic_exit import simulate  # noqa: E402
from services.common.metrics import (deflated_sharpe_ratio,  # noqa: E402
                                     paired_exposure_baseline)
from src.gold_rl import backtest as bt  # noqa: E402
from src.gold_rl.indicators import build_daily_features  # noqa: E402

PUBLIC_DATA = REPO / "usdcop-trading-dashboard" / "public" / "data"
SID = "gold_dynamic_exit"
NAME = "Gold · Dynamic exit (señal + Chandelier 3×ATR)"
TRAIL_M = 3.0          # prior congelado §8.7 — NO tocar
TRIALS_PROGRAM = 74    # trials acumulados del programa (§8.8)
INITIAL = 10000.0


def _load_publisher():
    p = REPO / "src" / "contracts" / "strategy_manifest.py"
    spec = importlib.util.spec_from_file_location("strategy_manifest", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["strategy_manifest"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_trades(raw: list[dict], daily: pd.DataFrame) -> list[dict]:
    """PnL por trade NETO desde la serie diaria (costos + swap incluidos) — la tabla de
    trades DEBE componer a la misma equity que la curva (honestidad: sin PnL bruto)."""
    out, tid = [], 0
    dts = pd.to_datetime(daily["time"]).dt.tz_localize(None).values
    eqc = (1.0 + daily["strat_ret"]).cumprod().values  # curva acumulada (la misma del summary)
    for t in raw:
        tid += 1
        i0 = int(np.searchsorted(dts, pd.Timestamp(t["entry_time"]).tz_localize(None).to_datetime64()))
        i1 = int(np.searchsorted(dts, pd.Timestamp(t["exit_time"]).tz_localize(None).to_datetime64()))
        i0, i1 = min(i0, len(eqc) - 1), min(i1, len(eqc) - 1)
        base = eqc[i0 - 1] if i0 > 0 else 1.0
        g = float(eqc[i1] / base)  # neto exacto: mismo compuesto que la curva de equity
        t = {**t, "pnl_pct": (g - 1.0) * 100.0}
        eq_in = INITIAL * base
        equity = INITIAL * float(eqc[i1])
        out.append({
            "trade_id": tid,
            "timestamp": pd.Timestamp(t["entry_time"]).isoformat(),
            "exit_timestamp": pd.Timestamp(t["exit_time"]).isoformat(),
            "side": "LONG",
            "entry_price": round(t["entry_px"], 2),
            "exit_price": round(t["exit_px"], 2),
            "pnl_pct": round(t["pnl_pct"], 4),
            "pnl_usd": round(eq_in * (g - 1.0), 2),
            "equity_at_entry": round(eq_in, 2),
            "equity_at_exit": round(equity, 2),
            "leverage": round(t["size"], 3),
            "exit_reason": t["reason"],
            "hold_days": int(t["days"]),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--year", type=int, default=2026)
    a = ap.parse_args()

    df = pd.read_parquet(REPO / "seeds/latest/xauusd_daily_ohlcv.parquet").sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    feat = build_daily_features(df)
    d = simulate(feat, TRAIL_M)
    d = d.iloc[252:].reset_index(drop=True)  # warmup SMA252
    d["equity"] = (1.0 + d["strat_ret"]).cumprod()

    m = bt.metrics(d)
    boot = bt.block_bootstrap_pvalue(d["strat_ret"])
    trades = build_trades(d.attrs["trades"], d)

    # OOS-2025 slice (el juez del backtest, metodología COP) — los 5 gates de Vote-1 se
    # evalúan SOBRE EL OOS-2025, no sobre la ventana de 22 años (el gate de DD 20% está
    # dimensionado para un año de backtest, como en COP).
    y25 = d[(d["time"] >= "2025-01-01") & (d["time"] <= "2025-12-31")].copy()
    y25["equity"] = (1.0 + y25["strat_ret"]).cumprod()
    m25 = bt.metrics(y25)
    b25 = bt.block_bootstrap_pvalue(y25["strat_ret"])
    n_tr_25 = sum(1 for t in trades if str(t["exit_timestamp"])[:4] == "2025")
    gates = bt.evaluate_gates(m25, n_tr_25, b25)
    rec, conf = bt.recommendation(gates)

    # Honest gate: B1' + costos ×2/×3 (re-simulación real) + DSR trial-aware del PROGRAMA
    b1p = paired_exposure_baseline(d["position"].values, d["ret"].values, 252)
    stress = {}
    for k in (2.0, 3.0):
        dk = simulate(feat, TRAIL_M, cost_mult=k).iloc[252:]
        dk = dk.copy()
        dk["equity"] = (1.0 + dk["strat_ret"]).cumprod()
        stress[f"x{int(k)}"] = bt.metrics(dk)["calmar"]
    sr = d["strat_ret"].values
    from scipy import stats as sps
    srd = float(np.mean(sr) / np.std(sr))
    dsr = deflated_sharpe_ratio(srd, len(sr), TRIALS_PROGRAM, 0.5 / np.sqrt(252),
                                skew=float(sps.skew(sr)), kurtosis=float(sps.kurtosis(sr, fisher=False)))
    # Veredicto honesto: DSR<0.95 ⇒ nunca PROMOTE (constitución §2). Es ingeniería de riesgo
    # sobre beta (bate B1' en Calmar), no claim de alfa — experimental-paper, juez forward 2026.
    verdict_notes = []
    if not dsr.get("significant"):
        verdict_notes.append(f"DSR {dsr.get('dsr')} < 0.95 (trial-aware, N={TRIALS_PROGRAM}) — sin claim de alfa")
        if rec == "PROMOTE":
            rec = "REVIEW"
    if stress["x2"] <= 0:
        rec, _ = "REJECT", verdict_notes.append("muere a costos ×2")

    wins = [t for t in trades if t["pnl_pct"] > 0]
    gl = abs(sum(t["pnl_pct"] for t in trades if t["pnl_pct"] < 0))
    pf = round(sum(t["pnl_pct"] for t in wins) / gl, 3) if gl > 0 else None
    exit_reasons: dict[str, int] = {}
    for t in trades:
        exit_reasons[t["exit_reason"]] = exit_reasons.get(t["exit_reason"], 0) + 1

    bh_total = float(d["close"].iloc[-1] / d["close"].iloc[0] - 1.0)
    summary = {
        "strategy_id": SID, "strategy_name": NAME, "year": a.year,
        "initial_capital": INITIAL, "asset": "XAU/USD", "n_trading_days": int(len(d)),
        "strategies": {
            SID: {
                "final_equity": round(INITIAL * float(d["equity"].iloc[-1]), 2),
                "total_return_pct": m["total_return_pct"], "sharpe": m["sharpe"],
                "sortino": m["sortino"], "calmar": m["calmar"], "max_dd_pct": m["max_dd"],
                "win_rate_pct": round(100 * len(wins) / len(trades), 1) if trades else 0.0,
                "profit_factor": pf, "n_long": len(trades), "n_short": 0,
                "trading_days": int(len(d)), "exit_reasons": exit_reasons,
                "median_hold_days": int(np.median([t["hold_days"] for t in trades])) if trades else None,
            },
            "buy_and_hold": {"final_equity": round(INITIAL * (1 + bh_total), 2),
                             "total_return_pct": round(bh_total * 100, 2)},
        },
        "statistical_tests": {"p_value": boot["p_value"], "significant": boot["significant"],
                              "bootstrap_95ci_ann": [boot["ci_low"], boot["ci_high"]],
                              "deflated_sharpe": dsr},
        "backtest_recommendation": rec, "backtest_confidence": conf,
        "gates": gates,
        "verdict_notes": verdict_notes,
        "honest_gate": {"b1_prime": b1p,
                        "beats_b1_prime_calmar": bool(m["calmar"] > b1p["calmar"]),
                        "cost_stress": {"calmar_x2": stress["x2"], "calmar_x3": stress["x3"],
                                        "survives_2x": bool(stress["x2"] > 0)}},
        "oos": {"year": 2025, "n_trading_days": int(len(y25)), "metrics": m25,
                "statistical_tests": {"p_value": b25["p_value"], "significant": b25["significant"],
                                      "bootstrap_95ci_ann": [b25["ci_low"], b25["ci_high"]]}},
        "methodology": ("Entrada: votos SMA{63,126,252} ≥ 2/3 (causal t-1). Salida dinámica: "
                        "señal muere ∨ Chandelier trailing 3.0×ATR14 (prior ex-ante §8.7, "
                        "hold variable 9d-97d). Vol-target 10%, cap 1.5. Diseño ≤2024, OOS=2025. "
                        "Ingeniería de riesgo sobre beta — sin claim de alfa (DSR<0.95)."),
    }
    trades_file = {"strategy_id": SID, "strategy_name": NAME, "initial_capital": INITIAL,
                   "date_range": {"start": str(d["time"].iloc[0].date()),
                                  "end": str(d["time"].iloc[-1].date())},
                   "trades": trades,
                   "summary": {"total_trades": len(trades),
                               "win_rate": summary["strategies"][SID]["win_rate_pct"],
                               "total_return_pct": m["total_return_pct"], "sharpe_ratio": m["sharpe"],
                               "max_drawdown_pct": m["max_dd"], "profit_factor": pf,
                               "p_value": boot["p_value"], "n_long": len(trades), "n_short": 0}}
    series = {"kind": "daily_equity", "initial_capital": INITIAL,
              "rows": [{"d": str(r_.time)[:10], "eq": round(INITIAL * float(r_.equity), 2)}
                       for r_ in d[["time", "equity"]].itertuples(index=False)]}

    print(f"[science] full: ret={m['total_return_pct']}% Sharpe={m['sharpe']} Calmar={m['calmar']} "
          f"MaxDD={m['max_dd']}% trades={len(trades)} | OOS2025: ret={m25['total_return_pct']}% "
          f"Sharpe={m25['sharpe']} p={b25['p_value']}")
    print(f"[honest] B1'={b1p['calmar']:.3f} vs {m['calmar']:.3f} beats={m['calmar'] > b1p['calmar']} | "
          f"stress x2={stress['x2']:.3f} x3={stress['x3']:.3f} | DSR={dsr.get('dsr')} -> {rec}")

    sm = _load_publisher()
    pub = sm.BundlePublisher(PUBLIC_DATA, generated_at=str(pd.Timestamp.utcnow().isoformat()))
    gates_meta = {"passed": sum(g["passed"] for g in gates), "of": len(gates), "recommendation": rec}
    r = pub.publish(strategy_id=SID, asset_id="xauusd", symbol="XAU/USD", display_name=NAME,
                    pipeline_type="rule_based", timeframe="daily", version=a.version, year=a.year,
                    summary=summary, trades=trades_file, gates=gates_meta,
                    headline={"return_pct": m["total_return_pct"], "sharpe": m["sharpe"],
                              "p_value": boot["p_value"]},
                    signals=series, status="experimental", refresh_registry=True)
    print(f"[publish] {SID} v{a.version}/{a.year} wrote_new={r.get('wrote_new_files')}")

    # Archivar todo lo que no sea campeón (1 por activo) y refrescar el registry.
    champions = {"smart_simple_v11", "btc_trend_b2", SID}
    strat_root = PUBLIC_DATA / "strategies"
    changed = []
    for man_path in strat_root.glob("*/manifest.json"):
        man = json.loads(man_path.read_text(encoding="utf-8"))
        sid = man.get("strategy_id")
        want = man.get("status") if sid in champions else "archived"
        if sid in champions and man.get("status") == "archived":
            want = "experimental"  # nunca dejar un campeón oculto
        if man.get("status") != want:
            man["status"] = want
            man_path.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
            changed.append(f"{sid}->{want}")
    builder = sm.RegistryBuilder(PUBLIC_DATA, generated_at=str(pd.Timestamp.utcnow().isoformat()))
    builder.write(builder.build(write_manifests=False))

    # ── Paquete Vote-2 (patrón BTC): approval_state_<sid>.json PENDING + summary/trades 2026 ──
    prod = PUBLIC_DATA / "production"
    prod.mkdir(exist_ok=True)
    ap_path = prod / f"approval_state_{SID}.json"
    existing = json.loads(ap_path.read_text(encoding="utf-8")) if ap_path.exists() else {}
    if existing.get("status") not in ("APPROVED", "LIVE"):  # nunca clobberear un voto ya emitido
        now = str(pd.Timestamp.utcnow().isoformat())
        ap = {
            "status": "PENDING_APPROVAL", "strategy": SID, "strategy_name": NAME,
            "backtest_year": 2025, "backtest_recommendation": rec,
            "backtest_confidence": conf, "gates": gates,
            "backtest_metrics": {"return_pct": m25["total_return_pct"], "sharpe": m25["sharpe"],
                                 "calmar": m25["calmar"], "max_dd_pct": abs(m25["max_dd"]),
                                 "p_value": b25["p_value"], "trades_2025": n_tr_25,
                                 "full_calmar": m["calmar"], "dsr": dsr.get("dsr"),
                                 "beats_b1_prime": bool(m["calmar"] > b1p["calmar"])},
            "verdict_notes": verdict_notes,
            "deploy_manifest": {"pipeline_type": "rule_based",
                                "script": "scripts/pipeline/publish_gold_dynexit.py",
                                "args": ["--version", a.version],
                                "config_path": "config/assets/xauusd.yaml",
                                "db_tables": [], "mode": "paper"},
            "approved_by": existing.get("approved_by"), "approved_at": existing.get("approved_at"),
            "reviewer_notes": existing.get("reviewer_notes"),
            "created_at": existing.get("created_at", now), "last_updated": now,
        }
        ap_path.write_text(json.dumps(ap, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[vote2] approval_state_{SID}.json -> PENDING_APPROVAL ({rec})")

    # summary/trades de producción 2026 (lo que /production muestra tras la aprobación)
    y26 = d[d["time"] >= "2026-01-01"].copy()
    if len(y26) > 20:
        y26["equity"] = (1.0 + y26["strat_ret"]).cumprod()
        m26 = bt.metrics(y26)
        tr26 = [t for t in trades if str(t["exit_timestamp"])[:4] == "2026"]
        w26 = [t for t in tr26 if t["pnl_pct"] > 0]
        bh26 = float(y26["close"].iloc[-1] / y26["close"].iloc[0] - 1.0) * 100
        sum26 = {"strategy_id": SID, "strategy_name": NAME, "year": 2026,
                 "initial_capital": INITIAL, "asset": "XAU/USD", "mode": "paper",
                 "chart_symbol": "XAUUSD",
                 "n_trading_days": int(len(y26)),
                 "strategies": {SID: {
                     "final_equity": round(INITIAL * float(y26["equity"].iloc[-1]), 2),
                     "total_return_pct": m26["total_return_pct"], "sharpe": m26["sharpe"],
                     "calmar": m26["calmar"], "max_dd_pct": m26["max_dd"],
                     "win_rate_pct": round(100 * len(w26) / len(tr26), 1) if tr26 else 0.0,
                     "profit_factor": None, "n_long": len(tr26), "n_short": 0,
                     "trading_days": int(len(y26)),
                     "exit_reasons": {r2: sum(1 for t in tr26 if t["exit_reason"] == r2)
                                      for r2 in {t["exit_reason"] for t in tr26}}},
                     "buy_and_hold": {"total_return_pct": round(bh26, 2)}}}
        (prod / f"summary_{SID}.json").write_text(json.dumps(sum26, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")
        (prod / "trades").mkdir(exist_ok=True)
        (prod / "trades" / f"{SID}.json").write_text(json.dumps(
            {"strategy_id": SID, "strategy_name": NAME, "initial_capital": INITIAL,
             "date_range": {"start": "2026-01-01", "end": str(d["time"].iloc[-1].date())},
             "trades": tr26,
             "summary": {"total_trades": len(tr26),
                         "win_rate": round(100 * len(w26) / len(tr26), 1) if tr26 else 0.0,
                         "total_return_pct": m26["total_return_pct"],
                         "sharpe_ratio": m26["sharpe"], "max_drawdown_pct": m26["max_dd"],
                         "profit_factor": None, "p_value": None,
                         "n_long": len(tr26), "n_short": 0}},
            indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[prod] summary_{SID}.json (2026 YTD {m26['total_return_pct']}%) + trades ({len(tr26)})")
    print(f"[registry] archived/normalized: {changed or 'sin cambios'}")
    reg = json.loads((PUBLIC_DATA / "registry.json").read_text(encoding="utf-8"))
    vis = [(s["strategy_id"], s.get("status")) for s in reg["strategies"] if s.get("status") != "archived"]
    print(f"[registry] visibles: {vis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
