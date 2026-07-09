"""BTC/USDT end-to-end pipeline runner (design Fases 2-7 → visible on the web).

Loads ingested BTC daily data -> causal features -> 4-regime classifier -> spot-exposure risk
layer + baselines (B1 HODL, B2 trend, S3 regime-gated) -> full-history backtest with gates +
regime attribution + block-bootstrap (design/SPEC-11) -> PUBLISH immutable bundles to the dynamic
registry (design/SPEC-12) so BTC appears in the dashboard dropdown as `BTCUSDT` with per-version
replay. Additive: never touches the COP or Gold bundles.

Honest gate (design §6): S3 (regime-gated) is only a "strategy" if it beats BOTH baselines OOS on
risk-adjusted terms (Sharpe/Calmar). Otherwise the honest answer is "HODL vol-target is the floor".

Usage:
    python scripts/pipeline/run_btc_pipeline.py                 # backtest + publish all 3
    python scripts/pipeline/run_btc_pipeline.py --no-publish    # science only, print metrics
    python scripts/pipeline/run_btc_pipeline.py --warmup 250 --year 2026
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# services/ may be absent in the Airflow container (volume not mounted). Fail-safe per
# quant-constitution: without DSR nothing can PROMOTE (verdict degrades to REVIEW).
try:
    from services.common.metrics import deflated_sharpe_ratio
except ModuleNotFoundError:  # Airflow container without services/ mount
    deflated_sharpe_ratio = None
from src.btc_strategy import backtest as bt
from src.btc_strategy import strategies as st
from src.btc_strategy.indicators import (build_daily_features, classify_regime, merge_funding_features,
                                         regime_transitions_per_year)

PUBLIC_DATA = REPO / "usdcop-trading-dashboard" / "public" / "data"


def _load_publisher():
    p = REPO / "src" / "contracts" / "strategy_manifest.py"
    spec = importlib.util.spec_from_file_location("strategy_manifest", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["strategy_manifest"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="seeds/latest/btcusdt_daily_ohlcv.parquet")
    ap.add_argument("--warmup", type=int, default=250, help="rows dropped for indicator warmup")
    ap.add_argument("--year", type=int, default=2026, help="bundle label year (full-history backtest)")
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--target-vol", type=float, default=st.TARGET_VOL)
    ap.add_argument("--phase", choices=["backtest", "production"], default="backtest",
                    help="production: additionally export btc_trend_b2's LIVE-year files to "
                         "public/data/production/ (paper mode; Vote-2 + L4b deploy flow)")
    a = ap.parse_args()

    seed_path = REPO / a.seed
    if not seed_path.exists():
        print(f"[error] seed not found: {a.seed}\n"
              f"        run: python scripts/data/ingest_btc_ohlcv.py --no-db")
        return 2

    df = pd.read_parquet(seed_path).sort_values("time").reset_index(drop=True)
    print(f"[data] {len(df)} daily bars {df['time'].min().date()} -> {df['time'].max().date()}")

    feat = build_daily_features(df)
    feat = merge_funding_features(feat)  # OLA 5: causal z_funding (graceful si falta seed)
    feat = classify_regime(feat, dwell=5)
    feat = feat.iloc[a.warmup:].reset_index(drop=True)  # drop warmup NaNs
    print(f"[regime] transitions/year = {regime_transitions_per_year(feat)} "
          f"(should be low; BTC cycle regimes last weeks/months)")
    print(f"[regime] distribution:\n{feat['regime'].value_counts().to_string()}")
    bh = feat[["time", "close"]]

    results = {}
    for sid, (name, intent_fn, ptype) in st.STRATEGIES.items():
        pos = st.build_positions(feat, intent_fn, target_vol=a.target_vol)
        res = bt.run_backtest(pos, sid, name, year=a.year, bh_df=bh, oos_year=2025)
        results[sid] = (res, ptype, name)
        m = res["metrics"]
        print(f"\n=== {sid} ({name}) ===")
        print(f"  return={m['total_return_pct']}%  CAGR={m['cagr']}%  Sharpe={m['sharpe']}  "
              f"Sortino={m['sortino']}  MaxDD={m['max_dd']}%  Calmar={m['calmar']}")
        print(f"  trades={res['trades']['summary']['total_trades']}  "
              f"WR={res['trades']['summary']['win_rate']}%  PF={res['trades']['summary']['profit_factor']}  "
              f"p={res['bootstrap']['p_value']}  rec={res['recommendation']}")
        oos = res["summary"].get("oos")
        if oos:
            om = oos["metrics"]
            print(f"  OOS {oos['year']}: return={om['total_return_pct']}%  Sharpe={om['sharpe']}  "
                  f"MaxDD={om['max_dd']}%  p={oos['statistical_tests']['p_value']} "
                  f"(n={oos['n_trading_days']}d)")
        print(f"  regime attribution: { {k: v['pnl_pct'] for k, v in res['attribution'].items()} }")

    # Deflated Sharpe (Bailey & López de Prado): deflate each strategy's PSR for the N=3 trials,
    # using per-period Sharpe dispersion across trials. DSR>0.95 is the trial-aware bar. Injected.
    n_trials = len(results)
    srs = [r[0]["summary"]["statistical_tests"].get("sharpe_per_period", 0.0) for r in results.values()]
    sr_std = float(np.std(srs, ddof=1)) if len(srs) > 1 else 0.0
    print(f"\n[trial-aware] N={n_trials} strategies, per-period Sharpe std={sr_std:.5f}")
    for sid, (res, _ptype, _name) in results.items():
        stt = res["summary"]["statistical_tests"]
        if deflated_sharpe_ratio is None:
            print(f"  [{sid}] DSR UNAVAILABLE (services/ not mounted) -> verdict capped at REVIEW")
            if res["recommendation"] == "PROMOTE":
                res["recommendation"] = "REVIEW"
                res["summary"]["backtest_recommendation"] = "REVIEW"
                res["summary"]["verdict_notes"] = ["DSR unavailable in this runtime (fail-safe)"]
            continue
        dsr = deflated_sharpe_ratio(stt.get("sharpe_per_period", 0.0), stt.get("n_obs", 0),
                                    n_trials=n_trials, trials_sharpe_std=sr_std,
                                    skew=stt.get("skew", 0.0), kurtosis=stt.get("kurtosis", 3.0))
        stt["deflated_sharpe"] = dsr
        print(f"  [{sid}] DSR={dsr['dsr']} (SR0={dsr['sr0']}, PSR={stt.get('psr')}, "
              f"sig@0.95={dsr['significant']})")
        # Honest verdict (OLA 2 H1): PROMOTE requires DSR>=0.95 AND beating B1'
        # (paired-exposure ~ constant-0.44 for BTC, not HODL 1.0) AND surviving 2x costs.
        hg = res["summary"].get("honest_gate", {})
        reasons = []
        if res["recommendation"] == "PROMOTE":
            if not dsr["significant"]:
                reasons.append(f"DSR {dsr['dsr']} < 0.95 (trial-aware bar)")
            if hg and not hg.get("beats_b1_prime_calmar", True):
                reasons.append("does not beat B1' (paired-exposure) on Calmar")
            if reasons:
                res["recommendation"] = "REVIEW"
            if hg and not hg.get("cost_stress", {}).get("survives_2x", True):
                res["recommendation"] = "REJECT"
                reasons.append("dies at 2x transaction costs")
            if reasons:
                res["summary"]["backtest_recommendation"] = res["recommendation"]
                res["summary"]["verdict_notes"] = reasons
                print(f"    -> verdict degraded to {res['recommendation']}: {'; '.join(reasons)}")

    bh_total = float(bh['close'].iloc[-1] / bh['close'].iloc[0] - 1.0) * 100
    print(f"\n[baseline] Buy & Hold (raw, no vol-target) full-history return = {bh_total:.1f}%")

    # honest baseline verdict (design §6): does the regime-gated strat beat BOTH baselines?
    b1 = results["btc_hodl_b1"][0]["metrics"]
    b2 = results["btc_trend_b2"][0]["metrics"]
    s3 = results["btc_exposure_s3"][0]["metrics"]
    print(f"\n[gate] Sharpe: B1={b1['sharpe']}  B2={b2['sharpe']}  S3={s3['sharpe']}")
    print(f"[gate] Calmar: B1={b1['calmar']}  B2={b2['calmar']}  S3={s3['calmar']}")
    beats = s3['sharpe'] > max(b1['sharpe'], b2['sharpe']) and s3['calmar'] > max(b1['calmar'], b2['calmar'])
    print(f"[gate] S3 {'BEATS' if beats else 'does NOT beat'} both baselines on Sharpe AND Calmar "
          f"-> {'promote candidate' if beats else 'HODL vol-target remains the honest floor'}")

    if a.no_publish:
        print("\n[publish] skipped (--no-publish)")
        return 0

    sm = _load_publisher()
    pub = sm.BundlePublisher(PUBLIC_DATA, generated_at=str(pd.Timestamp.utcnow().isoformat()))
    print(f"\n[publish] -> {PUBLIC_DATA}")
    for sid, (res, ptype, name) in results.items():
        gates = {"passed": sum(g["passed"] for g in res["gates"]), "of": len(res["gates"]),
                 "recommendation": res["recommendation"]}
        # Daily equity series → signals_<year>.json: the SSOT for window-dynamic
        # replay metrics in the frontend (trade records cannot represent partial windows).
        eqc = res["equity_curve"]
        series = {"kind": "daily_equity", "initial_capital": 10000.0,
                  "rows": [{"d": str(r_.time)[:10], "eq": round(10000.0 * float(r_.equity), 2)}
                           for r_ in eqc.itertuples(index=False)]}
        r = pub.publish(
            strategy_id=sid, asset_id="btcusdt", symbol="BTC/USDT", display_name=name,
            pipeline_type=ptype, timeframe="daily", version=a.version, year=a.year,
            summary=res["summary"], trades=res["trades"], gates=gates, headline=res["headline"],
            signals=series, status="experimental", refresh_registry=True,
        )
        print(f"  published {sid} v{a.version}/{a.year}  "
              f"wrote_new={r.get('wrote_new_files')} immutable_hit={r.get('immutable_hit')}")
    print("\n[done] BTC bundles published. Registry + manifests refreshed (additive).")

    # ── production-paper export (CTR-WITHDRAWAL-BTC-001) ─────────────────────────
    # Writes btc_trend_b2's LIVE-year (2026) files to public/data/production/ using the
    # per-strategy convention (summary_<sid>.json / approval_state_<sid>.json /
    # trades/<sid>.json) so /production can list it NEXT TO the COP singleton without
    # touching it. PAPER mode by contract: PreTradeGate simulates; withdrawal protocol
    # (.claude/specs/assets/btcusdt/WITHDRAWAL-PROTOCOL-BTC.md) governs graduation.
    if a.phase == "production":
        import json as _json
        from datetime import datetime as _dt, timezone as _tz

        sid = "btc_trend_b2"
        res, _ptype, name = results[sid]
        pos = st.build_positions(feat, st.STRATEGIES[sid][1], target_vol=a.target_vol)
        res26 = bt.run_backtest(pos, sid, name, year=a.year, bh_df=bh, oos_year=2026)
        oos26 = (res26["summary"].get("oos") or {})
        m26, stt26 = oos26.get("metrics", {}), oos26.get("statistical_tests", {})

        trades26 = [t for t in res["trades"]["trades"]
                    if str(t.get("timestamp", "")).startswith("2026")]
        wins = [t for t in trades26 if (t.get("pnl_usd") or 0) > 0]
        losses = [t for t in trades26 if (t.get("pnl_usd") or 0) < 0]
        gross_w = sum(t["pnl_usd"] for t in wins)
        gross_l = abs(sum(t["pnl_usd"] for t in losses))
        exit_reasons: dict[str, int] = {}
        for t in trades26:
            exit_reasons[t.get("exit_reason", "n/a")] = exit_reasons.get(t.get("exit_reason", "n/a"), 0) + 1

        ret26 = m26.get("total_return_pct")
        stats26 = {
            "final_equity": None if ret26 is None else round(10000.0 * (1 + ret26 / 100), 2),
            "total_return_pct": ret26,
            "sharpe": m26.get("sharpe"),
            "max_dd_pct": None if m26.get("max_dd") is None else abs(m26["max_dd"]),
            "win_rate_pct": None if not trades26 else round(100.0 * len(wins) / len(trades26), 1),
            "profit_factor": None if gross_l == 0 else round(gross_w / gross_l, 2),
            "trading_days": oos26.get("n_trading_days"),
            "exit_reasons": exit_reasons,
            "n_long": sum(1 for t in trades26 if t.get("side") == "LONG"),
            "n_short": sum(1 for t in trades26 if t.get("side") == "SHORT"),
            "calmar": m26.get("calmar"),
        }
        now = _dt.now(_tz.utc).isoformat()
        prod_dir = PUBLIC_DATA / "production"
        (prod_dir / "trades").mkdir(parents=True, exist_ok=True)

        summary_doc = {
            "strategy_id": sid, "strategy_name": name, "year": 2026, "mode": "paper",
            "chart_symbol": "BTCUSDT",
            "frozen_version": a.version, "generated_at": now,
            "strategies": {sid: stats26},
            "statistical_tests": stt26,
            "withdrawal_protocol": ".claude/specs/assets/btcusdt/WITHDRAWAL-PROTOCOL-BTC.md",
        }
        gates_list = res["gates"] + [{
            "gate": "deflated_sharpe", "label": "DSR trial-aware (>0.95)",
            "passed": bool(res["summary"]["statistical_tests"].get("deflated_sharpe", {}).get("significant")),
            "value": res["summary"]["statistical_tests"].get("deflated_sharpe", {}).get("dsr"),
            "threshold": 0.95,
        }]
        approval_doc = {
            "status": "PENDING_APPROVAL",
            "strategy": sid, "strategy_name": name, "backtest_year": a.year,
            "backtest_recommendation": res["recommendation"],
            "backtest_confidence": round(sum(1 for g in gates_list if g.get("passed")) / len(gates_list), 2),
            "gates": gates_list,
            "backtest_metrics": {
                "return_pct": res["metrics"]["total_return_pct"], "sharpe": res["metrics"]["sharpe"],
                "calmar": res["metrics"]["calmar"], "max_dd_pct": abs(res["metrics"]["max_dd"]),
                "p_value": res["bootstrap"]["p_value"],
                "oos_2025_return_pct": (res["summary"].get("oos") or {}).get("metrics", {}).get("total_return_pct"),
            },
            "deploy_manifest": {
                "pipeline_type": "rule_based",
                "script": "scripts/pipeline/run_btc_pipeline.py",
                "args": ["--phase", "production", "--version", a.version],
                "config_path": "config/assets/btcusdt.yaml",
                "db_tables": [],
                "mode": "paper",
            },
            "approved_by": None, "approved_at": None, "reviewer_notes": None,
            "created_at": now, "last_updated": now,
        }
        trades_doc = {"strategy_id": sid, "strategy_name": name, "year": 2026,
                      "initial_capital": 10000.0, "trades": trades26,
                      "summary": {"total_trades": len(trades26),
                                  "win_rate": stats26["win_rate_pct"],
                                  "profit_factor": stats26["profit_factor"]}}

        def _dump(path, doc):
            with open(path, "w", encoding="utf-8") as f:
                _json.dump(doc, f, indent=2, ensure_ascii=False, default=str)
            print(f"  [production] wrote {path}")

        _dump(prod_dir / f"summary_{sid}.json", summary_doc)
        _dump(prod_dir / f"approval_state_{sid}.json", approval_doc)
        _dump(prod_dir / "trades" / f"{sid}.json", trades_doc)
        print(f"\n[production] {sid} exported (PAPER). 2026 YTD: ret={ret26}% "
              f"sharpe={m26.get('sharpe')} trades={len(trades26)}. "
              f"Next: Vote-2 on /dashboard (selector {sid}) -> L4b deploy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
