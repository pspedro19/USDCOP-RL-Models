"""Gold end-to-end pipeline runner (SPECSGOLD Fases 1-6 → visible on the web).

Loads ingested Gold daily data -> features (SPEC-03) -> regime (SPEC-04) -> risk+strategies
(SPEC-06/07) -> walk-forward-style backtest with gates + regime attribution + bootstrap (SPEC-09)
-> PUBLISH immutable bundles to the dynamic registry (SPEC-12) so the Gold asset appears in the
dashboard dropdown with per-version replay. Additive: never touches the existing COP bundles.

Usage:
    python scripts/run_gold_pipeline.py                 # backtest + publish all strategies
    python scripts/run_gold_pipeline.py --no-publish    # science only, print metrics
    python scripts/run_gold_pipeline.py --warmup 300 --year 2026
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# services/ may be absent in the Airflow container (volume not mounted). Fail-safe per
# quant-constitution: without DSR nothing can PROMOTE (verdict degrades to REVIEW).
try:
    from services.common.metrics import deflated_sharpe_ratio
except ModuleNotFoundError:  # Airflow container without services/ mount
    deflated_sharpe_ratio = None
from src.gold_rl import backtest as bt
from src.gold_rl import strategies as st
from src.gold_rl.indicators import build_daily_features, classify_regime, regime_transitions_per_year

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
    ap.add_argument("--seed", default="seeds/latest/xauusd_daily_ohlcv.parquet")
    ap.add_argument("--warmup", type=int, default=252, help="rows dropped for indicator warmup")
    ap.add_argument("--year", type=int, default=2026, help="bundle label year")
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--target-vol", type=float, default=0.10)
    a = ap.parse_args()

    df = pd.read_parquet(REPO / a.seed).sort_values("time").reset_index(drop=True)
    print(f"[data] {len(df)} daily bars {df['time'].min().date()} -> {df['time'].max().date()}")

    feat = build_daily_features(df)

    # H-XAU-DXY-01: merge causal del DXY (T-1) desde MACRO_DAILY_CLEAN (fresco vía backfill).
    # merge_asof backward + shift(1) en el z: la decisión del día t usa DXY hasta t-1.
    try:
        import pandas as _pd
        _mc = _pd.read_parquet(REPO / "data" / "pipeline" / "04_cleaning" / "output" / "MACRO_DAILY_CLEAN.parquet")
        _dxy = _mc["FXRT_INDEX_DXY_USA_D_DXY"].dropna().rename("dxy")
        _dxy.index = _pd.to_datetime(_dxy.index).tz_localize(None)
        _ret20 = _dxy.pct_change(20)
        _z = ((_ret20 - _ret20.rolling(252, min_periods=60).mean())
              / _ret20.rolling(252, min_periods=60).std()).shift(1)  # causal
        _zdf = _z.rename("dxy_z20").reset_index()
        _zdf.columns = ["mdate", "dxy_z20"]  # index name varies (fecha/None) — force it
        _g = feat[["time"]].copy()
        _g["mdate"] = _pd.to_datetime(_g["time"]).dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
        _merged = _pd.merge_asof(_g.sort_values("mdate"), _zdf.sort_values("mdate"),
                                 on="mdate", direction="backward")
        feat["dxy_z20"] = _merged["dxy_z20"].values
        print(f"[dxy] merged: {feat['dxy_z20'].notna().sum()} bars with DXY z-score "
              f"(cobertura {feat.loc[feat['dxy_z20'].notna(),'time'].min()} ->)")
    except Exception as _e:  # noqa: BLE001 — tilt degrades to B1 without DXY
        print(f"[dxy] merge skipped ({_e}) — tilt = B1")

    # A10-01: read the ASSET's regime-gate thresholds from its profile (xauusd.yaml);
    # nulls fall back to an explicit, logged 0.5 pivot inside classify_regime —
    # never the COP values. Fitting real thresholds = registered trial (test D1).
    _rg = {}
    try:
        import yaml as _yaml
        _prof = _yaml.safe_load((REPO / "config" / "assets" / "xauusd.yaml").read_text(encoding="utf-8"))
        _rg = (_prof or {}).get("regime_gate") or {}
    except Exception as _e:  # noqa: BLE001 — profile is optional for the backtest runner
        print(f"[regime] asset profile not loaded ({_e}); using neutral 0.5 pivot")
    feat = classify_regime(
        feat,
        dwell=int(_rg.get("hysteresis_dwell_days", 4)),
        hurst_trending=_rg.get("hurst_trending"),
        hurst_mean_rev=_rg.get("hurst_mean_rev"),
    )
    feat = feat.iloc[a.warmup:].reset_index(drop=True)  # drop warmup NaNs
    print(f"[regime] transitions/year = {regime_transitions_per_year(feat)} "
          f"(should be low; Gold regimes last weeks)")
    print(f"[regime] distribution:\n{feat['regime'].value_counts().to_string()}")
    bh = feat[["time", "close"]]

    results = {}
    for sid, (name, dir_fn, ptype) in st.STRATEGIES.items():
        pos = st.build_positions(feat, dir_fn, target_vol=a.target_vol)
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

    # Deflated Sharpe (Bailey & López de Prado): deflate each strategy's PSR for the N=3 trials we
    # tested, using the dispersion of per-period Sharpe across trials. Trial-aware honesty — a lone
    # p<0.05 among several tries is easy; DSR>0.95 is the real bar. Injected into each summary.
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
        # ── Honest verdict (OLA 2 H1, resolves audit I-3): the recommendation is now
        # DSR-aware + B1'-aware + cost-stress-aware. A lone bootstrap p-value can no
        # longer produce PROMOTE (that's how gold_trend_b2 sat PROMOTE at DSR 0.921).
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

    # honest baseline verdict (STRATEGY §6): does the regime strategy beat BOTH baselines?
    b1 = results["gold_long_only_b1"][0]["metrics"]["sharpe"]
    b2 = results["gold_trend_b2"][0]["metrics"]["sharpe"]
    rg = results["gold_regime_gated_v1"][0]["metrics"]["sharpe"]
    print(f"\n[gate] Sharpe: B1={b1}  B2={b2}  regime-gated={rg}  "
          f"-> regime {'BEATS' if rg > max(b1, b2) else 'does NOT beat'} both baselines")

    if a.no_publish:
        print("\n[publish] skipped (--no-publish)")
        return 0

    sm = _load_publisher()
    pub = sm.BundlePublisher(PUBLIC_DATA, generated_at=str(pd.Timestamp.utcnow().isoformat()))
    print(f"\n[publish] -> {PUBLIC_DATA}")
    for sid, (res, ptype, name) in results.items():
        gates = {"passed": sum(g["passed"] for g in res["gates"]), "of": len(res["gates"]),
                 "recommendation": res["recommendation"]}
        eqc = res["equity_curve"]
        series = {"kind": "daily_equity", "initial_capital": 10000.0,
                  "rows": [{"d": str(r_.time)[:10], "eq": round(10000.0 * float(r_.equity), 2)}
                           for r_ in eqc.itertuples(index=False)]}
        r = pub.publish(
            strategy_id=sid, asset_id="xauusd", symbol="XAU/USD", display_name=name,
            pipeline_type=ptype, timeframe="daily", version=a.version, year=a.year,
            summary=res["summary"], trades=res["trades"], gates=gates, headline=res["headline"],
            signals=series, status="experimental", refresh_registry=True,
        )
        print(f"  published {sid} v{a.version}/{a.year}  "
              f"wrote_new={r.get('wrote_new_files')} immutable_hit={r.get('immutable_hit')}")
    print("\n[done] Gold bundles published. Registry + manifests refreshed (additive).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
