"""Run the S&P 500 strategy science stage from the asset SSOT.

This thin entrypoint is intentionally deterministic and Airflow-friendly: it
validates that the asset profile and experiment config agree, then delegates to
the strategy's backtest/gates implementation.  Real-feed promotion remains
blocked by the experiment gates.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--check", action="store_true", help="validate SSOT only")
    ap.add_argument("--synthetic", action="store_true",
                    help="EXPLICIT opt-in to the synthetic-data science run (never default; "
                         "plan SPX S0.1: the default path must be real data, fail-closed)")
    args = ap.parse_args()
    asset = yaml.safe_load((ROOT / "config/assets/spx500.yaml").read_text())
    exp = yaml.safe_load((ROOT / "config/forecast_experiments/spx500_regime_gated_v1.yaml").read_text())
    if asset.get("asset_id") != exp.get("asset_id") or asset.get("strategy_id") != exp.get("strategy_id"):
        raise RuntimeError("SPX500 asset/experiment SSOT mismatch")
    if args.check:
        print("SPX500 SSOT check: PASS (promotion remains blocked until real PIT/OOS evidence)")
        return 0
    import sys
    sys.path.insert(0, str(ROOT / "src/strategies/spx500_regime_gated_v1"))
    if args.synthetic:
        from run_strategy import main as run
        result = run(seed=args.seed)
        return 0 if isinstance(result, dict) else 1
    # DEFAULT = REAL DATA, FAIL-CLOSED (Codex audit P0 pipeline_blocked, verified: the old
    # default delegated to run_strategy.main() -> datagen.generate(), proving code, not market)
    from load_real import load_real
    df = load_real()  # raises if the snapshot is absent -- that IS the fail-closed contract
    if df is None or len(df) == 0:
        raise RuntimeError("load_real() returned no rows — refusing synthetic fallback")
    from policies import spx_regime_gated_v1
    w = spx_regime_gated_v1(df)
    print(f"SPX500 REAL run: {len(df)} filas {df['timestamp'].iloc[0]} -> "
          f"{df['timestamp'].iloc[-1]}, pesos no-nulos={int((w.abs() > 1e-9).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
