"""Build post-freeze 2026 SessionSpec objects without refitting anything."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args: argparse.Namespace) -> dict:
    # Set identity inputs before importing dataset: they are part of the frozen contract.
    os.environ["THESIS_PORTABLE"] = str(args.portable.resolve())
    os.environ["THESIS_SCALER_CONFIG"] = str(args.scaler.resolve())
    os.environ["THESIS_REGIME_CONFIG"] = str(args.regime.resolve())
    os.environ["THESIS_PARTITION_CONFIG"] = str(args.partition.resolve())
    os.environ["THESIS_MACRO_CLEAN"] = str(args.macro.resolve())

    from src.research.dataset import (
        MACRO_FEATURES,
        MARKET_FEATURES,
        _closes,
        load_portable,
    )
    from src.research.evaluation_mask import build_mask
    from src.research.features import attach_macro_features, build_market_features
    from src.research.regime_hmm import _levels_for_k, build_regime_observations
    from src.research.regime_portable import PortableRegimeModel
    from src.research.session_gym import SessionSpec

    data = load_portable(args.portable.resolve())
    m5 = pd.read_parquet(args.market)
    mask = build_mask()
    valid = set(mask.valid)
    start = pd.Timestamp(args.start).date()
    end = pd.Timestamp(args.end).date()
    target_dates = sorted(d for d in valid if start <= d <= end)
    features = build_market_features(m5, valid_sessions=valid)
    macro = attach_macro_features(sorted(valid))
    observations = build_regime_observations(m5, valid_sessions=valid).dropna().sort_index()
    model = PortableRegimeModel.load(args.regime.resolve())
    levels = _levels_for_k(model.k)
    specs = []
    dropped: dict[str, list[str]] = {}

    for date in target_dates:
        group = features[features["_session"] == date]
        if len(group) != 60:
            dropped.setdefault("incomplete", []).append(date.isoformat())
            continue
        prior = observations[observations.index.date < date]
        if len(prior) < 60:
            dropped.setdefault("no_prior_regime", []).append(date.isoformat())
            continue
        posterior = model.filtered_posterior(prior.to_numpy(dtype=float))
        macro_row = macro.reindex([pd.Timestamp(date)])[MACRO_FEATURES].iloc[0]
        if not np.isfinite(macro_row.to_numpy(dtype=float)).all():
            dropped.setdefault("sin_macro", []).append(date.isoformat())
            continue
        market_values = (group[MARKET_FEATURES].to_numpy(dtype=float) - data.scaler_mean) / data.scaler_scale
        macro_values = (macro_row.to_numpy(dtype=float) - data.macro_scaler_mean) / data.macro_scaler_scale
        context = np.clip(np.concatenate([macro_values, posterior]), -5.0, 5.0).astype(np.float32)
        specs.append(SessionSpec(
            date=date,
            close=_closes(m5, date),
            market=np.clip(market_values, -5.0, 5.0).astype(np.float32),
            context=context,
            spread_pips=float(np.dot(posterior, levels)),
        ))

    payload = {
        "schema_version": "confirmatory-forward-specs-v1",
        "portable_sha256": _digest(args.portable.resolve()),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "n_sessions": len(specs),
        "dates": [s.date.isoformat() for s in specs],
        "dropped": dropped,
        "refit": False,
        "hmm_fit_range": list(model.fit_range),
        "scaler_source": "portable_v4_stable",
        "macro_policy": "strict_prior_session",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump({"manifest": payload, "sessions": specs}, handle)
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portable", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--regime", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--macro", type=Path, required=True)
    parser.add_argument("--market", type=Path, default=ROOT / "seeds/latest/usdcop_m5_ohlcv.parquet")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(args)
    print(json.dumps({k: report[k] for k in ("start", "end", "n_sessions", "portable_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
