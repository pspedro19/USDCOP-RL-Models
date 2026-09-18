"""Reproduce archived selection posteriors before any OHLC representation diagnostic.

No strategy evaluation, fitted parameters, production export, network or secrets.
Run in a dedicated process: archived macro dependency injection is temporary and
is deliberately NOT an application-wide/live feature builder.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research import regime_hmm  # noqa: E402
from src.research.historical_hmm_audit import (  # noqa: E402
    HistoricalArchive,
    compare_stored,
    historical_model,
    posterior_path,
    recover_mask,
    representation_delta,
    safe_path,
    sha256,
    shifted_path,
)
from src.research.macro_asof import strict_asof  # noqa: E402

SOURCE_BINDINGS = ("src/research/regime_hmm.py", "src/research/regime_portable.py",
                   "src/research/macro_asof.py")
MODEL = "config/research/regime_hmm_frozen.json"
PORTABLE = "data/thesis/research_data_portable_v2.pkl"
SEED = "seeds/latest/usdcop_m5_ohlcv.parquet"
MACRO = "data/pipeline/04_cleaning/output/MACRO_RESEARCH_v2.parquet"
AVAILABILITY = "config/research/macro_availability.yaml"
MASK = "config/research/evaluation_mask.json"


def observations(archive, prices, valid):
    archive.read(MACRO)
    archive.read(AVAILABILITY)

    def archived_asof(targets, series, *, name):
        return strict_asof(targets, series, name=name,
                           availability_path=archive.object_path(AVAILABILITY))

    with patch.object(regime_hmm, "MACRO_CLEAN", archive.object_path(MACRO)), \
            patch.object(regime_hmm, "strict_asof", archived_asof):
        return regime_hmm.build_regime_observations(prices, valid_sessions=valid)


def filtered(model, obs):
    if list(obs.columns) != list(model.feature_names):
        raise ValueError("historical feature order mismatch")
    clean = obs.dropna()
    values, diagnostics = posterior_path(model, clean.to_numpy())
    frame = pd.DataFrame(values, index=clean.index,
                         columns=[f"state_{i}" for i in model.vol_order])
    shifted, source_dates = shifted_path(frame, min_context=60)
    diagnostics.update({"first_observation": str(clean.index[0].date()),
                        "last_observation": str(clean.index[-1].date()),
                        "excluded_incomplete_observation_dates": [str(x.date()) for x in
                                                                   obs.index.difference(clean.index)]})
    return shifted, source_dates, diagnostics


def audit(snapshot: Path, expected_sha256: str, *, repo: Path = ROOT):
    archive = HistoricalArchive(snapshot, expected_sha256)
    archive.bind_current_code(repo, SOURCE_BINDINGS)
    # Only trusted LOCAL project archives: pickle is executable. Verify bytes FIRST.
    # The blob contains all partitions; only selection contexts are evaluated here.
    blob = pickle.loads(archive.read(PORTABLE))
    model, model_report = historical_model(archive.json(MODEL), blob,
                                           model_sha256=archive.entries[MODEL]["sha256"])
    if model.k != 5:
        raise ValueError("C043 scope is the archived K5 model, not another candidate")
    specs = blob["selection"]
    dates = pd.DatetimeIndex([s.date for s in specs])
    if len(dates) != 226 or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("archived selection cohort must have 226 unique ordered dates")
    contexts = [np.asarray(s.context) for s in specs]
    if any(x.shape != (7,) or x.dtype != np.float32 for x in contexts):
        raise ValueError("archived context must have seven float32 coordinates")
    stored = np.vstack([x[-4:] for x in contexts])
    prices = pd.read_parquet(io.BytesIO(archive.read(SEED)))
    if "symbol" in prices:
        prices = prices[prices["symbol"].astype(str).str.upper().str.replace(
            "/", "", regex=False).eq("USDCOP")].copy()
    times = pd.to_datetime(prices["time"])
    valid, train_valid = recover_mask(archive.json(MASK), times.dt.date.unique())
    if not set(dates.date).issubset(valid):
        raise ValueError("selection date absent from archived valid mask")
    # No observations after the last selection session enter either forward scan.
    prices = prices[times.dt.date <= dates[-1].date()].copy()
    selection_prices = prices[pd.to_datetime(prices["time"]).dt.date.isin(set(dates.date))]
    ohlc = selection_prices[["open", "high", "low", "close"]].astype(float)
    flat_fraction = float(ohlc.eq(ohlc["close"], axis=0).all(axis=1).mean())
    original_obs = observations(archive, prices, valid)
    baseline, sources, diagnostics = filtered(model, original_obs)
    parity = compare_stored(baseline, dates, stored)
    parity["rows"] = [{"date": str(d.date()),
                       "observation_date": None if pd.isna(sources.get(d)) else str(sources[d].date()),
                       "max_abs_error": None if d not in baseline.index or
                       not np.isfinite(baseline.loc[d].to_numpy()[:4]).all() else
                       float(np.max(np.abs(baseline.loc[d].to_numpy()[:4] - x)))}
                      for d, x in zip(dates, stored, strict=True)]
    report = {
        "contract": "THESIS-HISTORICAL-HMM-DIAGNOSTIC-1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "retrospective_parameter_fixed_selection_only_not_confirmatory",
        "snapshot_sha256": archive.manifest_sha256,
        "runtime": {"python": platform.python_version(), **{x: version(x) for x in
                                                            ("numpy", "pandas", "scipy")}},
        "model": model_report,
        "cohort": {"partition": "selection", "n_sessions": len(dates),
                   "first": str(dates[0].date()), "last": str(dates[-1].date()),
                   "n_bars_selection": len(selection_prices),
                   "flat_ohlc_fraction_selection": flat_fraction,
                   "not_evaluated_cohorts": ["holdout", "2026", "forward"],
                   "archived_valid_count": len(valid), "archived_train_valid_count": len(train_valid)},
        "transform_contract": {"mask_before_daily_aggregation": True, "joint_dropna_features": 9,
                               "minimum_context": 60, "shift": "one_row_on_joint_clean_index",
                               "macro_join": "archived_strict_asof_not_publication_v2",
                               "filter_start": "first_complete_observation_not_fit_range"},
        "baseline_parity": parity, "baseline_numerics": diagnostics,
        "counterfactual": None, "status": "BASELINE_PARITY_FAILED",
        "proves": ["numerical_reproduction_only_if_baseline_parity_passes"],
        "does_not_prove": ["historical_point_in_time_availability", "official_source_authenticity",
                           "original_training_lineage", "PPO_checkpoint_lineage", "tradable_alpha",
                           "causal_share_of_vendor_bias", "approval_of_future_K_or_schema",
                           "sensitivity_in_2026_or_other_unmeasured_representations"],
        "network_used": False, "secrets_read": False, "training_performed": False,
        "strategy_returns_evaluated": False, "current_model_eligible": False,
    }
    if parity["passed"] and not diagnostics["fallback_rows"]:
        flattened_prices = prices.copy()
        for col in ("open", "high", "low"):
            flattened_prices[col] = flattened_prices["close"]
        flattened_obs = observations(archive, flattened_prices, valid)
        other, other_sources, other_diagnostics = filtered(model, flattened_obs)
        same_history = baseline.index.equals(other.index) and sources.equals(other_sources)
        report["counterfactual_numerics"] = other_diagnostics
        report["history_alignment"] = {
            "passed": same_history,
            "original_only_dates": [str(d.date()) for d in baseline.index.difference(other.index)],
            "flattened_only_dates": [str(d.date()) for d in other.index.difference(baseline.index)]}
        if same_history and not other_diagnostics["fallback_rows"]:
            result = representation_delta(baseline, other, dates, baseline_passed=True)
            result["intervention"] = "replace_each_bar_OHL_by_its_unchanged_close_in_entire_prefix"
            result["same_fitted_parameters"] = True
            original_clean, flattened_clean = original_obs.dropna(), flattened_obs.dropna()
            result["feature_changes_on_clean_history"] = {
                c: {"max_abs_difference": float((original_clean[c] - flattened_clean[c]).abs().max()),
                    "n_rows_different": int(((original_clean[c] - flattened_clean[c]).abs() > 1e-12).sum())}
                for c in model.feature_names}
            report["counterfactual"] = result
            report["status"] = "DIAGNOSTIC_REPRODUCED_NOT_SCIENTIFIC_CLOSURE"
        else:
            report["status"] = "COUNTERFACTUAL_HISTORY_OR_NUMERICS_BLOCKED"
    elif parity["passed"]:
        report["status"] = "BASELINE_NUMERICAL_FALLBACK_BLOCKED"
    archive.bind_current_code(repo, SOURCE_BINDINGS)
    # Detect archive changes while the dedicated-process injection was in use.
    for key in list(archive.used):
        archive.read(key)
    report["inputs_sha256"] = dict(sorted(archive.used.items()))
    report["runner_sha256"] = sha256(Path(__file__).read_bytes())
    report["helper_sha256"] = sha256((repo / "src/research/historical_hmm_audit.py").read_bytes())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--expected-snapshot-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = safe_path(args.output)
    if not output.is_relative_to(ROOT) or output.exists():
        parser.error("output must be a NEW file within this repository")
    report = audit(args.snapshot, args.expected_snapshot_sha)
    raw = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(raw)
    print(json.dumps({"status": report["status"], "output": str(output),
                      "baseline_parity": {k: v for k, v in report["baseline_parity"].items() if k != "rows"},
                      "counterfactual": {k: v for k, v in (report["counterfactual"] or {}).items()
                                         if k not in {"rows", "feature_changes_on_clean_history"}}}))
    return 0 if report["status"] == "DIAGNOSTIC_REPRODUCED_NOT_SCIENTIFIC_CLOSURE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
