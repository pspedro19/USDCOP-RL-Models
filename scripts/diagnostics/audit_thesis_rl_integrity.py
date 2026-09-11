"""Reproduce EXP-TESIS-RL-01 audit evidence without fitting or changing a policy.

Reads an explicit allowlist of local research artifacts. Perturbations are in memory.
Published holdout returns are inspected retrospectively, never used for selection.
An exit code of 0 means the measurements completed, NOT that research gates passed.
Use --output <new.json> to save; existing files are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import pickle
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
SEED = "seeds/latest/usdcop_m5_ohlcv.parquet"
MACRO = "data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet"
PORTABLE = "data/thesis/research_data_portable.pkl"
BLOCKS = ("development", "selection", "holdout")
CONFIGS = ("ppo_regime", "ppo_backbone")
SEEDS = (42, 123, 456, 789, 1337)
HASHES: dict[str, str] = {}


def source(relative: str) -> Path:
    path = ROOT / relative
    if relative not in HASHES:
        with path.open("rb") as fh:
            HASHES[relative] = hashlib.file_digest(fh, "sha256").hexdigest()
    return path


def read_json(relative: str):
    return json.loads(source(relative).read_text(encoding="utf-8"))


def standalone(relative: str, name: str):
    """Load the actual pure module, avoiding unrelated application initializers."""
    spec = importlib.util.spec_from_file_location(name, source(relative))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class AuditSession:
    """Data-only replacement for the pickle's SessionSpec; no model is loaded."""
    date: object
    close: np.ndarray
    market: np.ndarray
    context: np.ndarray
    spread_pips: float


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("src.research.session_gym", "SessionSpec"):
            return AuditSession
        allowed = {
            ("datetime", "date"): date,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        }
        core = "numpy._core" if int(np.__version__.split(".")[0]) >= 2 else "numpy.core"
        multiarray = importlib.import_module(core + ".multiarray")
        numeric = importlib.import_module(core + ".numeric")
        for prefix in ("numpy.core", "numpy._core"):
            allowed[(prefix + ".multiarray", "_reconstruct")] = multiarray._reconstruct
            allowed[(prefix + ".multiarray", "scalar")] = multiarray.scalar
            allowed[(prefix + ".numeric", "_frombuffer")] = numeric._frombuffer
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unapproved pickle global: {module}.{name}")
        return allowed[(module, name)]


def pct_metrics(returns: np.ndarray, operations: int) -> dict:
    returns = np.asarray(returns, dtype=float)
    sd = float(returns.std(ddof=1))
    return {
        "n_sessions": len(returns), "n_changes_excluding_terminal": operations,
        "sum_return_pct": float(100 * returns.sum()),
        "compounded_return_pct": float(100 * (np.prod(1 + returns) - 1)),
        # Match the existing thesis annualizer for reproduction, not endorsement.
        "sharpe_221": float(np.sqrt(221) * returns.mean() / sd)
        if operations >= 20 and sd > 0 else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("output exists; choose a new path (audit evidence is immutable)")
    if args.output and not args.output.resolve().is_relative_to(ROOT):
        parser.error("output must be inside this workspace")

    feats_mod = standalone("src/research/features.py", "thesis_audit_features")
    mask_mod = standalone("src/research/evaluation_mask.py", "thesis_audit_mask")
    source("config/trading_calendar.json")
    part = yaml.safe_load(source("config/research/partition.yaml").read_text(encoding="utf-8"))
    seed = pd.read_parquet(source(SEED)).sort_values("time").reset_index(drop=True)
    t = pd.to_datetime(seed.time)
    ohlc = seed[["open", "high", "low", "close"]]
    flat = ohlc.nunique(axis=1).eq(1)
    mask = mask_mod.build_mask(source(SEED))
    valid = set(mask.valid)
    daily = {str(d): g.close.to_numpy(dtype=float)
             for d, g in seed.assign(day=t.dt.date).groupby("day")}
    quality = {
        "rows": len(seed), "first": str(t.min()), "last": str(t.max()),
        "dtypes": seed.dtypes.astype(str).to_dict(),
        "null_ohlc": int(ohlc.isna().sum().sum()),
        "nonfinite_ohlc": int((~np.isfinite(ohlc)).sum().sum()),
        "nonpositive_ohlc": int((ohlc <= 0).sum().sum()),
        "duplicate_time_symbol": int(seed.duplicated(["time", "symbol"]).sum()),
        "invalid_ohlc": int(((seed.high < ohlc.max(axis=1))
                             | (seed.low > ohlc.min(axis=1))).sum()),
        "off_grid": int(((t.dt.minute % 5 != 0) | (t.dt.second != 0)
                         | (t.dt.microsecond != 0)).sum()),
        "off_session": int(((t.dt.hour < 8) | (t.dt.hour > 12)
                            | (t.dt.dayofweek > 4)).sum()),
        "flat_ohlc_pct": float(100 * flat.mean()),
        "nonzero_volume": int(seed.volume.ne(0).sum()),
        "by_year": {}, "mask_sha256": mask.sha256,
        "mask_counts": {}, "official_source_equality": "NOT_CERTIFIED",
    }
    for year in sorted(t.dt.year.unique()):
        take = t.dt.year.eq(year)
        quality["by_year"][str(year)] = {
            "bars": int(take.sum()), "flat_ohlc_pct": float(100 * flat[take].mean())}
    for block in BLOCKS:
        b = part["blocks"][block]
        quality["mask_counts"][block] = len(mask.in_block(b["start"], b["end"]))
    daily_reference = pd.read_parquet(source("seeds/latest/usdcop_daily_ohlcv.parquet")).set_index("time")
    aggregated = seed.groupby(t.dt.date).agg(open=("open", "first"), high=("high", "max"),
                                             low=("low", "min"), close=("close", "last"))
    aggregated.index = pd.to_datetime(aggregated.index)
    common = aggregated.index.intersection(daily_reference.index)
    difference = (aggregated.loc[common] - daily_reference.loc[common, aggregated.columns]).abs()
    quality["daily_reference_vs_intraday_aggregation"] = {
        "common_dates": len(common), "exact_all_ohlc_dates": int(difference.eq(0).all(axis=1).sum()),
        "max_abs_per_column": difference.max().to_dict(),
        "interpretation": "Internal reconciliation; not independent official-source validation.",
    }

    # Recompute the market vector on the complete raw series so this diagnostic
    # can compare legacy v1 portable sessions even when mask v2 excludes them.
    # This is retrospective evidence only; training/evaluation still consumes
    # the v2 mask.
    features = feats_mod.build_market_features(seed)
    market_names = [f for group in ("precio", "volatilidad", "tendencia", "temporal")
                    for f in feats_mod.GROUPS[group]]
    frozen = read_json("config/research/feature_scaler_frozen.json")
    mean, scale = np.asarray(frozen["mean"]), np.asarray(frozen["scale"])
    with source(PORTABLE).open("rb") as fh:
        portable = DataOnlyUnpickler(fh).load()
    raw_macro = pd.read_parquet(source(MACRO)).sort_index()
    brent_backup = pd.read_csv(source("data/backups/macro_fixes/brent_corrupt_20250925_20251219.csv"),
                              parse_dates=["fecha"]).set_index("fecha")
    macro_audit = {
        "rows": len(raw_macro), "columns": len(raw_macro.columns),
        "first": str(raw_macro.index.min()), "last": str(raw_macro.index.max()),
        "duplicate_dates": int(raw_macro.index.duplicated().sum()),
        "weekend_dates_since_2020": int(((raw_macro.index >= "2020-01-01")
                                         & (raw_macro.index.dayofweek > 4)).sum()),
        "brent_patch_rows": len(brent_backup),
        "brent_patch_vs_backup_max_abs": float((raw_macro.loc[brent_backup.index,
            "COMM_OIL_BRENT_GLB_D_BRENT"] - brent_backup.fred).abs().max()),
        "patch_warning": "Local backup parity only; code documents spot replacing futures.",
    }
    macro_actual = feats_mod.attach_macro_features(valid)
    feature_audit = {"frozen_scaler_vs_portable_max_abs": float(max(
        np.abs(mean - portable["scaler_mean"]).max(),
        np.abs(scale - portable["scaler_scale"]).max())), "blocks": {}}
    for block in BLOCKS:
        specs = portable[block]
        dates = [str(s.date) for s in specs]
        rows = features[features._session.astype(str).isin(dates)]
        recomputed = np.clip((rows[market_names].to_numpy() - mean) / scale, -5, 5)
        observed = np.concatenate([s.market for s in specs])
        expected_frame = macro_actual.reindex(pd.to_datetime(dates))
        available = expected_frame.notna().all(axis=1).to_numpy()
        expected_ctx = expected_frame.to_numpy(dtype=np.float32)
        observed_ctx = np.stack([s.context[:3] for s in specs])
        unclipped = (rows[market_names].to_numpy() - mean) / scale
        clip_rates = (np.abs(unclipped) >= 5).mean(axis=0) * 100
        raw = rows[market_names].to_numpy()
        b = part["blocks"][block]
        missing = sorted({str(d) for d in mask.in_block(b["start"], b["end"])} - set(dates))
        feature_audit["blocks"][block] = {
            "sessions": len(specs), "first": dates[0], "last": dates[-1],
            "dropped_dates_after_mask": missing,
            "market_vs_portable_max_abs": float(np.max(np.abs(recomputed.astype(np.float32) - observed))),
            # v1 portable artefacts may contain dates that the corrected strict
            # macro rule now rejects; compare only the causally available rows
            # and publish the remainder as a retrospective compatibility gap.
            "macro_vs_portable_max_abs": float(
                np.max(np.abs(expected_ctx[available, :3] - observed_ctx[available]))
            ) if available.any() else None,
            "macro_unavailable_under_strict_rule": int((~available).sum()),
            "clip_pct_per_feature": dict(zip(market_names, clip_rates.tolist(), strict=True)),
            "exact_zero_pct_per_feature": dict(zip(market_names, (100 * (raw == 0).mean(axis=0)).tolist(), strict=True)),
            "macro_std": observed_ctx.std(axis=0).tolist(),
        }

    # A causal input dated d must not respond to a value unavailable at open(d).
    day = pd.Timestamp("2023-06-15")
    macro_columns = ["COMM_OIL_BRENT_GLB_D_BRENT", "FXRT_INDEX_DXY_USA_D_DXY",
                     "FINC_RATE_IBR_OVERNIGHT_COL_D_IBR", "FINC_BOND_YIELD2Y_USA_D_DGS2"]
    mutated = raw_macro.copy()
    mutated.loc[day, macro_columns[:2]] *= 1.1
    mutated.loc[day, macro_columns[2]] += 1
    with patch.object(feats_mod.pd, "read_parquet", return_value=raw_macro):
        before = feats_mod.attach_macro_features([day]).iloc[0]
    with patch.object(feats_mod.pd, "read_parquet", return_value=mutated):
        after = feats_mod.attach_macro_features([day]).iloc[0]
    dxy = raw_macro[macro_columns[1]].dropna()
    dxy_prior = dxy.loc[dxy.index < day]
    dxy_expected = float(np.log(dxy_prior.iloc[-1] / dxy_prior.iloc[-2]))
    with patch.object(feats_mod.Path, "is_file", return_value=False):
        try:
            feats_mod.attach_macro_features([day])
        except FileNotFoundError:
            missing_macro = None
        else:  # pragma: no cover - fail-closed contract regression
            missing_macro = "unexpected_success"
    feature_audit["macro_same_day_perturbation"] = {
        "date": str(day.date()), "before": before.to_dict(), "after": after.to_dict(),
        "max_abs_change": float((after - before).abs().max()),
        "causality_gate_pass": bool(np.allclose(before, after, rtol=0, atol=1e-12)),
        "dxy_expected_previous_observation_return": dxy_expected,
        "missing_file_fails_closed": missing_macro is None,
        "note": "Previous observation alone does not certify publication time or vintage.",
    }
    first_bars = features.groupby("_session").head(1)
    feature_audit["opening_logret_nonzero_sessions"] = int(first_bars.logret_1.ne(0).sum())
    dev_dates = set(mask.in_block(part["blocks"]["development"]["start"],
                                part["blocks"]["development"]["end"]))
    dev_rows = features[features._session.isin(dev_dates)][market_names]
    recalculated_scale = dev_rows.std(ddof=0).to_numpy()
    recalculated_scale[recalculated_scale == 0] = 1
    feature_audit["fresh_dev_scaler_vs_frozen_max_abs"] = float(max(
        np.abs(dev_rows.mean().to_numpy() - mean).max(),
        np.abs(recalculated_scale - scale).max()))

    results = {}
    # decomposition_selection belongs to a refit diagnostic and is not the original
    # out-of-sample selection evaluation. Never silently combine those populations.
    for block in ("holdout",):
        decomp = read_json(f"outputs/thesis/decomposition_{block}.json")
        by_config, means = {}, {}
        common_dates = None
        for config in CONFIGS:
            gross, costs, net, operations, per_seed = [], [], [], [], {}
            directions = []
            max_gross_error = 0.0
            for seed_id in SEEDS:
                run = decomp["runs"][f"{config}_seed{seed_id}"]
                sessions = run["sessions"]
                dates = [s["date"] for s in sessions]
                if dates != sorted(set(dates)):
                    raise ValueError("duplicate or unordered decomposition dates")
                if common_dates is None:
                    common_dates = dates
                elif common_dates != dates:
                    raise ValueError("seed/config dates differ; cannot average positionally")
                g = np.array([s["gross_return"] for s in sessions])
                c = np.array([s["total_cost"] for s in sessions])
                n = np.array([s["daily_return"] for s in sessions])
                ops = sum(s["n_changes"] for s in sessions)
                direction = 0.0
                for session in sessions:
                    close = daily[session["date"]]
                    w = np.asarray(session["weights"])
                    r = close[1:] / close[:-1] - 1
                    max_gross_error = max(max_gross_error, abs(float(w @ r) - session["gross_return"]))
                    direction += w.mean() * r.sum()
                directions.append(direction)
                gross.append(g)
                costs.append(c)
                net.append(n)
                operations.append(ops)
                # The opening script writes refit holdout results into the original
                # model's JSON and identifies the producing model with this field.
                artifact = read_json(f"data/thesis/ppo/{config}_seed{seed_id}.json")
                if artifact.get("holdout_models") != "refit":
                    raise ValueError("holdout producer is not the expected refit model")
                evaluation = artifact[block]
                if evaluation["dates"] != dates:
                    raise ValueError("decomposition and evaluation dates differ")
                per_seed[str(seed_id)] = {
                    "gross": pct_metrics(g, ops), "net": pct_metrics(n, ops),
                    "net_identity_max_abs": float(np.max(np.abs(g - c - n))),
                    "evaluation_returns_max_abs": float(np.max(np.abs(n - evaluation["daily_returns"]))),
                }
            g, c, n = np.mean(gross, axis=0), np.mean(costs, axis=0), np.mean(net, axis=0)
            means[config] = n
            by_config[config] = {
                "per_seed": per_seed,
                "mean_seed_series_gross": pct_metrics(g, sum(operations)),
                "mean_seed_series_net": pct_metrics(n, sum(operations)),
                "published_weights_gross_identity_max_abs": max_gross_error,
                "stress_frozen_actions": {
                    str(k): pct_metrics(g - k * c, sum(operations)) for k in (1, 2, 3)},
                "ex_post_direction_sum_pct": float(100 * np.mean(directions)),
                "ex_post_timing_residual_sum_pct": float(100 * (g.sum() - np.mean(directions))),
                "attribution_warning": "Daily mean exposure uses future actions; not an opening-time trading signal.",
            }
        results[block] = {"configs": by_config,
                          "correlation_mean_seed_net_series": float(np.corrcoef(*means.values())[0, 1])}

    selection, selection_means, selection_dates = {}, {}, None
    for config in CONFIGS:
        runs, rows, ops = {}, [], 0
        for seed_id in SEEDS:
            evaluation = read_json(f"data/thesis/ppo/{config}_seed{seed_id}.json")["selection"]
            if selection_dates is None:
                selection_dates = evaluation["dates"]
            if (evaluation["dates"] != selection_dates
                    or selection_dates != sorted(set(selection_dates))):
                raise ValueError("selection dates are not identical, unique and ordered")
            n = np.array(evaluation["daily_returns"])
            rows.append(n)
            ops += evaluation["n_ops"]
            runs[str(seed_id)] = pct_metrics(n, evaluation["n_ops"])
        selection_means[config] = np.mean(rows, axis=0)
        selection[config] = {"per_seed_net": runs,
                             "mean_seed_series_net": pct_metrics(np.mean(rows, axis=0), ops)}
    results["selection"] = {
        "configs": selection,
        "correlation_mean_seed_net_series": float(np.corrcoef(*selection_means.values())[0, 1]),
        "provenance_note": "Original development-only models; refit selection decomposition excluded.",
    }

    # Published results must be valid interoperable JSON, not JavaScript-style NaN.
    invalid_json = {}
    for block in BLOCKS:
        path = source(f"outputs/thesis/statistics_{block}.json")
        constants = []
        json.loads(path.read_text(encoding="utf-8"), parse_constant=constants.append)
        invalid_json[block] = constants
    for relative in ("scripts/diagnostics/audit_thesis_rl_integrity.py",
                     "src/research/dataset.py", "src/research/session_gym.py",
                     "src/research/session_env.py", "src/research/cost_model.py",
                     "scripts/analysis/thesis_statistics.py", "outputs/thesis/holdout_opening.json",
                     ".claude/specs/planes/06-PRE-REGISTRATION.md",
                     ".claude/specs/assets/usdcop/HYPOTHESIS-REGISTRY.md"):
        source(relative)
    git = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
                         cwd=ROOT, text=True, capture_output=True, check=True)
    evidence = {
        "audit": "EXP-TESIS-RL-01", "measured_at_utc": datetime.now(UTC).isoformat(),
        "base_commit": git.stdout.strip(), "python": sys.version.split()[0],
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "scope": "retrospective audit of published data/returns; no training or policy selection",
        "quality": quality, "macro_data": macro_audit,
        "features": feature_audit, "published_return_reproduction": results,
        "nonstandard_json_constants": invalid_json, "sha256": HASHES,
        "limits": ["No full independent official-source reconciliation or original vintage proof.",
                   "No quote/fill data: executable spread, latency and bid-ask bounce unverified.",
                   "No rerun of the claimed lag/macro-policy counterfactuals; their artifacts were not supplied.",
                   "Bootstrap/PBO values require separate reproduction; this script reports return identities.",
                   "Sharpe of the average return series is not median seed Sharpe or a deployed action ensemble."],
    }
    payload = json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as fh:
            fh.write(payload)
        print(f"Evidence saved: {args.output}")
        print(f"Rows={len(seed)}; macro causality gate={feature_audit['macro_same_day_perturbation']['causality_gate_pass']}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
