"""Export the USD/COP forward-macro weekly OOS audit to one Excel workbook."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.weekly_forecasting_oos_canonical import build_frame

REPORTS = ROOT / "reports"
PIT_PATH = (
    ROOT / "data" / "pipeline" / "04_cleaning" / "output"
    / "USDCOP_FORWARD_MACRO_PIT.parquet"
)
MANIFEST_DIR = (
    ROOT / "data" / "pipeline" / "02_scrapers" / "storage" / "manifests"
)


def _read(prefix: str, suffix: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS / f"{prefix}_{suffix}.csv")


def _half_life_key(value: object) -> str:
    text = str(value).strip().lower()
    return text[:-2] if text.endswith(".0") else text


def selected_predictions(prefix: str, variant: str) -> pd.DataFrame:
    predictions = _read(prefix, "all_predictions")
    selections = _read(prefix, "selections")
    chosen: list[pd.DataFrame] = []
    for row in selections.itertuples(index=False):
        part = predictions[
            predictions["horizon"].eq(int(row.horizon))
            & predictions["half_life"].map(_half_life_key).eq(
                _half_life_key(row.half_life)
            )
        ].copy()
        part["threshold"] = float(row.threshold)
        part["prediction"] = (part["prob_up"] >= float(row.threshold)).astype(int)
        part["correct"] = part["prediction"].eq(part["actual"].astype(int))
        part["variant"] = variant
        chosen.append(part)
    result = pd.concat(chosen, ignore_index=True)
    result["origin_date"] = pd.to_datetime(result["origin_date"])
    result["target_date"] = pd.to_datetime(result["target_date"])
    return result


def wilson_interval(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return math.nan, math.nan
    p = correct / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return centre - margin, centre + margin


def add_intervals(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    intervals = [
        wilson_interval(int(round(row.da * row.n_weeks)), int(row.n_weeks))
        for row in result.itertuples(index=False)
    ]
    result["da_ci95_low"] = [item[0] for item in intervals]
    result["da_ci95_high"] = [item[1] for item in intervals]
    return result


def comparison_table(prefixes: dict[str, str]) -> pd.DataFrame:
    frames = []
    for variant, prefix in prefixes.items():
        frame = add_intervals(_read(prefix, "summary"))
        frame.insert(0, "variant", variant)
        frames.append(frame)
    long = pd.concat(frames, ignore_index=True)
    index = ["horizon", "period"]
    metrics = [
        "n_weeks", "da", "balanced_da", "brier", "historical_majority_da",
        "da_lift_vs_historical_majority", "always_down_da", "always_up_da",
        "best_constant_in_period_da", "pred_up_rate", "actual_up_rate",
        "up_recall", "down_recall", "da_ci95_low", "da_ci95_high",
    ]
    wide = long.pivot_table(index=index, columns="variant", values=metrics).reset_index()
    wide.columns = [
        "_".join(str(item) for item in column if str(item))
        if isinstance(column, tuple) else str(column)
        for column in wide.columns
    ]
    if {"da_baseline", "da_macro_pit"}.issubset(wide.columns):
        wide["da_delta_macro_pit_vs_baseline"] = wide["da_macro_pit"] - wide["da_baseline"]
        wide["balanced_delta_macro_pit_vs_baseline"] = (
            wide["balanced_da_macro_pit"] - wide["balanced_da_baseline"]
        )
        wide["brier_delta_macro_pit_vs_baseline"] = (
            wide["brier_macro_pit"] - wide["brier_baseline"]
        )
    return wide.sort_values(index).reset_index(drop=True)


def paired_audit(
    baseline: pd.DataFrame, macro: pd.DataFrame, *, bootstrap_reps: int = 5000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["week", "horizon", "origin_date", "target_date", "actual"]
    paired = baseline.merge(
        macro,
        on=keys,
        suffixes=("_baseline", "_macro_pit"),
        validate="one_to_one",
    )
    paired["year"] = paired["week"].astype(str).str[:4].astype(int)
    rows = []
    rng = np.random.default_rng(20260721)
    for (horizon, year), part in paired[paired["year"].isin([2025, 2026])].groupby(
        ["horizon", "year"]
    ):
        diff = (
            part["correct_macro_pit"].astype(int).to_numpy()
            - part["correct_baseline"].astype(int).to_numpy()
        )
        n = len(diff)
        block = max(1, int(math.ceil(int(horizon) / 5)))
        draws = np.empty(bootstrap_reps)
        for index in range(bootstrap_reps):
            starts = rng.integers(0, n, size=math.ceil(n / block))
            sampled = np.concatenate(
                [(np.arange(start, start + block) % n) for start in starts]
            )[:n]
            draws[index] = diff[sampled].mean()
        macro_only = int(
            (part["correct_macro_pit"] & ~part["correct_baseline"]).sum()
        )
        baseline_only = int(
            (~part["correct_macro_pit"] & part["correct_baseline"]).sum()
        )
        discordant = macro_only + baseline_only
        mcnemar_p = (
            float(binomtest(macro_only, discordant, 0.5).pvalue)
            if discordant else 1.0
        )
        rows.append(
            {
                "horizon": int(horizon),
                "year": int(year),
                "n_weeks": n,
                "overlap_block_weeks": block,
                "baseline_da": float(part["correct_baseline"].mean()),
                "macro_pit_da": float(part["correct_macro_pit"].mean()),
                "paired_da_delta": float(diff.mean()),
                "delta_block_bootstrap_ci95_low": float(np.quantile(draws, 0.025)),
                "delta_block_bootstrap_ci95_high": float(np.quantile(draws, 0.975)),
                "bootstrap_probability_delta_gt_0": float((draws > 0).mean()),
                "macro_only_correct": macro_only,
                "baseline_only_correct": baseline_only,
                "mcnemar_exact_p": mcnemar_p,
            }
        )
    keep = [
        "week", "horizon", "origin_date", "target_date", "actual", "year",
        "half_life_baseline", "threshold_baseline", "prob_up_baseline",
        "prediction_baseline", "correct_baseline", "half_life_macro_pit",
        "threshold_macro_pit", "prob_up_macro_pit", "prediction_macro_pit",
        "correct_macro_pit",
    ]
    return pd.DataFrame(rows), paired[keep].sort_values(["horizon", "origin_date"])


def feature_coverage() -> pd.DataFrame:
    frame, _, candidates = build_frame(include_forward_pit=True, promotion_only=False)
    pit_features = [item for item in candidates if item.startswith("pit_")]
    dates = pd.to_datetime(frame["date"])
    rows = []
    for feature in pit_features:
        non_null = frame[feature].notna()
        valid_dates = dates[non_null]
        for year in range(2020, 2027):
            mask = dates.dt.year.eq(year)
            rows.append(
                {
                    "feature": feature,
                    "year": year,
                    "rows": int(mask.sum()),
                    "non_null_rows": int((mask & non_null).sum()),
                    "coverage": float(frame.loc[mask, feature].notna().mean()) if mask.any() else math.nan,
                    "first_available": valid_dates.min(),
                    "last_available": valid_dates.max(),
                }
            )
    return pd.DataFrame(rows)


def source_coverage() -> pd.DataFrame:
    pit = pd.read_parquet(PIT_PATH)
    pit["observation_date"] = pd.to_datetime(pit["observation_date"])
    result = (
        pit.groupby(["source", "pit_vintage", "promotion_eligible"], dropna=False)
        .agg(
            rows=("series_id", "size"),
            series=("series_id", "nunique"),
            observations=("observation_date", "nunique"),
            first_observation=("observation_date", "min"),
            last_observation=("observation_date", "max"),
        )
        .reset_index()
    )
    return result


def manifest_history() -> pd.DataFrame:
    rows = []
    for path in sorted(MANIFEST_DIR.glob("usdcop-forward-macro-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        rows.append(
            {
                "run_id": payload.get("run_id"),
                "status": payload.get("status"),
                "sources": ",".join(payload.get("sources", [])),
                "start": payload.get("start"),
                "end": payload.get("end"),
                "rows_accepted": payload.get("rows_accepted"),
                "series_count": payload.get("series_count"),
                "documents_seen": payload.get("documents_seen"),
                "error_count": payload.get("error_count"),
                "sha256": payload.get("sha256"),
                "manifest": str(path.relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows)


def build(output: Path) -> Path:
    primary = {
        "baseline": "weekly_nested_baseline_20260721",
        "macro_pit": "weekly_nested_forward_pit_20260721",
        "strict_pit": "weekly_nested_forward_pit_strict_20260721",
    }
    multiyear = {
        "baseline": "weekly_nested_multiyear_baseline_20260721",
        "macro_pit": "weekly_nested_multiyear_forward_pit_20260721",
    }
    primary_comparison = comparison_table(primary)
    multiyear_comparison = comparison_table(multiyear)
    baseline_predictions = selected_predictions(primary["baseline"], "baseline")
    macro_predictions = selected_predictions(primary["macro_pit"], "macro_pit")
    paired_summary, paired_predictions = paired_audit(
        baseline_predictions, macro_predictions
    )
    selections = []
    for protocol, prefixes in (("primary_2024", primary), ("multiyear_2022_2024", multiyear)):
        for variant, prefix in prefixes.items():
            frame = _read(prefix, "selections")
            frame.insert(0, "variant", variant)
            frame.insert(0, "protocol", protocol)
            selections.append(frame)
    selection_table = pd.concat(selections, ignore_index=True)

    readme = pd.DataFrame(
        [
            ("Scope", "USD/COP weekly directional forecasts; H1/H5/H10/H15/H20/H25/H30 kept separate."),
            ("Primary protocol", "Features selected with labels maturing before 2024; 2024 tunes memory/threshold; 2025 locked OOS; 2026 expanding weekly retraining."),
            ("Robustness protocol", "Features frozen before 2022; 2022-2024 tune memory/threshold; 2025 locked OOS; 2026 expanding weekly retraining."),
            ("PIT contract", "Each feature is joined backward on available_at at 16:00 America/Bogota; daily values expire after 10 days and monthly values after 75 days."),
            ("Baseline meaning", "Historical-majority uses only the matured training labels available at each origin. Always-up/down and best in-period constant are reported separately."),
            ("Uncertainty", "DA has Wilson 95% intervals. Paired DA deltas use circular moving-block bootstrap with ceil(horizon/5) weekly blocks; McNemar is supplementary."),
            ("Promotion warning", "SFC Open Data and conservative-release rows are research-grade, not promotion eligible. strict_pit is the sensitivity excluding them."),
            ("No pooling", "Metrics are never pooled across horizons; 2026 sample counts fall with horizon because only realized targets are scored."),
        ],
        columns=["item", "detail"],
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    frames = {
        "README": readme,
        "primary_comparison": primary_comparison,
        "paired_significance": paired_summary,
        "multiyear_comparison": multiyear_comparison,
        "feature_selections": selection_table,
        "weekly_predictions": paired_predictions,
        "feature_coverage": feature_coverage(),
        "source_coverage": source_coverage(),
        "ingestion_manifests": manifest_history(),
    }
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet, frame in frames.items():
            export = frame.copy()
            for column in export.columns:
                if isinstance(export[column].dtype, pd.DatetimeTZDtype):
                    export[column] = export[column].dt.tz_localize(None)
            export.to_excel(writer, index=False, sheet_name=sheet)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cells in sheet.columns:
                width = min(max(max(len(str(cell.value or "")) for cell in cells) + 2, 11), 44)
                sheet.column_dimensions[cells[0].column_letter].width = width
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="reports/usdcop_forward_macro_weekly_oos_audit_20260721.xlsx",
    )
    arguments = parser.parse_args()
    print(build(ROOT / arguments.output))
