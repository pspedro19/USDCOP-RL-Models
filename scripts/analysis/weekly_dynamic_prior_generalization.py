"""Nested evaluation of a dynamically calibrated directional prior."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.analysis.weekly_forecasting_oos_canonical import build_frame

PRIOR_WINDOWS = (20, 60, 120, 252)
SHRINKAGE = 20.0


def logit(x: float) -> float:
    x = float(np.clip(x, 1e-5, 1 - 1e-5))
    return float(np.log(x / (1 - x)))


def expit(x: float) -> float:
    return float(1 / (1 + np.exp(-np.clip(x, -30, 30))))


def main() -> None:
    df, _, _ = build_frame()
    dates = pd.to_datetime(df["date"])
    date_to_idx = {d.date(): i for i, d in enumerate(dates)}
    predictions = pd.read_csv(
        ROOT / "reports/weekly_nested_generalization_all_predictions.csv"
    )
    predictions["origin_date"] = pd.to_datetime(predictions["origin_date"])
    adjusted_rows = []
    for row in predictions.itertuples(index=False):
        horizon = int(row.horizon)
        origin_idx = date_to_idx[row.origin_date.date()]
        future_return = np.log(df["close"].shift(-horizon) / df["close"])
        target = (future_return > 0).where(future_return.notna()).astype(float)
        matured_end = origin_idx - horizon + 1
        full_prior = float(row.train_up_rate)
        for window in PRIOR_WINDOWS:
            start = max(0, matured_end - window)
            recent = target.iloc[start:matured_end].dropna()
            recent_prior = float(
                (recent.sum() + SHRINKAGE * full_prior) / (len(recent) + SHRINKAGE)
            )
            adjusted = expit(logit(row.prob_up) + logit(recent_prior) - logit(full_prior))
            adjusted_rows.append({
                "week": row.week, "horizon": horizon,
                "half_life": row.half_life, "prior_window": window,
                "actual": int(row.actual), "prob_up_raw": row.prob_up,
                "prob_up_adjusted": adjusted, "recent_up_prior": recent_prior,
            })
    adjusted = pd.DataFrame(adjusted_rows)
    summaries = []
    selections = []
    for horizon, hframe in adjusted.groupby("horizon"):
        trials = []
        for (half_life, window), variant in hframe.groupby(["half_life", "prior_window"]):
            val = variant[variant["week"].str.startswith("2024")]
            pred = (val["prob_up_adjusted"] >= 0.5).astype(int)
            da = float((pred == val["actual"]).mean())
            ba = float(balanced_accuracy_score(val["actual"], pred))
            trials.append((0.5 * da + 0.5 * ba, ba, half_life, int(window)))
        best = max(trials, key=lambda x: (x[0], x[1], -x[3]))
        _, _, selected_half_life, selected_window = best
        chosen = hframe[
            (hframe["half_life"] == selected_half_life)
            & (hframe["prior_window"] == selected_window)
        ]
        selections.append({
            "horizon": horizon, "half_life": selected_half_life,
            "prior_window": selected_window, "validation_score_2024": best[0],
            "trial_count": len(trials),
        })
        for period, prefix in (
            ("2024_VALIDATION", "2024"), ("2025_OOS", "2025"),
            ("2026_FORWARD", "2026"),
        ):
            part = chosen[chosen["week"].str.startswith(prefix)]
            pred = (part["prob_up_adjusted"] >= 0.5).astype(int)
            actual = part["actual"].astype(int)
            summaries.append({
                "horizon": horizon, "period": period,
                "half_life": selected_half_life, "prior_window": selected_window,
                "n_weeks": len(part), "da": float((pred == actual).mean()),
                "balanced_da": float(balanced_accuracy_score(actual, pred)),
                "baseline_da": float((actual == 0).mean()),
                "pred_up_rate": float(pred.mean()),
                "actual_up_rate": float(actual.mean()),
            })
    report = ROOT / "reports"
    adjusted.to_csv(report / "weekly_dynamic_prior_predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(
        report / "weekly_dynamic_prior_selections.csv", index=False
    )
    summary = pd.DataFrame(summaries)
    summary.to_csv(report / "weekly_dynamic_prior_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
