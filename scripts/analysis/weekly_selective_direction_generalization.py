"""Nested selective-direction audit with frozen confidence thresholds."""
from pathlib import Path
import math
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[2]
CONFIDENCE = (0.50, 0.55, 0.60, 0.65, 0.70)


def wilson_lower(correct: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = correct / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def evaluate(frame: pd.DataFrame, confidence: float) -> dict:
    selected = frame[(frame.prob_up >= confidence) | (frame.prob_up <= 1 - confidence)].copy()
    if selected.empty:
        return {"n": 0, "coverage": 0.0, "da": 0.0, "balanced_da": 0.0,
                "pred_up_rate": 0.0, "wilson_lower": 0.0}
    pred = (selected.prob_up >= 0.5).astype(int)
    actual = selected.actual.astype(int)
    correct = int((pred == actual).sum())
    both_classes = actual.nunique() > 1 and pred.nunique() > 1
    balanced = balanced_accuracy_score(actual, pred) if both_classes else 0.5
    return {
        "n": len(selected), "coverage": len(selected) / len(frame),
        "da": correct / len(selected), "balanced_da": balanced,
        "pred_up_rate": pred.mean(), "wilson_lower": wilson_lower(correct, len(selected)),
    }


def main() -> None:
    raw = pd.read_csv(ROOT / "reports/weekly_nested_generalization_all_predictions.csv")
    summaries, selections = [], []
    for horizon, hframe in raw.groupby("horizon"):
        trials = []
        for half_life, variant in hframe.groupby("half_life"):
            validation = variant[variant.week.str.startswith("2024")]
            for confidence in CONFIDENCE:
                m = evaluate(validation, confidence)
                if m["n"] < 20 or m["pred_up_rate"] in (0.0, 1.0):
                    continue
                score = m["wilson_lower"] + 0.25 * (m["balanced_da"] - 0.5)
                trials.append((score, m["coverage"], half_life, confidence))
        if not trials:
            continue
        best = max(trials, key=lambda x: (x[0], x[1], -x[3]))
        _, _, selected_half_life, selected_confidence = best
        chosen = hframe[hframe.half_life == selected_half_life]
        selections.append({
            "horizon": horizon, "half_life": selected_half_life,
            "confidence": selected_confidence, "trial_count": len(trials),
            "validation_score_2024": best[0],
        })
        for period, prefix in (
            ("2024_VALIDATION", "2024"), ("2025_OOS", "2025"),
            ("2026_FORWARD", "2026"),
        ):
            m = evaluate(chosen[chosen.week.str.startswith(prefix)], selected_confidence)
            summaries.append({
                "horizon": horizon, "period": period,
                "half_life": selected_half_life, "confidence": selected_confidence,
                **m,
            })
    report = ROOT / "reports"
    pd.DataFrame(selections).to_csv(report / "weekly_selective_direction_selections.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(report / "weekly_selective_direction_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
