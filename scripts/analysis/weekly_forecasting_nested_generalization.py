"""Nested weekly USD/COP generalization audit.

Protocol:
* feature selection: data whose labels mature before 2024;
* model-memory and probability threshold selection: 2024 only;
* locked OOS: 2025;
* expanding weekly retraining and forward validation: 2026.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.weekly_forecasting_oos_canonical import build_frame

HORIZONS = (1, 5, 10, 15, 20, 25, 30)
HALF_LIVES = (None, 252, 504)
THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60)


def make_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    ret = np.log(df["close"].shift(-horizon) / df["close"])
    return (ret > 0).where(ret.notna()).astype(float)


def frozen_features(
    df: pd.DataFrame, candidates: list[str], horizon: int,
    cutoff_date: str = "2024-01-01", k: int = 12,
) -> list[str]:
    target = make_target(df, horizon)
    dates = pd.to_datetime(df["date"])
    cutoff = int(np.searchsorted(dates, pd.Timestamp(cutoff_date)))
    # A feature-selection label is eligible only if its target date is strictly
    # before the cutoff. The former +1 admitted one label maturing on the first
    # validation date.
    eligible = np.arange(0, max(0, cutoff - horizon))
    eligible = eligible[target.iloc[eligible].notna().to_numpy()]
    usable = [c for c in candidates if df.loc[eligible, c].notna().sum() >= 250]
    x = SimpleImputer(strategy="median").fit_transform(df.loc[eligible, usable])
    scores = mutual_info_classif(
        x, target.iloc[eligible].astype(int).to_numpy(), random_state=23
    )
    return [usable[i] for i in np.argsort(scores)[::-1][: min(k, len(usable))]]


def weekly_probabilities(
    df: pd.DataFrame, features: list[str], horizon: int, half_life: int | None,
    origin_start: str = "2024-01-01",
) -> pd.DataFrame:
    dates = pd.to_datetime(df["date"])
    origins = df.assign(week=dates.dt.strftime("%G-W%V")).groupby("week").tail(1)
    origins = origins[(origins["date"] >= origin_start)]
    target = make_target(df, horizon)
    rows = []
    for origin_idx, origin in origins.iterrows():
        target_idx = origin_idx + horizon
        if target_idx >= len(df) or pd.isna(target.iloc[origin_idx]):
            continue
        train = np.arange(0, max(0, origin_idx - horizon + 1))
        train = train[target.iloc[train].notna().to_numpy()]
        if len(train) < 300:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=0.1, max_iter=2000),
        )
        fit_kwargs = {}
        if half_life is not None:
            ages = train[-1] - train
            weights = np.exp(-np.log(2.0) * ages / half_life)
            fit_kwargs["logisticregression__sample_weight"] = weights
        model.fit(
            df.loc[train, features], target.iloc[train].astype(int), **fit_kwargs
        )
        probability = float(model.predict_proba(df.loc[[origin_idx], features])[0, 1])
        rows.append({
            "week": origin["week"], "origin_date": origin["date"],
            "target_date": df.loc[target_idx, "date"], "horizon": horizon,
            "half_life": "expanding" if half_life is None else str(half_life),
            "prob_up": probability, "actual": int(target.iloc[origin_idx]),
            "train_up_rate": float(target.iloc[train].mean()),
        })
    return pd.DataFrame(rows)


def metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    if frame.empty:
        return {
            "n_weeks": 0,
            "da": float("nan"),
            "balanced_da": float("nan"),
            "historical_majority_da": float("nan"),
            "da_lift_vs_historical_majority": float("nan"),
            "always_down_da": float("nan"),
            "always_up_da": float("nan"),
            "best_constant_in_period_da": float("nan"),
            "up_recall": float("nan"),
            "down_recall": float("nan"),
            "brier": float("nan"),
            "pred_up_rate": float("nan"),
            "actual_up_rate": float("nan"),
        }
    pred = (frame["prob_up"] >= threshold).astype(int)
    actual = frame["actual"].astype(int)
    historical_majority = (frame["train_up_rate"] >= 0.5).astype(int)
    always_down = float((actual == 0).mean())
    always_up = float((actual == 1).mean())
    historical_da = float((historical_majority == actual).mean())
    up = actual.eq(1)
    down = actual.eq(0)
    da = float((pred == actual).mean())
    return {
        "n_weeks": int(len(frame)),
        "da": da,
        "balanced_da": float(balanced_accuracy_score(actual, pred)),
        "historical_majority_da": historical_da,
        "da_lift_vs_historical_majority": da - historical_da,
        "always_down_da": always_down,
        "always_up_da": always_up,
        "best_constant_in_period_da": max(always_down, always_up),
        "up_recall": float(pred[up].mean()) if up.any() else float("nan"),
        "down_recall": float((1 - pred[down]).mean()) if down.any() else float("nan"),
        "brier": float(brier_score_loss(actual, frame["prob_up"])),
        "pred_up_rate": float(pred.mean()),
        "actual_up_rate": float(actual.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cutoff", default="2024-01-01")
    parser.add_argument("--validation-years", default="2024")
    parser.add_argument("--output-prefix", default="weekly_nested_generalization")
    parser.add_argument("--promotion-only", action="store_true")
    parser.add_argument("--exclude-forward-pit", action="store_true")
    args = parser.parse_args()
    validation_years = tuple(x.strip() for x in args.validation_years.split(",") if x.strip())
    df, _, candidates = build_frame(
        include_forward_pit=not args.exclude_forward_pit,
        promotion_only=args.promotion_only,
    )
    all_predictions = []
    selections = []
    summaries = []
    for horizon in HORIZONS:
        features = frozen_features(df, candidates, horizon, args.feature_cutoff)
        variant_frames = {}
        for half_life in HALF_LIVES:
            frame = weekly_probabilities(
                df, features, horizon, half_life, args.feature_cutoff
            )
            variant_frames[half_life] = frame
            all_predictions.append(frame.assign(selected_features="|".join(features)))

        candidates_2024 = []
        for half_life, frame in variant_frames.items():
            validation = frame[frame["week"].str[:4].isin(validation_years)]
            for threshold in THRESHOLDS:
                m = metrics(validation, threshold)
                # Equal weight to raw and balanced DA prevents majority-only selection.
                score = 0.5 * m["da"] + 0.5 * m["balanced_da"]
                candidates_2024.append((score, m["da"], m["balanced_da"], half_life, threshold))
        best = max(candidates_2024, key=lambda x: (x[0], x[2], -x[4]))
        _, _, _, selected_half_life, selected_threshold = best
        selected = variant_frames[selected_half_life]
        selections.append({
            "horizon": horizon,
            "half_life": "expanding" if selected_half_life is None else selected_half_life,
            "threshold": selected_threshold,
            "validation_score": best[0],
            "validation_years": ",".join(validation_years),
            "selected_features": "|".join(features),
            "trial_count": len(candidates_2024),
        })
        periods = [
            (f"{'_'.join(validation_years)}_VALIDATION", validation_years),
            ("2025_OOS", ("2025",)), ("2026_FORWARD", ("2026",)),
        ]
        for period, prefixes in periods:
            part = selected[selected["week"].str[:4].isin(prefixes)]
            row = {
                "horizon": horizon, "period": period,
                "half_life": "expanding" if selected_half_life is None else selected_half_life,
                "threshold": selected_threshold,
            }
            row.update(metrics(part, selected_threshold))
            summaries.append(row)

    report = ROOT / "reports"
    report.mkdir(exist_ok=True)
    pd.concat(all_predictions, ignore_index=True).to_csv(
        report / f"{args.output_prefix}_all_predictions.csv", index=False
    )
    pd.DataFrame(selections).to_csv(
        report / f"{args.output_prefix}_selections.csv", index=False
    )
    summary = pd.DataFrame(summaries)
    summary.to_csv(report / f"{args.output_prefix}_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
