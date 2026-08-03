from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analysis.usdcop_h1_latam_transport_v1 import (
    classification_metrics,
    combined_bootstrap,
)


def _contract() -> dict:
    return {
        "primary_estimand": {
            "block_length_weeks": 2,
            "bootstrap_samples": 500,
            "bootstrap_seed": 41,
        }
    }


def test_cluster_bootstrap_preserves_equal_asset_week_estimand() -> None:
    rows = []
    # Variable signals per week is the case that exposes accidental equal-week weighting.
    patterns = [
        ((1, 1, 0.40), (1, 0, 0.60)),
        ((0, 0, 0.60), (None, 1, 0.40)),
        ((1, 0, 0.40), (None, 0, 0.60)),
        ((0, 1, 0.60), (0, 0, 0.40)),
    ] * 3
    for week, pair in enumerate(patterns, start=1):
        for asset, (prediction, actual, train_up_rate) in zip(("USD/MXN", "USD/BRL"), pair):
            rows.append({
                "iso_week": f"2025-W{week:02d}",
                "asset": asset,
                "prediction": prediction,
                "actual": actual,
                "train_up_rate": train_up_rate,
            })
    frame = pd.DataFrame(rows)
    metrics = classification_metrics(frame)
    bootstrap = combined_bootstrap(frame, _contract())
    assert bootstrap["n_weeks"] == 12
    assert bootstrap["n_signal_asset_weeks"] == metrics["n_signals"]
    assert np.isclose(bootstrap["mean"], metrics["lift_vs_causal_majority"])
    assert bootstrap["valid_draws"] == 500


def test_cluster_bootstrap_is_deterministic_and_keeps_zero_signal_weeks() -> None:
    frame = pd.DataFrame([
        {
            "iso_week": f"2025-W{week:02d}",
            "asset": asset,
            "prediction": None if week % 3 == 0 else (week + offset) % 2,
            "actual": week % 2,
            "train_up_rate": 0.60,
        }
        for week in range(1, 13)
        for offset, asset in enumerate(("USD/MXN", "USD/BRL"))
    ])
    first = combined_bootstrap(frame, _contract())
    second = combined_bootstrap(frame, _contract())
    assert first == second
    assert first["n_weeks"] == 12
    assert first["n_signal_asset_weeks"] == 16


def test_classification_metrics_uses_only_committed_signals() -> None:
    frame = pd.DataFrame([
        {"prediction": 1, "actual": 1, "train_up_rate": 0.6},
        {"prediction": 0, "actual": 1, "train_up_rate": 0.6},
        {"prediction": None, "actual": 0, "train_up_rate": 0.6},
        {"prediction": 0, "actual": None, "train_up_rate": 0.4},
    ])
    result = classification_metrics(frame)
    assert result["n_total"] == 3
    assert result["n_signals"] == 2
    assert result["coverage"] == 2 / 3
    assert result["directional_accuracy"] == 0.5
