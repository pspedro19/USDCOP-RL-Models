#!/usr/bin/env python3
"""Fixed, causal supervised baseline for the v2 thesis dataset.

This is a single diagnostic arm, not a hyperparameter search: logistic regression is fit on
development sessions only, then frozen and evaluated on selection (or holdout when explicitly
requested for retrospective reporting). The threshold is fixed at 0.55 before evaluation.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.research.session_env import daily_series, run_session

ROOT = Path(__file__).resolve().parents[2]


def _training_matrix(sessions):
    xs, ys = [], []
    for spec in sessions:
        features = np.asarray(spec.market[:59], dtype=float)
        close = np.asarray(spec.close, dtype=float)
        returns = close[1:60] / close[:59] - 1.0
        for x, y in zip(features, returns):
            if np.isfinite(x).all() and np.isfinite(y):
                xs.append(x)
                ys.append(int(y > 0.0))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=int)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portable", type=Path, default=ROOT / "data/thesis/research_data_portable_v2.pkl")
    parser.add_argument("--block", choices=("selection", "holdout"), default="selection")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.portable.open("rb") as handle:
        portable = pickle.load(handle)
    train = portable["development"]
    evaluation = portable[args.block]
    x_train, y_train = _training_matrix(train)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42),
    )
    model.fit(x_train, y_train)

    results = []
    for spec in evaluation:
        probs = model.predict_proba(np.asarray(spec.market[:59], dtype=float))[:, 1]
        weights = np.where(probs >= 0.55, 1.0, np.where(probs <= 0.45, -1.0, 0.0))
        results.append(run_session(spec.close, weights, spec.spread_pips, date=spec.date))
    returns = daily_series(results)
    gross = np.asarray([r.gross_return for r in results], dtype=float)
    costs = np.asarray([r.total_cost for r in results], dtype=float)
    equity = np.cumprod(1.0 + returns)
    sd = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    ann = float(np.sqrt(221))
    sharpe = float(np.mean(returns) / sd * ann) if sd > 0 else 0.0
    payload = {
        "contract": "CTR-THESIS-SUPERVISED-V2-001",
        "classification": "diagnostic_retrospective",
        "dataset": str(args.portable),
        "train_block": "development",
        "evaluation_block": args.block,
        "model": "StandardScaler + LogisticRegression(C=1.0,class_weight=balanced)",
        "decision_thresholds": {"long": 0.55, "short": 0.45},
        "n_train_rows": int(len(y_train)),
        "n_sessions": int(len(results)),
        "total_return_pct": float((equity[-1] - 1.0) * 100.0),
        "sharpe_221": sharpe,
        "n_traded_sessions": int(sum(r.n_changes > 0 for r in results)),
        "daily_returns": [float(x) for x in returns],
        "daily_gross_returns": [float(x) for x in gross],
        "daily_costs": [float(x) for x in costs],
        "dates": [str(r.date) for r in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("evaluation_block", "n_sessions", "total_return_pct", "sharpe_221", "n_traded_sessions")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
