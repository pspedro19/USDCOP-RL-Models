#!/usr/bin/env python
"""Evaluate frozen PPO v4 checkpoints on the unopened 2024-2025 hold-out."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO  # noqa: E402

from scripts.analysis.thesis_train_ppo import evaluate, strip_regimes  # noqa: E402
from src.research.dataset import load_portable  # noqa: E402
from src.research.ppo_recipe import file_sha256  # noqa: E402

SEEDS = (42, 123, 456, 789, 1337)
CONFIGS = ("ppo_regime", "ppo_backbone")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portable", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = load_portable(args.portable.resolve())
    rows = []
    for config in CONFIGS:
        specs = data.holdout if config == "ppo_regime" else strip_regimes(data.holdout)
        for seed in SEEDS:
            checkpoint = args.checkpoint_dir / f"{config}_seed{seed}.zip"
            manifest = args.checkpoint_dir / f"{config}_seed{seed}.manifest.json"
            if not checkpoint.is_file() or not manifest.is_file():
                raise FileNotFoundError(f"missing frozen artifact for {config} seed {seed}")
            model = PPO.load(checkpoint, device="cpu")
            metrics = evaluate(model, specs)
            rows.append({
                "config": config,
                "seed": seed,
                "checkpoint_sha256": file_sha256(checkpoint),
                "manifest_sha256": file_sha256(manifest),
                "block": "confirmatory_holdout_2024_2025",
                "metrics": metrics,
            })

    aggregate = {}
    for config in CONFIGS:
        subset = [row["metrics"] for row in rows if row["config"] == config]
        returns = np.asarray([item["total_return"] for item in subset], dtype=float)
        sharpes = np.asarray([item["sharpe"] for item in subset], dtype=float)
        aggregate[config] = {
            "n_seeds": len(subset),
            "total_return_by_seed": returns.tolist(),
            "sharpe_by_seed": sharpes.tolist(),
            "median_total_return": float(np.median(returns)),
            "median_sharpe": float(np.median(sharpes)),
            "mean_total_return": float(np.mean(returns)),
            "mean_sharpe": float(np.mean(sharpes)),
            "positive_return_seeds": int(np.sum(returns > 0)),
        }
    payload = {
        "schema_version": "confirmatory-ppo-holdout-v4",
        "scope": "one_look_confirmatory",
        "portable_sha256": file_sha256(args.portable.resolve()),
        "partition": "2024-01-01..2025-12-31",
        "rows": rows,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
