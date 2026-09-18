"""Evaluate frozen PPO checkpoints on the post-freeze 2026 forward bundle."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO  # noqa: E402

from scripts.analysis.thesis_train_ppo import evaluate, strip_regimes  # noqa: E402
from src.research.ppo_recipe import file_sha256  # noqa: E402
from src.research.session_gym import EXPOSURE_LEVELS, SessionTradingEnv  # noqa: E402

SEEDS = (42, 123, 456, 789, 1337)
CONFIGS = ("ppo_regime", "ppo_backbone")


def _weights(model, specs):
    """Replay deterministic actions and return one 59-weight path per session."""
    env = SessionTradingEnv(specs, seed=0, shuffle=False)
    out = {}
    for _ in range(len(specs)):
        obs, _ = env.reset()
        session_date = env._spec.date.isoformat()
        path = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            path.append(float(EXPOSURE_LEVELS[int(action)]))
            obs, _, done, _, _ = env.step(int(action))
        if len(path) != 59:
            raise ValueError(f"{session_date}: expected 59 PPO decisions, got {len(path)}")
        out[session_date] = path
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--portable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actions-output", type=Path,
                        help="optional JSON with median per-session PPO weights for downstream hybrid")
    args = parser.parse_args()

    bundle = pickle.loads(args.specs.read_bytes())
    manifest = bundle.get("manifest", {})
    if manifest.get("portable_sha256") != file_sha256(args.portable.resolve()):
        raise ValueError("forward specs are bound to a different portable dataset")
    specs = bundle.get("sessions", [])
    if not specs:
        raise ValueError("forward specs contain no sessions")

    rows = []
    paths_by_config: dict[str, dict[int, dict[str, list[float]]]] = {
        config: {} for config in CONFIGS
    }
    for config in CONFIGS:
        config_specs = specs if config == "ppo_regime" else strip_regimes(specs)
        for seed in SEEDS:
            checkpoint = args.checkpoint_dir / f"{config}_seed{seed}.zip"
            manifest_path = args.checkpoint_dir / f"{config}_seed{seed}.manifest.json"
            if not checkpoint.is_file() or not manifest_path.is_file():
                raise FileNotFoundError(f"missing frozen artifact for {config} seed {seed}")
            model = PPO.load(checkpoint, device="cpu")
            metrics = evaluate(model, config_specs)
            paths_by_config[config][seed] = _weights(model, config_specs)
            rows.append({
                "config": config,
                "seed": seed,
                "checkpoint_sha256": file_sha256(checkpoint),
                "manifest_sha256": file_sha256(manifest_path),
                "block": "post_freeze_forward_2026_partial",
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
        "schema_version": "confirmatory-ppo-forward-v1",
        "scope": "post_freeze_forward_partial",
        "specs_sha256": file_sha256(args.specs.resolve()),
        "portable_sha256": file_sha256(args.portable.resolve()),
        "partition": {"start": manifest.get("start"), "end": manifest.get("end")},
        "n_sessions": len(specs),
        "rows": rows,
        "aggregate": aggregate,
    }
    if args.actions_output is not None:
        median_paths = {}
        for config in CONFIGS:
            dates = sorted(paths_by_config[config][SEEDS[0]])
            median_paths[config] = {
                date: np.median(
                    np.asarray([paths_by_config[config][seed][date] for seed in SEEDS], dtype=float),
                    axis=0,
                ).tolist()
                for date in dates
            }
        actions_payload = {
            "schema_version": "confirmatory-ppo-forward-actions-v1",
            "scope": "post_freeze_forward_partial",
            "specs_sha256": file_sha256(args.specs.resolve()),
            "portable_sha256": file_sha256(args.portable.resolve()),
            "n_sessions": len(specs),
            "aggregation": "componentwise_median_across_five_seeds",
            "median": median_paths,
        }
        args.actions_output.parent.mkdir(parents=True, exist_ok=True)
        args.actions_output.write_text(
            json.dumps(actions_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
