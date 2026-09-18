"""Reload frozen synthetic checkpoints and independently repeat their evaluator.

No learning, hyperparameter search, market data, or provider calls. This is an
independent evaluation loop over the shared, tested accounting engine, NOT an
independent economic engine. Only use locally generated, hash-verified artifacts:
SB3/VecNormalize loading deserializes trusted Python objects.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.cost_model import CostParameters  # noqa: E402
from src.research.ppo_recipe import (  # noqa: E402
    canonical_sha256, file_sha256, library_versions, sessions_sha256,
    training_code_hashes, write_immutable_json,
)
from src.research.session_env import OPERABLE_RETURNS  # noqa: E402
from src.research.session_gym import SessionTradingEnv  # noqa: E402
from src.research.synthetic_sessions import (  # noqa: E402
    SANITY_SEEDS, Fixture, fixture_manifest, sanity_session_splits,
)

REPLAY_TOLERANCE = 1e-12


def replay_sessions(model, normalizer, sessions, fixture: str) -> dict:
    """Own reset/predict/step loop; deliberately does not import the old evaluator."""
    if normalizer.training is not False or normalizer.norm_reward is not False:
        raise ValueError("replay requires a frozen evaluation normalizer")
    if normalizer.norm_obs is not False:
        raise ValueError("the frozen recipe does not normalize observations")
    contract = fixture_manifest(fixture)
    daily, gross, costs, exposure, actions = [], [], [], [], []
    changes = 0
    for spec in sessions:
        parameters = getattr(spec, "cost_parameters", None) or CostParameters()
        if (asdict(parameters) != contract["fees"]
                or float(spec.spread_pips) != contract["spread_cop_per_usd"]):
            raise ValueError("replay session fees differ from the frozen fixture")
        env = SessionTradingEnv([spec], seed=0, shuffle=False)
        try:
            observation, _ = env.reset()
            session_actions = []
            for bar in range(OPERABLE_RETURNS):
                normalized = normalizer.normalize_obs(observation)
                if not np.array_equal(normalized, observation):
                    raise ValueError("norm_obs=False unexpectedly changed the observation")
                action, _ = model.predict(normalized, deterministic=True)
                action = int(np.asarray(action).item())
                if not env.action_space.contains(action):
                    raise ValueError("checkpoint produced an invalid discrete action")
                session_actions.append(action)
                observation, _, terminated, truncated, _ = env.step(action)
                if truncated or terminated != (bar == OPERABLE_RETURNS - 1):
                    raise ValueError("replayed episode does not have exactly 59 decisions")
            result = env.last_result
            if result is None or result.date != spec.date:
                raise ValueError("replayed result does not match the declared session")
            daily.append(float(result.daily_return))
            gross.append(float(result.gross_return))
            costs.append(float(result.total_cost))
            exposure.append(float(result.mean_abs_exposure))
            changes += result.n_changes
            actions.append(session_actions)
        finally:
            env.close()
    daily_array = np.asarray(daily, dtype=float)
    if not len(daily_array) or not np.isfinite(daily_array).all():
        raise ValueError("empty or nonfinite policy replay")
    if fixture == "S2" and any(cost != 0.0 for cost in costs):
        raise ValueError("S2 replay charged a nonzero fee")
    return {
        "n_eval_sessions": len(daily), "dataset_sha256": sessions_sha256(sessions),
        "daily_returns": daily, "mean_net": float(daily_array.mean()),
        "mean_gross": float(np.mean(gross)), "mean_cost": float(np.mean(costs)),
        "mean_abs_exposure": float(np.mean(exposure)), "n_changes": int(changes),
        "actions_sha256": canonical_sha256(actions),
    }


def compare_replay(actual: dict, original: dict) -> dict:
    """Compare each day, not only summary means that can conceal reordered results."""
    replayed = np.asarray(actual["daily_returns"], dtype=float)
    recorded = np.asarray(original["daily_returns"], dtype=float)
    if (replayed.shape != recorded.shape or replayed.ndim != 1 or not replayed.size
            or not np.isfinite(replayed).all() or not np.isfinite(recorded).all()):
        raise ValueError("recorded and replayed daily series are not comparable")
    errors = {name: float(abs(actual[name] - original[name])) for name in (
        "mean_net", "mean_gross", "mean_cost", "mean_abs_exposure",
    )}
    daily_error = float(np.max(np.abs(replayed - recorded)))
    same_metadata = all(actual[name] == original[name] for name in (
        "n_eval_sessions", "dataset_sha256", "n_changes",
    ))
    return {
        "max_absolute_daily_return_difference": daily_error,
        "absolute_metric_differences": errors,
        "metadata_equal": same_metadata, "tolerance": REPLAY_TOLERANCE,
        "passed": bool(same_metadata and daily_error <= REPLAY_TOLERANCE
                       and all(np.isfinite(e) and e <= REPLAY_TOLERANCE
                               for e in errors.values())),
    }


def verify_protocol(report: Path, output: Path) -> dict:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from src.research.sanity_gate import require_sanity_pass

    if output.exists():
        raise FileExistsError("replay output is immutable; choose a new path")
    results_dir = output.parent / f"{output.stem}_runs"
    if results_dir.exists():
        raise FileExistsError("replay run directory already exists")
    # Completeness/seed identity, hashes, actual ZIP metadata, and regenerated
    # datasets/oracles are checked BEFORE deserializing our local checkpoints.
    protocol = require_sanity_pass(report)
    # Batch-one inference does not need the learner's parallel CPU workload.
    # No policy parameter is changed; the execution setting is recorded below.
    torch.set_num_threads(1)
    input_sha256 = file_sha256(report)
    frozen_sources = training_code_hashes()
    verifier_sha256 = file_sha256(Path(__file__))
    original_gate_sha256 = file_sha256(ROOT / "src/research/sanity_gate.py")
    rows, evidence = [], []
    for fixture in Fixture:
        by_seed = {row["seed"]: row for row in protocol["fixtures"][fixture.value]["rows"]}
        for seed in SANITY_SEEDS:
            original = by_seed[seed]
            _, unseen = sanity_session_splits(fixture, seed)
            if len(unseen) != 100 or sessions_sha256(unseen) != original["unseen_dataset_sha256"]:
                raise ValueError("replay did not regenerate the declared 100 unseen sessions")
            artifacts = original["artifacts"]
            for entry in artifacts.values():
                if file_sha256(Path(entry["path"])) != entry["sha256"]:
                    raise ValueError("an input artifact changed after sanity verification")
            carrier = DummyVecEnv([lambda: SessionTradingEnv(unseen, seed=0, shuffle=False)])
            normalizer = None
            try:
                normalizer = VecNormalize.load(artifacts["vecnormalize"]["path"], carrier)
                # These must already have been frozen by the producing runner.
                if normalizer.training is not False or normalizer.norm_reward is not False:
                    raise ValueError("saved normalizer is not in frozen evaluation mode")
                norm_contract = protocol["manifest"]["effective_recipe"]["vecnormalize"]
                if any(getattr(normalizer, name) != norm_contract[name]
                       for name in ("norm_obs", "clip_reward", "gamma")):
                    raise ValueError("saved normalizer contradicts its frozen recipe")
                model = PPO.load(artifacts["checkpoint"]["path"], device="cpu")
                if model.num_timesteps != original["timesteps_effective"] or model.seed != seed:
                    raise ValueError("loaded checkpoint seed/steps differ from original evidence")
                actual = replay_sessions(model, normalizer, unseen, fixture.value)
                if model.num_timesteps != original["timesteps_effective"]:
                    raise ValueError("replay unexpectedly changed the training step counter")
                comparison = compare_replay(actual, original["unseen"])
            finally:
                if normalizer is not None:
                    normalizer.close()
                else:
                    carrier.close()
            stable_inputs = (input_sha256 == file_sha256(report)
                             and frozen_sources == training_code_hashes()
                             and verifier_sha256 == file_sha256(Path(__file__))
                             and all(file_sha256(Path(e["path"])) == e["sha256"]
                                     for e in artifacts.values()))
            result = {
                "schema_version": "research-grade-sanity-policy-replay-v1",
                "synthetic_only": True, "market_evidence": False, "learning_steps": 0,
                "fixture": fixture.value, "seed": seed,
                "timesteps_effective": original["timesteps_effective"],
                "fingerprint": protocol["fingerprint"],
                "protocol_input": {"path": str(report.resolve()), "sha256": input_sha256},
                "artifacts": artifacts, "dataset_sha256": actual["dataset_sha256"],
                "original_unseen_sha256": canonical_sha256(original["unseen"]),
                "source_sha256": frozen_sources,
                "verifier_sha256": verifier_sha256, "sanity_gate_sha256": original_gate_sha256,
                "library_versions": library_versions(), "actual": actual,
                "execution": {"device": "cpu", "torch_num_threads": torch.get_num_threads(),
                              "deterministic_actions": True},
                "comparison": comparison, "inputs_unchanged": stable_inputs,
                "passed": bool(comparison["passed"] and stable_inputs),
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "scope": "Independent evaluator loop, shared accounting engine; not market evidence.",
            }
            path = results_dir / f"{fixture.value}_seed{seed}.json"
            write_immutable_json(path, result)
            rows.append(result)
            evidence.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
            print(json.dumps({"replayed": f"{fixture.value}/{seed}", "passed": result["passed"],
                              "max_abs_diff": comparison["max_absolute_daily_return_difference"]}),
                  flush=True)
    summary = {
        "schema_version": "research-grade-sanity-policy-replay-v1",
        "synthetic_only": True, "market_evidence": False, "learning_steps": 0,
        "fingerprint": protocol["fingerprint"], "n_runs": len(rows),
        "input_evidence": evidence, "rows": rows,
        "max_absolute_daily_return_difference": max(
            row["comparison"]["max_absolute_daily_return_difference"] for row in rows),
        "passed": len(rows) == 20 and all(row["passed"] for row in rows),
        "scope": "Checkpoint reload plus independent evaluator loop; shared accounting engine.",
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    write_immutable_json(output, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = verify_protocol(args.protocol, args.output)
    print(json.dumps({"output": str(args.output), "passed": summary["passed"],
                      "n_runs": summary["n_runs"],
                      "max_abs_diff": summary["max_absolute_daily_return_difference"]}))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
