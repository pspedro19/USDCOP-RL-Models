"""Fail-closed integrity check for the confirmatory PPO v4 bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SEEDS = (42, 123, 456, 789, 1337)
CONFIGS = ("ppo_regime", "ppo_backbone")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--portable", type=Path,
                        help="portable file bound to the holdout; enables stable hash check")
    parser.add_argument("--forward", type=Path,
                        help="optional post-freeze forward evaluation JSON")
    parser.add_argument("--forward-baselines", type=Path,
                        help="optional same-engine forward baseline JSON")
    parser.add_argument("--forward-llm-contexts", type=Path,
                        help="optional pre-call forward LLM context bundle")
    parser.add_argument("--forward-actions", type=Path,
                        help="optional median per-session PPO forward actions")
    args = parser.parse_args()

    errors: list[str] = []
    for config in CONFIGS:
        for seed in SEEDS:
            path = args.training / f"{config}_seed{seed}.json"
            if not path.exists():
                errors.append(f"missing training artifact: {path.name}")
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("timesteps") != 300_000:
                errors.append(f"{path.name}: timesteps != 300000")
            if payload.get("identity_unchanged") is not True:
                errors.append(f"{path.name}: identity_unchanged is not true")
            selection = payload.get("selection", {})
            if selection.get("n_sessions") != 226:
                errors.append(f"{path.name}: selection n_sessions != 226")

    holdout = json.loads(args.holdout.read_text(encoding="utf-8"))
    expected_portable = (
        hashlib.sha256(args.portable.read_bytes()).hexdigest()
        if args.portable is not None
        else "14cabbd270cbf1bb361f799920399ebc18844ccd392a1631d0572011b6ac1b9e"
    )
    if holdout.get("portable_sha256") != expected_portable:
        errors.append("holdout portable hash differs from the bound dataset")
    rows_by_config = {config: [] for config in CONFIGS}
    for row in holdout.get("rows", []):
        if row.get("config") in rows_by_config:
            rows_by_config[row["config"]].append(row)
    for config in CONFIGS:
        rows = rows_by_config[config]
        if len(rows) != len(SEEDS):
            errors.append(f"{config}: expected five holdout seed rows")
        for row in rows:
            metrics = row.get("metrics", {})
            if metrics.get("n_sessions") != 420:
                errors.append(
                    f"{config}/seed{row.get('seed')}: holdout n_sessions != 420"
                )

    statistics = json.loads(args.statistics.read_text(encoding="utf-8"))
    if statistics.get("schema_version") != "confirmatory-ppo-report-v4":
        errors.append("statistics schema version is not confirmatory-ppo-report-v4")
    if statistics.get("n_trials_used_for_dsr") != 115:
        errors.append("statistics trial count changed unexpectedly")

    if args.forward is not None:
        forward = json.loads(args.forward.read_text(encoding="utf-8"))
        if forward.get("schema_version") != "confirmatory-ppo-forward-v1":
            errors.append("forward schema version mismatch")
        if forward.get("portable_sha256") != expected_portable:
            errors.append("forward portable hash differs from the bound dataset")
        if forward.get("n_sessions") != 162:
            errors.append("forward session count is not 162 for the current data cut")
        for config in CONFIGS:
            aggregate = forward.get("aggregate", {}).get(config, {})
            if aggregate.get("n_seeds") != 5:
                errors.append(f"forward {config}: expected five seed rows")

    if args.forward_baselines is not None:
        baselines = json.loads(args.forward_baselines.read_text(encoding="utf-8"))
        if baselines.get("schema_version") != "confirmatory-forward-baselines-v1":
            errors.append("forward baseline schema version mismatch")
        if baselines.get("portable_sha256") != expected_portable:
            errors.append("forward baseline portable hash differs from bound dataset")
        if baselines.get("n_sessions") != 162 or len(baselines.get("rows", [])) != 8:
            errors.append("forward baseline count/session count mismatch")

    if args.forward_llm_contexts is not None:
        sessions: set[str] = set()
        rows = 0
        for line in args.forward_llm_contexts.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            sessions.add(row.get("session_date"))
            if row.get("dataset_block") != "forward":
                errors.append("forward LLM context has non-forward dataset_block")
            if row.get("retrospective") is not False:
                errors.append("forward LLM context is marked retrospective")
            if row.get("dataset_sha256") != expected_portable:
                errors.append("forward LLM context portable hash differs")
        if rows != 162 * 59 or len(sessions) != 162:
            errors.append("forward LLM context count/session count mismatch")

    if args.forward_actions is not None:
        actions = json.loads(args.forward_actions.read_text(encoding="utf-8"))
        if actions.get("schema_version") != "confirmatory-ppo-forward-actions-v1":
            errors.append("forward PPO actions schema version mismatch")
        if actions.get("portable_sha256") != expected_portable:
            errors.append("forward PPO actions portable hash differs")
        if actions.get("n_sessions") != 162:
            errors.append("forward PPO actions session count is not 162")
        median = actions.get("median", {})
        for config in CONFIGS:
            paths = median.get(config, {})
            if len(paths) != 162 or any(len(path) != 59 for path in paths.values()):
                errors.append(f"forward PPO actions invalid paths for {config}")

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("CONFIRMATORY_ARTIFACT_GATE_PASS")
    print("training=10x300000; selection_sessions=226; holdout_sessions=420; identity=true")
    print(f"statistics={args.statistics.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
