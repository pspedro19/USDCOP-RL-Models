"""Versioned S1-S4 controls using the SAME effective PPO recipe as market training.

Only flat_init_no_turn is authorized here. No search, stale resume or overwrite:
old synthetic artifacts remain historical evidence, not gates for this version.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.ppo_recipe import (  # noqa: E402
    FROZEN_SANITY_PROBE, build_ppo, canonical_sha256,
    file_sha256, sessions_sha256, write_immutable_json,
)
from src.research.session_gym import SessionTradingEnv  # noqa: E402
from src.research.synthetic_sessions import (  # noqa: E402
    Fixture, SANITY_SEEDS, UNSEEN_SEED_OFFSET, oracle_result,
    sanity_protocol_manifest, sanity_session_splits, validate_sanity_manifest,
)

SEEDS = SANITY_SEEDS
PROBES = (FROZEN_SANITY_PROBE,)


def _evaluate(model, sessions, fixture: Fixture) -> dict:
    env = SessionTradingEnv(sessions, seed=0, shuffle=False)
    results = []
    for _ in sessions:
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(int(action))
        results.append(env.last_result)
    daily = np.array([r.daily_return for r in results], dtype=float)
    oracle = np.array([oracle_result(fixture, s).daily_return for s in sessions])
    return {
        "n_eval_sessions": len(results), "dataset_sha256": sessions_sha256(sessions),
        "mean_net": float(daily.mean()), "median_net": float(np.median(daily)),
        "mean_gross": float(np.mean([r.gross_return for r in results])),
        "mean_cost": float(np.mean([r.total_cost for r in results])),
        "mean_abs_exposure": float(np.mean([r.mean_abs_exposure for r in results])),
        "n_changes": int(sum(r.n_changes for r in results)),
        "oracle_mean_net": float(oracle.mean()), "daily_returns": daily.tolist(),
    }


def _passes(fixture: Fixture, stats: dict) -> bool:
    if fixture in (Fixture.NOISE_WITH_COST, Fixture.SIGNAL_BELOW_COST):
        return bool(stats["mean_abs_exposure"] < 0.1 and stats["mean_net"] > -0.005)
    if fixture == Fixture.SIGNAL_PLANTED:
        return bool(stats["mean_cost"] == 0.0 and stats["oracle_mean_net"] > 0.0
                    and stats["mean_net"] >= 0.70 * stats["oracle_mean_net"])
    return bool(stats["mean_net"] > 0.0)


def _train_one(fixture: Fixture, seed: int, timesteps: int, probe: str,
               checkpoint_dir: Path | None = None,
               resume: Path | None = None) -> dict:
    if resume is not None:
        raise ValueError("legacy resume lacks a frozen manifest; start a new immutable run")
    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir is required for auditable synthetic evidence")
    manifest = sanity_protocol_manifest(probe, timesteps)
    fingerprint = canonical_sha256(manifest)
    training, unseen = sanity_session_splits(fixture, seed)
    tag = f"{fixture.value}_{probe}_seed{seed}"
    paths = {"checkpoint": checkpoint_dir / f"{tag}.zip",
             "vecnormalize": checkpoint_dir / f"{tag}.vecnorm.pkl",
             "manifest": checkpoint_dir / f"{tag}.manifest.json",
             "result": checkpoint_dir / f"{tag}.result.json"}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"immutable synthetic run already reserved: {tag}")
    run_manifest = {
        "schema_version": "research-grade-sanity-v1", "manifest": manifest,
        "fingerprint": fingerprint, "fixture": fixture.value, "seed": seed,
        "train_generation_seed": seed, "unseen_generation_seed": seed + UNSEEN_SEED_OFFSET,
        "training_dataset_sha256": sessions_sha256(training),
        "unseen_dataset_sha256": sessions_sha256(unseen),
        "frozen_at_utc": datetime.now(UTC).isoformat(),
    }
    # Exclusive reservation before training: failed attempts remain visible.
    write_immutable_json(paths["manifest"], run_manifest)
    model, norm, _ = build_ppo(training, seed=seed, probe=probe)
    try:
        model.learn(total_timesteps=timesteps, progress_bar=False)
        norm.training = False
        norm.norm_reward = False
        model.save(paths["checkpoint"])
        norm.save(str(paths["vecnormalize"]))
        train_stats = _evaluate(model, training[:100], fixture)
        unseen_stats = _evaluate(model, unseen, fixture)
        identity_unchanged = fingerprint == canonical_sha256(sanity_protocol_manifest(probe, timesteps))
        row = {
            "seed": seed, "fixture": fixture.value, "probe": probe,
            "fingerprint": fingerprint, "timesteps_requested": timesteps,
            "timesteps_effective": int(model.num_timesteps),
            "identity_unchanged": identity_unchanged,
            "training_dataset_sha256": run_manifest["training_dataset_sha256"],
            "unseen_dataset_sha256": run_manifest["unseen_dataset_sha256"],
            "train_generation_seed": seed,
            "unseen_generation_seed": run_manifest["unseen_generation_seed"],
            "train": train_stats, "unseen": unseen_stats,
            "train_pass": _passes(fixture, train_stats),
            "unseen_pass": _passes(fixture, unseen_stats),
            "passed": bool(identity_unchanged and _passes(fixture, train_stats)
                           and _passes(fixture, unseen_stats)),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "artifacts": {name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
                          for name, path in paths.items() if name != "result"},
        }
        write_immutable_json(paths["result"], row)
        print(json.dumps({"seed_completed": seed, "fixture": fixture.value,
                          "passed": row["passed"], "unseen_mean_net": unseen_stats["mean_net"],
                          "result": str(paths["result"])}), flush=True)
        return row
    finally:
        norm.close()


def run(fixture: Fixture | str, seeds: tuple[int, ...] = SEEDS,
        timesteps: int = 100_000, probe: str = FROZEN_SANITY_PROBE,
        checkpoint_dir: Path | None = None, resume: Path | None = None) -> dict:
    fixture = Fixture(fixture)
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(SEEDS):
        raise ValueError("sanity seeds must be unique members of the frozen five-seed set")
    manifest = sanity_protocol_manifest(probe, timesteps)
    rows = [_train_one(fixture, seed, timesteps, probe,
                       checkpoint_dir=checkpoint_dir, resume=resume) for seed in seeds]
    complete = set(seeds) == set(SEEDS)
    passing = [r["seed"] for r in rows if r["passed"]]
    evidence = []
    if checkpoint_dir is not None:
        for seed in seeds:
            path = checkpoint_dir / f"{fixture.value}_{probe}_seed{seed}.result.json"
            evidence.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    return {
        "schema_version": "research-grade-sanity-v1", "protocol": "fixture",
        "synthetic_only": True, "market_evidence": False, "market_trials_charged": 0,
        "fixture": fixture.value, "probe": probe, "manifest": manifest,
        "fingerprint": canonical_sha256(manifest), "seeds": list(seeds), "rows": rows,
        "complete": complete, "passed_seeds": passing, "input_evidence": evidence,
        "passed": bool(complete and len(passing) >= 4),
        "interpretation": "Optimizer controls only; unseen synthetic sessions are not market evidence.",
    }


def aggregate_fixture_reports(paths: list[Path]) -> dict:
    """Combine S1-S4 without training; reject stale identities and duplicate fixtures."""
    reports = {}
    evidence = []
    fingerprint = None
    manifest = None
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        manifest = validate_sanity_manifest(report)
        fixture = report.get("fixture")
        if fixture not in {f.value for f in Fixture} or fixture in reports:
            raise ValueError("duplicate or unknown fixture report")
        if report.get("synthetic_only") is not True or report.get("market_evidence") is not False:
            raise ValueError("not a synthetic-only report")
        if fingerprint is not None and report["fingerprint"] != fingerprint:
            raise ValueError("mixed sanity protocol fingerprints")
        fingerprint = report["fingerprint"]
        rows = report.get("rows", [])
        if len(rows) != 5 or {r.get("seed") for r in rows} != set(SEEDS):
            raise ValueError("fixture lacks five unique frozen seeds")
        raw_rows = {}
        for entry in report.get("input_evidence", []):
            raw_path = Path(entry["path"])
            if file_sha256(raw_path) != entry.get("sha256"):
                raise ValueError("seed result hash differs from fixture aggregate")
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            if raw.get("seed") in raw_rows:
                raise ValueError("duplicate seed evidence")
            raw_rows[raw.get("seed")] = raw
        if set(raw_rows) != set(SEEDS):
            raise ValueError("fixture lacks five raw seed result artifacts")
        passing = 0
        for row in rows:
            if row != raw_rows[row["seed"]]:
                raise ValueError("fixture rows differ from raw seed results")
            if row.get("fingerprint") != fingerprint:
                raise ValueError("seed run fingerprint differs from aggregate")
            if row.get("train_generation_seed") == row.get("unseen_generation_seed"):
                raise ValueError("unseen evaluation reuses the training generator seed")
            if row.get("unseen_generation_seed") != row["seed"] + UNSEEN_SEED_OFFSET:
                raise ValueError("unseen seed does not follow the frozen protocol")
            if row.get("training_dataset_sha256") == row.get("unseen_dataset_sha256"):
                raise ValueError("unseen evaluation repeats the training dataset")
            if set(row.get("artifacts", {})) != {"checkpoint", "vecnormalize", "manifest"}:
                raise ValueError("missing checkpoint/normalizer/frozen manifest evidence")
            for entry in row["artifacts"].values():
                if file_sha256(Path(entry["path"])) != entry.get("sha256"):
                    raise ValueError("seed artifact was changed after evaluation")
            run_manifest = json.loads(Path(row["artifacts"]["manifest"]["path"]).read_text(encoding="utf-8"))
            validate_sanity_manifest(run_manifest)
            if any(run_manifest.get(key) != row.get(key) for key in (
                "fingerprint", "seed", "fixture", "training_dataset_sha256", "unseen_dataset_sha256",
                "train_generation_seed", "unseen_generation_seed",
            )):
                raise ValueError("run manifest does not bind the evaluated data and seed")
            if (row.get("timesteps_requested") != manifest["timesteps_requested"]
                    or row.get("timesteps_effective", 0) < manifest["timesteps_requested"]):
                raise ValueError("incomplete or different training budget")
            if row["train"]["n_eval_sessions"] != 100 or row["unseen"]["n_eval_sessions"] != 100:
                raise ValueError("incomplete train/unseen evaluation")
            seed_pass = (row.get("identity_unchanged") is True
                         and _passes(Fixture(fixture), row["train"])
                         and _passes(Fixture(fixture), row["unseen"]))
            if row.get("passed") is not seed_pass:
                raise ValueError("seed verdict disagrees with recorded train/unseen statistics")
            passing += int(seed_pass)
        if report.get("passed") is not (passing >= 4):
            raise ValueError("fixture verdict disagrees with its five seeds")
        reports[fixture] = report
        evidence.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    if set(reports) != {f.value for f in Fixture}:
        raise ValueError("S1-S4 protocol requires all four fixture reports")
    return {
        "schema_version": "research-grade-sanity-v1", "protocol": "S1-S4",
        "synthetic_only": True, "market_evidence": False, "market_trials_charged": 0,
        "probe": manifest["probe"], "selected_probe": manifest["probe"],
        "manifest": manifest, "fingerprint": fingerprint, "seeds": list(SEEDS),
        "fixtures": reports, "input_evidence": evidence,
        "complete": True, "passed": all(r["passed"] for r in reports.values()),
    }


def run_protocol(seeds: tuple[int, ...] = SEEDS, timesteps: int = 100_000, *,
                 checkpoint_dir: Path | None = None) -> dict:
    """Frozen recipe only. The former automatic search is deliberately retired."""
    if checkpoint_dir is None:
        raise ValueError("protocol requires a new checkpoint directory")
    paths = []
    for fixture in Fixture:
        path = checkpoint_dir / f"{fixture.value}.json"
        if path.exists():
            raise FileExistsError(path)
        report = run(fixture, seeds, timesteps, FROZEN_SANITY_PROBE, checkpoint_dir)
        write_immutable_json(path, report)
        paths.append(path)
        print(json.dumps({"fixture_completed": fixture.value, "passed": report["passed"],
                          "passed_seeds": report["passed_seeds"], "artifact": str(path)}), flush=True)
    return aggregate_fixture_reports(paths)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", choices=[f.value for f in Fixture])
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="legacy resume is rejected; provenance is incomplete")
    parser.add_argument("--probe", choices=[FROZEN_SANITY_PROBE], default=FROZEN_SANITY_PROBE)
    parser.add_argument("--protocol", action="store_true", help="four fixtures, one frozen recipe; no search")
    parser.add_argument("--aggregate", type=Path, nargs="+", help="aggregate four immutable fixture reports")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; immutable evidence cannot be overwritten")
    if args.resume is not None:
        parser.error("legacy resume is not supported by the versioned protocol")
    if sum((args.fixture is not None, args.protocol, args.aggregate is not None)) != 1:
        parser.error("choose exactly one of --fixture, --protocol or --aggregate")
    if args.protocol and args.seed is not None:
        parser.error("the full protocol requires all five frozen seeds")
    seeds = (args.seed,) if args.seed is not None else SEEDS
    checkpoint_dir = args.checkpoint_dir or args.output.parent / f"{args.output.stem}_runs"
    if args.aggregate:
        report = aggregate_fixture_reports(args.aggregate)
    elif args.protocol:
        report = run_protocol(seeds, args.timesteps, checkpoint_dir=checkpoint_dir)
    else:
        report = run(args.fixture, seeds, args.timesteps, args.probe, checkpoint_dir)
    write_immutable_json(args.output, report)
    print(json.dumps({"output": str(args.output), "passed": report["passed"],
                      "fingerprint": report["fingerprint"]}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
