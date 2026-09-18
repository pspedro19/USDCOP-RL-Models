"""Adversarial gate checks. Inert checkpoints here are TEST FIXTURES, never results."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from src.research.ppo_recipe import canonical_sha256, file_sha256, sessions_sha256
from src.research.sanity_gate import _passes, _verify_checkpoint, require_sanity_pass
from src.research.synthetic_sessions import (
    Fixture, SANITY_SEEDS, UNSEEN_SEED_OFFSET, oracle_result,
    sanity_protocol_manifest, sanity_session_splits,
)


def _write(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _stats(fixture, sessions):
    results = [oracle_result(fixture, spec) for spec in sessions]
    daily = [r.daily_return for r in results]
    return {"n_eval_sessions": 100, "dataset_sha256": sessions_sha256(sessions),
            "daily_returns": daily, "mean_net": float(np.mean(daily)),
            "median_net": float(np.median(daily)),
            "mean_gross": float(np.mean([r.gross_return for r in results])),
            "mean_cost": float(np.mean([r.total_cost for r in results])),
            "mean_abs_exposure": float(np.mean([r.mean_abs_exposure for r in results])),
            "oracle_mean_net": float(np.mean(daily))}


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """Generate a self-consistent unit-test evidence graph without model fitting."""
    directory = tmp_path_factory.mktemp("controlled-sanity-gate")
    manifest = sanity_protocol_manifest()
    fingerprint = canonical_sha256(manifest)
    reports, input_evidence = {}, []
    for fixture in Fixture:
        rows = []
        for seed in SANITY_SEEDS:
            training, unseen = sanity_session_splits(fixture, seed)
            run_manifest = {"schema_version": "research-grade-sanity-v1", "manifest": manifest,
                            "fingerprint": fingerprint, "seed": seed, "fixture": fixture.value,
                            "train_generation_seed": seed,
                            "unseen_generation_seed": seed + UNSEEN_SEED_OFFSET,
                            "training_dataset_sha256": sessions_sha256(training),
                            "unseen_dataset_sha256": sessions_sha256(unseen)}
            tag = f"{fixture.value}-{seed}"
            manifest_path = directory / f"{tag}.manifest.json"
            _write(manifest_path, run_manifest)
            checkpoint = directory / f"{tag}.zip"
            metadata = dict(manifest["effective_recipe"]["ppo_kwargs"], seed=seed,
                            num_timesteps=102_400,
                            policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]},
                                           "activation_fn": "<class 'torch.nn.modules.activation.Tanh'>"})
            with zipfile.ZipFile(checkpoint, "w") as archive:
                archive.writestr("data", json.dumps(metadata))
                archive.writestr("policy.pth", b"inert unit-test weights; not trained")
            normalizer = directory / f"{tag}.pkl"
            normalizer.write_bytes(b"inert unit-test normalizer; never unpickled")
            row = {k: v for k, v in run_manifest.items() if k not in ("schema_version", "manifest")}
            row.update(probe=manifest["probe"], timesteps_requested=100_000,
                       timesteps_effective=102_400, identity_unchanged=True,
                       train=_stats(fixture, training[:100]), unseen=_stats(fixture, unseen),
                       train_pass=True, unseen_pass=True, passed=True,
                       artifacts={name: {"path": str(path), "sha256": file_sha256(path)}
                                  for name, path in (("checkpoint", checkpoint),
                                                     ("vecnormalize", normalizer), ("manifest", manifest_path))})
            rows.append(row)
        reports[fixture.value] = {"schema_version": "research-grade-sanity-v1",
                                  "manifest": manifest, "fingerprint": fingerprint,
                                  "fixture": fixture.value, "rows": rows, "passed": True}
        fixture_path = directory / f"{fixture.value}.json"
        _write(fixture_path, reports[fixture.value])
        input_evidence.append({"path": str(fixture_path), "sha256": file_sha256(fixture_path)})
    report = directory / "protocol.json"
    payload = {"schema_version": "research-grade-sanity-v1", "manifest": manifest,
               "fingerprint": fingerprint, "protocol": "S1-S4", "synthetic_only": True,
               "market_evidence": False, "market_trials_charged": 0,
               "fixtures": reports, "input_evidence": input_evidence, "passed": True}
    _write(report, payload)
    return report, payload


def _mutated_bundle(bundle, fixture, mutate):
    """Rehash both levels so tests exercise semantics rather than a trivial stale hash."""
    report, original = bundle
    payload = deepcopy(original)
    subreport = payload["fixtures"][fixture]
    mutate(subreport)
    path = report.parent / f"{fixture}.json"
    _write(path, subreport)
    for evidence in payload["input_evidence"]:
        if Path(evidence["path"]) == path:
            evidence["sha256"] = file_sha256(path)
    _write(report, payload)


def _restore_bundle(bundle):
    report, payload = bundle
    for name, subreport in payload["fixtures"].items():
        _write(report.parent / f"{name}.json", subreport)
    _write(report, payload)


def test_valid_controlled_evidence_graph_passes_without_training(bundle):
    report, _ = bundle
    result = require_sanity_pass(report)
    assert result["selected_probe"] == "flat_init_no_turn"


def test_legacy_green_booleans_are_not_evidence(tmp_path):
    path = tmp_path / "legacy.json"
    _write(path, {"protocol": "S1-S4", "passed": True, "synthetic_only": True,
                  "market_trials_charged": 0, "market_evidence": False,
                  "fixtures": {f.value: {"passed": True} for f in Fixture}})
    with pytest.raises(RuntimeError, match="lacks"):
        require_sanity_pass(path)


def test_current_manifest_but_missing_artifacts_is_rejected(tmp_path):
    manifest = sanity_protocol_manifest()
    path = tmp_path / "empty.json"
    _write(path, {"schema_version": "research-grade-sanity-v1", "manifest": manifest,
                  "fingerprint": canonical_sha256(manifest), "protocol": "S1-S4",
                  "synthetic_only": True, "market_evidence": False, "market_trials_charged": 0,
                  "fixtures": {f.value: {"passed": True} for f in Fixture}, "passed": True})
    with pytest.raises(RuntimeError, match="hashed fixture"):
        require_sanity_pass(path)


@pytest.mark.parametrize("cost,exposure", [(-0.1, 0.0), (0.0, -0.1), (0.0, 1.1)])
def test_nonphysical_but_finite_metrics_are_rejected(cost, exposure):
    stats = {"mean_net": 0.0, "mean_gross": cost, "mean_cost": cost,
             "mean_abs_exposure": exposure, "oracle_mean_net": 0.0,
             "daily_returns": [0.0] * 100, "n_eval_sessions": 100}
    with pytest.raises(ValueError, match="physical bounds"):
        _passes("S1", stats)


def test_rehashed_lower_oracle_cannot_make_s2_pass(bundle):
    def mutate(subreport):
        subreport["rows"][0]["unseen"]["oracle_mean_net"] *= 0.01
    try:
        _mutated_bundle(bundle, "S2", mutate)
        with pytest.raises(RuntimeError, match="oracle differs"):
            require_sanity_pass(bundle[0])
    finally:
        _restore_bundle(bundle)


def test_rehashed_metrics_not_reproduced_by_daily_returns_are_rejected(bundle):
    def mutate(subreport):
        subreport["rows"][0]["unseen"]["mean_net"] += 0.02
    try:
        _mutated_bundle(bundle, "S1", mutate)
        with pytest.raises(RuntimeError, match="daily evidence"):
            require_sanity_pass(bundle[0])
    finally:
        _restore_bundle(bundle)


def test_rehashed_checkpoint_metadata_cannot_lie_about_training_steps(bundle, tmp_path):
    _, payload = bundle
    row = payload["fixtures"]["S1"]["rows"][0]
    original = Path(row["artifacts"]["checkpoint"]["path"])
    path = tmp_path / "wrong_steps.zip"
    with zipfile.ZipFile(original) as archive:
        metadata = json.loads(archive.read("data"))
    metadata["num_timesteps"] = 1_000
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps(metadata))
        archive.writestr("policy.pth", b"inert test")
    with pytest.raises(ValueError, match="effective steps"):
        _verify_checkpoint(path, row, payload["manifest"])
