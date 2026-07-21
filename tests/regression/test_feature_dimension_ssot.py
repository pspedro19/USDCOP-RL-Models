"""The feature dimension must have exactly one authoritative value.

Contract: CTR-FEAT-SSOT-001 (audit FEAT-001 / FEAT-P0-01)

Three files disagree about how many features this system has, and each has its own set of
consumers that never talk to each other:

    config/feature_registry.yaml   observation_space.total_dimension = 15
        -> src/features/registry.py, airflow/dags/contracts/pipeline_contracts.py,
           scripts/ops/backup/backup_master.py
    config/experiment_ssot.yaml    observation_dim = 20   ("EXP-B-001: 15 -> 20")
        -> src/config/{experiment_loader,pipeline_config}.py, src/core/constants.py,
           src/backtest/engine/unified_backtest_engine.py
    config/pipeline_ssot.yaml      18 market + 9 state feature entries

EXP-B-001 raised the dimension to 20 and updated the experiment SSOT, but the registry --
which literally calls itself "SSOT for feature definitions" -- still says 15. So a validator
built on the registry will reject a valid 20-dim dataset, or wave through a legacy 15-dim one
as production. Feature ORDER is positional: a mismatch does not raise, it silently feeds the
model the wrong column in each slot. That is the worst kind of failure -- it produces numbers.

This test is expected to FAIL until the conflict is resolved. That is deliberate. Deleting or
xfail-ing it re-hides a live inconsistency between training and inference; the fix is to pick
one authority and make the others derive from it. The RL track being deprioritized is a reason
to leave the dimension alone, NOT a reason to stop the drift from being visible.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load(rel: str) -> dict:
    p = ROOT / rel
    if not p.is_file():
        pytest.skip(f"{rel} absent")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def declared_dimensions() -> dict[str, int]:
    dims: dict[str, int] = {}

    reg = _load("config/feature_registry.yaml")
    obs = reg.get("observation_space") or {}
    if "total_dimension" in obs:
        dims["config/feature_registry.yaml::observation_space.total_dimension"] = int(
            obs["total_dimension"])

    exp = _load("config/experiment_ssot.yaml")
    for section in exp.values():
        if isinstance(section, dict) and "observation_dim" in section:
            dims["config/experiment_ssot.yaml::observation_dim"] = int(section["observation_dim"])
            break

    return dims


def test_feature_dimension_has_a_single_authority():
    dims = declared_dimensions()
    if len(dims) < 2:
        pytest.skip("fewer than two declarations found; nothing to reconcile")

    distinct = set(dims.values())
    assert len(distinct) == 1, (
        "The feature dimension is declared with conflicting values:\n  "
        + "\n  ".join(f"{k} = {v}" for k, v in sorted(dims.items()))
        + "\n\nFeature order is positional, so a mismatch does not raise -- it feeds the model "
          "the wrong column in each slot and still returns a number. Pick ONE authority and "
          "make the others read from it; do not silence this test."
    )
