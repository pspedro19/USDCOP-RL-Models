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

RESOLVED 2026-07-21: feature_registry.yaml is now a DERIVED VIEW of experiment_ssot.yaml
(20 dims, EXP-B-001 order), and DataQualityGate's EXPECTED_FEATURES was moved off the
superseded 13-predictor generation. This test stays as the tripwire: if either side drifts
again, it goes red again. The stronger order-level check below pins the full sequence, not
just the count — feature order is positional, and a same-length reshuffle is the failure
mode a count check cannot see.
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


def test_registry_order_matches_experiment_ssot():
    """The registry's canonical order must BE the experiment SSOT's order, element by element.

    Equal counts with different orders is the worst case: every consumer passes the dimension
    check and every feature lands in the wrong slot.
    """
    reg = _load("config/feature_registry.yaml")
    exp = _load("config/experiment_ssot.yaml")
    reg_order = (reg.get("observation_space") or {}).get("order") or []
    exp_feats = [f["name"] for f in exp.get("features", []) if isinstance(f, dict)]
    if not reg_order or not exp_feats:
        pytest.skip("one side lacks an explicit order")
    assert reg_order == exp_feats, (
        "feature_registry.yaml order diverged from experiment_ssot.yaml. The registry is a "
        "derived view -- regenerate it from the SSOT, never hand-edit the order."
    )

