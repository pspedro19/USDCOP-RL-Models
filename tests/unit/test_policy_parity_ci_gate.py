"""CI must prove parity before a policy can become executable (C-010)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.validation import check_policy_parity as gate


REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "fabric-contracts.yml"


def _spec(policy_id: str, status: str) -> dict:
    return {"id": policy_id, "inputs": {}, "migration": {"status": status}}


def test_ci_zero_eligible_is_explicitly_inert(monkeypatch, capsys):
    monkeypatch.setattr(
        gate,
        "load_all_policy_specs",
        lambda: [_spec("record_only", "SPEC_ONLY"), _spec("pending", "PARITY_PENDING")],
    )

    assert gate.main(["--ci-eligible"]) == 0
    assert "0 specs elegibles" in capsys.readouterr().out


def test_ci_eligible_policy_without_harness_is_red(monkeypatch, capsys):
    monkeypatch.setattr(
        gate, "load_all_policy_specs", lambda: [_spec("unharnessed", "PARITY_GREEN")]
    )
    monkeypatch.setattr(gate, "CHECKS", {})

    assert gate.main(["--ci-eligible"]) == 1
    assert "sin arnés de paridad" in capsys.readouterr().out


def test_ci_eligible_policy_with_missing_frozen_data_is_red(monkeypatch, capsys):
    monkeypatch.setattr(
        gate, "load_all_policy_specs", lambda: [_spec("missing_data", "CUTOVER")]
    )

    def unavailable(_spec):
        raise gate.DataUnavailable("fixture congelada ausente")

    monkeypatch.setattr(gate, "CHECKS", {"missing_data": unavailable})
    monkeypatch.setattr(gate, "load_policy_spec", lambda _path: _spec("missing_data", "CUTOVER"))

    assert gate.main(["--ci-eligible"]) == 1
    output = capsys.readouterr().out
    assert "[FAIL] missing_data" in output
    assert "fixture congelada ausente" in output


def test_ci_eligible_policy_with_divergence_is_red(monkeypatch, capsys):
    monkeypatch.setattr(
        gate, "load_all_policy_specs", lambda: [_spec("divergent", "PARITY_GREEN")]
    )
    monkeypatch.setattr(
        gate,
        "CHECKS",
        {"divergent": lambda _spec: (np.array([0.0]), np.array([1.0]))},
    )
    monkeypatch.setattr(gate, "load_policy_spec", lambda _path: _spec("divergent", "PARITY_GREEN"))

    assert gate.main(["--ci-eligible"]) == 1
    assert "barras divergen" in capsys.readouterr().out


def test_fabric_workflow_invokes_strict_eligible_parity_gate():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert (
        "python scripts/validation/check_policy_parity.py --ci-eligible" in workflow
    ), "PARITY_GREEN must be mechanically checked by the fabric CI workflow"
