"""BL-27: el allocator consume su SSOT sin inventar caps ni autoridad paralela."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.portfolio.allocator import (
    AllocationError,
    AllocatorV1,
    CvxpyBudgetOptimizer,
    InfeasibleAllocation,
)


SSOT = Path("config/book/allocator_v1.yaml")


def _config(tmp_path: Path, mutate=None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load(SSOT.read_text(encoding="utf-8"))
    if mutate is not None:
        mutate(raw)
    path = tmp_path / "allocator.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _allocator(path: Path) -> AllocatorV1:
    return AllocatorV1.from_config(
        path,
        sleeve_caps={"a1": 1.0},
        asset_caps={"asset-a": 1.0},
    )


class _RecordingOptimizer:
    def __init__(self) -> None:
        self.requests = []

    def solve(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            raise InfeasibleAllocation("exercise configured relaxation")
        return {"a1": 0.2}


def _constrained_kwargs() -> dict:
    return {
        "volatility": {"a1": 0.1},
        "side": {"a1": 1},
        "sleeve_asset": {"a1": "asset-a"},
        "multipliers": {
            "a1": {
                "forward": 1.0,
                "liquidity": 1.0,
                "diversification": 1.0,
                "operations": 1.0,
                "drawdown": 1.0,
            }
        },
        "previous_budgets": {"a1": 0.2},
        "covariance": [[0.01]],
    }


def test_from_config_binds_solver_gross_and_declared_controls(tmp_path: Path) -> None:
    allocator = _allocator(_config(tmp_path))

    assert allocator.gross_cap == 1.0
    assert isinstance(allocator.optimizer, CvxpyBudgetOptimizer)
    assert allocator.optimizer.solver == "CLARABEL"
    assert allocator.config_source == str((tmp_path / "allocator.yaml").resolve())


def test_configured_controls_reach_request_when_call_omits_them(tmp_path: Path) -> None:
    allocator = _allocator(_config(tmp_path))
    optimizer = _RecordingOptimizer()
    allocator.optimizer = optimizer

    allocator.allocate_constrained(**_constrained_kwargs())

    request = optimizer.requests[0]
    assert request.target_vol == pytest.approx(0.10)
    assert request.turnover_budget == pytest.approx(0.20)
    assert optimizer.requests[1].turnover_budget == pytest.approx(0.30)


def test_configured_instance_rejects_divergent_call_override(tmp_path: Path) -> None:
    allocator = _allocator(_config(tmp_path))
    allocator.optimizer = _RecordingOptimizer()

    with pytest.raises(AllocationError, match="diverges from configured SSOT"):
        allocator.allocate_constrained(
            **_constrained_kwargs(),
            target_vol=0.20,
            turnover_budget=0.20,
            turnover_relaxation_limit=0.30,
        )


def test_direct_constructor_keeps_explicit_controls_and_legacy_novelty() -> None:
    allocator = AllocatorV1(
        sleeve_caps={"a1": 1.0}, gross_cap=1.0, asset_caps={"asset-a": 1.0},
        optimizer=_RecordingOptimizer(),
    )
    assert allocator.novelty_gate(0.50, 0.10) is True
    with pytest.raises(AllocationError, match="target_vol must be provided"):
        allocator.allocate_constrained(**_constrained_kwargs())


def test_mutating_ssot_novelty_threshold_changes_verdict(tmp_path: Path) -> None:
    baseline = _allocator(_config(tmp_path / "base"))

    def stricter(raw):
        raw["novelty_gate"]["max_correlation_lt"] = 0.40
        raw["novelty_gate"]["delta_information_ratio_gt"] = 0.20

    configured = _allocator(_config(tmp_path / "strict", stricter))
    assert baseline.novelty_gate(0.50, 0.10) is True
    assert configured.novelty_gate(0.50, 0.10) is False


def test_mutating_ssot_multiplier_range_changes_validation(tmp_path: Path) -> None:
    def narrower(raw):
        raw["multipliers"]["diversification"]["max"] = 1.0

    allocator = _allocator(_config(tmp_path, narrower))
    kwargs = _constrained_kwargs()
    kwargs["multipliers"] = deepcopy(kwargs["multipliers"])
    kwargs["multipliers"]["a1"]["diversification"] = 1.05
    with pytest.raises(AllocationError, match=r"diversification.*\[0.7,1\]"):
        allocator.allocate(**{k: kwargs[k] for k in (
            "volatility", "side", "sleeve_asset", "multipliers"
        )})


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda raw: raw.update({"unexpected": 1}), "root keys"),
        (lambda raw: raw.pop("constraints"), "root keys"),
        (lambda raw: raw["constraints"].update({"gross_cap": True}), "gross_cap"),
        (lambda raw: raw.update({"contract": "OTHER"}), "contract"),
        (lambda raw: raw["shadow"].update({"minimum_periods": 25}), "at least 26"),
        (lambda raw: raw["shadow"].update({"capital_enabled": True}), "capital_enabled"),
        (
            lambda raw: raw["optimization"].update({"objective": "other"}),
            "optimization.objective",
        ),
        (lambda raw: raw["fallbacks"].reverse(), "fallbacks"),
        (lambda raw: raw.update({"fallbacks": None}), "fallbacks must be a list"),
        (lambda raw: raw["prohibited"].pop(), "prohibited"),
        (
            lambda raw: raw["novelty_gate"].update({"max_correlation_lt": float("nan")}),
            "max_correlation_lt",
        ),
    ],
)
def test_from_config_fails_closed_on_invalid_schema(tmp_path: Path, mutate, message: str) -> None:
    with pytest.raises(AllocationError, match=message):
        _allocator(_config(tmp_path, mutate))


def test_caps_remain_explicit_and_are_not_inferred(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="sleeve_caps"):
        AllocatorV1.from_config(_config(tmp_path))
