"""SPX DSR uses the governed registry count, never the global spend cap."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from src.metrics.trial_count import TrialCountError, read_n_trials_total


ROOT = Path(__file__).resolve().parents[2]
STRATEGY_DIR = ROOT / "src/strategies/spx500_regime_gated_v1"


def _registry(path: Path, value: str) -> Path:
    path.write_text(f"---\nn_trials_total: {value}\n---\n# registry\n", encoding="utf-8")
    return path


def _runner():
    return importlib.import_module("src.strategies.spx500_regime_gated_v1.run_strategy")


def test_reader_propagates_the_registry_value(tmp_path: Path) -> None:
    path = _registry(tmp_path / "registry.md", "17")
    assert read_n_trials_total(path) == 17
    _registry(path, "23")
    assert read_n_trials_total(path) == 23


@pytest.mark.parametrize(
    "body",
    [
        "no front matter",
        "---\nother: 17\n---\n",
        "---\nn_trials_total: true\n---\n",
        "---\nn_trials_total: 0\n---\n",
    ],
)
def test_reader_fails_closed_for_an_unusable_registry(tmp_path: Path, body: str) -> None:
    path = tmp_path / "registry.md"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(TrialCountError):
        read_n_trials_total(path)


def test_spx_runner_passes_the_governed_count_to_dsr(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _runner()
    path = _registry(tmp_path / "registry.md", "23")
    calls: list[int] = []

    def dsr_spy(candidate, family_sharpes, n_trials):
        calls.append(n_trials)
        return 0.42

    monkeypatch.setattr(runner, "TRIAL_REGISTRY", path)
    monkeypatch.setattr(runner, "dsr_from_family", dsr_spy)

    result, audited_n = runner._governed_dsr(object(), np.array([0.1, 0.2]))
    assert result == 0.42
    assert audited_n == 23
    assert calls == [23]


def test_spx_runner_fails_before_dsr_when_registry_is_missing(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _runner()
    calls: list[int] = []
    monkeypatch.setattr(runner, "TRIAL_REGISTRY", tmp_path / "missing.md")
    monkeypatch.setattr(
        runner,
        "dsr_from_family",
        lambda *_args: calls.append(1),
    )

    with pytest.raises(TrialCountError):
        runner._governed_dsr(object(), np.array([0.1, 0.2]))
    assert calls == []


def test_repository_spx_count_is_not_the_global_spend_cap() -> None:
    runner_path = STRATEGY_DIR / "run_strategy.py"
    source = runner_path.read_text(encoding="utf-8")
    assert "N_MAX_STUDY" not in source
    assert read_n_trials_total(
        ROOT / ".claude/specs/assets/spx500/HYPOTHESIS-REGISTRY.md"
    ) != 989


def test_profitability_harness_delegates_to_the_same_reader() -> None:
    harness = importlib.import_module("scripts.analysis.profitability_evidence")
    result = harness.trial_count("spx500")
    assert result["n_trials_total"] == read_n_trials_total(
        ROOT / ".claude/specs/assets/spx500/HYPOTHESIS-REGISTRY.md"
    )
