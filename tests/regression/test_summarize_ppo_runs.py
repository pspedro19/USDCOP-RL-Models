import json
from pathlib import Path

from scripts.diagnostics.summarize_ppo_runs import summarize


def _run(path: Path, config: str, seed: int) -> None:
    payload = {
        "config": config,
        "seed": seed,
        "timesteps": 300000,
        "development": {"total_return": 0.1, "sharpe": 1.0, "n_ops": 2},
        "selection": {"total_return": -0.1, "sharpe": -1.0, "n_ops": 3},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_summary_is_incomplete_without_all_5x2_runs(tmp_path: Path):
    _run(tmp_path / "ppo_regime_seed42.json", "ppo_regime", 42)
    report = summarize(tmp_path)
    assert report["valid_runs"] == 1
    assert report["complete_matrix"] is False
    assert len(report["missing_runs"]) == 9
    assert report["confirmatory"] is False

