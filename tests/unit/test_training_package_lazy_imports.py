from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


EXPECTED_EXPORTS = {
    "ActionDistributionCallback",
    "DATA_SPLIT_CONFIG",
    "DataSplitConfig",
    "DefaultRewardStrategy",
    "ENVIRONMENT_CONFIG",
    "EnvironmentConfig",
    "EnvironmentFactory",
    "EnvObservationBuilder",
    "INDICATOR_CONFIG",
    "IndicatorConfig",
    "MLFLOW_CONFIG",
    "MLflowConfig",
    "MetricsCallback",
    "MultiSeedConfig",
    "MultiSeedResult",
    "MultiSeedTrainer",
    "NETWORK_CONFIG",
    "NetworkConfig",
    "PPOConfig",
    "PPOHyperparameters",
    "PPOTrainer",
    "PPO_HYPERPARAMETERS",
    "PortfolioState",
    "Position",
    "ProgressCallback",
    "RewardCalculator",
    "RewardConfig",
    "RewardStrategy",
    "RewardStrategyAdapter",
    "RewardStrategyRegistry",
    "StepResult",
    "TradingAction",
    "TradingEnvConfig",
    "TradingEnvironment",
    "TrainingConfig",
    "TrainingEngine",
    "TrainingRequest",
    "TrainingResult",
    "compute_file_hash",
    "compute_json_hash",
    "create_training_env",
    "get_project_root",
    "get_training_config",
    "load_config_from_yaml",
    "run_training",
    "set_reproducible_seeds",
    "train_ppo",
    "train_with_multiple_seeds",
    "validate_config",
}

REPO = Path(__file__).resolve().parents[2]


def _run_clean_interpreter(body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", body],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )


def test_config_import_does_not_load_the_optional_rl_stack() -> None:
    script = r'''
import importlib.abc
import pathlib
import sys

repo = pathlib.Path.cwd()
sys.path.insert(0, str(repo))

class BlockOptionalRL(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "stable_baselines3" or fullname.startswith("stable_baselines3."):
            raise ModuleNotFoundError(fullname, name=fullname)
        if fullname == "gymnasium" or fullname.startswith("gymnasium."):
            raise ModuleNotFoundError(fullname, name=fullname)
        return None

sys.meta_path.insert(0, BlockOptionalRL())
from src.training.config import EnvironmentConfig
assert EnvironmentConfig.__module__ == "src.training.config"
assert not any(name == "stable_baselines3" or name.startswith("stable_baselines3.") for name in sys.modules)
assert not any(name == "gymnasium" or name.startswith("gymnasium.") for name in sys.modules)
'''
    result = _run_clean_interpreter(script)
    assert result.returncode == 0, result.stderr


def test_requesting_an_rl_export_still_exposes_the_missing_dependency() -> None:
    script = r'''
import importlib.abc
import pathlib
import sys

repo = pathlib.Path.cwd()
sys.path.insert(0, str(repo))

class BlockOptionalRL(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "stable_baselines3" or fullname.startswith("stable_baselines3."):
            raise ModuleNotFoundError(fullname, name=fullname)
        if fullname == "gymnasium" or fullname.startswith("gymnasium."):
            raise ModuleNotFoundError(fullname, name=fullname)
        return None

sys.meta_path.insert(0, BlockOptionalRL())
import src.training as training
try:
    training.EnvironmentFactory
except ModuleNotFoundError as exc:
    assert exc.name.split(".")[0] in {"stable_baselines3", "gymnasium"}
else:
    raise AssertionError("EnvironmentFactory hid a missing optional RL dependency")
'''
    result = _run_clean_interpreter(script)
    assert result.returncode == 0, result.stderr


def test_public_exports_and_introspection_are_preserved() -> None:
    import src.training as training

    assert set(training.__all__) == EXPECTED_EXPORTS
    assert EXPECTED_EXPORTS <= set(dir(training))


def test_lazy_export_is_cached_and_keeps_its_identity() -> None:
    import src.training as training
    from src.training.reward_calculator import RewardCalculator

    assert training.RewardCalculator is RewardCalculator
    assert training.__dict__["RewardCalculator"] is RewardCalculator


def test_unknown_export_raises_attribute_error() -> None:
    import src.training as training

    with pytest.raises(AttributeError, match="has no attribute"):
        training.not_a_training_export


def test_heavy_public_reexports_keep_identity_when_ml_dependencies_exist() -> None:
    pytest.importorskip("stable_baselines3")

    import src.training as training
    from src.training.engine import TrainingEngine
    from src.training.environments import EnvironmentFactory
    from src.training.trainers import PPOTrainer

    assert training.TrainingEngine is TrainingEngine
    assert training.EnvironmentFactory is EnvironmentFactory
    assert training.PPOTrainer is PPOTrainer
