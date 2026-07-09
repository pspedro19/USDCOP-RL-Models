"""
Regression: `src.services` package after removing the dead NRT services (audit A2-02).

Guards that:
  1. The still-live BacktestFeatureBuilder API remains importable (the ONLY real
     consumer is scripts/build_backtest_features.py).
  2. The removed NRT services are gone — no module, no export — so they can't be
     silently resurrected again.
"""
import importlib
import importlib.util

import pytest


def test_backtest_feature_builder_still_exported():
    """The live service must remain importable from the package root."""
    services = importlib.import_module("src.services")
    assert hasattr(services, "BacktestFeatureBuilder")
    assert hasattr(services, "FeatureBuildConfig")
    assert set(services.__all__) == {"BacktestFeatureBuilder", "FeatureBuildConfig"}


def test_backtest_feature_builder_submodule_import_path():
    """The exact import used by scripts/build_backtest_features.py must work."""
    mod = importlib.import_module("src.services.backtest_feature_builder")
    assert hasattr(mod, "BacktestFeatureBuilder")
    assert hasattr(mod, "FeatureBuildConfig")


@pytest.mark.parametrize(
    "dead_module",
    [
        "src.services.l1_nrt_data_service",
        "src.services.l5_nrt_inference_service",
    ],
)
def test_dead_nrt_modules_are_gone(dead_module):
    """The removed NRT service modules must not exist anymore."""
    assert importlib.util.find_spec(dead_module) is None, (
        f"{dead_module} was deleted in audit A2-02 but is importable again"
    )


@pytest.mark.parametrize(
    "dead_symbol",
    ["L1NRTDataService", "L5NRTInferenceService", "L1NRTConfig", "L5NRTConfig"],
)
def test_dead_nrt_symbols_not_exported(dead_symbol):
    """The package must no longer export the removed NRT classes."""
    services = importlib.import_module("src.services")
    assert not hasattr(services, dead_symbol), (
        f"{dead_symbol} should have been removed from src.services (audit A2-02)"
    )
