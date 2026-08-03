from __future__ import annotations

import subprocess
import sys


def test_metric_formula_import_does_not_eagerly_load_engine_or_joblib() -> None:
    code = """
import sys
from src.metrics.formulas import sharpe_ratio
assert callable(sharpe_ratio)
assert 'src.metrics.engine' not in sys.modules
assert 'joblib' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_metric_persistence_import_does_not_eagerly_load_engine_or_joblib() -> None:
    code = """
import sys
from src.metrics.persistence import persist_metric_event
assert callable(persist_metric_event)
assert 'src.metrics.engine' not in sys.modules
assert 'joblib' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_metric_package_keeps_unknown_attributes_fail_closed() -> None:
    import src.metrics as metrics

    try:
        metrics.not_a_metric_api
    except AttributeError as exc:
        assert "not_a_metric_api" in str(exc)
    else:
        raise AssertionError("unknown metric API was accepted")
