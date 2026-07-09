"""
Regression: deprecated RL-L4 DAGs must stay out of the active registry (audit A2-01/07/09).

Guards that the deprecated ids are declared in DEPRECATED_DAGS and never leak back
into get_all_dag_ids(), so consumers can't treat them as live again.
"""
import sys
from pathlib import Path

import pytest

# dag_registry lives under airflow/dags/contracts and is imported as `contracts.*`
_DAGS = str(Path(__file__).resolve().parents[2] / "airflow" / "dags")
if _DAGS not in sys.path:
    sys.path.insert(0, _DAGS)


@pytest.fixture(scope="module")
def registry():
    return __import__("contracts.dag_registry", fromlist=["*"])


def test_deprecated_set_contains_the_three_rl_l4_dags(registry):
    dep = registry.DEPRECATED_DAGS
    assert registry.RL_L4_EXPERIMENT_RUNNER in dep
    assert registry.RL_L4_BACKTEST_VALIDATION in dep
    assert registry.RL_L4_SCHEDULED_RETRAINING in dep


def test_deprecated_dags_not_in_active_list(registry):
    active = set(registry.get_all_dag_ids())
    leaked = registry.DEPRECATED_DAGS & active
    assert not leaked, f"deprecated DAGs leaked into get_all_dag_ids(): {leaked}"


def test_active_list_has_no_duplicates(registry):
    ids = registry.get_all_dag_ids()
    assert len(ids) == len(set(ids)), "get_all_dag_ids() must not contain duplicates"
