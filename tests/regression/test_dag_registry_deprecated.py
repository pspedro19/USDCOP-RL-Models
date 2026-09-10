"""
Regression: deprecated RL-L4 DAGs must stay out of the active registry (audit A2-01/07/09).

Guards that the deprecated ids are declared in DEPRECATED_DAGS and never leak back
into get_all_dag_ids(), so consumers can't treat them as live again.
"""
import sys
from pathlib import Path

import pytest

# `dag_registry` vive en airflow/dags/contracts. NO se importa por `sys.path`: el repo
# tiene SIETE paquetes llamados `contracts` y el orden del path decide cual gana, asi
# que `pytest tests/regression/` pasaba pero `pytest tests/` (= `make test`) fallaba.
# Ver tests/regression/_dag_module_loader.py.
from tests.regression._dag_module_loader import import_dag_module


@pytest.fixture(scope="module")
def registry():
    return import_dag_module("contracts.dag_registry")


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
