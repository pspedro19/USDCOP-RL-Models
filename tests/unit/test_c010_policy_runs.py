# -*- coding: utf-8 -*-
"""C-010 R3: la cadena gobernada solo existe para referencias ELEGIBLES.

`resolve_feature_snapshot` aplica `available_at <= decision_cutoff` y hasta ahora
tenia **cero llamadores productivos** (`AUDIT-CLAUDE-wiring-gap.md`). C-010 le da
uno: el factory emite `resolve_snapshot -> evaluate -> publish` para cada
`policy_runs[].policy_id` cuyo `migration.status` sea `PARITY_GREEN|CUTOVER`,
ramificando por `engine.type` y **nunca** por `strategy_id`.

Invariantes que fijan estos candados:

1. Sin `policy_runs` declarados el grafo de tareas es **identico** — hoy es el
   caso, y la ausencia de delta no depende de criterio.
2. `SPEC_ONLY`/`PARITY_PENDING` producen **cero tareas**, no un skip verde.
3. IDs duplicados o desconocidos al loader SSOT fallan **al parsear**, no en
   runtime.
4. La cadena **llama** a `resolve_feature_snapshot`; retirar ese caller muerde.
5. Un `available_at` posterior al `decision_cutoff` **no llega a publicar**.

Promover un `migration.status` es acto exclusivo del operador (C-010 amendment):
estos tests LEEN estados, jamas los escriben.

Airflow real no esta instalado en el entorno local (`import airflow` resuelve al
directorio del repo), asi que se stubbea siguiendo el patron ya usado en
`tests/regression/test_watchdog_singleton_spawn.py`.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FACTORY = REPO / "airflow" / "dags" / "asset_pipeline_factory.py"


class _StubOperator:
    def __init__(self, *args, **kwargs):
        self.task_id = kwargs.get("task_id")
        self.python_callable = kwargs.get("python_callable")


def _load_factory():
    """Cargar el factory con Airflow stubbeado (patron del repo)."""
    stub_airflow = types.ModuleType("airflow")
    stub_airflow.DAG = object
    stub_operators = types.ModuleType("airflow.operators")
    stub_python = types.ModuleType("airflow.operators.python")
    stub_python.PythonOperator = _StubOperator
    stub_utils = types.ModuleType("airflow.utils")
    stub_dates = types.ModuleType("airflow.utils.dates")
    stub_dates.days_ago = lambda n: None
    stub_trigger = types.ModuleType("airflow.utils.trigger_rule")
    stub_trigger.TriggerRule = types.SimpleNamespace(ALL_SUCCESS="all_success")

    keys = {
        "airflow": stub_airflow,
        "airflow.operators": stub_operators,
        "airflow.operators.python": stub_python,
        "airflow.utils": stub_utils,
        "airflow.utils.dates": stub_dates,
        "airflow.utils.trigger_rule": stub_trigger,
    }
    saved = {k: sys.modules.get(k) for k in keys}
    sys.modules.update(keys)
    try:
        spec = importlib.util.spec_from_file_location("_c010_factory", FACTORY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


@pytest.fixture(scope="module")
def factory():
    return _load_factory()


def test_no_policy_runs_means_no_tasks_and_no_imports(factory):
    """Invariante 1: el arbol actual no gana ni una tarea."""
    assert factory.resolve_policy_runs({}) == []
    assert factory.resolve_policy_runs({"policy_runs": []}) == []


def test_current_configs_declare_no_policy_runs(factory):
    """El delta cero de hoy es un hecho del config, no una opinion."""
    config = factory._load_config()
    for asset_id, spec in (config.get("assets") or {}).items():
        assert not (spec.get("policy_runs") or []), (
            f"{asset_id} declara policy_runs; este candado debe revisarse a conciencia"
        )


def test_ineligible_status_yields_zero_tasks_not_a_green_skip(factory):
    """Invariante 2: los cuatro specs vigentes son inertes, y deben serlo."""
    from src.strategies.policies.loader import load_all_policy_specs

    every_id = [str(spec["id"]) for spec in load_all_policy_specs()]
    resolved = factory.resolve_policy_runs(
        {"policy_runs": [{"policy_id": pid} for pid in every_id]}
    )
    assert resolved == [], (
        "ningun spec vigente es PARITY_GREEN/CUTOVER; emitir tareas seria activar "
        f"una policy sin decision del operador (resuelto={resolved})"
    )


def test_duplicate_policy_id_fails_at_parse(factory):
    with pytest.raises(factory.PolicyRunConfigError, match="duplicado"):
        factory.resolve_policy_runs(
            {"policy_runs": [{"policy_id": "btc_hodl_b1"}, {"policy_id": "btc_hodl_b1"}]}
        )


def test_unknown_policy_id_fails_at_parse(factory):
    with pytest.raises(factory.PolicyRunConfigError, match="loader SSOT no conoce"):
        factory.resolve_policy_runs({"policy_runs": [{"policy_id": "no_existe_v9"}]})


def test_malformed_entry_fails_at_parse(factory):
    with pytest.raises(factory.PolicyRunConfigError):
        factory.resolve_policy_runs({"policy_runs": ["btc_hodl_b1"]})


def test_chain_actually_calls_resolve_feature_snapshot(factory, monkeypatch):
    """Invariante 4: el candado causal del caller.

    Si alguien retira la llamada, esto cae — que es justo lo que faltaba: el
    mecanismo existia y nadie lo invocaba.
    """
    called = {}

    def _spy(observations, *, decision_cutoff):
        called["observations"] = observations
        called["cutoff"] = decision_cutoff
        return {"rsi_9": 1.0}

    import src.orchestration.feature_snapshot as snap

    monkeypatch.setattr(snap, "resolve_feature_snapshot", _spy)

    class _TI:
        def xcom_pull(self, key=None, task_ids=None):
            if key and key.startswith("observations"):
                return {"rsi_9": {"value": 1.0, "available_at": "2026-01-01T00:00:00+00:00"}}
            if key and key.startswith("decision_cutoff"):
                return "2026-01-02T00:00:00+00:00"
            return None

    result = factory.make_resolve_snapshot("btc_hodl_b1")(ti=_TI())
    assert called, "la tarea no invoco resolve_feature_snapshot"
    assert result == {"rsi_9": 1.0}


def test_feature_available_after_cutoff_never_reaches_publish(factory):
    """Invariante 5: el cutoff corta antes de evaluar, luego no se publica."""
    from src.orchestration.feature_snapshot import FeatureSnapshotError

    class _TI:
        def xcom_pull(self, key=None, task_ids=None):
            if key and key.startswith("observations"):
                return {"rsi_9": {"value": 1.0, "available_at": "2026-06-01T00:00:00+00:00"}}
            if key and key.startswith("decision_cutoff"):
                return "2026-01-01T00:00:00+00:00"
            return None

    with pytest.raises(FeatureSnapshotError, match="exceeds decision_cutoff"):
        factory.make_resolve_snapshot("btc_hodl_b1")(ti=_TI())


def test_missing_inputs_fail_closed_instead_of_evaluating_blind(factory):
    class _TI:
        def xcom_pull(self, key=None, task_ids=None):
            return None

    with pytest.raises(factory.PolicyRunConfigError, match="no se evalua a ciegas"):
        factory.make_resolve_snapshot("btc_hodl_b1")(ti=_TI())

    with pytest.raises(factory.PolicyRunConfigError, match="no se publica"):
        factory.make_publish_signal("btc_hodl_b1")(ti=_TI())
