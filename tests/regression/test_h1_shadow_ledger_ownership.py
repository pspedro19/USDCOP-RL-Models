"""Un experimento RETIRADO no puede sobrescribir el ledger prospectivo de su sucesor.

Origen (2026-07-31, evidencia en el repo, no hipotesis):

- `.claude/specs/assets/usdcop/HYPOTHESIS-REGISTRY.md` declara `usdcop_h1_regime_shadow_v1`
  **retirado con 0 predicciones y 0 outcomes**.
- Su modulo `airflow/dags/forecast_h1_regime_shadow.py` sigue en disco, con cron de viernes
  15:30 COT y **las mismas tags** que su sucesor v2 (`forecasting/research/shadow/usdcop/h1`),
  o sea que un unico "despausa las shadow" los enciende a los dos.
- Su generador escribe **el mismo fichero** que hoy posee v2
  (`public/forecasting/usdcop/h1_regime_shadow_index.json`), cuyo propio campo `supersedes`
  nombra a v1.
- El orden de tareas del DAG es `generate >> verify`, y `verify_shadow_contract` solo
  comprueba la coherencia INTERNA del documento recien escrito. Cuando verifica, el
  sobrescrito ya ocurrio.

La constitucion (§5) hace del forward prospectivo el unico juez limpio de v11. Perder esa
cadena por reactivar un experimento retirado es un dato falso con aspecto de dato bueno.

La mutacion que pone estos tests en rojo es quitar la llamada a `assert_ledger_is_ours()`
al principio de `run_shadow_generator`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DAG_PATH = REPO_ROOT / "airflow" / "dags" / "forecast_h1_regime_shadow.py"


def _load_dag_module():
    """Carga el DAG por RUTA con Airflow stubbeado.

    El repo tiene su propio directorio `airflow/`, que ensombrece al paquete real, asi que
    se inyectan las dependencias en `sys.modules` antes de ejecutar el modulo.
    """

    class _StubDAG:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _StubOperator:
        def __init__(self, *args, **kwargs):
            pass

        def __rshift__(self, other):
            return other

        def __lshift__(self, other):
            return other

    stub_airflow = types.ModuleType("airflow")
    stub_airflow.DAG = _StubDAG
    stub_operators = types.ModuleType("airflow.operators")
    stub_python = types.ModuleType("airflow.operators.python")
    stub_python.PythonOperator = _StubOperator
    stub_utils = types.ModuleType("airflow.utils")
    stub_dates = types.ModuleType("airflow.utils.dates")
    stub_dates.days_ago = lambda n: None

    keys = (
        "airflow",
        "airflow.operators",
        "airflow.operators.python",
        "airflow.utils",
        "airflow.utils.dates",
    )
    saved = {k: sys.modules.get(k) for k in keys}
    sys.modules["airflow"] = stub_airflow
    sys.modules["airflow.operators"] = stub_operators
    sys.modules["airflow.operators.python"] = stub_python
    sys.modules["airflow.utils"] = stub_utils
    sys.modules["airflow.utils.dates"] = stub_dates
    try:
        spec = importlib.util.spec_from_file_location("_h1_regime_shadow_v1_under_test", DAG_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


@pytest.fixture
def dag_module():
    return _load_dag_module()


def _write_index(module, tmp_path: Path, payload: dict) -> Path:
    index = tmp_path / "h1_regime_shadow_index.json"
    index.write_text(json.dumps(payload), encoding="utf-8")
    module.INDEX = index
    return index


def test_refuses_when_the_ledger_belongs_to_the_successor(dag_module, tmp_path):
    """El caso real: el ledger en disco declara v2 y este DAG es v1."""
    _write_index(dag_module, tmp_path, {"experiment_id": "usdcop_h1_regime_shadow_v2"})
    with pytest.raises(RuntimeError, match="REFUSING TO RUN"):
        dag_module.assert_ledger_is_ours()


def test_the_refusal_happens_before_the_subprocess(dag_module, tmp_path, monkeypatch):
    """No basta con que exista el guard: tiene que cortar ANTES de escribir.

    Si `run_shadow_generator` llegara a `subprocess.run`, el generador ya habria
    reescrito el indice — que es exactamente el fallo que este candado existe para evitar.
    """
    _write_index(dag_module, tmp_path, {"experiment_id": "usdcop_h1_regime_shadow_v2"})

    def _explode(*args, **kwargs):  # pragma: no cover - debe no ejecutarse
        raise AssertionError("subprocess.run se ejecuto pese al ledger ajeno")

    monkeypatch.setattr(dag_module.subprocess, "run", _explode)
    with pytest.raises(RuntimeError, match="REFUSING TO RUN"):
        dag_module.run_shadow_generator()


def test_an_unlabelled_ledger_is_treated_as_foreign(dag_module, tmp_path):
    """Fail-closed: sin `experiment_id` no hay prueba de propiedad, y eso no es permiso."""
    _write_index(dag_module, tmp_path, {"records": [], "status": "awaiting_first_commit"})
    with pytest.raises(RuntimeError, match="REFUSING TO RUN"):
        dag_module.assert_ledger_is_ours()


def test_a_corrupt_ledger_is_not_a_free_pass(dag_module, tmp_path):
    """Un JSON ilegible tampoco autoriza: se para, no se asume propiedad."""
    index = tmp_path / "h1_regime_shadow_index.json"
    index.write_text("{not json", encoding="utf-8")
    dag_module.INDEX = index
    with pytest.raises(RuntimeError, match="cannot read the ledger"):
        dag_module.assert_ledger_is_ours()


def test_its_own_ledger_is_allowed(dag_module, tmp_path):
    """Control positivo anti-vacuidad: si el guard rechazara TODO, no probaria nada."""
    _write_index(dag_module, tmp_path, {"experiment_id": "usdcop_h1_regime_shadow_v1"})
    dag_module.assert_ledger_is_ours()  # no debe lanzar


def test_a_missing_ledger_is_allowed(dag_module, tmp_path):
    """Sin fichero no hay nada que sobrescribir; el guard no debe bloquear el bootstrap."""
    dag_module.INDEX = tmp_path / "no-existe.json"
    dag_module.assert_ledger_is_ours()  # no debe lanzar


def test_the_live_ledger_on_disk_is_owned_by_v2_not_v1():
    """La premisa del candado, medida contra el fichero REAL del repo.

    Si algun dia el ledger vuelve a declarar v1, este test cae y obliga a revisar la
    deprecacion en vez de dejar el candado protegiendo una situacion que ya no existe.
    """
    index = (
        REPO_ROOT
        / "usdcop-trading-dashboard"
        / "public"
        / "forecasting"
        / "usdcop"
        / "h1_regime_shadow_index.json"
    )
    if not index.is_file():
        pytest.skip("el ledger publicado no esta en este checkout")
    document = json.loads(index.read_text(encoding="utf-8"))
    assert document.get("experiment_id") == "usdcop_h1_regime_shadow_v2"
    assert document.get("supersedes") == "usdcop_h1_regime_shadow_v1"


def test_v1_is_deprecated_and_v2_is_active_in_the_registry():
    """El candado del modulo y el registro deben contar la MISMA historia."""
    sys.path.insert(0, str(REPO_ROOT / "airflow" / "dags"))
    try:
        from contracts import dag_registry as reg
    except ImportError:  # pragma: no cover
        pytest.skip("dag_registry no importable en este entorno")
    assert "forecast_h1_regime_shadow" in reg.DEPRECATED_DAGS
    active = set(reg.get_active_dag_ids())
    assert "forecast_h1_regime_shadow" not in active
    assert "forecast_h1_regime_shadow_v2" in active
