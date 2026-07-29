"""El corta-circuitos por drawdown de H5-L6 debe poder dispararse.

Origen (2026-07-29): `forecast_h5_l6_weekly_monitor.py` comparaba
`metrics["running_max_dd_pct"]` — construido sumando `forecast_h5_executions.week_pnl_pct`,
que sus productores escriben como FRACCION DECIMAL — contra el umbral del SSOT
`guardrails.circuit_breaker.max_drawdown_pct: 12.0`, que esta en PUNTOS PORCENTUALES.
Un drawdown real del -12 % llegaba como -0.12 y `-0.12 <= -12.0` es FALSO: la condicion
era insatisfacible y el corta-circuitos llevaba muerto desde que existe.
Evidencia por columna: `.claude/coordination/integration/PCT-COLUMNS-INVENTORY.md` §2.1, §3.4.

Estos tests fijan tres cosas:
  1. un drawdown del -12 % dispara (limite exacto) y uno del -5 % no;
  2. dispara IGUAL bajo las dos convenciones de unidad — la del proyecto es una decision
     abierta (BL-42), asi que el monitor deduce la escala de los precios de la propia fila
     en vez de asumirla;
  3. si la escala no se puede determinar, o dos productores escriben unidades distintas
     sobre la misma tabla, se LANZA. Un guardarrail inevaluable no se evalua en silencio.

La mutacion que los pone rojos es volver a comparar el valor crudo contra el umbral.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MONITOR_PATH = REPO_ROOT / "airflow" / "dags" / "forecast_h5_l6_weekly_monitor.py"

# El SSOT congelado del que sale el umbral que se esta probando.
SSOT_PATH = REPO_ROOT / "config" / "execution" / "smart_simple_v1.yaml"


# =============================================================================
# Carga del DAG con Airflow stubbeado (mismo patron que test_watchdog_singleton_spawn.py)
# =============================================================================

def _load_monitor():
    """Carga el DAG por RUTA, con Airflow y los helpers del contenedor stubbeados.

    El repo tiene su propio directorio `airflow/`, que ensombrece al paquete real, y los
    modulos `utils.*` / `contracts.*` solo estan en el PYTHONPATH dentro del contenedor
    (`sys.path.insert(0, '/opt/airflow')`). Por eso se inyectan en `sys.modules` antes de
    ejecutar el modulo, en vez de confiar en el import normal.
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

        # El fichero encadena tareas con `>>`; devolver el otro extremo permite `a >> b >> c`.
        def __rshift__(self, other):
            return other

        def __rrshift__(self, other):
            return self

    class _StubTriggerRule:
        ALL_DONE = "all_done"

    stubs: dict[str, types.ModuleType] = {}

    stubs["airflow"] = types.ModuleType("airflow")
    stubs["airflow"].DAG = _StubDAG
    stubs["airflow.operators"] = types.ModuleType("airflow.operators")
    stubs["airflow.operators.python"] = types.ModuleType("airflow.operators.python")
    stubs["airflow.operators.python"].PythonOperator = _StubOperator
    stubs["airflow.utils"] = types.ModuleType("airflow.utils")
    stubs["airflow.utils.trigger_rule"] = types.ModuleType("airflow.utils.trigger_rule")
    stubs["airflow.utils.trigger_rule"].TriggerRule = _StubTriggerRule

    stubs["utils"] = types.ModuleType("utils")
    stubs["utils.run_status"] = types.ModuleType("utils.run_status")
    stubs["utils.run_status"].honest_leaf = lambda fn=None: fn
    stubs["utils.dag_common"] = types.ModuleType("utils.dag_common")
    stubs["utils.dag_common"].get_db_connection = lambda *a, **k: pytest.fail(
        "el test debe inyectar su propia conexion"
    )

    stubs["contracts"] = types.ModuleType("contracts")
    stubs["contracts.dag_registry"] = types.ModuleType("contracts.dag_registry")
    stubs["contracts.dag_registry"].FORECAST_H5_L6_WEEKLY_MONITOR = "forecast_h5_l6_weekly_monitor"
    stubs["contracts.dag_registry"].get_dag_tags = lambda dag_id: []

    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("_h5_l6_monitor_under_test", MONITOR_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


# =============================================================================
# Dobles de DB / XCom
# =============================================================================

class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql, params=None):
        assert "forecast_h5_executions" in sql

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows
        self.closed = False

    def cursor(self):
        return _FakeCursor(self._rows)

    def close(self):
        self.closed = True


class _FakeTI:
    """XCom en memoria: `compute_metrics` publica y `check_gates` consume."""

    def __init__(self, store):
        self.store = store

    def xcom_pull(self, key=None, task_ids=None):
        return self.store.get(key)

    def xcom_push(self, key=None, value=None):
        self.store[key] = value


ENTRY_PRICE = 1000.0


def _week_row(stored_value, true_return_decimal, direction=1, leverage=1.0):
    """Una fila de `forecast_h5_executions`.

    `true_return_decimal` fija los PRECIOS (la verdad independiente y verificable);
    `stored_value` es lo que la columna `week_pnl_pct` afirma — el campo cuya unidad
    esta en disputa. Separarlos es justamente lo que permite deducir la escala.
    """
    exit_price = ENTRY_PRICE * (1.0 + direction * true_return_decimal / leverage)
    return ("2026-01-05", direction, stored_value, ENTRY_PRICE, exit_price, leverage)


def _rows_from_decimal_returns(returns_decimal, stored_scale=1.0):
    """`stored_scale=1.0` => la DB guarda decimales (convencion actual);
    `stored_scale=100.0` => la DB guarda puntos porcentuales (convencion post-BL-42)."""
    return [_week_row(r * stored_scale, r) for r in returns_decimal]


def _run_monitor(monitor, rows):
    """Encadena compute_metrics -> check_gates sobre un historico dado."""
    monitor.get_db_connection = lambda *a, **k: _FakeConn(rows)
    monitor.PROJECT_ROOT = REPO_ROOT  # lee el SSOT congelado REAL, no una copia del test

    store = {"results": {"found": True, "week_pnl_pct": rows[-1][2]}}
    ti = _FakeTI(store)
    metrics = monitor.compute_metrics(ti=ti)
    gates = monitor.check_gates(ti=ti)
    return metrics, gates


@pytest.fixture(scope="module")
def monitor():
    return _load_monitor()


# =============================================================================
# El umbral que se esta probando sale del SSOT, no de una constante del test
# =============================================================================

def test_threshold_under_test_is_the_frozen_ssot_one(monitor):
    """Si alguien cambia el umbral del SSOT, estos tests deben dejar de mentir."""
    import yaml

    with open(SSOT_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["guardrails"]["circuit_breaker"]["max_drawdown_pct"] == 12.0, (
        "el umbral del SSOT ya no es 12.0; actualiza las series de estos tests"
    )


# =============================================================================
# 1. El defecto: un -12 % real tiene que disparar
# =============================================================================

def test_a_real_twelve_percent_drawdown_trips_the_breaker(monitor):
    """ROJO con el codigo anterior: -0.12 <= -12.0 es FALSO, nunca disparaba.

    Cuatro semanas de -3 % = -12 % de drawdown acumulado, exactamente el umbral.
    Son 4 perdidas consecutivas (< 5), asi que el unico guardarrail que puede
    dispararse aqui es el de drawdown: el test no puede aprobar por el motivo erroneo.
    """
    rows = _rows_from_decimal_returns([-0.03, -0.03, -0.03, -0.03])
    metrics, gates = _run_monitor(monitor, rows)

    assert metrics["consecutive_losses"] == 4, "no debe disparar por perdidas consecutivas"
    assert gates["circuit_breaker"] is True, (
        "un drawdown REAL del -12 % no disparo el corta-circuitos: la comparacion contra "
        f"max_drawdown_pct=12.0 se hizo con running_max_dd_pct={metrics['running_max_dd_pct']}, "
        "que jamas puede ser <= -12.0. El guardarrail esta muerto."
    )
    assert any("Cumulative DD" in a for a in gates["alarms"])
    assert metrics["running_max_dd_pct"] == pytest.approx(-12.0), (
        "running_max_dd_pct debe publicarse en PUNTOS PORCENTUALES (-12.0), que es la "
        "unidad del umbral del SSOT y la del dato historico ya escrito"
    )


def test_the_exact_boundary_trips(monitor):
    """El limite es inclusivo (`<=`): justo en -12.00 dispara, en -11.99 no."""
    at_threshold = _rows_from_decimal_returns([-0.12])
    just_inside = _rows_from_decimal_returns([-0.1199])

    _, gates_at = _run_monitor(monitor, at_threshold)
    metrics_inside, gates_inside = _run_monitor(monitor, just_inside)

    assert gates_at["circuit_breaker"] is True
    assert metrics_inside["running_max_dd_pct"] == pytest.approx(-11.99)
    assert gates_inside["circuit_breaker"] is False


def test_a_drawdown_clearly_below_the_threshold_does_not_trip(monitor):
    """-5 % esta claramente por debajo: ni dispara ni levanta alarma de DD."""
    rows = _rows_from_decimal_returns([-0.02, -0.03])
    metrics, gates = _run_monitor(monitor, rows)

    assert metrics["running_max_dd_pct"] == pytest.approx(-5.0)
    assert gates["circuit_breaker"] is False
    assert not any("Cumulative DD" in a for a in gates["alarms"])


# =============================================================================
# 2. Independencia de la convencion de unidades (decision abierta del operador)
# =============================================================================

def test_it_also_trips_when_the_column_stores_percentage_points(monitor):
    """Si BL-42 fase 2 aterriza y la columna pasa a puntos porcentuales, sigue disparando.

    Mismos precios, mismo drawdown real; solo cambia lo que la columna afirma. Un arreglo
    que hubiera hardcodeado `* 100` fallaria aqui multiplicando por segunda vez.
    """
    rows = _rows_from_decimal_returns([-0.03] * 4, stored_scale=100.0)
    metrics, gates = _run_monitor(monitor, rows)

    assert metrics["return_scale_to_points"] == 1.0
    assert metrics["running_max_dd_pct"] == pytest.approx(-12.0)
    assert gates["circuit_breaker"] is True


def test_the_resolved_scale_is_published_for_the_comparison(monitor):
    """La unidad viaja DECLARADA junto al valor; el umbral no se compara contra un anonimo."""
    rows = _rows_from_decimal_returns([-0.03] * 4)
    metrics, _ = _run_monitor(monitor, rows)
    assert metrics["return_scale_to_points"] == 100.0


# =============================================================================
# 3. Fail-loud: lo indeterminable no se compara en silencio
# =============================================================================

def test_two_writers_with_different_units_raise_instead_of_comparing(monitor):
    """El escenario que predice el inventario (§3.4): dos productores, dos unidades.

    Mezclar decimales y puntos porcentuales en la misma columna hace que CUALQUIER
    drawdown agregado sea basura. Debe gritar, no devolver un veredicto.
    """
    rows = _rows_from_decimal_returns([-0.03, -0.03])
    rows += _rows_from_decimal_returns([-0.03, -0.03], stored_scale=100.0)

    with pytest.raises(monitor.ReturnUnitContractError, match="DOS unidades"):
        _run_monitor(monitor, rows)


def test_a_stored_value_matching_neither_scale_raises(monitor):
    """Si el valor almacenado no casa con los precios en ninguna escala, no hay unidad."""
    rows = [_week_row(stored_value=42.0, true_return_decimal=-0.03)]

    with pytest.raises(monitor.ReturnUnitContractError, match="0 filas concluyentes"):
        _run_monitor(monitor, rows)


def test_gates_refuse_metrics_without_a_declared_unit(monitor):
    """Perimetro: cualquier futuro productor de metricas debe declarar la unidad.

    Rojo si alguien vuelve a comparar `running_max_dd_pct` contra el umbral sin saber
    en que escala viene — que es exactamente el defecto original.
    """
    monitor.PROJECT_ROOT = REPO_ROOT
    store = {
        "results": {"found": True, "week_pnl_pct": -0.12},
        "metrics": {
            "n_weeks": 4,
            "running_max_dd_pct": -0.12,   # unidad desconocida
            "long_pct_8w": 0.0,
            "consecutive_losses": 1,
            # sin `return_scale_to_points`
        },
    }
    with pytest.raises(monitor.ReturnUnitContractError, match="return_scale_to_points"):
        monitor.check_gates(ti=_FakeTI(store))
