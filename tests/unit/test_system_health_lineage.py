# -*- coding: utf-8 -*-
"""BL-25 — el reloj de DATOS mira el LINAJE de la estrategia viva, no sólo la frescura.

QUÉ CIERRA. `evaluate_data_clock` medía frescura de **fuentes** (m5, macro, seeds) y
**no consultaba** `lineage.node` ni `lineage.strategy_node`, aunque BL-24 ya había
entregado `status VALID/STALE/INVALIDATED` y los roles por estrategia. Consecuencia:
**un dato fresco cuyo nodo está `INVALIDATED` pasaba como «ok»** — el reloj daba verde
sobre linaje degradado, que es justo lo que FABRIC §23 quiere impedir.

POR QUÉ SÓLO `INPUT` Y `SIGNAL`. Son los roles que el sistema **materializa hoy**
(medido en la DB viva: 3 `INPUT` + 1 `SIGNAL` para `H5_PRODUCTION_STRATEGY_ID`).
`FEATURE` y `MODEL` existen en el `CHECK` de la tabla pero **nadie los enlaza**:
exigirlos pondría rojo algo que nadie ha prometido, y **un rojo falso gasta la misma
credibilidad que un verde falso** — la misma razón por la que `smart_simple_v11` quedó
fuera del gate cross-SSOT.

UNA CONFUSIÓN QUE VALE LA PENA DEJAR ESCRITA. `smart_simple_v11` es **dos cosas**: el
id de la estrategia de **producción** del track H5 (`h5_strategy_identity.py`) y un
**policy spec** `SPEC_ONLY` en `config/policies/`. Los nodos de linaje pertenecen al
**pipeline vivo**, no a la migración pendiente al motor de políticas. Yo mismo los
confundí al revisar el shape, y el nombre compartido lo invita.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DAG = REPO / "airflow" / "dags" / "control_system_health.py"

# El DAG importa `contracts.dag_registry`, que vive en `airflow/dags/contracts/` y NO
# en `src/contracts/`. Hay DOS paquetes que se llaman `contracts` y cuál gana depende
# del ORDEN de `sys.path` — que otras fixtures del conftest mutan globalmente
# (`tests/conftest.py:1000` hace `sys.path.insert(0, src)` dentro de una fixture).
# Depender de ese orden haría que este fichero pasara o fallara según qué otro test
# corriera antes: la clase de fragilidad que ya nos mordió con `test_publish_link`.
#
# Así que NO se pelea con el path: se carga el módulo por su ruta explícita y se
# registra, de modo que el `import contracts.dag_registry` del DAG lo encuentre ya
# resuelto pase lo que pase.
_DAGS_DIR = REPO / "airflow" / "dags"


def _preload_dag_contracts() -> None:
    if "contracts.dag_registry" in sys.modules:
        return
    pkg_init = _DAGS_DIR / "contracts" / "__init__.py"
    pkg_spec = importlib.util.spec_from_file_location(
        "contracts", pkg_init, submodule_search_locations=[str(pkg_init.parent)]
    )
    pkg = importlib.util.module_from_spec(pkg_spec)
    sys.modules["contracts"] = pkg
    pkg_spec.loader.exec_module(pkg)

    mod_spec = importlib.util.spec_from_file_location(
        "contracts.dag_registry", _DAGS_DIR / "contracts" / "dag_registry.py"
    )
    mod = importlib.util.module_from_spec(mod_spec)
    sys.modules["contracts.dag_registry"] = mod
    mod_spec.loader.exec_module(mod)


class _StubOperator:
    def __init__(self, *a, **k):
        self.task_id = k.get("task_id")

    def __rshift__(self, other):
        return other

    def __rrshift__(self, other):
        # El DAG hace `[t1, t2, t3] >> t_publish`: la lista no sabe encadenar, asi
        # que la derecha tiene que aceptar el reflejado. Sin esto el modulo entero
        # revienta al importarse y ningun candado llega a ejecutarse.
        return self


class _StubDag:
    """`DAG` usable como context manager. `object` no vale: el modulo hace
    `with DAG(...)` a nivel superior y `object()` no acepta kwargs."""

    def __init__(self, *a, **k):
        self.dag_id = k.get("dag_id")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _load_dag_module():
    """Cargar el DAG con Airflow stubbeado (patrón del repo)."""
    stub_airflow = types.ModuleType("airflow")
    stub_airflow.DAG = _StubDag
    stub_ops = types.ModuleType("airflow.operators")
    stub_py = types.ModuleType("airflow.operators.python")
    stub_py.PythonOperator = _StubOperator
    stub_utils = types.ModuleType("airflow.utils")
    stub_dates = types.ModuleType("airflow.utils.dates")
    stub_dates.days_ago = lambda n: None
    keys = {
        "airflow": stub_airflow,
        "airflow.operators": stub_ops,
        "airflow.operators.python": stub_py,
        "airflow.utils": stub_utils,
        "airflow.utils.dates": stub_dates,
    }
    _preload_dag_contracts()
    saved = {k: sys.modules.get(k) for k in keys}
    sys.modules.update(keys)
    try:
        spec = importlib.util.spec_from_file_location("_bl25_health", DAG)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@pytest.fixture(scope="module")
def health():
    return _load_dag_module()


class _Cursor:
    """Cursor que responde a la consulta de linaje y neutraliza las demás.

    Las otras consultas del reloj (m5, macro) devuelven `None` a propósito: sus
    probes quedan `missing` y **no interfieren** con lo que se juzga aquí, que es
    exclusivamente el bloque de linaje.
    """

    def __init__(self, filas_linaje, *, params_vistos):
        self._filas = filas_linaje
        self._params = params_vistos
        self._ultima_es_linaje = False

    def execute(self, sql, params=None):
        self._ultima_es_linaje = "lineage.strategy_node" in sql
        if self._ultima_es_linaje:
            # El id NO puede venir interpolado en el SQL: eso es lo que se fija.
            self._params.append((sql, params))

    def fetchall(self):
        return list(self._filas) if self._ultima_es_linaje else []

    def fetchone(self):
        return None


class _Conn:
    def __init__(self, cursor):
        self._c = cursor

    def cursor(self):
        return self._c

    def close(self):
        pass


def _probes(health, monkeypatch, filas, params_vistos=None):
    params_vistos = params_vistos if params_vistos is not None else []
    monkeypatch.setattr(
        health, "_get_db_connection",
        lambda *a, **k: _Conn(_Cursor(filas, params_vistos=params_vistos)),
    )
    # Rutas de fichero inexistentes: sus probes serán `missing`, no interfieren.
    monkeypatch.setattr(health, "DAILY_SEED", REPO / "no_existe_seed.parquet", raising=False)
    monkeypatch.setattr(health, "MACRO_CLEAN", REPO / "no_existe_macro.parquet", raising=False)
    capturados: dict = {}

    class _TI:
        def xcom_push(self, key=None, value=None):
            capturados[key] = value

    health.evaluate_data_clock(ti=_TI())
    return capturados


def _leer_probes(health, monkeypatch, filas):
    """Devuelve `{nombre: (status, active)}` sólo de los probes de linaje."""
    vistos = []
    monkeypatch.setattr(
        health, "_get_db_connection",
        lambda *a, **k: _Conn(_Cursor(filas, params_vistos=vistos)),
    )
    monkeypatch.setattr(health, "DAILY_SEED", REPO / "no_existe_seed.parquet", raising=False)
    monkeypatch.setattr(health, "MACRO_CLEAN", REPO / "no_existe_macro.parquet", raising=False)

    recogidos = []
    import src.monitoring.system_health_contract as contrato
    _Real = contrato.DataProbe

    class _Spy(_Real):
        def __init__(self, name, status, active_component=True, details=None):
            super().__init__(name, status, active_component, details or {})
            recogidos.append(self)

    # Se parchea el MODULO DE ORIGEN, no el del DAG: `evaluate_data_clock` hace
    # `from src.monitoring.system_health_contract import DataProbe` DENTRO de la
    # funcion, asi que el nombre se resuelve en cada llamada contra el origen y un
    # `setattr` sobre el modulo del DAG no lo verian nunca. Parcharlo ahi habria
    # dejado el espia sin recoger nada -- y el test verde por vacio.
    monkeypatch.setattr(contrato, "DataProbe", _Spy)
    try:
        health.evaluate_data_clock()
    except Exception:
        pass
    return (
        {p.name: (p.status, p.active_component) for p in recogidos
         if p.name.startswith("lineage_")},
        vistos,
    )


VALID_COMPLETO = [("INPUT", "VALID", 3), ("SIGNAL", "VALID", 1)]


def test_lineage_all_valid_is_ok(health, monkeypatch) -> None:
    """El caso sano: ambos roles enlazados y `VALID` ⇒ `ok`.

    Sin esta mitad, un probe cableado a `stale` pasaría los tests de degradación y
    el reloj estaría permanentemente rojo sin que nadie lo notase.
    """
    probes, _ = _leer_probes(health, monkeypatch, VALID_COMPLETO)
    assert probes == {"lineage_input": ("ok", True), "lineage_signal": ("ok", True)}


@pytest.mark.parametrize("estado", ["STALE", "INVALIDATED"])
def test_a_degraded_node_makes_the_active_probe_stale(health, monkeypatch, estado) -> None:
    """`STALE`/`INVALIDATED` en un nodo enlazado ⇒ probe ACTIVO degradado.

    Es el defecto que cierra BL-25: hasta aquí, un dato **fresco** cuyo nodo estaba
    `INVALIDATED` pasaba como «ok», porque el reloj no miraba el linaje.

    `active_component=True` importa tanto como el status: el motor sólo falla cerrado
    para componentes activos, así que marcarlo diagnóstico lo degradaría a un aviso.
    """
    probes, _ = _leer_probes(
        health, monkeypatch, [("INPUT", estado, 1), ("SIGNAL", "VALID", 1)]
    )
    assert probes["lineage_input"] == ("stale", True)
    assert probes["lineage_signal"] == ("ok", True), "SIGNAL sano no debe contagiarse"


def test_mutating_the_degraded_status_back_to_valid_loses_the_red(health, monkeypatch) -> None:
    """La mutación causal que pidió el shape: `STALE → VALID` pierde el rojo.

    Es lo que distingue «el probe mide el estado» de «el probe siempre dice stale».
    """
    degradado, _ = _leer_probes(health, monkeypatch, [("INPUT", "STALE", 1), ("SIGNAL", "VALID", 1)])
    sano, _ = _leer_probes(health, monkeypatch, VALID_COMPLETO)
    assert degradado["lineage_input"][0] == "stale"
    assert sano["lineage_input"][0] == "ok"


@pytest.mark.parametrize("ausente", ["INPUT", "SIGNAL"])
def test_a_missing_role_is_missing_ACTIVE_not_silence(health, monkeypatch, ausente) -> None:
    """Perder los links NO puede dar verde (anti-vacuidad).

    Si el rol no está enlazado, el probe no encontraría nada degradado **porque no
    encontraría nada** — la forma más cara de verde. Se exige `missing` y ACTIVO.

    Rojo con: saltarse el rol ausente en vez de emitir su probe.
    """
    filas = [(r, "VALID", 1) for r in ("INPUT", "SIGNAL") if r != ausente]
    probes, _ = _leer_probes(health, monkeypatch, filas)
    assert probes[f"lineage_{ausente.lower()}"] == ("missing", True)


def test_no_lineage_at_all_reports_both_roles_missing(health, monkeypatch) -> None:
    """Grafo vacío ⇒ los DOS roles `missing` activos, no ausencia de probes."""
    probes, _ = _leer_probes(health, monkeypatch, [])
    assert probes == {
        "lineage_input": ("missing", True),
        "lineage_signal": ("missing", True),
    }


def test_the_strategy_id_is_a_bound_parameter_never_interpolated(health, monkeypatch) -> None:
    """`strategy_id` va parametrizado, jamás concatenado en el SQL.

    Una tabla de linaje es exactamente donde no quieres construir SQL por
    concatenación; y además un id interpolado rompería el plan cacheado por cada
    estrategia distinta. Se comprueba que el literal NO aparece en el texto de la
    consulta y que SÍ llega como parámetro ligado.
    """
    from src.contracts.h5_strategy_identity import H5_PRODUCTION_STRATEGY_ID

    _, vistos = _leer_probes(health, monkeypatch, VALID_COMPLETO)
    assert vistos, "la consulta de linaje no llegó a ejecutarse"
    sql, params = vistos[0]
    assert H5_PRODUCTION_STRATEGY_ID not in sql, (
        f"el strategy_id aparece INTERPOLADO en el SQL: {sql}"
    )
    assert params == (H5_PRODUCTION_STRATEGY_ID,), (
        f"el strategy_id debe llegar como parámetro ligado, llegó {params!r}"
    )


def test_only_the_roles_the_system_actually_materialises_are_required(health) -> None:
    """`INPUT` y `SIGNAL`, ni más ni menos — y la razón, fijada.

    `FEATURE`/`MODEL` están en el `CHECK` de la tabla pero **nadie los enlaza** hoy.
    Exigirlos convertiría una promesa que nadie hizo en un rojo permanente; el día que
    se enlacen, este candado obliga a decidirlo a conciencia en vez de heredarlo.
    """
    assert health.LINEAGE_REQUIRED_ROLES == ("INPUT", "SIGNAL")
    assert health.LINEAGE_DEGRADED_STATUSES == ("STALE", "INVALIDATED")
