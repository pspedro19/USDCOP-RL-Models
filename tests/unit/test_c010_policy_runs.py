# -*- coding: utf-8 -*-
"""C-010 R3: la cadena gobernada solo existe para referencias ELEGIBLES.

`resolve_feature_snapshot` aplica `available_at <= decision_cutoff` y hasta ahora
tenia **cero llamadores productivos** (`AUDIT-CLAUDE-wiring-gap.md`). C-010 le da
uno: el factory emite `resolve_snapshot -> evaluate -> publish` para cada
`policy_runs[].policy_id` cuyo `migration.status` sea `PARITY_GREEN|CUTOVER`,
ramificando por `engine.type` y **nunca** por `strategy_id`.

Invariantes que fijan estos candados:

1. Sin `policy_runs` declarados el grafo de tareas es **identico**, y esa ausencia
   de delta no depende de criterio.
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


#: DAG stub "actual" mientras se ejecuta `with dag:` — asi los operadores se
#: registran solos, igual que en Airflow real.
_ABIERTO: list = []


class _StubOperator:
    def __init__(self, *args, **kwargs):
        self.task_id = kwargs.get("task_id")
        self.python_callable = kwargs.get("python_callable")
        self.trigger_rule = kwargs.get("trigger_rule")
        if _ABIERTO:
            _ABIERTO[-1]._registrar(self)

    def __rshift__(self, other):
        """Registrar la ARISTA. Sin esto el grafo no es observable y la cadena
        `resolve -> validate -> evaluate -> publish` solo se podria comprobar
        leyendo el fuente como texto — que no prueba que se ejecute asi."""
        if _ABIERTO:
            _ABIERTO[-1].aristas.append((self.task_id, other.task_id))
        return other


class _StubDag:
    def __init__(self, *args, **kwargs):
        self.dag_id = kwargs.get("dag_id")
        self.tasks: dict[str, _StubOperator] = {}
        self.aristas: list[tuple[str, str]] = []

    def _registrar(self, task):
        self.tasks[task.task_id] = task

    def __enter__(self):
        _ABIERTO.append(self)
        return self

    def __exit__(self, *exc):
        _ABIERTO.pop()
        return False

    def get_task(self, task_id):
        return self.tasks[task_id]


def _load_factory():
    """Cargar el factory con Airflow stubbeado (patron del repo)."""
    stub_airflow = types.ModuleType("airflow")
    stub_airflow.DAG = _StubDag
    stub_operators = types.ModuleType("airflow.operators")
    stub_python = types.ModuleType("airflow.operators.python")
    stub_python.PythonOperator = _StubOperator
    stub_utils = types.ModuleType("airflow.utils")
    stub_dates = types.ModuleType("airflow.utils.dates")
    stub_dates.days_ago = lambda n: None
    stub_trigger = types.ModuleType("airflow.utils.trigger_rule")
    stub_trigger.TriggerRule = types.SimpleNamespace(
        ALL_SUCCESS="all_success", ALL_DONE="all_done"
    )

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


SSOT_CONFIG = REPO / "config" / "assets" / "pipelines.yaml"
POLICY_SPEC_DIR = REPO / "config" / "policies"


def _statuses_read_independently() -> dict[str, str | None]:
    """`policy_id -> migration.status`, leido DIRECTAMENTE de los YAML.

    A proposito **no** pasa por `load_all_policy_specs()` ni por
    `resolve_policy_runs`: si lo esperado y lo observado salieran del mismo
    resolver, el test compararia una funcion consigo misma y daria verde con el
    resolver roto. Es el error exacto que cometi en BL-20 (CXD-584) y no se repite.
    """
    import yaml

    out: dict[str, str | None] = {}
    for path in sorted(POLICY_SPEC_DIR.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        policy_id = doc.get("id")
        assert isinstance(policy_id, str) and policy_id, f"{path.name}: spec sin `id`"
        out[policy_id] = (doc.get("migration") or {}).get("status")
    return out


def test_every_declared_policy_run_has_a_registered_eligible_decision(factory, monkeypatch):
    """El SSOT solo declara para ejecucion policies con decision registrada.

    ESTE TEST ERA UN VERDE VACUO (CLD-553, concedido en CXD-593). Afirmaba que
    ningun activo declaraba `policy_runs` y pasaba **sin comprobar nada**:
    `CONFIG_PATH` apunta a `/opt/airflow/...`, que es la ruta del CONTENEDOR y no
    existe en host; `_load_config()` se traga el FileNotFoundError con
    `except Exception: return {}` y el bucle iteraba sobre un dict vacio. Mientras
    tanto `spx500` **si** declara `policy_runs` desde `04fa2dd2`. O sea: el candado
    que debia avisar del cambio no podia verlo.

    Dos reparaciones, y ninguna es aflojar la asercion:
      1. se INYECTA el `CONFIG_PATH` real del repo, y si el fichero no se puede
         leer el test **falla** — un juez que no ve a su sujeto no absuelve;
      2. el invariante deja de ser "nadie declara nada" (caducado el dia que el
         operador promovio spx500) y pasa a ser el que de verdad protege:
         *nada se declara para ejecucion sin una decision de migracion elegible
         registrada en su spec*.
    """
    monkeypatch.setattr(factory, "CONFIG_PATH", SSOT_CONFIG)
    config = factory._load_config()
    assert config, (
        f"`_load_config()` devolvio vacio leyendo {SSOT_CONFIG}. Antes esto daba VERDE "
        f"porque el bucle no llegaba a iterar; ahora es rojo a proposito."
    )
    assets = config.get("assets") or {}
    assert assets, "el SSOT no expone `assets`: el candado se quedaria sin sujeto"

    statuses = _statuses_read_independently()
    declarados = [
        (asset_id, entry.get("policy_id"))
        for asset_id, spec in assets.items()
        for entry in (spec.get("policy_runs") or [])
    ]
    assert declarados, (
        "ningun activo declara `policy_runs`. No es ilegal, pero deja este candado sin "
        "nada que juzgar: si se retiro la declaracion de spx500 (registrada en `04fa2dd2`) "
        "hay que retirar tambien el candado a conciencia, no dejarlo pasando en vacio."
    )

    for asset_id, policy_id in declarados:
        assert policy_id in statuses, (
            f"{asset_id} declara `{policy_id}`, que no existe en {POLICY_SPEC_DIR.name}/"
        )
        assert statuses[policy_id] in factory.ELIGIBLE_MIGRATION_STATES, (
            f"{asset_id} declara `{policy_id}` para ejecucion con "
            f"migration.status={statuses[policy_id]!r}: seria activar una policy sin "
            f"decision del operador. Elegibles: {sorted(factory.ELIGIBLE_MIGRATION_STATES)}"
        )


def test_resolution_obeys_the_registered_status_not_the_declaration(factory):
    """Invariante 2: declarar no es activar; lo que activa es el estado registrado.

    Se le pasan al resolver **todas** las policies del repo y se exige que devuelva
    exactamente las elegibles — con lo esperado leido aparte del YAML. La version
    anterior exigia `resolved == []` sobre la premisa "los cuatro specs vigentes son
    inertes", que caduco cuando spx500 llego a PARITY_GREEN: llevaba en rojo sin que
    nadie lo mirara, y el rojo era correcto.
    """
    statuses = _statuses_read_independently()
    elegibles = sorted(
        pid for pid, st in statuses.items() if st in factory.ELIGIBLE_MIGRATION_STATES
    )
    inertes = sorted(
        pid for pid, st in statuses.items() if st not in factory.ELIGIBLE_MIGRATION_STATES
    )
    # Anti-vacuidad de las DOS particiones: con una sola poblada el test no
    # distingue "filtra bien" de "devuelve todo" ni de "devuelve nada".
    assert elegibles and inertes, (
        f"el corpus de specs perdio una de las dos particiones (elegibles={elegibles}, "
        f"inertes={inertes}); asi este candado no puede probar que el filtro filtra"
    )

    resolved = factory.resolve_policy_runs(
        {"policy_runs": [{"policy_id": pid} for pid in sorted(statuses)]}
    )
    assert sorted(r["policy_id"] for r in resolved) == elegibles, (
        f"resuelto={sorted(r['policy_id'] for r in resolved)} vs elegibles por estado "
        f"registrado={elegibles}"
    )


def test_demoting_the_eligible_policy_yields_zero_tasks_not_a_green_skip(factory, monkeypatch):
    """Mutacion causal (CXD-593 c): la MISMA policy, degradada, emite CERO tareas.

    Sin mutacion resuelve; con `PARITY_PENDING` no resuelve. Ese delta es lo que
    prueba que quien manda es el estado y no la declaracion: un test que solo
    mirase el estado feliz de hoy pasaria igual con el filtro cableado a `True`.
    """
    import copy

    import src.strategies.policies.loader as loader

    declarado = {"policy_runs": [{"policy_id": "spx500_daily_ma200_v1"}]}

    base = factory.resolve_policy_runs(declarado)
    assert [r["policy_id"] for r in base] == ["spx500_daily_ma200_v1"], (
        "la mutacion no probaria nada si el estado sano ya diera cero tareas"
    )

    original = loader.load_all_policy_specs

    def _degradada():
        specs = copy.deepcopy(list(original()))
        for spec in specs:
            if str(spec.get("id")) == "spx500_daily_ma200_v1":
                spec.setdefault("migration", {})["status"] = "PARITY_PENDING"
        return specs

    monkeypatch.setattr(loader, "load_all_policy_specs", _degradada)
    assert factory.resolve_policy_runs(declarado) == [], (
        "una policy degradada a PARITY_PENDING siguio emitiendo cadena: el filtro por "
        "`migration.status` no es causal"
    )


def _dag_de_spx500(factory):
    """Construir el DAG real de `spx500` con el SSOT real y devolverlo observable."""
    import yaml

    config = yaml.safe_load(SSOT_CONFIG.read_text(encoding="utf-8")) or {}
    spec = (config.get("assets") or {})["spx500"]
    return factory._build_asset_dag("spx500", spec, "usdcop-trading-dashboard/public/data")


def test_the_governed_chain_has_the_four_declared_links_in_order(factory):
    """R3: `resolve -> validate -> evaluate -> publish` es OBSERVABLE en el grafo.

    Hasta R3 el segundo eslabon vivia DENTRO del tercero: una validacion fallida no
    se distinguia de un fallo de evaluacion. Ahora es tarea propia, y este candado
    mira las ARISTAS del DAG construido —no el texto del fuente—, porque lo que
    importa es como queda cableado, no como esta escrito.

    Rojo con: quitar `chain[1]` del encadenado, o reordenar validate y evaluate.
    """
    dag = _dag_de_spx500(factory)
    pid = "spx500_daily_ma200_v1"
    esperado = [
        f"policy_{pid}_resolve_snapshot",
        f"policy_{pid}_validate_inputs",
        f"policy_{pid}_evaluate",
        f"policy_{pid}_publish",
    ]
    for task_id in esperado:
        assert task_id in dag.tasks, (
            f"la cadena gobernada no emitio `{task_id}`. Emitidas: {sorted(dag.tasks)}"
        )
    consecutivas = list(zip(esperado, esperado[1:]))
    faltan = [par for par in consecutivas if par not in dag.aristas]
    assert not faltan, (
        f"la cadena no esta encadenada en el orden declarado; faltan las aristas "
        f"{faltan}. Aristas observadas: {[a for a in dag.aristas if 'policy_' in a[0]]}"
    )
    # Y cuelga de `verify`: publicar señales antes de verificar el registry seria
    # emitir decisiones sobre un bundle no verificado.
    assert ("l6_verify_registry", esperado[0]) in dag.aristas, (
        "la cadena de politica no cuelga de l6_verify_registry"
    )


def test_validate_link_is_not_decorative_evaluate_honours_its_degraded_decision(factory):
    """El segundo eslabon DECIDE: si degrada a FLAT, evaluate no recalcula.

    Un `validate_inputs` que corre y cuyo resultado se ignora es peor que no tenerlo,
    porque el grafo muestra un control que no controla. Aqui se comprueba el unico
    hecho que lo hace real: la decision que sale de evaluate ES la degradada.
    """
    degradada = {"direction": "FLAT", "reason_codes": ["INPUT_STALE"], "_marca": "de-validate"}

    class _TI:
        def xcom_pull(self, key=None, task_ids=None):
            if task_ids and task_ids.endswith("_resolve_snapshot"):
                return {"rsi_9": 1.0}
            if task_ids and task_ids.endswith("_validate_inputs"):
                return degradada
            return None

    salida = factory.make_evaluate_policy("spx500_daily_ma200_v1")(ti=_TI())
    assert salida is degradada, (
        "evaluate ignoro la decision degradada del eslabon de validacion y siguio "
        "adelante: el control seria decorativo"
    )


def _spec_con(**cambios_engine):
    """Clonar los specs reales cambiando SOLO `engine.*` de spx500.

    Se materializa AQUI, antes de monkeypatchear: si se evaluara perezosamente
    dentro del lambda ya sustituido se llamaria a si misma (me paso: RecursionError).
    """
    import copy

    import src.strategies.policies.loader as loader

    specs = copy.deepcopy(list(loader.load_all_policy_specs()))
    for spec in specs:
        if str(spec.get("id")) == "spx500_daily_ma200_v1":
            spec.setdefault("engine", {}).update(cambios_engine)
    return specs


def test_retrain_other_than_never_fails_closed_instead_of_silently_skipping_train(
    factory, monkeypatch
):
    """La cadena no sabe entrenar; omitirlo en silencio daria decisiones caducadas.

    Una policy que declara reentrenamiento y recibe una cadena SIN tarea de train
    publicaria señales de un modelo viejo con el grafo entero en verde. Falla cerrado.
    """
    import src.strategies.policies.loader as loader

    specs = _spec_con(retrain="weekly")  # materializado ANTES del parche
    monkeypatch.setattr(loader, "load_all_policy_specs", lambda: specs)
    with pytest.raises(factory.PolicyRunConfigError, match="retrain"):
        factory.resolve_policy_runs(
            {"policy_runs": [{"policy_id": "spx500_daily_ma200_v1"}]}
        )


def test_retrain_is_read_from_engine_not_from_the_spec_root(factory, monkeypatch):
    """Candado de un bug MIO de hoy, declarado en CLD-553.

    Escribi la guarda leyendo `retrain` de la RAIZ del spec, donde no vive: daba
    `None` para las cuatro policies y habria disparado el fail-closed en TODAS las
    elegibles. Una guarda que se activa siempre no protege, bloquea — y el sintoma
    (cero cadenas emitidas) se parece muchisimo a "no hay nada declarado".

    Este caso es el discriminante: `engine.retrain=never` (legitimo) con un
    `retrain: weekly` señuelo en la raiz. Leyendo bien, resuelve; leyendo de la raiz,
    revienta.
    """
    import src.strategies.policies.loader as loader

    specs = _spec_con(retrain="never")
    for spec in specs:
        if str(spec.get("id")) == "spx500_daily_ma200_v1":
            spec["retrain"] = "weekly"  # señuelo en la raiz

    monkeypatch.setattr(loader, "load_all_policy_specs", lambda: specs)
    resuelto = factory.resolve_policy_runs(
        {"policy_runs": [{"policy_id": "spx500_daily_ma200_v1"}]}
    )
    assert [r["policy_id"] for r in resuelto] == ["spx500_daily_ma200_v1"], (
        "la guarda leyo `retrain` de la raiz del spec en vez de `engine.retrain`: "
        "bloquearia toda policy elegible"
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
