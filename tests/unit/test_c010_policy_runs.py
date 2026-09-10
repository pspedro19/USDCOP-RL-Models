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

    # CORRECCION DE ESTE CANDADO (2026-08-06, democion de spx500 a PARITY_PENDING).
    # Antes exigia que TODA referencia declarada tuviera estado elegible. Era MAS
    # ESTRICTO QUE EL DISENO: `resolve_policy_runs` acepta a proposito una referencia
    # inelegible y la deja INERTE ("cero tareas, nunca un skip verde"), porque
    # declarar es apuntar a una policy, no activarla. Con aquella version, demover
    # una policy —el acto de gobierno correcto mientras se revalida— obligaba ademas
    # a editar el SSOT de pipelines, o el candado se ponia rojo por hacer lo debido.
    #
    # El invariante que SI protege, y que se conserva entero: nada se EJECUTA sin
    # decision elegible registrada. Se comprueba contra el resolver, que es quien
    # decide, en vez de contra la declaracion, que solo referencia.
    for asset_id, policy_id in declarados:
        emitidas = factory.resolve_policy_runs(
            {"policy_runs": [{"policy_id": policy_id}]}
        )
        elegible = statuses[policy_id] in factory.ELIGIBLE_MIGRATION_STATES
        assert bool(emitidas) is elegible, (
            f"{asset_id}/{policy_id}: status={statuses[policy_id]!r} (elegible="
            f"{elegible}) pero el resolver emitio {emitidas}. Una policy no elegible "
            f"debe quedar INERTE, y una elegible debe emitir: cualquiera de las dos "
            f"al reves activa o apaga una estrategia sin decision registrada"
        )


def test_resolution_obeys_the_registered_status_not_the_declaration(factory, monkeypatch):
    """Invariante 2: declarar no es activar; lo que activa es el estado registrado.

    Se le pasan al resolver **todas** las policies del repo y se exige que devuelva
    exactamente las elegibles — con lo esperado leido aparte del YAML. La version
    anterior exigia `resolved == []` sobre la premisa "los cuatro specs vigentes son
    inertes", que caduco cuando spx500 llego a PARITY_GREEN: llevaba en rojo sin que
    nadie lo mirara, y el rojo era correcto.
    """
    # Tras la democion de spx500 (decision C) NO queda ninguna policy elegible en el
    # repo, asi que sobre el estado real este test se quedaria con una sola particion
    # y su propia guarda anti-vacuidad lo dice: no distinguiria "filtra bien" de
    # "no devuelve nada". Se promueve UNA en memoria para tener las dos pobladas.
    #
    # Lo esperado se sigue leyendo del `migration.status` del corpus, NO de
    # `resolve_policy_runs`: la independencia que importa es respecto a la funcion
    # bajo prueba, y esa se conserva entera.
    _elegible_en_memoria(factory, monkeypatch)
    from src.strategies.policies.loader import load_all_policy_specs

    statuses = {
        str(s["id"]): (s.get("migration") or {}).get("status")
        for s in load_all_policy_specs()
    }
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


def test_demoting_the_eligible_policy_yields_zero_tasks_not_a_green_skip(factory, monkeypatch):  # noqa: E501
    """Mutacion causal (CXD-593 c): la MISMA policy, degradada, emite CERO tareas.

    Sin mutacion resuelve; con `PARITY_PENDING` no resuelve. Ese delta es lo que
    prueba que quien manda es el estado y no la declaracion: un test que solo
    mirase el estado feliz de hoy pasaria igual con el filtro cableado a `True`.
    """
    import copy

    import src.strategies.policies.loader as loader

    declarado = {"policy_runs": [{"policy_id": "spx500_daily_ma200_v1"}]}

    _elegible_en_memoria(factory, monkeypatch)   # sujeto: ver la nota del helper
    base = factory.resolve_policy_runs(declarado)
    assert [r["policy_id"] for r in base] == ["spx500_daily_ma200_v1"], (
        "la mutacion no probaria nada si el estado sano ya diera cero tareas"
    )

    original = loader.load_all_policy_specs   # ya es la version promovida en memoria

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


def test_the_governed_chain_has_the_four_declared_links_in_order(factory, monkeypatch):
    """R3: `resolve -> validate -> evaluate -> publish` es OBSERVABLE en el grafo.

    Hasta R3 el segundo eslabon vivia DENTRO del tercero: una validacion fallida no
    se distinguia de un fallo de evaluacion. Ahora es tarea propia, y este candado
    mira las ARISTAS del DAG construido —no el texto del fuente—, porque lo que
    importa es como queda cableado, no como esta escrito.

    Rojo con: quitar `chain[1]` del encadenado, o reordenar validate y evaluate.
    """
    _elegible_en_memoria(factory, monkeypatch)
    dag = _dag_de_spx500(factory)
    pid = "spx500_daily_ma200_v1"
    # CINCO eslabones desde C2: el primero MATERIALIZA el snapshot. Antes la cadena
    # empezaba en `resolve_snapshot`, que hace `xcom_pull` de `observations::` — y
    # ninguna tarea productiva los ponia. Cuatro eslabones que ninguna corrida podia
    # atravesar: el mecanismo existia y no tenia productor.
    esperado = [
        f"policy_{pid}_produce_observations",
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


#: Probe END-TO-END (CXD-598). Los candados de R3 miraban las ARISTAS del grafo y
#: mi test de `validate` tomaba la salida temprana de `evaluate`, asi que **ningun
#: test ejecutaba el callable real por el camino sano**. Resultado: tres eslabones
#: que no podian correr —`build_policy(policy_id)` recibia un id donde espera un
#: spec, `context.get("ctx")` era siempre None, y publish llamaba `load_policy_spec`
#: con un id donde espera una ruta— con 14 tests en verde. Estructura observable no
#: es ejecucion observable, y solo lo demuestra invocar el callable de verdad.
PID = "spx500_daily_ma200_v1"
CTX_INTERVALO = {"data_interval_end": "2026-07-24T00:00:00+00:00"}


CUTOFF = "2026-07-24T21:00:00+00:00"


class _TIProbe:
    """`ti` REAL: xcom_push/pull de verdad, para que el hecho de frescura viaje
    por el mismo canal que en Airflow en vez de entrar por una clave magica."""

    def __init__(self, observations, cutoff=CUTOFF, degradada=None, decision=None):
        self._obs, self._cutoff = observations, cutoff
        self._degradada, self._decision = degradada, decision
        self.pushed: dict = {}

    def xcom_push(self, key=None, value=None):
        self.pushed[key] = value

    def xcom_pull(self, key=None, task_ids=None):
        if key and key.startswith("observations"):
            return self._obs
        if key and key.startswith("decision_cutoff"):
            return self._cutoff
        if key in self.pushed:
            return self.pushed[key]
        if task_ids and task_ids.endswith("_resolve_snapshot"):
            return {n: o["value"] for n, o in self._obs.items()}
        if task_ids and task_ids.endswith("_validate_inputs"):
            return self._degradada
        if task_ids and task_ids.endswith("_evaluate"):
            return self._decision
        return None


def _obs(close_at: str, ma_at: str | None = None) -> dict:
    """Observaciones con `available_at` REAL, y **por feature**.

    El parametro por feature existe porque la version anterior ponia el MISMO sello
    en todas: con edades homogeneas el test no distingue `max(available_at)` de
    `min(available_at)`, asi que no podia cazar que R5 midiera la observacion mas
    NUEVA. Fixture que no puede fallar en la dimension que importa — la misma clase
    de defecto que CODEX ya me encontro en BL-20 (CXD-576).
    """
    return {
        "close": {"value": 5200.0, "available_at": close_at},
        "ma_200": {"value": 5000.0, "available_at": ma_at or close_at},
    }


OBS_FRESCAS = _obs("2026-07-24T20:00:00+00:00")   # ambas 1h antes del cutoff
OBS_VIEJAS = _obs("2026-07-18T20:00:00+00:00")    # ambas 6 dias antes
#: El caso de CXD-603: `close` de hace 1h, `ma_200` de hace 6 dias.
OBS_MIXTAS = _obs("2026-07-24T20:00:00+00:00", "2026-07-18T20:00:00+00:00")
SNAPSHOT_SANO = {"close": 5200.0, "ma_200": 5000.0}


def _elegible_en_memoria(factory, monkeypatch, policy_id=None):
    """Promover una policy a `PARITY_GREEN` **solo en memoria**, para tener sujeto.

    Desde la democion de `spx500_daily_ma200_v1` a `PARITY_PENDING` (decision C) NO
    HAY NINGUNA policy elegible en el repo, asi que las pruebas de la cadena se
    quedaron sin nada que ejercitar. La alternativa —re-promover el spec real para
    que los tests tengan sujeto— seria exactamente lo prohibido: la promocion es
    acto EXCLUSIVO del operador, y hacerla desde un test para poner verde una suite
    es la peor version posible de tocar un candado.

    Asi que se promueve una COPIA en memoria y se re-congela su hash, igual que hace
    `_con_umbral`. El fichero del repo no se toca; lo que se prueba es el mecanismo
    de la cadena, no el estado de gobierno de esa policy.
    """
    import copy

    import src.strategies.policies.loader as loader

    specs = copy.deepcopy(list(loader.load_all_policy_specs()))
    for spec in specs:
        if str(spec.get("id")) == (policy_id or PID):
            spec.setdefault("migration", {})["status"] = "PARITY_GREEN"
            spec.setdefault("governance", {})["policy_hash"] = (
                loader.canonical_policy_hash(spec)
            )
    monkeypatch.setattr(loader, "load_all_policy_specs", lambda: specs)


def _con_umbral(factory, monkeypatch, edad="P1D", policy_id=None):
    """Declarar `inputs.max_snapshot_age` SOLO en memoria.

    No se escribe en el spec real a proposito: un umbral de frescura decide CUANDO
    opera la estrategia, asi que fijarlo es una declaracion economica de la policy
    y no un detalle de fontaneria que yo pueda elegir para que un test pase
    (`quant-constitution.md` §1). Aqui se declara para demostrar que el MECANISMO
    funciona; que el spec no lo declare es un hecho aparte, y tiene su propio test.
    """
    import copy

    import src.strategies.policies.loader as loader

    specs = copy.deepcopy(list(loader.load_all_policy_specs()))
    for spec in specs:
        if str(spec.get("id")) == (policy_id or PID):
            spec.setdefault("inputs", {})["max_snapshot_age"] = edad
            # Elegible en memoria: ver `_elegible_en_memoria`. Sin esto, tras la
            # democion no queda ninguna policy que la cadena pueda ejercitar.
            spec.setdefault("migration", {})["status"] = "PARITY_GREEN"
            # RE-CONGELAR. `max_snapshot_age` entra en `canonical_policy_payload`
            # (CXD-610), asi que declararlo CAMBIA la identidad de la policy y el
            # muro de congelacion rechaza el spec si el hash declarado se queda
            # atras. Antes esta fixture inyectaba el umbral y dejaba el hash viejo:
            # un estado que en la realidad NO PUEDE EXISTIR. El muro lo cazo en
            # cuanto la clave empezo a contar, que es exactamente su trabajo.
            spec.setdefault("governance", {})["policy_hash"] = (
                loader.canonical_policy_hash(spec)
            )
    monkeypatch.setattr(loader, "load_all_policy_specs", lambda: specs)


def test_validate_link_actually_runs_end_to_end_without_monkeypatching_build_policy(factory, monkeypatch):
    """El camino SANO atraviesa `validate` con spec y contexto REALES.

    Sin monkeypatch de `build_policy` ni `ctx` magico: si el resolver id->spec o la
    construccion del `PolicyContext` estan mal, esto revienta. Antes reventaba.
    """
    _con_umbral(factory, monkeypatch)
    ti = _TIProbe(OBS_FRESCAS)
    factory.make_resolve_snapshot(PID)(ti=ti, **CTX_INTERVALO)   # produce la frescura
    salida = factory.make_validate_inputs(PID)(ti=ti, **CTX_INTERVALO)
    assert salida is None, (
        f"inputs validos deben dejar seguir la cadena (None), se devolvio {salida!r}"
    )


def test_evaluate_link_actually_produces_a_decision_end_to_end(factory, monkeypatch):
    """El camino sano llega a una decision REAL, no a un AttributeError."""
    _con_umbral(factory, monkeypatch)
    ti = _TIProbe(OBS_FRESCAS)
    factory.make_resolve_snapshot(PID)(ti=ti, **CTX_INTERVALO)
    decision = factory.make_evaluate_policy(PID)(ti=ti, **CTX_INTERVALO)
    assert decision is not None
    assert getattr(decision, "direction", None) in {"LONG", "FLAT", "SHORT"}, (
        f"la evaluacion no produjo una decision con direccion: {decision!r}"
    )


def test_stale_snapshot_honours_the_DECLARED_flat_not_the_runner_default(factory, monkeypatch):
    """`stale_input_policy: FLAT` del spec manda sobre el default del runner.

    La cadena no pasaba ningun fallback, asi que aplicaba `FAIL_CLOSED` por defecto:
    un snapshot stale habria **cerrado la tarea** en vez de emitir el FLAT explicito
    que la policy declara. El invariante 9 es "sin default, sin freeze" y la cadena
    estaba corriendo justo el default (CXD-598, evidencia 3).

    Rojo con: dejar de pasar `**_declared_fallbacks(spec)` en `make_validate_inputs`.
    """
    _con_umbral(factory, monkeypatch, edad="P1D")
    ti = _TIProbe(OBS_VIEJAS)                       # 6 dias > P1D declarado
    factory.make_resolve_snapshot(PID)(ti=ti, **CTX_INTERVALO)
    assert ti.pushed[f"{factory.STALENESS_XCOM_KEY}::{PID}"] is True, (
        "la frescura no se derivo de la evidencia; el resto del test seria teatro"
    )
    degradada = factory.make_validate_inputs(PID)(ti=ti, **CTX_INTERVALO)
    assert degradada is not None, "un snapshot stale no puede pasar como valido"
    assert degradada.direction == "FLAT"
    assert "INPUT_STALE" in degradada.reason_codes


def test_missing_feature_honours_the_DECLARED_fail_closed(factory, monkeypatch):
    """El otro fallback declarado (`missing_input_policy: FAIL_CLOSED`) tambien se aplica.

    Se comprueba el par completo a proposito: si alguien pasara ambos como `FLAT`
    "para que no falle", este test lo caza — el spec declara valores DISTINTOS para
    los dos, y esa asimetria es justamente lo que un default uniforme borra.
    """
    with pytest.raises(ValueError):
        _con_umbral(factory, monkeypatch)
        ti = _TIProbe({"close": {"value": 5200.0,
                                 "available_at": "2026-07-24T20:00:00+00:00"}})
        factory.make_resolve_snapshot(PID)(ti=ti, **CTX_INTERVALO)
        factory.make_validate_inputs(PID)(ti=ti, **CTX_INTERVALO)  # falta `ma_200`


def test_no_spec_declares_a_freshness_threshold_so_the_chain_fails_closed(factory):
    """ESTADO DE PRODUCCION HOY: nadie declara `inputs.max_snapshot_age`.

    R4 resolvia esto con `snapshot_is_stale=False` por defecto y yo lo describi
    como "limite declarado". CXD-600 lo rechazo y tenia razon: **no es un limite
    pasivo, fabrica un hecho de frescura**. Toda corrida productiva afirmaba "el
    dato esta fresco" sin medir nada, el `stale_input_policy: FLAT` del spec era
    inalcanzable, y un snapshot viejo se habria evaluado como nuevo.

    Este test fija las DOS mitades de la verdad:
      1. ninguno de los cuatro specs declara umbral -- leido del YAML, aparte;
      2. por tanto la cadena FALLA CERRADA, que es lo correcto: sin criterio de
         frescura no se decide. Declarar el umbral es una decision de la policy
         (cambia CUANDO opera) y no del orquestador, asi que no me lo invento.

    Rojo con: reponer cualquier default de frescura en `_policy_context` o en
    `_derive_staleness`.
    """
    import yaml

    declaran = [
        doc["id"]
        for path in sorted(POLICY_SPEC_DIR.glob("*.yaml"))
        for doc in [yaml.safe_load(path.read_text(encoding="utf-8")) or {}]
        if (doc.get("inputs") or {}).get("max_snapshot_age") is not None
    ]
    assert not declaran, (
        f"{declaran} ya declara(n) `max_snapshot_age`: actualiza este candado y "
        f"comprueba que la derivacion de frescura se ejercita de verdad en produccion"
    )

    ti = _TIProbe(OBS_FRESCAS)
    with pytest.raises(factory.PolicyRunConfigError, match="max_snapshot_age"):
        factory.make_resolve_snapshot(PID)(ti=ti, **CTX_INTERVALO)


def test_complete_required_inputs_with_unmeasured_freshness_fail_closed(factory, monkeypatch):
    """CXD-606 §4: el transporte de `None` no puede abrir un bypass.

    Desde R7 el hecho de frescura puede valer `None` ("no medible porque faltan
    requeridas") y **se transporta** en vez de abortar, para que el
    `missing_input_policy` declarado resuelva. El riesgo evidente de esa concesion
    es que un `None` se cuele con las requeridas COMPLETAS y nadie mida nada. Aqui
    las dos requeridas estan presentes y frescas, pero `resolve` no llego a correr:
    tiene que fallar cerrado.

    Ojo a la capa: antes fallaba en `_policy_context` con `PolicyRunConfigError`;
    ahora falla en el RUNNER con `ValueError`, porque el contexto ya no juzga. La
    asercion se mueve de capa, no se afloja: sigue exigiendo que no se evalue.

    Rojo con: reponer un default `False` en `validate_policy_inputs`.
    """
    _con_umbral(factory, monkeypatch)
    ti = _TIProbe(OBS_FRESCAS)          # NO se corre `resolve`: no hay hecho
    with pytest.raises(ValueError, match="freshness was never measured"):
        factory.make_validate_inputs(PID)(ti=ti, **CTX_INTERVALO)


def test_one_fresh_feature_cannot_launder_a_stale_one(factory, monkeypatch):
    """Un snapshot vale lo que su observacion MAS VIEJA (CXD-603).

    R5 derivaba la frescura con `max(available_at)` — la observacion mas NUEVA — asi
    que `close` de hace una hora blanqueaba una `ma_200` de hace seis dias y la policy
    operaba con un input caducado creyendolo fresco. Y los 22 tests en verde no podian
    verlo porque la fixture ponia el MISMO sello en todas las features: con edades
    homogeneas, `max` y `min` son indistinguibles.

    La regla correcta: stale si CUALQUIER observacion requerida excede el umbral.

    Rojo con: `min(...)` -> `max(...)` en `_derive_staleness`.
    """
    _con_umbral(factory, monkeypatch, edad="P1D")
    ti = _TIProbe(OBS_MIXTAS)
    factory.make_resolve_snapshot(PID)(ti=ti, **CTX_INTERVALO)
    assert ti.pushed[f"{factory.STALENESS_XCOM_KEY}::{PID}"] is True, (
        "una feature fresca blanqueo a una vieja: el snapshot se declaro fresco con "
        "`ma_200` de hace 6 dias y umbral P1D"
    )
    degradada = factory.make_validate_inputs(PID)(ti=ti, **CTX_INTERVALO)
    assert degradada is not None and degradada.direction == "FLAT", (
        "el snapshot mixto debia degradar a FLAT por el fallback declarado"
    )


def test_all_fresh_is_still_fresh_so_the_lock_is_not_just_always_stale(factory, monkeypatch):
    """La otra mitad del par: si TODAS son frescas, no se declara stale.

    Sin este, `_derive_staleness` cableada a `True` pasaria el test de arriba y la
    cadena no operaria nunca — un candado que solo comprueba una direccion no
    distingue "mide bien" de "siempre dice stale".
    """
    _con_umbral(factory, monkeypatch, edad="P1D")
    ti = _TIProbe(OBS_FRESCAS)
    factory.make_resolve_snapshot(PID)(ti=ti, **CTX_INTERVALO)
    assert ti.pushed[f"{factory.STALENESS_XCOM_KEY}::{PID}"] is False


@pytest.mark.parametrize("declarado", ["P", "PT"])
def test_a_duration_without_magnitude_is_rejected_not_read_as_zero(factory, declarado):
    """AUTOAUDITORIA (no vino de un rechazo): "P"/"PT" se aceptaban como cero.

    Devolvian `timedelta(0)` en silencio, o sea que una declaracion malformada
    pasaba por un umbral valido. La direccion era fail-safe (todo stale), pero
    aceptar basura callando es como se cuela una config sin sentido creyendo que
    hay criterio. `P0D` SI es legitimo -- declara "mismo instante" -- asi que se
    rechaza el vacio, jamas el cero.
    """
    with pytest.raises(factory.PolicyRunConfigError, match="magnitud"):
        factory._declared_max_snapshot_age(
            {"id": "x", "inputs": {"max_snapshot_age": declarado}}
        )


def test_zero_duration_is_legitimate_and_still_accepted(factory):
    """La otra mitad: `P0D` no es basura, es un umbral exigente. No se rechaza."""
    from datetime import timedelta

    assert factory._declared_max_snapshot_age(
        {"id": "x", "inputs": {"max_snapshot_age": "P0D"}}
    ) == timedelta(0)


# RETIRADO (CXD-605): aqui vivia `test_an_optional_feature_cannot_block_a_decision...`.
# Fijaba que una opcional vieja no declarase stale el snapshot. La idea es correcta,
# pero la implementacion que lo hacia verde permitia que la EDAD de una opcional
# reclasificara la ausencia total de las requeridas (stale/FLAT en vez del missing
# declarado). Un candado que fija una semantica aun no acordada es peor que su
# ausencia: da por zanjado lo que esta en discusion (propuesta en CLD-563).


@pytest.mark.parametrize("policy_id", ["gold_trend_simple", "btc_hodl_b1"])
@pytest.mark.parametrize("edad_opcional,etiqueta", [
    ("2026-06-01T00:00:00+00:00", "opcional VIEJA"),
    ("2026-07-24T20:00:00+00:00", "opcional FRESCA"),
])
def test_an_optional_features_age_cannot_reclassify_a_missing_required_set(
    factory, monkeypatch, policy_id, edad_opcional, etiqueta
):
    """CXD-606 §3, con los specs REALES que tienen opcionales: Gold y BTC.

    ESTE ES EL DEFECTO QUE ME ENCONTRO CODEX (CXD-605). Con `stale` evaluado antes
    que `missing`, y midiendo opcionales como fallback, la EDAD de un dato que la
    policy declara OPCIONAL decidia la CATEGORIA de la ausencia total del nucleo
    requerido: opcional vieja -> FLAT/INPUT_STALE; opcional fresca -> el missing
    FAIL_CLOSED declarado. Mismo estado de datos, dos veredictos, y el que decidia
    era el dato que la policy dice no necesitar.

    Se usan Gold y BTC a proposito y no un spec sintetico: yo afirme que "los cuatro
    specs declaran `optional_features: []`" mirando solo spx500, y sobre esa frase
    falsa construi el argumento de que el cambio no afectaba a nadie. Afectaba a estos
    dos. El candado se instrumenta donde el defecto vive.

    Rojo con: volver a evaluar `stale` antes que `missing` en `validate_policy_inputs`.
    """
    _con_umbral(factory, monkeypatch, edad="P1D", policy_id=policy_id)
    ti = _TIProbe({"regime_risk_mult": {"value": 1.0, "available_at": edad_opcional}})
    factory.make_resolve_snapshot(policy_id)(ti=ti, **CTX_INTERVALO)
    assert ti.pushed[f"{factory.STALENESS_XCOM_KEY}::{policy_id}"] is None, (
        f"{etiqueta}: sin requeridas la frescura NO es medible; devolver un bool aqui "
        f"es fabricar el hecho"
    )
    # Gold y BTC declaran `missing_input_policy: FAIL_CLOSED`: las DOS edades deben
    # dar EXACTAMENTE eso, y no una degradacion stale.
    with pytest.raises(ValueError, match="invalid policy inputs") as exc:
        factory.make_validate_inputs(policy_id)(ti=ti, **CTX_INTERVALO)
    assert "INPUT_STALE" not in str(exc.value), (
        f"{etiqueta}: la ausencia del nucleo requerido se reclasifico como problema "
        f"de frescura"
    )


# --- Candados del RUNNER, no del factory -------------------------------------
#
# POR QUE EXISTEN. Escribi primero los candados de arriba (Gold/BTC via factory) y
# medi que las mutaciones "stale antes que missing" y "default False en el runner"
# **no los ponian rojos**: el `None` que devuelve el factory tapa el fallo del
# runner, porque un `stale` que vale None no dispara el chequeo aunque se evalue
# primero. Es decir, el orden estaba defendido por via INDIRECTA — y una defensa
# indirecta no es un candado, es una coincidencia. Estos dos ejercen el contrato
# del runner de frente.


class _PoliticaFalsa:
    """Policy minima que declara inputs invalidos: aisla el ORDEN del veredicto."""

    sleeve_id = "x"
    version = "1.0.0"
    policy_hash = "sha256:" + "ab" * 32
    policy_version_id = "x:1.0.0"

    def validate_inputs(self, snapshot):
        return ["falta `close`"]

    def evaluate(self, snapshot, ctx):  # pragma: no cover - no debe llegar
        raise AssertionError("no se evalua con inputs invalidos")


def test_runner_resolves_missing_before_stale_when_BOTH_are_true():
    """El orden acordado (CXD-606), observado de frente.

    Snapshot que esta **a la vez** incompleto y marcado stale. Con
    `missing_input_policy: FLAT` y `stale_input_policy: FLAT` los dos caminos
    degradan, asi que el veredicto no distingue... salvo por el REASON CODE. Eso es
    justo lo que decide la precedencia: sin datos requeridos, el problema es la
    ausencia, no la edad.

    Rojo con: mover el bloque `stale` delante del bloque `missing`.
    """
    from src.contracts.policy import PolicyContext
    from src.policy_engine import validate_policy_inputs

    ctx = PolicyContext(as_of="2026-07-24T21:00:00+00:00",
                        extras={"snapshot_is_stale": True})
    degradada = validate_policy_inputs(
        _PoliticaFalsa(), {"otra": 1.0}, ctx,
        missing_input_policy="FLAT", stale_input_policy="FLAT",
    )
    assert degradada is not None
    assert degradada.reason_codes == ("INPUT_MISSING",), (
        f"con inputs incompletos Y stale, gano la frescura: {degradada.reason_codes}. "
        f"No se puede juzgar la edad de un dato que no esta"
    )


def test_runner_refuses_an_absent_freshness_key_instead_of_defaulting_to_fresh():
    """La ausencia de la clave no es "no esta stale" (CXD-606 §4).

    `_policy_context` siempre pone la clave, asi que via factory este caso no se
    alcanza — y por eso la mutacion del default no ponia nada rojo. Pero
    `validate_policy_inputs` es API publica con otros consumidores, y ahi un
    `.get(..., False)` convierte "nadie midio" en "esta fresco".

    Rojo con: `context.extras.get("snapshot_is_stale", False)`.
    """
    from src.contracts.policy import PolicyContext
    from src.policy_engine import validate_policy_inputs

    class _Valida(_PoliticaFalsa):
        def validate_inputs(self, snapshot):
            return []

        def evaluate(self, snapshot, ctx):
            raise AssertionError("no debe evaluarse sin hecho de frescura")

    ctx = PolicyContext(as_of="2026-07-24T21:00:00+00:00", extras={})
    with pytest.raises(ValueError, match="freshness was never measured"):
        validate_policy_inputs(_Valida(), {"close": 1.0}, ctx)


def test_publish_refuses_to_publish_without_evidence_backing_the_decision(factory):
    """Antes de la DB, el gate del techo de evidencia (CXD-620 §2).

    `publish` es por donde la señal ESCAPA, asi que ahi se comprueba que la evidencia
    sostiene lo que el estado reclama. Sin observaciones no hay nada que juzgar, y una
    señal cuya evidencia no se puede juzgar no se publica.
    """
    class _SinObs:
        def xcom_pull(self, key=None, task_ids=None):
            if task_ids and task_ids.endswith("_evaluate"):
                return object()
            return None

    with pytest.raises(factory.PolicyRunConfigError, match="sin observaciones"):
        factory.make_publish_signal(PID)(
            ti=_SinObs(), data_interval_start="2026-07-17T00:00:00+00:00", **CTX_INTERVALO
        )


def test_publish_link_resolves_its_spec_and_only_stops_at_the_db_boundary(factory, monkeypatch):
    """El TERCER eslabon roto, que el rechazo no llego a nombrar.

    `make_publish_signal` llamaba `load_policy_spec(policy_id)`, y ese recibe una
    RUTA: habria muerto con FileNotFoundError. Y detras habia un cuarto crash:
    `_canonical_instrument_id` leia `spec["asset"]` como mapping cuando en los
    CUATRO specs vigentes es una cadena, asi que `(spec.get("asset") or {}).get("id")`
    reventaba siempre y el `or spec.get("asset")` de detras era codigo inalcanzable.

    LIMITE DECLARADO de este candado: no publica de verdad, porque
    `_canonical_instrument_id` necesita `reference.instrument` en una DB viva y aqui
    no hay stack. Lo que fija es que el fallo ocurra en la FRONTERA DE DB y no antes:
    si vuelve a romperse la resolucion de spec o de activo, el error deja de ser el
    del acceso a datos y esto cae. Cubrir la publicacion completa exige el stack y
    queda declarado como pendiente, no simulado aqui.
    """
    # Promovida en memoria: con `PARITY_PENDING` el gate del techo corta antes (su
    # estado no declara que evidencia reclama), y este candado mira la frontera de DB.
    _elegible_en_memoria(factory, monkeypatch)
    obs_con_sello = {
        k: {**v, "provenance": "available_at_reconstructed:close+P1D"}
        for k, v in OBS_FRESCAS.items()
    }
    ti = _TIProbe(obs_con_sello, decision=object())
    with pytest.raises(Exception) as exc:
        factory.make_publish_signal(PID)(
            ti=ti, data_interval_start="2026-07-17T00:00:00+00:00", **CTX_INTERVALO
        )
    tipo, mensaje = type(exc.value).__name__, str(exc.value)
    # Se afirma por NEGACION, no por el error concreto. La version anterior exigia
    # `ModuleNotFoundError: utils`, y eso ataba el candado al ENTORNO: en la suite
    # ANCHA algo pone `airflow/dags` en `sys.path`, `utils` SI importa, publish avanza
    # mas alla de la frontera de DB y el test fallaba con otro error. Pasaba en focal y
    # fallaba en ancha -- yo mismo declare esta fragilidad en CLD-555 como el punto mas
    # debil de la entrega, y se confirmo.
    #
    # Lo que este candado vigila es que NO se rompa la resolucion de spec/activo; que
    # despues muera en el import de DB o mas alla depende de si hay stack, y eso no es
    # asunto suyo.
    prohibidos = ("'str' object has no attribute 'get'", "No such file or directory")
    assert not any(p in mensaje for p in prohibidos), (
        f"publish fallo por resolucion de spec/activo, que es lo que este candado "
        f"vigila: {tipo}: {mensaje[:200]}"
    )
    assert tipo != "PolicyRunConfigError" or "sin observaciones" not in mensaje, (
        "el gate del techo corto antes: este candado mira mas abajo"
    )


def test_context_without_data_interval_fails_closed_instead_of_dating_a_decision_now(factory):
    """Sin `data_interval_*` no hay `as_of`: no se inventa con `now()`.

    Fechar la decision con la hora de ejecucion haria que dos re-ejecuciones de la
    misma fecha produjeran registros distintos y el replay dejaria de ser replay.
    """
    with pytest.raises(factory.PolicyRunConfigError, match="data_interval"):
        factory.make_validate_inputs(PID)(ti=_TIProbe(OBS_FRESCAS))


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
            # Elegible SOLO en memoria (ver `_elegible_en_memoria`): tras la democion
            # no queda ninguna policy elegible con la que ejercitar el resolver.
            spec.setdefault("migration", {})["status"] = "PARITY_GREEN"
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
        def __init__(self):
            self.pushed = {}

        def xcom_push(self, key=None, value=None):
            self.pushed[key] = value

        def xcom_pull(self, key=None, task_ids=None):
            if key and key.startswith("observations"):
                return {"rsi_9": {"value": 1.0, "available_at": "2026-01-01T00:00:00+00:00"}}
            if key and key.startswith("decision_cutoff"):
                return "2026-01-02T00:00:00+00:00"
            return self.pushed.get(key)

    # R5: la tarea ademas DERIVA la frescura, y eso exige umbral declarado.
    _con_umbral(factory, monkeypatch, edad="P7D", policy_id="btc_hodl_b1")
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
