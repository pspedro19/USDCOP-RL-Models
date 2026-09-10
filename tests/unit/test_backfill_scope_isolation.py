"""El `conf` del backfill L0 tiene que GOBERNAR, no decorar (CXD-515, carril CLAUDE).

Incidente: el run `codex_bl40_usdmxn_20260805T0015` se lanzó con
`conf={"symbols":["USD/MXN"]}` y terminó `SUCCESS`, pero las tres tareas `process_*`
ejecutaron gap detection (COP 7 gaps, MXN 2, BRL 35) y `export_seeds` reescribió los cuatro
parquets trackeados, que hubo que restaurar a HEAD. `get_target_symbols` calculaba la lista
correcta y **nadie la consultaba**: cada `process_*` leía su propio `params['symbol']`.

Estos candados son CONDUCTUALES sin Airflow instalado: se extraen los `FunctionDef` del DAG
por AST y se ejecutan en un namespace con XCom, DB y `AirflowSkipException` simulados. Así se
ejerce la ruta real de decisión sin importar el módulo (el `conftest` del repo saca Apache
Airflow de `sys.modules` a propósito, porque `airflow/` del repo tiene que ganar).

Lo que fijan:
  1. fuera de alcance => `AirflowSkipException` (estado Airflow `skipped`), NUNCA un
     `success` con `bars_backfilled: 0`. Un SUCCESS vacío afirma algo sobre los datos;
     `skipped` dice la verdad: no se miró.
  2. sin XCom de alcance => se levanta. Caer a `ALL_SYMBOLS` es cómo se perdió el aislamiento.
  3. `export_seeds` exporta sólo el alcance y NO reescribe el seed unificado en un run parcial.
  4. `validate_results` informa sólo del alcance y lo declara en el propio reporte.
"""

from __future__ import annotations

import ast
import logging as _logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DAG_PATH = ROOT / "airflow" / "dags" / "l0_ohlcv_backfill.py"
DAG_SOURCE = DAG_PATH.read_text(encoding="utf-8")

ALL_SYMBOLS = ["USD/COP", "USD/MXN", "USD/BRL"]


class FakeSkip(Exception):
    """Sustituto de `AirflowSkipException`: lo que Airflow traduce a estado `skipped`."""


def _load(*names: str) -> dict:
    """Ejecuta SOLO las funciones pedidas del DAG, con dependencias simuladas."""
    tree = ast.parse(DAG_SOURCE)
    wanted = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {n.name for n in wanted} == set(names), (
        f"faltan funciones en el DAG: {set(names) - {n.name for n in wanted}}"
    )
    ns: dict = {
        "AirflowSkipException": FakeSkip,
        "ALL_SYMBOLS": ALL_SYMBOLS,
        "logging": _logging,
        "pd": pd,
        "Path": Path,
        "os": os,
        "List": list,
        "__builtins__": __builtins__,
    }
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(DAG_PATH), "exec"), ns)
    return ns


class _TI:
    def __init__(self, store: dict):
        self._store = store

    def xcom_pull(self, key=None, task_ids=None):
        return self._store.get((key, task_ids))

    def xcom_push(self, key=None, value=None):
        self._store[(key, None)] = value


def _context(scope, symbol=None, force=False):
    store = {("force_backfill", "health_check"): force}
    if scope is not None:
        store[("target_symbols", "health_check")] = scope
    ctx = {"ti": _TI(store), "dag_run": SimpleNamespace(conf={})}
    if symbol is not None:
        ctx["params"] = {"symbol": symbol}
    return ctx


class _Cursor:
    def __init__(self, rows_by_symbol, asked):
        self._rows = rows_by_symbol
        self._asked = asked
        self._symbol = None
        self._unified = False

    def execute(self, sql, params=None):
        self._unified = params is None
        self._symbol = params[0] if params else None
        if self._symbol:
            self._asked.append(self._symbol)

    def fetchall(self):
        if self._unified:
            return [r for rows in self._rows.values() for r in rows]
        return self._rows.get(self._symbol, [])

    def fetchone(self):
        return (len(self._rows.get(self._symbol, [])), None, None)

    def close(self):
        return None


class _Conn:
    def __init__(self, rows_by_symbol):
        self._rows = rows_by_symbol
        self.asked: list = []

    def cursor(self):
        return _Cursor(self._rows, self.asked)

    def close(self):
        return None


# ------------------------------------------------------------------ 1: skip real
def test_a_symbol_outside_the_scope_is_skipped_not_a_hollow_success():
    """El fallo exacto de CXD-515: COP corriendo en un run de solo USD/MXN."""
    aperturas = []
    ns = _load("resolve_scope", "process_symbol")
    ns["get_db_connection"] = lambda *a, **k: aperturas.append("db") or _Conn({})
    ns["SYMBOL_CONFIG"] = {s: {"seed_path": Path("nope")} for s in ALL_SYMBOLS}

    with pytest.raises(FakeSkip) as exc:
        ns["process_symbol"](**_context(["USD/MXN"], symbol="USD/COP"))

    assert "USD/COP" in str(exc.value) and "USD/MXN" in str(exc.value)
    assert aperturas == [], "se abrio la DB antes de comprobar el alcance"


def test_the_symbol_inside_the_scope_is_not_skipped():
    """El candado no puede apagar el trabajo legitimo: MXN debe pasar del guard."""
    ns = _load("resolve_scope", "process_symbol")
    marca = []
    ns["get_db_connection"] = lambda *a, **k: marca.append("db") or _Conn({})
    ns["SYMBOL_CONFIG"] = {}  # fuerza el retorno 'unknown_symbol' justo despues del guard

    resultado = ns["process_symbol"](**_context(["USD/MXN"], symbol="USD/MXN"))

    assert resultado["status"] == "error" and resultado["reason"] == "unknown_symbol"
    assert marca == [], "el early-return de simbolo desconocido no debe abrir la DB"


# --------------------------------------------------- 2: fail-closed sin alcance
@pytest.mark.parametrize("scope", [None, []])
def test_missing_or_empty_scope_raises_instead_of_defaulting_to_every_pair(scope):
    ns = _load("resolve_scope")
    with pytest.raises(ValueError, match="target_symbols"):
        ns["resolve_scope"](_context(scope, symbol="USD/COP"))


def test_scope_default_to_all_symbols_is_absent_from_the_resolver():
    """Candado estructural por AST: el resolver no puede recuperar el default permisivo.

    Se mira el ÁRBOL, no el texto: la propia explicación de por qué el default es peligroso
    nombra `ALL_SYMBOLS`, y un grep se dispararía con el comentario que documenta la regla.
    """
    resolver = next(
        node
        for node in ast.parse(DAG_SOURCE).body
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_scope"
    )
    referencias = {
        n.id for n in ast.walk(resolver) if isinstance(n, ast.Name) and n.id == "ALL_SYMBOLS"
    }
    assert not referencias, (
        "resolve_scope no puede caer a ALL_SYMBOLS: convierte 'no se mi alcance' en 'todos'"
    )


# --------------------------------------------------------- 3: export acotado
def test_export_touches_only_the_scope_and_skips_the_unified_seed(tmp_path, monkeypatch):
    ns = _load("resolve_scope", "export_seeds")
    conn = _Conn({"USD/MXN": [(pd.Timestamp("2026-01-05", tz="UTC"), "USD/MXN", 1, 2, 0, 1, 5)]})
    ns["get_db_connection"] = lambda *a, **k: conn
    monkeypatch.setenv("AIRFLOW_HOME", str(tmp_path))

    result = ns["export_seeds"](**_context(["USD/MXN"]))

    escritos = {p.name for p in (tmp_path / "seeds" / "latest").glob("*.parquet")}
    assert escritos == {"usdmxn_m5_ohlcv.parquet"}, (
        "un run acotado no puede reescribir los seeds de los pares que no miro"
    )
    assert "USD/COP" not in conn.asked and "USD/BRL" not in conn.asked
    unified = [e for e in result["exported"] if e["symbol"] == "ALL"]
    assert unified and unified[0]["skipped"] == "partial_scope"


def test_export_still_writes_the_unified_seed_when_the_run_covers_all_pairs(
    tmp_path, monkeypatch
):
    ns = _load("resolve_scope", "export_seeds")
    rows = {
        s: [(pd.Timestamp("2026-01-05", tz="UTC"), s, 1, 2, 0, 1, 5)] for s in ALL_SYMBOLS
    }
    ns["get_db_connection"] = lambda *a, **k: _Conn(rows)
    monkeypatch.setenv("AIRFLOW_HOME", str(tmp_path))

    result = ns["export_seeds"](**_context(list(ALL_SYMBOLS)))

    escritos = {p.name for p in (tmp_path / "seeds" / "latest").glob("*.parquet")}
    assert "fx_multi_m5_ohlcv.parquet" in escritos
    unified = [e for e in result["exported"] if e["symbol"] == "ALL"]
    assert unified and "skipped" not in unified[0]


# ------------------------------------------------- 4: la evidencia dice su alcance
def test_validation_reports_only_the_scope_and_declares_it():
    ns = _load("resolve_scope", "validate_results")
    ns["get_db_connection"] = lambda *a, **k: _Conn({})

    report = ns["validate_results"](**_context(["USD/MXN"]))

    assert report["_scope"] == ["USD/MXN"]
    assert set(report) - {"_scope"} == {"USD/MXN"}


# ------------------------------------------- 5: la forma del `conf` se valida (CXD-518)
def _conf_context(conf):
    return {"ti": _TI({}), "dag_run": SimpleNamespace(conf=conf)}


def test_a_string_scope_is_rejected_instead_of_iterated_as_characters():
    """`symbols="USD/MXN"` es iterable: sin validar, el filtro compara SUBCADENAS."""
    ns = _load("_validated_scope", "get_target_symbols")
    with pytest.raises(ValueError, match="lista de s"):
        ns["get_target_symbols"](_conf_context({"symbols": "USD/MXN"}))


def test_an_empty_scope_fails_at_health_check_not_task_by_task():
    ns = _load("_validated_scope", "get_target_symbols")
    with pytest.raises(ValueError, match="vac"):
        ns["get_target_symbols"](_conf_context({"symbols": []}))


def test_an_unknown_symbol_is_rejected_instead_of_skipping_everything():
    """`['FOO']` dejaria las tres tareas skipped: verde por vacuidad."""
    ns = _load("_validated_scope", "get_target_symbols")
    with pytest.raises(ValueError, match="no gobierna"):
        ns["get_target_symbols"](_conf_context({"symbols": ["FOO"]}))


def test_duplicates_are_rejected_so_the_export_does_not_write_twice():
    ns = _load("_validated_scope", "get_target_symbols")
    with pytest.raises(ValueError, match="duplicados"):
        ns["get_target_symbols"](_conf_context({"symbols": ["USD/MXN", "USD/MXN"]}))


def test_non_string_members_are_rejected():
    ns = _load("_validated_scope", "get_target_symbols")
    with pytest.raises(ValueError, match="no-string"):
        ns["get_target_symbols"](_conf_context({"symbols": ["USD/MXN", 7]}))


def test_a_valid_scope_and_the_legacy_single_symbol_key_still_work():
    ns = _load("_validated_scope", "get_target_symbols")
    assert ns["get_target_symbols"](_conf_context({"symbols": ["USD/MXN"]})) == ["USD/MXN"]
    assert ns["get_target_symbols"](_conf_context({"symbol": "USD/BRL"})) == ["USD/BRL"]
    with pytest.raises(ValueError, match="no gobierna"):
        ns["get_target_symbols"](_conf_context({"symbol": "FOO"}))


def test_no_conf_still_means_all_three_pairs():
    """El default permisivo es legitimo AQUI: nadie declaro un alcance acotado."""
    ns = _load("_validated_scope", "get_target_symbols")
    assert ns["get_target_symbols"](_conf_context({})) == ALL_SYMBOLS
    assert ns["get_target_symbols"](_conf_context(None)) == ALL_SYMBOLS


# --------------- 6: la semantica del GRAFO, no solo de los callables (CXD-521)
def _operator_calls() -> dict:
    """`python_callable` -> kwargs literales de cada `PythonOperator` del DAG.

    Los tests de arriba ejercen funciones AISLADAS y por eso NO vieron el defecto de
    CXD-521: con el `trigger_rule` por defecto (`all_success`) el skip intencional de COP
    hacia inalcanzable a MXN, y el run terminaba SUCCESS habiendo ejecutado solo un skip.
    Aqui se mira el cableado.
    """
    calls = {}
    for node in ast.walk(ast.parse(DAG_SOURCE)):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "PythonOperator"):
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        callable_node = kwargs.get("python_callable")
        nombre = getattr(callable_node, "id", None)
        if nombre:
            calls[nombre] = kwargs
    return calls


def test_symbol_tasks_continue_past_an_intentional_upstream_skip():
    """El defecto exacto de CXD-521: `all_success` + skip de alcance = run vacio."""
    kwargs = _operator_calls()["process_symbol"]
    regla = kwargs.get("trigger_rule")
    assert regla is not None, (
        "process_* no puede usar el trigger_rule por defecto: la cadena es secuencial, asi "
        "que un skipped de alcance aguas arriba haria inalcanzable el simbolo pedido"
    )
    assert isinstance(regla, ast.Constant) and regla.value == "none_failed", (
        f"process_* debe usar 'none_failed' (continua ante skipped, para ante failed); "
        f"declara {getattr(regla, 'value', regla)!r}"
    )


def test_the_sequential_pool_is_preserved_so_the_fix_does_not_flood_the_api():
    """`none_failed` no debe convertirse en 'ejecuta los tres en paralelo'."""
    kwargs = _operator_calls()["process_symbol"]
    pool = kwargs.get("pool")
    assert isinstance(pool, ast.Constant) and pool.value == "api_requests"
    assert ">> task_export" in DAG_SOURCE or "task_export" in DAG_SOURCE


def test_export_and_validate_keep_their_own_permissive_rule():
    calls = _operator_calls()
    for nombre in ("export_seeds", "validate_results"):
        regla = calls[nombre].get("trigger_rule")
        assert isinstance(regla, ast.Constant), f"{nombre} sin trigger_rule explicito"
        assert regla.value == "none_failed_min_one_success", (
            f"{nombre} debe exigir al menos un exito: si TODO se salto, no hay nada que "
            f"exportar ni validar; declara {regla.value!r}"
        )


def test_export_hangs_off_every_symbol_not_just_the_last_one():
    """Con un solo upstream skipped, `min_one_success` salta export igual.

    Encontrado midiendo el grafo REAL en el contenedor: `export_seeds` colgaba solo de
    `process_usd_brl`. En un run acotado a USD/MXN, BRL queda `skipped` => cero exitos entre
    los upstream directos => export y validate se saltan aunque MXN hubiera corrido bien.
    El fan-in es lo que hace que `none_failed_min_one_success` signifique lo que dice.
    """
    chain = DAG_SOURCE.split("# Chain: health")[1]
    code = "\n".join(l for l in chain.splitlines() if not l.lstrip().startswith("#"))
    assert "for task in symbol_tasks:" in code and "task >> task_export" in code, (
        "export debe depender de los tres process, no solo de symbol_tasks[-1]"
    )
    assert "symbol_tasks[-1] >> task_export" not in code
    # La secuencia por pool de API se conserva.
    assert "symbol_tasks[i] >> symbol_tasks[i + 1]" in code


# ------- 7: un backfill 100% rechazado no puede reportarse SUCCESS (log real 401)
def _gap_namespace(fetch, inserta=0):
    """`process_symbol` con la deteccion de huecos y la API simuladas."""
    from datetime import date as _date

    ns = _load("resolve_scope", "process_symbol")
    ns["get_db_connection"] = lambda *a, **k: _Conn({})
    ns["SYMBOL_CONFIG"] = {s: {"seed_path": Path("nope")} for s in ALL_SYMBOLS}
    ns["get_data_date_range"] = lambda conn, sym: (_date(2026, 1, 2), _date(2026, 8, 4))
    ns["get_all_trading_days"] = lambda a, b: [_date(2026, 1, 2)]
    ns["get_bars_per_day"] = lambda conn, sym: {}
    ns["detect_all_gaps"] = lambda *a: [_date(2026, 3, 27)]
    ns["group_consecutive_gaps"] = lambda gaps: [
        {"start_date": _date(2026, 3, 27), "end_date": _date(2026, 3, 27), "days_missing": 1},
        {"start_date": _date(2026, 7, 29), "end_date": _date(2026, 8, 4), "days_missing": 6},
    ]
    ns["fetch_ohlcv_data"] = fetch
    ns["filter_market_hours"] = lambda df: df
    ns["insert_ohlcv_batch"] = lambda conn, df: inserta
    ns["API_RATE_DELAY_SECONDS"] = 0
    ns["time"] = SimpleNamespace(sleep=lambda *_a: None)
    ns["datetime"] = __import__("datetime").datetime
    ns["COT_TZ"] = __import__("datetime").timezone.utc
    return ns


def _un_401(*_a, **_k):
    raise RuntimeError("401 Client Error: Unauthorized for url: https://api.twelvedata.com/...")


def test_every_rejected_fetch_fails_the_task_instead_of_reporting_success():
    """Log real del run codex_bl40_usdmxn_20260805T0110: dos 401 y `status: ok` + SUCCESS."""
    ns = _gap_namespace(_un_401)
    with pytest.raises(RuntimeError, match="SIN examinar"):
        ns["process_symbol"](**_context(["USD/MXN"], symbol="USD/MXN"))


def test_the_failure_names_the_first_real_cause():
    ns = _gap_namespace(_un_401)
    with pytest.raises(RuntimeError, match="401"):
        ns["process_symbol"](**_context(["USD/MXN"], symbol="USD/MXN"))


def test_a_gap_the_api_serves_empty_is_a_legitimate_zero_and_stays_ok():
    """El discriminador es el ERROR, no el cero: festivo o fuera de historia es `ok`."""
    ns = _gap_namespace(lambda *a, **k: pd.DataFrame())
    resultado = ns["process_symbol"](**_context(["USD/MXN"], symbol="USD/MXN"))
    assert resultado["status"] == "ok"
    assert resultado["bars_backfilled"] == 0
    assert resultado["fetch_errors"] == 0


def test_a_partial_backfill_also_fails_and_persists_what_it_got():
    """Concedido a CXD-530: mi semantica anterior devolvia SUCCESS con un rango sin examinar.

    Con un hueco fallido y otro insertado, `status: ok` dejaba el rango fallido invisible: ningun
    estado de Airflow lo declaraba y export/validate corrian igual. Como los inserts son
    idempotentes, levantar DESPUES de persistir es gratis y el retry recupera lo que falta.
    """
    llamadas = {"n": 0}
    insertadas = []

    def fetch_mixto(*_a, **_k):
        llamadas["n"] += 1
        if llamadas["n"] == 1:
            raise RuntimeError("401 Client Error: Unauthorized")
        return pd.DataFrame([{"time": pd.Timestamp("2026-07-29", tz="UTC"), "close": 1.0}])

    ns = _gap_namespace(fetch_mixto)
    ns["insert_ohlcv_batch"] = lambda conn, df: insertadas.append(len(df)) or 7

    with pytest.raises(RuntimeError, match="SIN examinar"):
        ns["process_symbol"](**_context(["USD/MXN"], symbol="USD/MXN"))

    # Lo que si se pudo traer quedo PERSISTIDO antes de levantar: el retry no repite trabajo.
    assert insertadas == [1], "las barras del hueco que si respondio deben persistirse"


def test_the_partial_failure_message_carries_count_and_first_cause():
    ns = _gap_namespace(_un_401)
    with pytest.raises(RuntimeError, match=r"2 de 2 fetch fallaron"):
        ns["process_symbol"](**_context(["USD/MXN"], symbol="USD/MXN"))


def test_the_log_does_not_say_complete_when_a_fetch_failed():
    """`Backfill complete` con errores dentro es la misma mentira en el log."""
    cuerpo = DAG_SOURCE.split("def process_symbol")[1].split("\ndef ")[0]
    assert "Backfill {veredicto}" in cuerpo
    assert "'complete' if not fetch_errors" in cuerpo


def test_an_internal_error_status_no_longer_returns_as_success():
    """`status: 'error'` devuelto normalmente era SUCCESS en Airflow."""
    ns = _gap_namespace(_un_401)
    ns["get_data_date_range"] = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("DB caida"))
    with pytest.raises(RuntimeError, match="procesamiento fallido"):
        ns["process_symbol"](**_context(["USD/MXN"], symbol="USD/MXN"))
