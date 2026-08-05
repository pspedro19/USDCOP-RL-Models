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
