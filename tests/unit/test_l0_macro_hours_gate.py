# -*- coding: utf-8 -*-
"""El gate de horario de `core_l0_04_macro_update` decide por la ventana lógica.

EL DEFECTO, medido en producción el 2026-08-06
----------------------------------------------
`check_market_hours` decidía con `datetime.now(tz)` — el reloj del momento en que se
ejecuta la tarea — en vez de la ventana que la corrida representa.

Lo observado: la corrida `scheduled__2026-08-06T16:00:00+00:00` (11:00 COT, DENTRO de
horario) se ejecutó a las 15:54 COT porque el scheduler acababa de recrearse. El gate la
cortó, y con ella `extract_all_sources`, `upsert_all` y `update_is_complete`. **El DagRun
quedó en `success`** — `skipped` no es `failed`. Resultado: `macro_indicators_daily` con
9 días sin una fila nueva (umbral de frescura: 7) mientras Airflow mostraba verde.

Que los extractores estaban sanos se probó ejecutando la MISMA corrida con
`force_run=true`: 26.326 → 26.336 filas, `max(fecha)` 2026-07-28 → 2026-08-07, y las 12
columnas macro de vuelta a fecha corriente. O sea: no era el scraper ni la credencial.
Era el gate apagándolos en silencio.

Cualquier corrida que se ejecute tarde —scheduler caído, reintento, catchup, backfill—
caía en lo mismo, de forma permanente y sin señal.

QUÉ FIJA ESTE FICHERO
---------------------
Que la decisión dependa de `data_interval_end`/`logical_date` y NO del reloj de pared, y
que el fallback al reloj sólo exista si el contexto llega sin fecha lógica.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

pytz = pytest.importorskip("pytz")

REPO = Path(__file__).resolve().parents[2]
DAG = REPO / "airflow" / "dags" / "l0_macro_update.py"
FUENTE = DAG.read_text(encoding="utf-8")

BOGOTA = pytz.timezone("America/Bogota")


def _gate():
    """Extrae `check_market_hours` sin importar el DAG entero (necesitaría Airflow)."""
    ini = FUENTE.index("def check_market_hours(")
    fin = FUENTE.index("# TASK 3: EXTRACT ALL SOURCES")
    cuerpo = FUENTE[ini:fin]
    # El cierre del bloque incluye el separador de comentarios; recortarlo no cambia el código.
    cuerpo = cuerpo[: cuerpo.rindex("return True") + len("return True")]
    ns: dict = {
        "datetime": datetime,
        "logger": SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None),
        "MARKET_HOURS_START": 8,
        "MARKET_HOURS_END": 13,
    }
    exec(compile(cuerpo, str(DAG), "exec"), ns)
    return ns["check_market_hours"]


def _contexto(interval_end, conf=None):
    return {
        "dag_run": SimpleNamespace(conf=conf or {}),
        "data_interval_end": interval_end,
        "logical_date": interval_end,
    }


def test_a_late_run_of_a_valid_window_still_proceeds() -> None:
    """El caso exacto que se perdió: ventana 11:00 COT, ejecutada a las 15:54 COT.

    Rojo con la versión anterior: decidía con `datetime.now()`, que a las 15:54 está
    fuera de 8-13 y devolvía False.
    """
    ventana = BOGOTA.localize(datetime(2026, 8, 6, 11, 0))  # jueves, dentro de horario
    assert _gate()(**_contexto(ventana)) is True, (
        "una corrida cuya ventana lógica es válida se saltó por ejecutarse tarde: es el "
        "defecto que dejó macro 9 días sin escribir con el DAG en verde"
    )


def test_a_window_genuinely_outside_hours_is_skipped() -> None:
    """El gate no se vuelve permisivo: una ventana de las 19:00 COT sigue cortando."""
    ventana = BOGOTA.localize(datetime(2026, 8, 6, 19, 0))
    assert _gate()(**_contexto(ventana)) is False


def test_a_weekend_window_is_skipped() -> None:
    """Sábado dentro del rango horario, pero fin de semana: no hay mercado."""
    ventana = BOGOTA.localize(datetime(2026, 8, 8, 10, 0))  # sábado
    assert _gate()(**_contexto(ventana)) is False


def test_the_window_is_read_in_bogota_even_if_it_arrives_in_utc() -> None:
    """Airflow entrega el contexto en UTC; 16:00Z son las 11:00 COT y debe pasar.

    Comparar la hora UTC contra un rango expresado en COT es la misma clase de error de
    convención que ya nos costó una reconciliación entera: dos relojes a los dos lados de
    la misma comparación.
    """
    ventana = pytz.UTC.localize(datetime(2026, 8, 6, 16, 0))
    assert _gate()(**_contexto(ventana)) is True, (
        "no convirtió a America/Bogota: 16:00Z se leyó como 16:00 COT y cayó fuera"
    )


def test_force_run_still_bypasses() -> None:
    """La salida declarada sigue existiendo — es la que permitió diagnosticar esto."""
    ventana = BOGOTA.localize(datetime(2026, 8, 6, 22, 0))
    assert _gate()(**_contexto(ventana, conf={"force_run": True})) is True


def test_the_gate_does_not_decide_with_the_wall_clock() -> None:
    """`datetime.now` sólo puede aparecer como fallback, nunca como la decisión.

    Rojo con: devolver `now = datetime.now(tz)` a la línea de la referencia.

    Se cuenta sobre el AST y no sobre el texto: buscar la cadena `datetime.now(` casaba
    con las TRES menciones del docstring que explica el defecto, y el test fallaba por
    su propia prosa. Es el mismo fallo que ya me costó un `grep` sobre comentarios en el
    workflow de CI — medir el texto cuando lo que importa es el código.
    """
    import ast

    ini = FUENTE.index("def check_market_hours(")
    fin = FUENTE.index("# TASK 3: EXTRACT ALL SOURCES")
    cuerpo = FUENTE[ini:fin]

    arbol = ast.parse(cuerpo[: cuerpo.rindex("return True") + len("return True")])
    llamadas = [
        n for n in ast.walk(arbol)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "now"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "datetime"
    ]
    assert len(llamadas) == 1, (
        f"esperaba una única llamada a datetime.now (el fallback), encontré {len(llamadas)}"
    )
    assert "referencia = datetime.now(tz)" in cuerpo, (
        "el reloj de pared volvió a ser la fuente de la decisión en vez del fallback"
    )
    assert "context.get('data_interval_end')" in cuerpo, (
        "la decisión ya no se toma sobre la ventana lógica de la corrida"
    )
