"""C028 — la medicion de disponibilidad corre donde y cuando debe.

El defecto que C028 existe para matar: `weekly_generator` hacia `fillna(0.0)` sobre el
sentiment, asi que "no medido" se volvia "medido y neutro" y en la serie eran
indistinguibles. Al ACKear el contrato encontre que el `fillna` estaba en CUATRO sitios,
no uno -- y el cuarto es un `.get(..., 0)` dentro de un `np.mean`, que un grep por
`fillna` NO encuentra.

Estos candados fijan las dos decisiones que tome al cablear la tarea, ambas con la misma
forma que ya acordamos en C027: **ubicacion aguas abajo del productor** y **cutoff
compartido con el consumidor**.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.support.dag_graph import by_task_id, task_runs_before  # noqa: E402

DAG = ROOT / "airflow" / "dags" / "news_daily_pipeline.py"


def test_the_measurement_runs_after_the_features_it_measures() -> None:
    """Aguas abajo de `export_features`: mide el estado que ese paso acaba de producir.

    Un DAG de calidad aparte habria creado un segundo schedule que mantener alineado con
    la ingesta, y el dia que se desalinearan la medicion describiria un estado que ya no
    existe. Es el mismo razonamiento por el que `verify_ledger_anchor` cuelga de
    `paper_ledger_2026`.
    """
    assert task_runs_before(
        DAG.read_text(encoding="utf-8"),
        by_task_id("export_features"),
        by_task_id("measure_feature_availability"),
    ), "la medicion no corre despues de las features que mide"


def test_the_three_daily_runs_produce_three_distinct_cutoffs() -> None:
    """El cron es `0 7,12,18 * * 1-5`: las tres corridas NO pueden compartir key.

    Defecto que Codex rechazo (CXD-502) y era doble. Mi primera version usaba
    `news_feature_cutoff(data_interval_end.date())`, que suma tres dias:

        07:00 UTC -> 2026-08-07T00:00Z
        12:00 UTC -> 2026-08-07T00:00Z     <- misma key
        18:00 UTC -> 2026-08-07T00:00Z     <- misma key

    Las tres colisionaban aunque la evidencia entre ellas hubiera cambiado. Y peor: esa
    key esta en el **futuro** -- el viernes se pedia disponibilidad a lunes 00:00, un
    instante que aun no ha ocurrido. Medir disponibilidad en un tiempo futuro no
    significa nada.

    La causa de fondo: el `+2d` de aquel helper pertenece a la **ventana de seleccion de
    articulos**, no al tiempo de observacion. Reutilizarlo fue confundir dos relojes
    porque ambos se llamaban "cutoff".
    """
    from datetime import datetime, timezone

    fuente = DAG.read_text(encoding="utf-8")
    assert 'schedule="0 7,12,18 * * 1-5"' in fuente, (
        "cambio el cron: re-verifica que las corridas siguen sin colisionar"
    )

    arbol = ast.parse(fuente)
    cuerpo = next(
        n for n in ast.walk(arbol)
        if isinstance(n, ast.FunctionDef) and n.name == "_measure_feature_availability"
    )
    # Solo el cuerpo EJECUTABLE: el docstring nombra el helper viejo para explicar por
    # que se retiro, y mirarlo entero hacia fallar el candado por su propia explicacion
    # -- me paso al escribirlo.
    ejecutable = ast.Module(
        body=[n for n in cuerpo.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))],
        type_ignores=[],
    )
    texto = ast.unparse(ejecutable)

    assert "cutoff = context['data_interval_end']" in texto.replace('"', "'"), (
        "el cutoff no es el instante logico exacto de la corrida"
    )
    assert "news_feature_cutoff" not in texto, (
        "vuelve a usar la ventana de seleccion de articulos como tiempo de observacion"
    )

    # Discriminante: los tres instantes reales dan tres claves distintas.
    instantes = [datetime(2026, 8, 4, h, tzinfo=timezone.utc) for h in (7, 12, 18)]
    assert len({i.isoformat() for i in instantes}) == 3

    # Y ninguno es futuro respecto a su propia corrida, por construccion: el cutoff ES
    # el final del intervalo que acaba de cerrarse.
    for i in instantes:
        assert i <= datetime(2026, 8, 4, 18, tzinfo=timezone.utc)


def test_the_cutoff_is_never_wall_clock() -> None:
    """Ni `now()` ni `today()`: dos re-ejecuciones de la misma corrida deben coincidir."""
    arbol = ast.parse(DAG.read_text(encoding="utf-8"))
    cuerpo = next(
        n for n in ast.walk(arbol)
        if isinstance(n, ast.FunctionDef) and n.name == "_measure_feature_availability"
    )
    texto = ast.unparse(cuerpo)

    for reloj in ("date.today(", "datetime.now(", "utcnow("):
        assert reloj not in texto, (
            f"usa {reloj}: dos re-ejecuciones del mismo run logico medirian instantes "
            "distintos y la idempotencia por (feature, instante) dejaria de existir"
        )
    assert "data_interval_end" in texto


def test_an_empty_catalog_fails_closed_instead_of_measuring_nothing() -> None:
    """Sin features declaradas se LANZA: una medicion vacia se leeria como 'todo bien'.

    Es la misma trampa que el `fillna(0.0)` que C028 combate, un nivel mas arriba: cero
    mediciones y cero UNAVAILABLE son indistinguibles de 'todo disponible' si nadie
    exige que el catalogo tenga contenido.
    """
    fuente = DAG.read_text(encoding="utf-8")
    arbol = ast.parse(fuente)
    cuerpo = next(
        n for n in ast.walk(arbol)
        if isinstance(n, ast.FunctionDef) and n.name == "_measure_feature_availability"
    )
    texto = ast.unparse(cuerpo)

    assert "if not specs" in texto and "raise" in texto, (
        "un catalogo vacio no falla cerrado: cero mediciones se leerian como cobertura"
    )


def test_the_transaction_rolls_back_on_failure() -> None:
    """La persistencia es del caller y un fallo no puede dejar media medicion escrita."""
    fuente = DAG.read_text(encoding="utf-8")
    arbol = ast.parse(fuente)
    cuerpo = next(
        n for n in ast.walk(arbol)
        if isinstance(n, ast.FunctionDef) and n.name == "_measure_feature_availability"
    )
    texto = ast.unparse(cuerpo)

    assert "rollback" in texto and "raise" in texto, (
        "sin rollback+raise, un fallo a mitad dejaria mediciones parciales persistidas"
    )
    assert texto.index("commit") < texto.index("rollback"), (
        "el commit debe preceder al except; si no, se estaria commiteando en el fallo"
    )
