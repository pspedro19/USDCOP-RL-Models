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


def test_the_cutoff_is_shared_with_the_consumer_not_wall_clock() -> None:
    """El corte sale de `news_feature_cutoff`, el MISMO helper que usa el consumidor.

    Si midieramos con "ahora" y el weekly leyera otra ventana, habria features marcadas
    disponibles que el consumidor no ve, y al reves. Es el gemelo del
    `quality_observed_at` de C027: alli acordamos re-evaluar con el instante original por
    exactamente la misma razon.
    """
    fuente = DAG.read_text(encoding="utf-8")
    arbol = ast.parse(fuente)
    cuerpo = next(
        n for n in ast.walk(arbol)
        if isinstance(n, ast.FunctionDef) and n.name == "_measure_feature_availability"
    )
    texto = ast.unparse(cuerpo)

    assert "news_feature_cutoff" in texto, (
        "el cutoff no usa el helper compartido: la medicion y el consumo miraran "
        "ventanas distintas"
    )
    for reloj in ("date.today(", "datetime.now(", "utcnow("):
        assert reloj not in texto, (
            f"usa {reloj}: dos re-ejecuciones de la misma fecha logica medirian ventanas "
            "distintas y la idempotencia por (feature, instante) dejaria de significar nada"
        )
    assert "data_interval_end" in texto, "el corte no se deriva de la fecha logica"


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
