"""C-010 R3 / BL-45 — la tarea `publish` debe SATISFACER la firma real de publish_signal.

Defecto que encontre en mi propio factory al intentar cumplir la condicion de Codex de
"un policy_run **observado atravesando** resolve->evaluate->publish": la cadena emitia
las tres tareas y la tercera **crasheaba con TypeError**, porque llamaba
`publish_signal(decision)` mientras la funcion exige cinco keyword-args obligatorios.

Una cadena que existe en el grafo y no puede atravesarse es **peor** que no tenerla: el
DAG la muestra, el tablero la cuenta, y solo se descubre al ejecutar. Es la variante
mas cara del patron que este ciclo persigue -- no un gate que no protege, sino un
consumidor que no puede consumir.

El candado compara la firma REAL con lo que el sitio de llamada aporta, asi que si
`publish_signal` gana un parametro obligatorio manana, esto se pone rojo en vez de
esperar a la primera ejecucion.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FACTORY = ROOT / "airflow" / "dags" / "asset_pipeline_factory.py"


def _publish_call_kwargs() -> set[str]:
    """Keywords que el factory pasa a `publish_signal`, leidas del AST."""
    arbol = ast.parse(FACTORY.read_text(encoding="utf-8"))
    for nodo in ast.walk(arbol):
        if (
            isinstance(nodo, ast.Call)
            and isinstance(nodo.func, ast.Name)
            and nodo.func.id == "publish_signal"
        ):
            return {k.arg for k in nodo.keywords if k.arg}
    return set()


def test_the_publish_task_supplies_every_required_keyword() -> None:
    """Todo keyword-only obligatorio de `publish_signal` llega desde el factory."""
    from src.policy_engine import publish_signal

    firma = inspect.signature(publish_signal).parameters
    obligatorios = {
        nombre
        for nombre, parametro in firma.items()
        if parametro.kind is inspect.Parameter.KEYWORD_ONLY
        and parametro.default is inspect.Parameter.empty
    }

    aportados = _publish_call_kwargs()
    faltantes = obligatorios - aportados

    assert not faltantes, (
        f"la tarea `publish` no aporta {sorted(faltantes)}: emitiria la tarea y "
        "crasheria con TypeError al ejecutarla — una cadena que el grafo muestra y "
        "que no puede atravesarse"
    )


def test_created_at_is_the_logical_instant_not_wall_clock() -> None:
    """`created_at` no puede ser `now()`: el replay dejaria de reproducir.

    Dos re-ejecuciones de la misma fecha logica deben producir el mismo registro. Con
    reloj de pared, cada corrida escribiria una señal distinta para la misma decision, y
    comparar dos replays seria imposible — el mismo principio por el que C023 excluye
    `generated_at` del hash.
    """
    fuente = FACTORY.read_text(encoding="utf-8")
    inicio = fuente.index("def make_publish_signal")
    cuerpo = fuente[inicio : fuente.index("\ndef ", inicio + 10)]

    for reloj in ("datetime.now(", "utcnow(", "time.time("):
        assert reloj not in cuerpo, (
            f"la tarea `publish` usa {reloj}: dos re-ejecuciones de la misma fecha "
            "logica produzirian registros distintos y el replay dejaria de serlo"
        )
    assert "data_interval_end" in cuerpo, (
        "`created_at` no se deriva del intervalo logico de la corrida"
    )


def test_the_instrument_id_comes_from_the_spine_not_from_a_name_convention() -> None:
    """El `instrument_id` se resuelve contra `reference.instrument`, no se adivina.

    `asset_id` y `canonical_symbol` son cosas distintas; adivinar cual toca es
    exactamente como se rompieron los joins que BL-37 existe para arreglar.
    """
    fuente = FACTORY.read_text(encoding="utf-8")

    assert "reference.instrument" in fuente, (
        "el factory no consulta la espina para el instrument_id: lo estaria infiriendo"
    )
    assert "_canonical_instrument_id" in fuente
