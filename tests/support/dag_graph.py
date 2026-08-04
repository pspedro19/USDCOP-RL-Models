"""Lectura causal del grafo de un DAG de Airflow, para candados de orden.

Nace de un defecto concreto y repetido: comprobar el orden de tareas **buscando texto**.
Codex lo encontró en mi candado de BL-16 (CXD-442) —retirar la tarea del grafo la dejaba
huérfana y el test seguía verde— y al revisar su BL-18 encontré el mismo patrón en el
suyo (CLD-444): con la cadena escrita en un comentario, su batería daba 3P mientras el
scheduler mostraba `upstream de persist_governed_sharpe: []`.

Los dos casos comparten la raíz: *presencia no es causalidad*. Un candado que busca
`'a >> b'` en el fuente aprueba un comentario y rechaza un reordenamiento equivalente;
falla en las dos direcciones.

Este módulo resuelve **alcanzabilidad** sobre las aristas `>>` parseadas del AST, y
identifica cada tarea por lo que **hace** (`python_callable`, `task_id`) y nunca por el
nombre de su variable — renombrar `t_governance` no debe burlar un candado.

Por qué AST y no `DagBag`: cargar el DAG real exige Airflow instalado y las variables de
entorno del scheduler, así que un candado basado en `DagBag` se salta en CI justo cuando
más falta hace. El AST se lee en cualquier sitio. Cuando el scheduler está disponible,
verificar además contra él es un complemento — no un sustituto.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Callable, Mapping


def _rshift_leaves(node: ast.AST, lado: str) -> set[str]:
    """Nombres en el extremo `lado` de una expresión `a >> b >> [c, d]`."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.RShift):
        return _rshift_leaves(node.right if lado == "right" else node.left, lado)
    if isinstance(node, (ast.List, ast.Tuple)):
        return {e.id for e in node.elts if isinstance(e, ast.Name)}
    if isinstance(node, ast.Name):
        return {node.id}
    return set()


def dependency_edges(tree: ast.AST) -> set[tuple[str, str]]:
    """Aristas `upstream → downstream` declaradas con el operador `>>` de Airflow.

    `a >> b >> c` se parsea como `((a >> b) >> c)`, así que recorrer todos los `BinOp`
    del árbol produce las dos aristas sin tratar el anidamiento como caso especial. Un
    `>>` dentro de un comentario o de una cadena no existe para el parser, que es
    justamente la propiedad que un `in source` no tiene.
    """
    aristas: set[tuple[str, str]] = set()
    for nodo in ast.walk(tree):
        if isinstance(nodo, ast.BinOp) and isinstance(nodo.op, ast.RShift):
            for arriba in _rshift_leaves(nodo.left, "right"):
                for abajo in _rshift_leaves(nodo.right, "left"):
                    aristas.add((arriba, abajo))
    return aristas


def task_var_by(
    tree: ast.AST, predicado: Callable[[Mapping[str, ast.AST]], bool]
) -> str | None:
    """Variable del operador cuyo `Call(...)` satisface `predicado(kwargs)`.

    El predicado recibe los keyword-args **sin evaluar** (nodos AST), de modo que se
    identifica la tarea por su `python_callable` o su `task_id` — lo que hace— y nunca
    por cómo se llama la variable que la sostiene.
    """
    for nodo in ast.walk(tree):
        if (
            isinstance(nodo, ast.Assign)
            and len(nodo.targets) == 1
            and isinstance(nodo.targets[0], ast.Name)
            and isinstance(nodo.value, ast.Call)
        ):
            kwargs = {k.arg: k.value for k in nodo.value.keywords if k.arg}
            if predicado(kwargs):
                return nodo.targets[0].id
    return None


def by_task_id(task_id: str) -> Callable[[Mapping[str, ast.AST]], bool]:
    """Predicado: la tarea cuyo `task_id=` es exactamente `task_id`."""

    def _predicado(kwargs: Mapping[str, ast.AST]) -> bool:
        nodo = kwargs.get("task_id")
        return isinstance(nodo, ast.Constant) and nodo.value == task_id

    return _predicado


def by_callable(nombre: str) -> Callable[[Mapping[str, ast.AST]], bool]:
    """Predicado: la tarea cuyo `python_callable=` es la función `nombre`."""

    def _predicado(kwargs: Mapping[str, ast.AST]) -> bool:
        nodo = kwargs.get("python_callable")
        return isinstance(nodo, ast.Name) and nodo.id == nombre

    return _predicado


def reaches(aristas: set[tuple[str, str]], origen: str, destino: str) -> bool:
    """¿Existe un camino `origen → … → destino` en el grafo de dependencias?"""
    alcanzados, frontera = {origen}, [origen]
    while frontera:
        actual = frontera.pop()
        for arriba, abajo in aristas:
            if arriba == actual and abajo not in alcanzados:
                alcanzados.add(abajo)
                frontera.append(abajo)
    return destino in alcanzados


def task_runs_before(
    source: str,
    upstream: Callable[[Mapping[str, ast.AST]], bool],
    downstream: Callable[[Mapping[str, ast.AST]], bool],
) -> bool:
    """¿La tarea `upstream` es antecesora real de `downstream`?

    Devuelve `False` si cualquiera de las dos no existe o si no hay camino entre ellas
    — incluido el caso que motivó este módulo: la tarea **definida pero desenlazada**.
    """
    tree = ast.parse(source)
    arriba = task_var_by(tree, upstream)
    abajo = task_var_by(tree, downstream)
    if arriba is None or abajo is None:
        return False
    return reaches(dependency_edges(tree), arriba, abajo)


def dag_source(nombre_fichero: str, dags_dir: Path | None = None) -> str:
    """Fuente de un DAG por nombre de fichero, para no repetir la ruta en cada test."""
    base = dags_dir or Path(__file__).resolve().parents[2] / "airflow" / "dags"
    return (base / nombre_fichero).read_text(encoding="utf-8")
