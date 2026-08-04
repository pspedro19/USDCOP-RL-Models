"""BL-16 — el gate de gobernanza debe tener un llamador productivo.

`validate_declaration()` valida las 96 combinaciones nominales contra las 26 legales
y su paridad con los `CHECK` vivos de `control.strategy_declaration` quedó verificada
en `1e805c73`. Pero hasta este incremento **ningún código productivo lo invocaba**:
un mecanismo correcto que no protege nada en ejecución.

Lo que estos candados exigen:

1. que exista una función que **lea la declaración del SSOT de la estrategia** — no que
   la invente —, y
2. que **falle cerrado** cuando la estrategia no declara gobernanza. Una estrategia sin
   declaración no es un caso neutro: es el caso más inválido de todos, porque nadie
   afirmó en qué estado opera.

Lo que NO exigen: elegir el `capital_tier` de una estrategia. Eso es decisión de
gobierno (`PAPER` admite `ZERO` y `SHADOW`) y el gate la reclama, no la suple.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: SSOT real de la estrategia en producción H5.
SSOT_V11 = ROOT / "config" / "execution" / "smart_simple_v1.yaml"


def test_strategy_declaration_module_exists() -> None:
    """El lector debe existir como superficie productiva, no como helper de test."""
    from src.governance import strategy_declaration  # noqa: F401


def test_gate_fails_closed_when_the_strategy_does_not_declare_governance(
    tmp_path: Path,
) -> None:
    """Sin bloque `governance:`, el gate BLOQUEA — no asume un estado por defecto.

    Es la propiedad central de BL-16: *rechazar antes de ejecutar un DAG*. Una
    estrategia indeclarada no puede colarse por omisión.
    """
    from src.governance.declaration import DeclarationError
    from src.governance.strategy_declaration import declaration_from_strategy_config

    sin_declarar = tmp_path / "sin_governance.yaml"
    sin_declarar.write_text("executor:\n  status: paper\n", encoding="utf-8")

    with pytest.raises(DeclarationError, match="no declara"):
        declaration_from_strategy_config(sin_declarar)


def test_gate_rejects_an_illegal_combination_declared_in_config(tmp_path: Path) -> None:
    """Una combinación ilegal declarada explícitamente también se rechaza.

    `PAPER` sólo admite `ZERO`/`SHADOW`; declarar `FULL` debe morir en el gate y no
    llegar nunca a un DAG.
    """
    from src.governance.declaration import DeclarationError
    from src.governance.strategy_declaration import declaration_from_strategy_config

    ilegal = tmp_path / "ilegal.yaml"
    ilegal.write_text(
        "governance:\n"
        "  research_state: PAPER\n"
        "  capital_tier: FULL\n"
        "  operational_state: NOMINAL\n",
        encoding="utf-8",
    )

    with pytest.raises(DeclarationError):
        declaration_from_strategy_config(ilegal)


def test_gate_accepts_a_legal_combination_and_derives_dag_eligibility(
    tmp_path: Path,
) -> None:
    """Una combinación legal pasa, y `dag_declared` se DERIVA, no se declara a mano.

    Que un estado sea elegible para DAG lo dice `DAG_ELIGIBLE_STATES`, no el YAML:
    dejarlo escribir en config permitiría declarar `dag_declared: true` sobre un estado
    que no lo es.
    """
    from src.governance.strategy_declaration import declaration_from_strategy_config

    legal = tmp_path / "legal.yaml"
    legal.write_text(
        "governance:\n"
        "  research_state: PAPER\n"
        "  capital_tier: SHADOW\n"
        "  operational_state: NOMINAL\n",
        encoding="utf-8",
    )

    decl = declaration_from_strategy_config(legal)
    assert decl.research_state.value == "PAPER"
    assert decl.capital_tier.value == "SHADOW"
    assert decl.dag_declared is True  # PAPER ∈ DAG_ELIGIBLE_STATES


def test_production_strategy_ssot_is_the_declared_integration_point() -> None:
    """El SSOT real de v11 es el punto de integración que pide BL-16.

    Este candado no exige que ya declare gobernanza —esa es una decisión de gobierno
    pendiente—, sino que el fichero exista y sea el que el gate consulta. Cuando el
    bloque `governance:` se añada, `test_production_strategy_declares_governance`
    pasará de xfail a verde sin tocar el lector.
    """
    assert SSOT_V11.is_file(), f"SSOT de la estrategia H5 ausente: {SSOT_V11}"


def test_production_strategy_declares_governance() -> None:
    """El SSOT real de v11 declara gobernanza y el gate la valida.

    `capital_tier: SHADOW` no se eligió: es la convención de la spec normativa,
    cuyo ejemplo canónico para paper es PAPER+SHADOW+NOMINAL
    (`04-CTR-QLAB-FABRIC-004.md:410-411`), y coincide con que la estrategia
    paper-tradea de verdad — ejecuciones espejadas en `forecast_h5_paper_trading`.
    """
    from src.governance.strategy_declaration import declaration_from_strategy_config

    decl = declaration_from_strategy_config(SSOT_V11)
    assert decl.research_state.value == "PAPER"
    assert decl.capital_tier.value == "SHADOW"
    assert decl.operational_state.value == "NOMINAL"
    assert decl.dag_declared is True


#: DAG productivo de señal H5 — el consumidor real del gate.
DAG_H5 = ROOT / "airflow" / "dags" / "forecast_h5_l5_weekly_signal.py"


def _rshift_leaves(node: ast.AST, lado: str) -> set[str]:
    """Nombres en el extremo `lado` de una expresión `a >> b >> [c, d]`."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.RShift):
        return _rshift_leaves(node.right if lado == "right" else node.left, lado)
    if isinstance(node, (ast.List, ast.Tuple)):
        return {e.id for e in node.elts if isinstance(e, ast.Name)}
    if isinstance(node, ast.Name):
        return {node.id}
    return set()


def _dependency_edges(tree: ast.AST) -> set[tuple[str, str]]:
    """Aristas upstream→downstream declaradas con el operador `>>` de Airflow."""
    aristas: set[tuple[str, str]] = set()
    for nodo in ast.walk(tree):
        if isinstance(nodo, ast.BinOp) and isinstance(nodo.op, ast.RShift):
            for arriba in _rshift_leaves(nodo.left, "right"):
                for abajo in _rshift_leaves(nodo.right, "left"):
                    aristas.add((arriba, abajo))
    return aristas


def _task_var_by(tree: ast.AST, predicado) -> str | None:
    """Variable del operador cuyo `Call(...)` satisface `predicado(kwargs)`.

    Se identifica por lo que la tarea HACE (su `python_callable` / `task_id`), nunca
    por cómo se llama la variable: renombrar `t_governance` no debe burlar el candado.
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


def _gate_precede_a_la_senal(source: str) -> bool:
    """¿La tarea del gate es upstream real de la que produce la señal?

    Causal a propósito: no busca texto, resuelve alcanzabilidad sobre el grafo de
    dependencias. Una tarea definida pero **huérfana** (no enlazada con `>>`) devuelve
    `False`, que es exactamente el agujero que CXD-442 encontró en la versión textual.
    """
    tree = ast.parse(source)

    gate = _task_var_by(
        tree,
        lambda kw: isinstance(kw.get("python_callable"), ast.Name)
        and kw["python_callable"].id == "assert_governance_declaration",
    )
    senal = _task_var_by(
        tree,
        lambda kw: isinstance(kw.get("task_id"), ast.Constant)
        and kw["task_id"].value == "generate_signal",
    )
    if gate is None or senal is None:
        return False

    aristas = _dependency_edges(tree)
    alcanzados, frontera = {gate}, [gate]
    while frontera:
        actual = frontera.pop()
        for arriba, abajo in aristas:
            if arriba == actual and abajo not in alcanzados:
                alcanzados.add(abajo)
                frontera.append(abajo)
    return senal in alcanzados


def test_production_dag_gates_on_the_declaration_before_producing_signals() -> None:
    """El gate debe ser upstream REAL de la señal, no una tarea presente y suelta.

    Sin este candado el motor de gobernanza vuelve a ser correcto y no invocado, que
    es el estado del que este BL sale. La versión textual anterior de este test daba
    verde con la tarea huérfana (CXD-442): comprobaba presencia, que no es causalidad.
    """
    assert _gate_precede_a_la_senal(DAG_H5.read_text(encoding="utf-8")), (
        "el gate de gobernanza no es upstream de 'generate_signal': el DAG podría "
        "producir señal sin haber validado la declaración"
    )


def test_the_order_lock_fails_when_the_gate_is_unlinked() -> None:
    """Fail-first permanente: desenlazar el gate debe poner el candado en rojo.

    Este es el par del candado anterior y la razón de que exista. Un test de orden que
    nunca se demuestra capaz de fallar no prueba nada; aquí la demostración queda
    versionada, y se ejecuta en cada corrida en vez de haber ocurrido una sola vez.
    """
    huerfano = DAG_H5.read_text(encoding="utf-8").replace(
        "t_wait_l3 >> t_governance >> t_check", "t_wait_l3 >> t_check"
    )
    assert "t_wait_l3 >> t_check" in huerfano, "la mutación de prueba no se aplicó"
    assert not _gate_precede_a_la_senal(huerfano), (
        "con el gate desenlazado el candado sigue verde: vuelve a medir presencia, "
        "no orden de ejecución"
    )
