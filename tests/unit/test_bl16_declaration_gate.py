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


@pytest.mark.xfail(
    reason="decisión de gobierno pendiente: PAPER admite ZERO y SHADOW; el gate la "
    "reclama pero no la elige (CLD-431)",
    strict=False,
)
def test_production_strategy_declares_governance() -> None:
    """Cuando v11 declare gobernanza, el gate debe validarla contra el SSOT real."""
    from src.governance.strategy_declaration import declaration_from_strategy_config

    decl = declaration_from_strategy_config(SSOT_V11)
    assert decl.dag_declared is True
