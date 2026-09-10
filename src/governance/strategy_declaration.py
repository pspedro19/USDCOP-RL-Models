"""BL-16 — lee la declaración de gobernanza de una estrategia desde su SSOT.

`validate_declaration()` (`src/governance/declaration.py`) valida las 96 combinaciones
nominales contra las 26 legales, y su paridad con los `CHECK` vivos de
`control.strategy_declaration` quedó verificada en `1e805c73`. Lo que faltaba —y lo que
BL-16 pide— es un **llamador productivo**: un mecanismo correcto que nadie invoca no
protege nada en ejecución.

Este módulo es esa costura. Su regla de diseño es una sola:

    **lee lo declarado; no infiere lo no declarado.**

Por eso una estrategia sin bloque ``governance:`` **no obtiene un estado por defecto**:
obtiene un `DeclarationError`. Un default silencioso convertiría "nadie lo declaró" en
"está permitido", que es el fallo que este backlog existe para impedir.

Contract: CTR-QLAB-FABRIC-004 (BL-16) · Date: 2026-08-05
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from src.governance.declaration import (
    DAG_ELIGIBLE_STATES,
    DeclarationError,
    GovernanceDeclaration,
    ResearchState,
)

#: Clave del bloque declarativo dentro del SSOT de la estrategia.
GOVERNANCE_KEY = "governance"

#: Campos que el SSOT declara. `dag_declared` NO está aquí a propósito: se deriva de
#: `DAG_ELIGIBLE_STATES`. Dejarlo en config permitiría declarar `dag_declared: true`
#: sobre un estado que no es elegible — una contradicción escribible a mano.
_DECLARED_FIELDS = ("research_state", "capital_tier", "operational_state", "exit_checklist")


def _read_governance_block(config_path: Path) -> Mapping[str, Any]:
    """Devuelve el bloque ``governance:`` o falla cerrado si no existe."""
    if not config_path.is_file():
        raise DeclarationError(f"SSOT de estrategia ausente: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - defensa de parseo
        raise DeclarationError(f"SSOT ilegible ({config_path}): {exc}") from exc

    block = raw.get(GOVERNANCE_KEY)
    if not isinstance(block, Mapping) or not block:
        raise DeclarationError(
            f"{config_path.name} no declara gobernanza: falta el bloque "
            f"'{GOVERNANCE_KEY}' con research_state/capital_tier/operational_state. "
            "Una estrategia indeclarada no puede ejecutar un DAG"
        )
    return block


def declaration_from_strategy_config(config_path: str | Path) -> GovernanceDeclaration:
    """Construye y **valida** la declaración de gobernanza declarada en el SSOT.

    Falla cerrado en los tres casos que importan: SSOT ausente, sin bloque
    ``governance:``, o con una combinación que las reglas no admiten.

    ``dag_declared`` se deriva de ``DAG_ELIGIBLE_STATES``; el YAML no puede fijarlo.
    """
    path = Path(config_path)
    block = _read_governance_block(path)

    payload: dict[str, Any] = {
        campo: block[campo] for campo in _DECLARED_FIELDS if campo in block
    }

    estado = payload.get("research_state")
    if not isinstance(estado, str):
        raise DeclarationError(
            f"{path.name}: 'research_state' es obligatorio en el bloque de gobernanza"
        )

    try:
        payload["dag_declared"] = ResearchState(estado) in DAG_ELIGIBLE_STATES
    except ValueError as exc:
        raise DeclarationError(f"{path.name}: research_state desconocido {estado!r}") from exc

    return GovernanceDeclaration.from_mapping(payload)


def assert_strategy_may_run_dag(config_path: str | Path) -> GovernanceDeclaration:
    """Gate para el arranque de un DAG: valida la declaración o **aborta**.

    Pensado para invocarse como primera tarea de un DAG de estrategia. Si la
    declaración no existe o es ilegal, el `DeclarationError` detiene el DAG **antes**
    de que produzca señales — que es exactamente lo que BL-16 pide.
    """
    declaration = declaration_from_strategy_config(config_path)
    if declaration.dag_declared is not True:
        raise DeclarationError(
            f"{Path(config_path).name}: research_state "
            f"'{declaration.research_state.value}' no es elegible para ejecutar un DAG "
            f"(elegibles: {sorted(s.value for s in DAG_ELIGIBLE_STATES)})"
        )
    return declaration
