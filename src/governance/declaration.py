"""Fail-closed legality matrix for strategy declarations (BL-16).

The three dimensions deliberately remain independent. In particular,
``QUARANTINED`` does not rewrite the capital tier: retaining the intended tier is
required to resume safely after recovery, while order admission blocks openings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping


class DeclarationError(ValueError):
    """Raised when a declaration violates the constitutional state matrix."""


class ResearchState(StrEnum):
    DECLARED = "DECLARED"
    SCREENED = "SCREENED"
    DESIGN_RUN = "DESIGN_RUN"
    FROZEN = "FROZEN"
    PAPER = "PAPER"
    CHAMPION = "CHAMPION"
    RETIRING = "RETIRING"
    WITHDRAWN = "WITHDRAWN"


class CapitalTier(StrEnum):
    ZERO = "ZERO"
    SHADOW = "SHADOW"
    CANARY = "CANARY"
    FULL = "FULL"
    REDUCED = "REDUCED"
    EXIT_ONLY = "EXIT_ONLY"


class OperationalState(StrEnum):
    NOMINAL = "NOMINAL"
    QUARANTINED = "QUARANTINED"


_ZERO_ONLY = {
    ResearchState.DECLARED,
    ResearchState.SCREENED,
    ResearchState.DESIGN_RUN,
    ResearchState.FROZEN,
}

LEGAL_CAPITAL_TIERS: dict[ResearchState, frozenset[CapitalTier]] = {
    **{state: frozenset({CapitalTier.ZERO}) for state in _ZERO_ONLY},
    ResearchState.PAPER: frozenset({CapitalTier.ZERO, CapitalTier.SHADOW}),
    ResearchState.CHAMPION: frozenset(
        {
            CapitalTier.ZERO,
            CapitalTier.SHADOW,
            CapitalTier.CANARY,
            CapitalTier.FULL,
            CapitalTier.REDUCED,
        }
    ),
    ResearchState.RETIRING: frozenset({CapitalTier.EXIT_ONLY}),
    ResearchState.WITHDRAWN: frozenset({CapitalTier.ZERO}),
}

DAG_ELIGIBLE_STATES = frozenset(
    {
        ResearchState.FROZEN,
        ResearchState.PAPER,
        ResearchState.CHAMPION,
        ResearchState.RETIRING,
    }
)


def _closed_enum(enum_type: type[StrEnum], value: Any, field: str) -> StrEnum:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise DeclarationError(f"{field} must be a string")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise DeclarationError(
            f"{field}={value!r} is not one of {[member.value for member in enum_type]}"
        ) from exc


@dataclass(frozen=True, slots=True)
class GovernanceDeclaration:
    research_state: ResearchState
    capital_tier: CapitalTier
    operational_state: OperationalState
    exit_checklist: str | None = None
    dag_declared: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.research_state, ResearchState):
            raise DeclarationError("research_state must be a ResearchState")
        if not isinstance(self.capital_tier, CapitalTier):
            raise DeclarationError("capital_tier must be a CapitalTier")
        if not isinstance(self.operational_state, OperationalState):
            raise DeclarationError("operational_state must be an OperationalState")
        if self.dag_declared is not None and type(self.dag_declared) is not bool:
            raise DeclarationError("dag_declared must be a boolean when present")
        if self.exit_checklist is not None and not isinstance(self.exit_checklist, str):
            raise DeclarationError("exit_checklist must be a string when present")
        validate_declaration(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GovernanceDeclaration":
        allowed = {
            "research_state",
            "capital_tier",
            "operational_state",
            "exit_checklist",
            "dag_declared",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise DeclarationError(f"unknown declaration fields: {unknown}")
        missing = sorted(
            {"research_state", "capital_tier", "operational_state"} - set(raw)
        )
        if missing:
            raise DeclarationError(f"missing declaration fields: {missing}")
        dag_declared = raw.get("dag_declared")
        if dag_declared is not None and type(dag_declared) is not bool:
            raise DeclarationError("dag_declared must be a boolean when present")
        exit_checklist = raw.get("exit_checklist")
        if exit_checklist is not None and not isinstance(exit_checklist, str):
            raise DeclarationError("exit_checklist must be a string when present")
        declaration = cls(
            research_state=_closed_enum(
                ResearchState, raw["research_state"], "research_state"
            ),
            capital_tier=_closed_enum(CapitalTier, raw["capital_tier"], "capital_tier"),
            operational_state=_closed_enum(
                OperationalState, raw["operational_state"], "operational_state"
            ),
            exit_checklist=exit_checklist,
            dag_declared=dag_declared,
        )
        return declaration

    @property
    def blocks_new_orders(self) -> bool:
        return self.operational_state is OperationalState.QUARANTINED

    @property
    def dag_should_exist(self) -> bool:
        return self.research_state in DAG_ELIGIBLE_STATES


def validate_declaration(
    declaration: GovernanceDeclaration | Mapping[str, Any],
) -> GovernanceDeclaration:
    """Validate all 96 nominal combinations against the 26 legal combinations."""

    if not isinstance(declaration, GovernanceDeclaration):
        return GovernanceDeclaration.from_mapping(declaration)

    allowed = LEGAL_CAPITAL_TIERS[declaration.research_state]
    if declaration.capital_tier not in allowed:
        raise DeclarationError(
            f"{declaration.research_state.value} cannot use "
            f"{declaration.capital_tier.value}; allowed={sorted(x.value for x in allowed)}"
        )
    if (
        declaration.research_state is ResearchState.WITHDRAWN
        and declaration.exit_checklist != "PASS"
    ):
        raise DeclarationError("WITHDRAWN requires exit_checklist=PASS")
    if (
        declaration.dag_declared is not None
        and declaration.dag_declared is not declaration.dag_should_exist
    ):
        expectation = "must" if declaration.dag_should_exist else "must not"
        raise DeclarationError(
            f"a DAG {expectation} exist for {declaration.research_state.value}"
        )
    return declaration
