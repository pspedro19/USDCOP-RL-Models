"""Constitutional governance primitives.

The package is intentionally independent from Airflow and application services so
declarations can be rejected before a DAG is imported or executed.
"""

from src.governance.declaration import (
    CapitalTier,
    GovernanceDeclaration,
    OperationalState,
    ResearchState,
    validate_declaration,
)

__all__ = [
    "CapitalTier",
    "GovernanceDeclaration",
    "OperationalState",
    "ResearchState",
    "validate_declaration",
]
