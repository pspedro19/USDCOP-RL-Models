"""Strangler migration control plane (BL-31, FABRIC §29).

Public surface:

    from src.strangler import load_plan, ParityLedger, evaluate_advance, parity_table

The USD/COP chain migrates layer by layer with parity evidence; this package owns the
plan, the append-only evidence ledger and the advance gates. It never sends an order and
never trains a model.
"""

from src.strangler.contracts import (
    CONTRACT_ID,
    CONTRACT_VERSION,
    LAYER_ORDER,
    MONEY_LAYER,
    AcceptanceAttestation,
    AcceptanceCriterion,
    CriterionStatus,
    ExecutionReadiness,
    GateDecision,
    HashKind,
    LayerPlan,
    LayerState,
    LayerStatus,
    LayerTransition,
    MigrationLayer,
    ParityObservation,
    ParityVerdict,
    RecordType,
    RollbackPlan,
    SensorMigration,
    StranglerContractError,
    StranglerPlan,
)
from src.strangler.gates import (
    acceptance_gaps,
    derive_states,
    evaluate_advance,
    layer_status,
    migrated_at,
    parity_table,
    summarize,
)
from src.strangler.parity import (
    ParityError,
    ParityLedger,
    green_streak,
    hash_artifact,
    observe_parity,
)
from src.strangler.plan import DEFAULT_PLAN_PATH, load_plan, parse_plan

__all__ = [
    "AcceptanceAttestation",
    "AcceptanceCriterion",
    "CONTRACT_ID",
    "CONTRACT_VERSION",
    "CriterionStatus",
    "DEFAULT_PLAN_PATH",
    "ExecutionReadiness",
    "GateDecision",
    "HashKind",
    "LAYER_ORDER",
    "LayerPlan",
    "LayerState",
    "LayerStatus",
    "LayerTransition",
    "MONEY_LAYER",
    "MigrationLayer",
    "ParityError",
    "ParityLedger",
    "ParityObservation",
    "ParityVerdict",
    "RecordType",
    "RollbackPlan",
    "SensorMigration",
    "StranglerContractError",
    "StranglerPlan",
    "acceptance_gaps",
    "derive_states",
    "evaluate_advance",
    "green_streak",
    "hash_artifact",
    "layer_status",
    "load_plan",
    "migrated_at",
    "observe_parity",
    "parity_table",
    "parse_plan",
    "summarize",
]
