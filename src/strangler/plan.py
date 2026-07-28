"""Loader for the declarative strangler plan (BL-31).

The plan is the SSOT of *what* migrates and *in what order*; the ledger is the SSOT of
*what actually happened*. Loading is strict: an unknown key is a typo that would silently
weaken a gate, so it raises instead of being ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from src.strangler.contracts import (
    AcceptanceCriterion,
    HashKind,
    LayerPlan,
    RollbackPlan,
    SensorMigration,
    StranglerContractError,
    StranglerPlan,
    as_mapping,
    parse_enum,
    parse_layer,
)

#: Default location of the USD/COP plan.
DEFAULT_PLAN_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "migration" / "strangler_usdcop.yaml"
)

_PLAN_KEYS = {
    "asset",
    "contract_id",
    "version",
    "execute_requires_days_after_last_migration",
    "ab_cohort",
    "layers",
    "acceptance_criteria",
}
_LAYER_KEYS = {
    "layer",
    "legacy_anchors",
    "artifacts",
    "required_hash_kind",
    "candidate_generator",
    "min_parallel_days",
    "min_observations",
    "sensor_migrations",
    "rollback",
    "caveat",
}
_ROLLBACK_KEYS = {"trigger", "steps", "data_loss", "rehearsed"}
_SENSOR_KEYS = {"sensor_ref", "external_dag_id", "replacement_asset"}
_CRITERION_KEYS = {"id", "text", "owner"}


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], what: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise StranglerContractError(f"{what}: unknown key(s) {unknown}")


def _rollback(payload: Any) -> RollbackPlan:
    data = as_mapping(payload, "rollback")
    _reject_unknown(data, _ROLLBACK_KEYS, "rollback")
    if "trigger" not in data or "steps" not in data:
        raise StranglerContractError("rollback requires 'trigger' and 'steps'")
    steps = data["steps"]
    if isinstance(steps, str) or not isinstance(steps, (list, tuple)):
        raise StranglerContractError("rollback.steps must be a list of strings")
    return RollbackPlan(
        trigger=data["trigger"],
        steps=tuple(steps),
        data_loss=bool(data.get("data_loss", False)),
        rehearsed=bool(data.get("rehearsed", False)),
    )


def _sensor(payload: Any) -> SensorMigration:
    data = as_mapping(payload, "sensor_migration")
    _reject_unknown(data, _SENSOR_KEYS, "sensor_migration")
    missing = sorted(_SENSOR_KEYS - set(data))
    if missing:
        raise StranglerContractError(f"sensor_migration missing {missing}")
    return SensorMigration(
        sensor_ref=data["sensor_ref"],
        external_dag_id=data["external_dag_id"],
        replacement_asset=data["replacement_asset"],
    )


def _layer(payload: Any) -> LayerPlan:
    data = as_mapping(payload, "layer")
    _reject_unknown(data, _LAYER_KEYS, "layer")
    for required in ("layer", "legacy_anchors", "artifacts", "required_hash_kind", "rollback"):
        if required not in data:
            raise StranglerContractError(f"layer entry missing '{required}'")
    sensors = data.get("sensor_migrations") or []
    if isinstance(sensors, str) or not isinstance(sensors, (list, tuple)):
        raise StranglerContractError("sensor_migrations must be a list")
    return LayerPlan(
        layer=parse_layer(data["layer"]),
        legacy_anchors=tuple(data["legacy_anchors"] or ()),
        artifacts=tuple(data["artifacts"] or ()),
        required_hash_kind=parse_enum(
            HashKind, data["required_hash_kind"], "required_hash_kind"
        ),
        rollback=_rollback(data["rollback"]),
        candidate_generator=data.get("candidate_generator"),
        min_parallel_days=int(data.get("min_parallel_days", 14)),
        min_observations=int(data.get("min_observations", 2)),
        sensor_migrations=tuple(_sensor(s) for s in sensors),
        caveat=data.get("caveat"),
    )


def _criterion(payload: Any) -> AcceptanceCriterion:
    data = as_mapping(payload, "acceptance_criterion")
    _reject_unknown(data, _CRITERION_KEYS, "acceptance_criterion")
    missing = sorted(_CRITERION_KEYS - set(data))
    if missing:
        raise StranglerContractError(f"acceptance_criterion missing {missing}")
    return AcceptanceCriterion(id=int(data["id"]), text=data["text"], owner=data["owner"])


def parse_plan(payload: Any) -> StranglerPlan:
    data = as_mapping(payload, "plan")
    _reject_unknown(data, _PLAN_KEYS, "plan")
    for required in ("asset", "contract_id", "version", "layers", "acceptance_criteria"):
        if required not in data:
            raise StranglerContractError(f"plan missing '{required}'")
    layers = data["layers"]
    if isinstance(layers, str) or not isinstance(layers, (list, tuple)):
        raise StranglerContractError("plan.layers must be a list")
    criteria = data["acceptance_criteria"]
    if isinstance(criteria, str) or not isinstance(criteria, (list, tuple)):
        raise StranglerContractError("plan.acceptance_criteria must be a list")
    return StranglerPlan(
        asset=data["asset"],
        contract_id=data["contract_id"],
        version=str(data["version"]),
        layers=tuple(_layer(entry) for entry in layers),
        acceptance_criteria=tuple(_criterion(entry) for entry in criteria),
        execute_requires_days_after_last_migration=int(
            data.get("execute_requires_days_after_last_migration", 30)
        ),
        ab_cohort=tuple(data.get("ab_cohort") or ()),
    )


def load_plan(path: str | Path | None = None) -> StranglerPlan:
    plan_path = Path(path) if path is not None else DEFAULT_PLAN_PATH
    if not plan_path.is_file():
        raise StranglerContractError(f"strangler plan not found: {plan_path}")
    payload = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    return parse_plan(payload)
