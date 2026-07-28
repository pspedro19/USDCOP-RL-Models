"""Domain-separated fingerprints for the control-plane spine (BL-17)."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from src.identity.canonical import canonical_json_bytes


def _fingerprint(namespace: str, payload: Mapping[str, Any]) -> str:
    envelope = {"namespace": namespace, "version": 1, "payload": payload}
    return "sha256:" + hashlib.sha256(canonical_json_bytes(envelope)).hexdigest()


def spec_fingerprint(
    *,
    strategy_spec: Mapping[str, Any],
    data_snapshot_id: str,
    feature_snapshot_id: str,
    config_hash: str,
    model_or_policy_hash: str,
    calendar_hash: str,
    cost_model_hash: str,
    dependency_lock_hash: str,
    container_image_digest: str,
) -> str:
    return _fingerprint(
        "spec",
        {
            "strategy_spec": strategy_spec,
            "data_snapshot_id": data_snapshot_id,
            "feature_snapshot_id": feature_snapshot_id,
            "config_hash": config_hash,
            "model_or_policy_hash": model_or_policy_hash,
            "calendar_hash": calendar_hash,
            "cost_model_hash": cost_model_hash,
            "dependency_lock_hash": dependency_lock_hash,
            "container_image_digest": container_image_digest,
        },
    )


def decision_fingerprint(
    *,
    spec_fingerprint_value: str,
    as_of: str,
    decision_inputs: Mapping[str, Any] | Sequence[Any],
) -> str:
    return _fingerprint(
        "decision",
        {
            "spec_fingerprint": spec_fingerprint_value,
            "as_of": as_of,
            "decision_inputs": decision_inputs,
        },
    )


def execution_fingerprint(
    *,
    decision_fingerprint_value: str,
    env: str,
    account_id: str,
    broker_id: str,
    order_policy_hash: str,
) -> str:
    return _fingerprint(
        "execution",
        {
            "decision_fingerprint": decision_fingerprint_value,
            "env": env,
            "account_id": account_id,
            "broker_id": broker_id,
            "order_policy_hash": order_policy_hash,
        },
    )


def derivation_id(
    *,
    inputs: Mapping[str, Any] | Sequence[Any],
    code_hash: str,
    params: Mapping[str, Any],
) -> str:
    return _fingerprint(
        "derivation",
        {"inputs": inputs, "code_hash": code_hash, "params": params},
    )
