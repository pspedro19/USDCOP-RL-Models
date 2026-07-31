#!/usr/bin/env python3
"""Validate the static BL-41 cutover contract without contacting DB or Vault.

The gate validates whether repository metadata is internally safe.  It cannot
attest external readiness and deliberately treats a blocked contract as a valid
state.  Enabling cutover requires every named precondition, nonempty evidence,
and explicit operator authorization.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "governance" / "bl41_secret_cutover.yaml"

REQUIRED_PRECONDITIONS = {
    "external_secret_store_canary",
    "legacy_credential_tables_empty_under_lock",
    "runtime_roles_non_superuser",
    "migration_runner_fail_closed",
    "consumers_cutover_coordinated",
    "operator_authorization",
}
LEGACY_RELATIONS = {
    "public.exchange_credentials",
    "public.sb_exchange_credentials",
    "public.user_exchange_keys",
}
REQUIRED_CONSUMERS = {
    "services/signalbridge_api",
    "usdcop-trading-dashboard",
}
REFERENCE_FIELDS = {
    "id",
    "user_id",
    "provider",
    "display_label",
    "secret_backend",
    "secret_reference",
    "status",
    "is_testnet",
    "created_at",
    "updated_at",
}
FORBIDDEN_SECRET_KEY = re.compile(
    r"(^|_)(api_key|api_secret|encrypted|ciphertext|passphrase|password|access_token|"
    r"refresh_token|secret_value|credential_value|fingerprint|mask)(_|$)",
    flags=re.IGNORECASE,
)


def _mapping(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    if not all(isinstance(key, str) for key in value):
        return None
    return value


def _string_set(value: object) -> set[str] | None:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        return None
    if not all(isinstance(item, str) and item.strip() for item in value):
        return None
    return set(value)


def _has_evidence(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return bool(value)
    return False


def _secret_material_key_paths(value: object, prefix: str = "root") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}"
            if FORBIDDEN_SECRET_KEY.search(key):
                paths.append(path)
            paths.extend(_secret_material_key_paths(child, path))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, child in enumerate(value):
            paths.extend(_secret_material_key_paths(child, f"{prefix}[{index}]"))
    return paths


def validate_document(payload: object) -> list[str]:
    """Return all static contract violations; an empty list means internally valid."""

    errors: list[str] = []
    document = _mapping(payload)
    if document is None:
        return ["root document must be a string-keyed mapping"]

    expected_scalars = {
        "schema_version": "1.0.0",
        "backlog_id": "BL-41",
        "decision_contract": "C-007",
        "planned_migration": "database/migrations/069_secret_external_account.sql",
    }
    for key, expected in expected_scalars.items():
        if document.get(key) != expected:
            errors.append(f"{key} must be {expected!r}")

    target = _mapping(document.get("target"))
    if target is None:
        errors.append("target must be a mapping")
    else:
        if target.get("relation") != "secret.external_account":
            errors.append("target relation must be secret.external_account")
        if target.get("stores_references_only") is not True:
            errors.append("target must store references only")
        if target.get("secret_material_backend") != "external_vault_or_kms":
            errors.append("target backend must be external_vault_or_kms")
        fields = _string_set(target.get("reference_fields"))
        if fields != REFERENCE_FIELDS:
            errors.append("target reference_fields must match the C-007 reference-only shape")

    for path in _secret_material_key_paths(document):
        errors.append(f"forbidden secret-material key: {path}")

    preconditions = _mapping(document.get("preconditions"))
    all_ready = False
    if preconditions is None:
        errors.append("preconditions must be a mapping")
    else:
        missing = REQUIRED_PRECONDITIONS - set(preconditions)
        if missing:
            errors.append(f"missing preconditions: {sorted(missing)}")

        readiness: list[bool] = []
        for name, raw_condition in preconditions.items():
            condition = _mapping(raw_condition)
            if condition is None:
                errors.append(f"precondition {name} must be a mapping")
                readiness.append(False)
                continue
            ready = condition.get("ready")
            if not isinstance(ready, bool):
                errors.append(f"precondition {name}.ready must be boolean")
                readiness.append(False)
                continue
            readiness.append(ready)
            if ready and not _has_evidence(condition.get("evidence")):
                errors.append(f"precondition {name} is ready without nonempty evidence")

        all_ready = bool(readiness) and len(readiness) == len(preconditions) and all(readiness)

        legacy = _mapping(preconditions.get("legacy_credential_tables_empty_under_lock"))
        relations = _string_set(legacy.get("relations")) if legacy else None
        if relations != LEGACY_RELATIONS:
            errors.append("legacy credential relations must match the three C-007 sources")

        consumers = _mapping(preconditions.get("consumers_cutover_coordinated"))
        named_consumers = _string_set(consumers.get("consumers")) if consumers else None
        if named_consumers is None or not named_consumers >= REQUIRED_CONSUMERS:
            errors.append("consumer cutover must include SignalBridge and dashboard")

    allowed = document.get("cutover_allowed")
    status = document.get("status")
    if not isinstance(allowed, bool):
        errors.append("cutover_allowed must be boolean")
    elif allowed:
        if status != "READY_FOR_CUTOVER":
            errors.append("enabled cutover requires status READY_FOR_CUTOVER")
        if not all_ready:
            errors.append("enabled cutover requires all preconditions ready with evidence")
    elif status != "BLOCKED_OPERATOR":
        errors.append("disabled cutover requires status BLOCKED_OPERATOR")

    return errors


def cutover_may_proceed(payload: object) -> bool:
    """Return static eligibility; this never substitutes for the operator action."""

    document = _mapping(payload)
    return bool(
        document
        and document.get("cutover_allowed") is True
        and document.get("status") == "READY_FOR_CUTOVER"
        and not validate_document(document)
    )


def validate_repository_state(payload: object, root: Path) -> list[str]:
    """Reject a DDL file while the fail-closed contract remains blocked."""

    errors: list[str] = []
    document = _mapping(payload)
    if document is None:
        return ["root document must be a string-keyed mapping"]
    migration = document.get("planned_migration")
    if not isinstance(migration, str):
        return errors
    relative = Path(migration)
    if relative.is_absolute() or ".." in relative.parts:
        return ["planned_migration must be a repository-relative path"]
    if document.get("cutover_allowed") is False and (root / relative).exists():
        errors.append(f"premature migration exists while cutover is blocked: {migration}")
    return errors


def load_document(path: Path) -> object:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)

    payload = load_document(args.config)
    errors = [*validate_document(payload), *validate_repository_state(payload, args.root)]
    if errors:
        for error in errors:
            print(f"BL-41 static gate ERROR: {error}", file=sys.stderr)
        return 1

    document = _mapping(payload)
    assert document is not None
    allowed = str(document["cutover_allowed"]).lower()
    print(f"BL-41 static gate OK: {document['status']} (cutover_allowed={allowed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
