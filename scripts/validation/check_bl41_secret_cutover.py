#!/usr/bin/env python3
"""Validate the static BL-41 preflight contract without contacting DB or Vault.

Repository metadata can describe and bind evidence, but it cannot authorize a
credential cutover.  This gate therefore accepts the blocked state and rejects
``cutover_allowed: true`` unconditionally.  A future runtime gate must verify
the external systems and the operator action independently.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
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
EVIDENCE_KINDS = {
    "external_secret_store_canary": "external_secret_store_canary",
    "legacy_credential_tables_empty_under_lock": "locked_relation_count",
    "runtime_roles_non_superuser": "database_role_catalog",
    "migration_runner_fail_closed": "migration_runner_dry_run",
    "consumers_cutover_coordinated": "consumer_cutover_ack",
    "operator_authorization": "operator_change_approval",
}
EVIDENCE_FIELDS = {
    "subject",
    "kind",
    "artifact_path",
    "sha256",
    "observed_at",
}
EVIDENCE_ROOT = Path(".claude/evidence/bl41")
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
FORBIDDEN_SECRET_KEY = re.compile(
    r"(^|_)(api_key|api_secret|encrypted|ciphertext|passphrase|password|access_token|"
    r"refresh_token|secret_value|credential_value|fingerprint|mask)(_|$)",
    flags=re.IGNORECASE,
)
FORBIDDEN_EVIDENCE_FILENAMES = re.compile(
    r"^(?:\.env(?:\..*)?|credentials.*\.json|service-account.*\.json)$",
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


def _evidence_path_error(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return "artifact_path must be a nonempty repository-relative path"
    relative = Path(value)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        return "artifact_path must stay inside the repository"
    lowered_parts = {part.lower() for part in relative.parts}
    if "secrets" in lowered_parts or ".git" in lowered_parts:
        return "artifact_path may not target secrets or Git internals"
    if FORBIDDEN_EVIDENCE_FILENAMES.fullmatch(relative.name):
        return "artifact_path may not target an environment or credential file"
    if relative.suffix.lower() in {".pem", ".key"}:
        return "artifact_path may not target private-key material"
    if not relative.is_relative_to(EVIDENCE_ROOT):
        return "artifact_path must live under .claude/evidence/bl41"
    return None


def _evidence_errors(name: str, value: object) -> list[str]:
    evidence = _mapping(value)
    if evidence is None:
        return [f"precondition {name}.evidence must be a typed mapping"]

    errors: list[str] = []
    fields = set(evidence)
    if fields != EVIDENCE_FIELDS:
        errors.append(
            f"precondition {name}.evidence fields must be {sorted(EVIDENCE_FIELDS)}"
        )
    if evidence.get("subject") != name:
        errors.append(f"precondition {name}.evidence subject must match the precondition")
    if evidence.get("kind") != EVIDENCE_KINDS.get(name):
        errors.append(f"precondition {name}.evidence kind is not the reviewed kind")

    path_error = _evidence_path_error(evidence.get("artifact_path"))
    if path_error:
        errors.append(f"precondition {name}.evidence {path_error}")

    digest = evidence.get("sha256")
    if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
        errors.append(f"precondition {name}.evidence sha256 must be sha256:<64 lowercase hex>")

    observed_at = evidence.get("observed_at")
    if not isinstance(observed_at, str):
        errors.append(f"precondition {name}.evidence observed_at must be an ISO timestamp")
    else:
        try:
            parsed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
        if parsed is None or parsed.tzinfo is None or parsed.utcoffset() is None:
            errors.append(
                f"precondition {name}.evidence observed_at must include a UTC offset"
            )
    return errors


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
        "gate_mode": "STATIC_PREFLIGHT_ONLY",
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
    if preconditions is None:
        errors.append("preconditions must be a mapping")
    else:
        missing = REQUIRED_PRECONDITIONS - set(preconditions)
        if missing:
            errors.append(f"missing preconditions: {sorted(missing)}")
        extra = set(preconditions) - REQUIRED_PRECONDITIONS
        if extra:
            errors.append(f"unexpected preconditions: {sorted(extra)}")

        for name, raw_condition in preconditions.items():
            condition = _mapping(raw_condition)
            if condition is None:
                errors.append(f"precondition {name} must be a mapping")
                continue
            ready = condition.get("ready")
            if not isinstance(ready, bool):
                errors.append(f"precondition {name}.ready must be boolean")
                continue
            evidence = condition.get("evidence")
            if ready:
                errors.extend(_evidence_errors(name, evidence))
            elif evidence is not None:
                errors.append(f"precondition {name} is not ready but carries active evidence")

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
        errors.append("static preflight metadata cannot authorize cutover")
    if status != "BLOCKED_OPERATOR":
        errors.append("static preflight status must remain BLOCKED_OPERATOR")

    return errors


def cutover_may_proceed(payload: object) -> bool:
    """Static repository metadata never grants cutover authority."""

    _ = payload
    return False


def validate_repository_state(payload: object, root: Path) -> list[str]:
    """Bind ready evidence to safe repository files and reject premature DDL."""

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

    preconditions = _mapping(document.get("preconditions"))
    if preconditions is None:
        return errors
    resolved_root = root.resolve()
    for name, raw_condition in preconditions.items():
        condition = _mapping(raw_condition)
        if condition is None or condition.get("ready") is not True:
            continue
        evidence = _mapping(condition.get("evidence"))
        if evidence is None:
            continue
        artifact_path = evidence.get("artifact_path")
        if _evidence_path_error(artifact_path) is not None:
            continue
        assert isinstance(artifact_path, str)
        artifact = (resolved_root / artifact_path).resolve()
        if not artifact.is_relative_to(resolved_root):
            errors.append(f"precondition {name} evidence escapes repository")
            continue
        if not artifact.is_file():
            errors.append(f"precondition {name} evidence artifact does not exist: {artifact_path}")
            continue
        expected_digest = evidence.get("sha256")
        actual_digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
        if expected_digest != actual_digest:
            errors.append(f"precondition {name} evidence sha256 does not match artifact")
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
