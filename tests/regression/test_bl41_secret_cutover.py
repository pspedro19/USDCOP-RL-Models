"""BL-41 static cutover guard.

The real database/Vault cutover is an operator action.  These tests only prove
that repository state cannot advertise readiness while evidence is absent or a
PostgreSQL credential value is part of the proposed target.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import yaml

from scripts.validation.check_bl41_secret_cutover import (
    EVIDENCE_KINDS,
    cutover_may_proceed,
    main,
    validate_document,
    validate_repository_state,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "governance" / "bl41_secret_cutover.yaml"
VALIDATOR = ROOT / "scripts" / "validation" / "check_bl41_secret_cutover.py"


def _document() -> dict[str, object]:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _attach_typed_evidence(payload: dict[str, object], root: Path) -> None:
    preconditions = payload["preconditions"]
    assert isinstance(preconditions, dict)
    for name, condition in preconditions.items():
        assert isinstance(condition, dict)
        relative = Path(".claude") / "evidence" / "bl41" / f"{name}.json"
        artifact = root / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(
            f'{{"precondition":"{name}","result":"PASS"}}\n',
            encoding="utf-8",
        )
        digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
        condition["ready"] = True
        condition["evidence"] = {
            "subject": name,
            "kind": EVIDENCE_KINDS[name],
            "artifact_path": relative.as_posix(),
            "sha256": digest,
            "observed_at": "2026-07-31T14:00:00-05:00",
        }


def test_repository_contract_is_valid_but_stays_blocked() -> None:
    payload = _document()
    assert validate_document(payload) == []
    assert validate_repository_state(payload, ROOT) == []
    assert payload["status"] == "BLOCKED_OPERATOR"
    assert payload["cutover_allowed"] is False
    assert cutover_may_proceed(payload) is False


def test_cutover_cannot_be_enabled_while_any_precondition_is_false() -> None:
    payload = _document()
    payload["cutover_allowed"] = True
    payload["status"] = "READY_FOR_CUTOVER"

    errors = validate_document(payload)
    assert any("static preflight metadata cannot authorize" in error for error in errors)
    assert any("status must remain BLOCKED_OPERATOR" in error for error in errors)
    assert cutover_may_proceed(payload) is False


def test_ready_precondition_requires_nonempty_evidence() -> None:
    payload = _document()
    preconditions = payload["preconditions"]
    assert isinstance(preconditions, dict)
    preconditions["external_secret_store_canary"]["ready"] = True

    errors = validate_document(payload)
    assert any("external_secret_store_canary" in error and "evidence" in error for error in errors)


def test_typed_evidence_can_be_bound_but_never_authorizes_cutover(tmp_path: Path) -> None:
    payload = _document()
    _attach_typed_evidence(payload, tmp_path)

    assert validate_document(payload) == []
    assert validate_repository_state(payload, tmp_path) == []
    assert cutover_may_proceed(payload) is False

    without_operator = deepcopy(payload)
    without_operator["preconditions"]["operator_authorization"]["evidence"] = None
    errors = validate_document(without_operator)
    assert any("operator_authorization.evidence" in error for error in errors)


def test_plain_ok_evidence_cannot_buy_cutover_authority() -> None:
    payload = _document()
    preconditions = payload["preconditions"]
    assert isinstance(preconditions, dict)
    for condition in preconditions.values():
        assert isinstance(condition, dict)
        condition["ready"] = True
        condition["evidence"] = "ok"
    payload["status"] = "READY_FOR_CUTOVER"
    payload["cutover_allowed"] = True

    errors = validate_document(payload)
    assert sum("evidence must be a typed mapping" in error for error in errors) == 6
    assert any("static preflight metadata cannot authorize" in error for error in errors)
    assert cutover_may_proceed(payload) is False


def test_evidence_subject_path_and_hash_are_bound(tmp_path: Path) -> None:
    payload = _document()
    _attach_typed_evidence(payload, tmp_path)
    preconditions = payload["preconditions"]
    assert isinstance(preconditions, dict)
    canary = preconditions["external_secret_store_canary"]
    assert isinstance(canary, dict)
    evidence = canary["evidence"]
    assert isinstance(evidence, dict)

    evidence["subject"] = "operator_authorization"
    assert any("subject must match" in error for error in validate_document(payload))
    evidence["subject"] = "external_secret_store_canary"
    evidence["sha256"] = "sha256:" + "0" * 64
    assert any(
        "sha256 does not match artifact" in error
        for error in validate_repository_state(payload, tmp_path)
    )
    evidence["artifact_path"] = ".env"
    assert any("environment or credential file" in error for error in validate_document(payload))


def test_target_rejects_secret_material_columns_and_public_schema() -> None:
    payload = _document()
    target = payload["target"]
    assert isinstance(target, dict)
    target["api_secret"] = "do-not-store"
    target["relation"] = "public.external_account"

    errors = validate_document(payload)
    assert any("forbidden secret-material key" in error for error in errors)
    assert any("secret.external_account" in error for error in errors)


def test_all_legacy_credential_relations_are_named() -> None:
    payload = _document()
    relations = payload["preconditions"]["legacy_credential_tables_empty_under_lock"]["relations"]
    relations.remove("public.user_exchange_keys")

    errors = validate_document(payload)
    assert any("legacy credential relations" in error for error in errors)


def test_blocked_contract_rejects_a_premature_migration_file(tmp_path: Path) -> None:
    payload = _document()
    migration = tmp_path / payload["planned_migration"]
    migration.parent.mkdir(parents=True)
    migration.write_text("-- premature DDL", encoding="utf-8")

    errors = validate_repository_state(payload, tmp_path)
    assert any("premature migration" in error for error in errors)


def test_validator_is_static_and_cli_reports_blocked_state(capsys: object) -> None:
    source = VALIDATOR.read_text(encoding="utf-8")
    for forbidden in ("psycopg2", "hvac", "dotenv", "os.environ", "subprocess"):
        assert forbidden not in source

    assert main(["--config", str(CONFIG), "--root", str(ROOT)]) == 0
    output = capsys.readouterr().out
    assert "BL-41 static gate OK" in output
    assert "BLOCKED_OPERATOR" in output
    assert "cutover_allowed=false" in output
