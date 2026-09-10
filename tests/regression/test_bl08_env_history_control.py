"""BL-08 local Git-history evidence must match the repository being tested.

The probe intentionally asks Git only for path/revision reachability.  It never
reads, prints, hashes, or otherwise materializes the historical ``.env`` bytes.
Remote visibility is operator-attested and is outside this local probe.
"""

from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "config" / "governance" / "security_incident_env_history.yaml"
LOCAL_FIELDS = {
    "env_tracked_now",
    "env_present_in_local_history",
    "env_blob_recoverable_locally",
    "local_history_rewritten",
}


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def _load_control() -> dict[str, object]:
    payload = yaml.safe_load(CONTROL.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _measure_local_history(root: Path, payload: dict[str, object]) -> dict[str, bool]:
    affected = payload.get("affected_history")
    assert isinstance(affected, list) and affected
    revisions = [entry["revision"] for entry in affected if isinstance(entry, dict)]
    assert revisions and all(isinstance(revision, str) for revision in revisions)

    tracked = _git(root, "ls-files", "--error-unmatch", "--", ".env").returncode == 0
    history = _git(root, "log", "--all", "--format=%H", "--", ".env")
    assert history.returncode == 0, history.stderr
    present = bool(history.stdout.splitlines())
    recoverable = any(
        _git(root, "cat-file", "-e", f"{revision}:.env").returncode == 0
        for revision in revisions
    )
    return {
        "env_tracked_now": tracked,
        "env_present_in_local_history": present,
        "env_blob_recoverable_locally": recoverable,
        "local_history_rewritten": not present and not recoverable,
    }


def _mismatches(payload: dict[str, object], measured: dict[str, bool]) -> list[str]:
    declared = payload.get("local_repository_evidence")
    if not isinstance(declared, dict) or set(declared) != LOCAL_FIELDS:
        return ["local_repository_evidence must contain exactly the schema 1.1 fields"]
    return [
        f"{field}: declared={declared[field]!r}, measured={value!r}"
        for field, value in measured.items()
        if declared.get(field) is not value
    ]


def test_local_history_control_matches_full_clone_bidirectionally() -> None:
    payload = _load_control()
    assert payload["schema_version"] == "1.1.0"
    shallow = _git(ROOT, "rev-parse", "--is-shallow-repository")
    assert shallow.returncode == 0, shallow.stderr
    assert shallow.stdout.strip() == "false", (
        "BL-08 history evidence requires a full clone; CI must checkout with fetch-depth: 0"
    )

    measured = _measure_local_history(ROOT, payload)
    assert _mismatches(payload, measured) == []


def test_gate_rejects_optimistic_and_pessimistic_stale_evidence() -> None:
    payload = _load_control()
    measured = _measure_local_history(ROOT, payload)
    for field, actual in measured.items():
        stale = copy.deepcopy(payload)
        stale["local_repository_evidence"][field] = not actual
        assert any(field in error for error in _mismatches(stale, measured))


def test_current_control_turns_red_in_a_purged_clone_fixture(tmp_path: Path) -> None:
    """After a real purge, today's pessimistic declaration must become stale."""
    assert _git(tmp_path, "init").returncode == 0
    assert _git(tmp_path, "config", "user.email", "bl08-gate@example.invalid").returncode == 0
    assert _git(tmp_path, "config", "user.name", "BL-08 gate").returncode == 0
    (tmp_path / "README.md").write_text("purged fixture\n", encoding="utf-8")
    assert _git(tmp_path, "add", "README.md").returncode == 0
    assert _git(tmp_path, "commit", "-m", "history without env").returncode == 0

    payload = _load_control()
    measured = _measure_local_history(tmp_path, payload)
    assert measured == {
        "env_tracked_now": False,
        "env_present_in_local_history": False,
        "env_blob_recoverable_locally": False,
        "local_history_rewritten": True,
    }
    errors = _mismatches(payload, measured)
    assert {error.split(":", 1)[0] for error in errors} == {
        "env_present_in_local_history",
        "env_blob_recoverable_locally",
        "local_history_rewritten",
    }


def _assert_remote_attestation_boundary(payload: dict[str, object]) -> None:
    remote = payload.get("remote_repository_evidence")
    assert isinstance(remote, dict)
    assert set(remote) == {"source", "observed_visibility"}
    assert remote["source"] == "OPERATOR_ATTESTATION"
    assert remote["observed_visibility"] in {"public", "private"}
    assert payload["release_policy"]["push_allowed"] is False
    assert payload["status"] == "BLOCKED_OPERATOR"


def test_remote_visibility_is_operator_attested_not_locally_derived() -> None:
    _assert_remote_attestation_boundary(_load_control())


def test_private_operator_attestation_is_a_legal_transition_but_does_not_enable_push() -> None:
    payload = copy.deepcopy(_load_control())
    payload["remote_repository_evidence"]["observed_visibility"] = "private"
    _assert_remote_attestation_boundary(payload)


def test_remote_attestation_rejects_an_unknown_visibility_value() -> None:
    payload = copy.deepcopy(_load_control())
    payload["remote_repository_evidence"]["observed_visibility"] = "maybe_private"
    with pytest.raises(AssertionError):
        _assert_remote_attestation_boundary(payload)
