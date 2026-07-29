"""Append-only, idempotent research-family and trial ledger operations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

GENESIS_HASH = "0" * 64
TRIAL_ID_PATTERN = re.compile(r"^(FT|AT)-\d{4}$")
CONTENT_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class QLabError(RuntimeError):
    pass


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _utc_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise QLabError("audit timestamps must be valid ISO-8601 values") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise QLabError("audit timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _utc_iso(value: str) -> str:
    return _utc_datetime(value).isoformat().replace("+00:00", "Z")


@contextmanager
def _exclusive_lock(
    path: Path, timeout_seconds: float = 15.0, stale_after_seconds: float = 300.0
) -> Iterator[None]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"{os.getpid()}\n".encode())
            os.close(descriptor)
            break
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > stale_after_seconds:
                    path.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise QLabError(f"ledger lock timeout: {path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


@dataclass(frozen=True)
class TrialCharge:
    trial_id: str
    family: str
    asset: str
    cluster: str
    kind: str
    variant: str
    cutoff: str
    result: str = "pending"
    env: str = "screening"
    label: str = "qlab"
    source: str | None = None
    code_hash: str | None = None
    data_hash: str | None = None
    available_at_field: str | None = None
    n_rows: int | None = None
    max_available_at: str | None = None
    note: str | None = None


class TrialLedger:
    def __init__(self, path: Path):
        self.path = path

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise QLabError(f"{self.path}:{number}: invalid JSON") from exc
        return rows

    def charge(self, charge: TrialCharge) -> dict[str, Any]:
        match = (
            TRIAL_ID_PATTERN.fullmatch(charge.trial_id)
            if isinstance(charge.trial_id, str)
            else None
        )
        expected_family = (
            "FT" if charge.kind == "forecast" else "AT" if charge.kind == "action" else None
        )
        if match is None or expected_family is None or match.group(1) != expected_family:
            raise QLabError(
                "trial_id must match ^(FT|AT)-\\d{4}$ and agree with "
                "forecast=FT or action=AT"
            )
        if not charge.cutoff:
            raise QLabError("screening trials require an explicit cutoff")

        audit_values = (
            charge.available_at_field,
            charge.n_rows,
            charge.max_available_at,
        )
        if charge.source is not None:
            if any(value is None for value in audit_values):
                raise QLabError(
                    "sourced screening trials require available_at_field, "
                    "n_rows and max_available_at"
                )
            if charge.available_at_field != "available_at":
                raise QLabError("sourced screening trials require canonical available_at")
            if (
                not isinstance(charge.n_rows, int)
                or isinstance(charge.n_rows, bool)
                or charge.n_rows < 1
            ):
                raise QLabError("sourced screening trials require n_rows >= 1")
            if (
                not isinstance(charge.data_hash, str)
                or CONTENT_HASH_PATTERN.fullmatch(charge.data_hash) is None
            ):
                raise QLabError("sourced screening trials require a canonical sha256 hash")
            cutoff_instant = _utc_datetime(charge.cutoff)
            maximum_instant = _utc_datetime(str(charge.max_available_at))
            if (
                _utc_iso(charge.cutoff) != charge.cutoff
                or _utc_iso(str(charge.max_available_at))
                != charge.max_available_at
            ):
                raise QLabError("audit timestamps must use canonical UTC Z form")
            if maximum_instant > cutoff_instant:
                raise QLabError("max_available_at cannot exceed cutoff")
        elif any(value is not None for value in audit_values):
            raise QLabError("audit fields require a sourced screening trial")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive_lock(self.path.with_suffix(".lock")):
            rows = self.rows()
            existing = next((row for row in rows if row.get("trial_id") == charge.trial_id), None)
            if existing:
                immutable = {
                    "family": charge.family,
                    "asset": charge.asset,
                    "cluster": charge.cluster,
                    "kind": charge.kind,
                    "variant": charge.variant,
                    "cutoff": charge.cutoff,
                    "result": charge.result,
                    "env": charge.env,
                    "label": charge.label,
                    "source": charge.source,
                    "code_hash": charge.code_hash,
                    "data_hash": charge.data_hash,
                    "available_at_field": charge.available_at_field,
                    "n_rows": charge.n_rows,
                    "max_available_at": charge.max_available_at,
                    "note": charge.note,
                }
                if any(existing.get(key) != value for key, value in immutable.items()):
                    raise QLabError(f"{charge.trial_id} already exists with a different payload")
                return existing
            payload: dict[str, Any] = {
                "trial_id": charge.trial_id,
                "family": charge.family,
                "asset": charge.asset,
                "cluster": charge.cluster,
                "kind": charge.kind,
                "variant": charge.variant,
                "cutoff": charge.cutoff,
                "result": charge.result,
                "env": charge.env,
                "label": charge.label,
                "source": charge.source,
                "code_hash": charge.code_hash,
                "data_hash": charge.data_hash,
                "available_at_field": charge.available_at_field,
                "n_rows": charge.n_rows,
                "max_available_at": charge.max_available_at,
                "note": charge.note,
                "charged_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "N_family": sum(row.get("family") == charge.family for row in rows) + 1,
                "N_cluster": sum(row.get("cluster") == charge.cluster for row in rows) + 1,
                "N_global": len(rows) + 1,
                "prev_hash": rows[-1]["line_hash"] if rows else GENESIS_HASH,
            }
            payload["line_hash"] = hashlib.sha256(_canonical(payload)).hexdigest()
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(_canonical(payload).decode("utf-8") + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return payload


class FamilyStore:
    ALLOWED_TRANSITIONS = {
        "DECLARED": {"SCREENING", "CLOSED"},
        "SCREENING": {"FROZEN", "CLOSED"},
        "FROZEN": {"PROMOTED", "CLOSED"},
        "PROMOTED": {"CLOSED"},
        "CLOSED": set(),
    }

    def __init__(self, root: Path):
        self.root = root

    def path(self, family_id: str) -> Path:
        if not family_id.replace("_", "").replace("-", "").isalnum():
            raise QLabError("invalid family_id")
        return self.root / f"{family_id}.yaml"

    def load(self, family_id: str) -> dict[str, Any]:
        path = self.path(family_id)
        if not path.exists():
            raise QLabError(f"undeclared family: {family_id}")
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    @staticmethod
    def _write(path: Path, document: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = yaml.safe_dump(
            dict(document), sort_keys=False, allow_unicode=True
        )
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def declare(self, payload: Mapping[str, Any]) -> Path:
        family_id = str(payload["family_id"])
        path = self.path(family_id)
        if path.exists():
            existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            immutable = ("family_id", "kind", "cluster_id", "asset", "question", "bar")
            if any(existing.get(key) != payload.get(key) for key in immutable):
                raise QLabError(f"conflicting declaration for {family_id}")
            return path
        document = {
            **dict(payload),
            "state": "DECLARED",
            "declared_at": datetime.now(timezone.utc).date().isoformat(),
            "cells": [],
            "trials_charged": 0,
            "closed": False,
        }
        self._write(path, document)
        return path

    def record_trial(self, family_id: str, row: Mapping[str, Any]) -> Path:
        """Idempotently project a charged ledger row into its family manifest."""
        path = self.path(family_id)
        with _exclusive_lock(path.with_suffix(".lock")):
            document = self.load(family_id)
            if document.get("kind") != row.get("kind"):
                raise QLabError("ledger kind conflicts with family declaration")
            if document.get("cluster_id") != row.get("cluster"):
                raise QLabError("ledger cluster conflicts with family declaration")
            cells = list(document.get("cells") or [])
            existing = next(
                (cell for cell in cells if cell.get("trial_id") == row.get("trial_id")),
                None,
            )
            projected = {
                "asset": row.get("asset"),
                "variant": row.get("variant"),
                "cell_kind": "atomic",
                "trial_id": row.get("trial_id"),
                "n_trials": 1,
                "status": "EVALUATED",
                "result": row.get("result"),
                "cutoff": row.get("cutoff"),
                "cutoff_class": "declared_in_cell",
                "source": row.get("source"),
                "note": row.get("note"),
            }
            if existing is not None:
                if any(existing.get(key) != value for key, value in projected.items()):
                    raise QLabError(
                        f"{row.get('trial_id')} family cell has a different payload"
                    )
                return path
            cells.append(projected)
            document["cells"] = cells
            document["trials_charged"] = int(document.get("trials_charged", 0)) + 1
            self._write(path, document)
        return path

    def transition(self, family_id: str, state: str, *, reason: str) -> Path:
        path = self.path(family_id)
        document = self.load(family_id)
        current = str(document.get("state", "DECLARED")).upper()
        target = state.upper()
        if target not in self.ALLOWED_TRANSITIONS.get(current, set()):
            raise QLabError(f"illegal family transition {current} -> {target}")
        document["state"] = target
        document["state_changed_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        document["state_change_reason"] = reason
        document["closed"] = target == "CLOSED"
        if target == "CLOSED":
            document["closure_note"] = reason
        self._write(path, document)
        return path
