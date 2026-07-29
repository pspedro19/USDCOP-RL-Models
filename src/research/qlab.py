"""Append-only, idempotent research-family and trial ledger operations."""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

GENESIS_HASH = "0" * 64


class QLabError(RuntimeError):
    pass


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


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
                    "note": charge.note,
                }
                if any(existing.get(key) != value for key, value in immutable.items()):
                    raise QLabError(f"{charge.trial_id} already exists with a different payload")
                return existing
            prefix = "FT-" if charge.kind == "forecast" else "AT-" if charge.kind == "action" else None
            if prefix is None or not charge.trial_id.startswith(prefix):
                raise QLabError("trial_id prefix must match forecast=FT or action=AT")
            if not charge.cutoff:
                raise QLabError("screening trials require an explicit cutoff")
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
