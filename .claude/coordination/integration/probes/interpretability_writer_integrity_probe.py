"""Adversarial red probe for the BL-20 immutable artifact writer.

The probe exercises integrity properties that the happy-path idempotency tests
do not cover:

1. A stored payload cannot be altered while retaining its old ``artifact_id``.
2. ``supersedes`` is part of the cryptographic identity of the new artifact.
3. Two divergent first publishers cannot both report success for one path.

It writes only below a temporary directory.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _namespace(name: str, relative_path: str) -> None:
    package = ModuleType(name)
    package.__package__ = name
    package.__path__ = [str(REPO_ROOT / relative_path)]  # type: ignore[attr-defined]
    sys.modules[name] = package


_namespace("src", "src")
_namespace("src.contracts", "src/contracts")

import scripts.analysis.generate_interpretability as writer  # noqa: E402


def _payload(value: int) -> dict:
    return {
        "nota": writer.NOTA,
        "generated_at": "2026-07-28T00:00:00+00:00",
        "value": value,
    }


failures: list[str] = []

with tempfile.TemporaryDirectory(prefix="bl20-integrity-") as tmp:
    original_root = writer.OUT_ROOT
    original_replace = writer.os.replace
    try:
        writer.OUT_ROOT = Path(tmp) / "tamper"
        path = writer._write("zoo", "usdcop", "probe", "v1", _payload(1))
        tampered = json.loads(path.read_text(encoding="utf-8"))
        tampered["value"] = 999
        path.write_text(json.dumps(tampered), encoding="utf-8")

        detected = False
        try:
            writer._write("zoo", "usdcop", "probe", "v1", _payload(1))
        except writer.ArtifactConflictError:
            detected = True
        print(json.dumps({"case": "stored_payload_tamper", "detected": detected}))
        if not detected:
            failures.append("stored payload was trusted solely by its retained artifact_id")

        identity_a = writer._artifact_identity(
            {"value": 2, "supersedes": "sha256:" + "a" * 64}
        )
        identity_b = writer._artifact_identity(
            {"value": 2, "supersedes": "sha256:" + "b" * 64}
        )
        supersedes_bound = identity_a != identity_b
        print(
            json.dumps(
                {
                    "case": "supersedes_bound_to_identity",
                    "bound": supersedes_bound,
                    "identity_a": identity_a,
                    "identity_b": identity_b,
                }
            )
        )
        if not supersedes_bound:
            failures.append("supersedes is excluded from artifact identity")

        writer.OUT_ROOT = Path(tmp) / "race"
        barrier = threading.Barrier(2)

        def racing_replace(source: str | bytes, destination: str | bytes) -> None:
            barrier.wait(timeout=5)
            original_replace(source, destination)

        writer.os.replace = racing_replace

        def publish(value: int) -> str:
            try:
                writer._write("zoo", "usdcop", "probe", "v1", _payload(value))
                return "success"
            except writer.ArtifactConflictError:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = sorted(pool.map(publish, (1, 2)))
        race_safe = outcomes == ["conflict", "success"]
        print(
            json.dumps(
                {
                    "case": "divergent_first_publishers",
                    "outcomes": outcomes,
                    "race_safe": race_safe,
                }
            )
        )
        if not race_safe:
            failures.append("divergent concurrent publishers both reported success")
    finally:
        writer.OUT_ROOT = original_root
        writer.os.replace = original_replace

if failures:
    print(json.dumps({"result": "RED", "failures": failures}, ensure_ascii=False))
    raise SystemExit(1)

print(json.dumps({"result": "GREEN"}))
