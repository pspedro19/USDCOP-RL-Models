"""Python side of the policy timestamp parity red probe."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[4]


def _namespace(name: str, relative_path: str) -> None:
    package = ModuleType(name)
    package.__package__ = name
    package.__path__ = [str(REPO_ROOT / relative_path)]  # type: ignore[attr-defined]
    sys.modules[name] = package


_namespace("src", "src")
_namespace("src.contracts", "src/contracts")

from src.contracts.policy import require_iso_timestamp  # noqa: E402


CASES = (
    ("year_0000", "0000-01-01T00:00:00Z", False),
    ("year_0001", "0001-01-01T00:00:00Z", True),
    ("leap_1900", "1900-02-29T00:00:00Z", False),
    ("leap_2000", "2000-02-29T00:00:00Z", True),
)


failures = 0
for case_id, value, expected in CASES:
    try:
        require_iso_timestamp("as_of", value)
        actual = True
    except ValueError:
        actual = False
    print({"id": case_id, "value": value, "expected": expected, "actual": actual})
    failures += actual != expected

raise SystemExit(1 if failures else 0)
