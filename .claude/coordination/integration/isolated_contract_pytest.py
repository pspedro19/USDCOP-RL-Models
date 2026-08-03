"""Run contract tests without importing the application's eager package roots.

This is an audit harness, not an application shim.  ``src/__init__.py`` and
``src/contracts/__init__.py`` currently import optional runtime dependencies
such as pytz/joblib during test collection.  Contract tests do not need those
dependencies, so this runner exposes the real package directories as namespace
packages and then delegates to pytest.

The normal test command remains the acceptance target.  A green run through
this file proves contract behaviour only; it does not excuse collection errors
from the repository's regular pytest entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _namespace(name: str, relative_path: str) -> None:
    package = ModuleType(name)
    package.__package__ = name
    package.__path__ = [str(REPO_ROOT / relative_path)]  # type: ignore[attr-defined]
    sys.modules[name] = package


def main(arguments: list[str]) -> int:
    _namespace("src", "src")
    _namespace("src.contracts", "src/contracts")
    _namespace("src.strategies", "src/strategies")
    return pytest.main(arguments)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
