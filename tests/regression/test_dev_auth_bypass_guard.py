"""The dev auth bypass must never be on where security assertions are believed.

Contract: CTR-RBAC-001

`services/signalbridge_api/app/middleware/auth.py` returns a `DevUser` with
``role = "admin"`` for EVERY request when `SIGNALBRIDGE_DEV_MODE=true`, ignoring the
presented token. It is guarded against production (`and not settings.is_production`),
so it is not a production vulnerability.

The damage is subtler: with the bypass on, the QA suites reported

    FAIL F6 system/kill DENIED to subscriber — status=200
    FAIL N3 approve as non-admin DENIED (403) — status=200

and those look exactly like RBAC being broken. They were not: with the bypass off the
same suites returned 14/14 and 15/17, kill-switch 403 and non-admin approve 403.

The real hazard is the inverse case. Under the bypass those assertions **cannot fail for
the right reason** — a genuine authorization regression would be indistinguishable from
the bypass, so a green run would mean nothing. This test makes the compose default
explicit so nobody re-enables it and then trusts a security run.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILES = [
    ROOT / "docker-compose.compact.yml",
    ROOT / "docker-compose.yml",
]

PATTERN = re.compile(r"SIGNALBRIDGE_DEV_MODE\s*[:=]\s*[\"']?(\w+)[\"']?")


@pytest.mark.parametrize(
    "compose", [p for p in COMPOSE_FILES if p.is_file()], ids=lambda p: p.name
)
def test_dev_auth_bypass_is_off_by_default(compose: Path):
    values = PATTERN.findall(compose.read_text(encoding="utf-8", errors="replace"))
    enabled = [v for v in values if v.lower() == "true"]
    assert not enabled, (
        f"{compose.name} enables SIGNALBRIDGE_DEV_MODE. That makes every caller an admin, "
        "so the RBAC assertions in qa:functional / qa:registration pass regardless of the "
        "actual authorization code — a real regression would look identical to the bypass. "
        "Keep it false and opt in explicitly for local work."
    )


def test_bypass_still_refuses_to_run_in_production():
    """The production guard is the reason this is not a vulnerability — pin it."""
    src = (ROOT / "services" / "signalbridge_api" / "app" / "middleware" / "auth.py").read_text(
        encoding="utf-8", errors="replace"
    )
    assert "not settings.is_production" in src, (
        "the DEV_MODE flag lost its production guard — SIGNALBRIDGE_DEV_MODE would then "
        "be able to disable authentication on a live deployment"
    )


def test_devuser_admin_role_is_documented_as_deliberate():
    """DevUser is a full admin on purpose; make sure that stays visible to readers."""
    src = (ROOT / "services" / "signalbridge_api" / "app" / "middleware" / "auth.py").read_text(
        encoding="utf-8", errors="replace"
    )
    assert 'self.role = "admin"' in src
    assert "Dev bypass is a full admin" in src, (
        "the comment explaining why DevUser is admin was removed; without it the next "
        "reader cannot tell this is intentional rather than a privilege bug"
    )
