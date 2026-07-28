"""BL-06 / FABRIC §25 — muralla frontend: forecasting no aprueba ni ejecuta.

Contract: the forecasting surface is DIAGNOSTIC (BL-13: `surface: action|diagnostic`) —
it exists to be looked at, never to place or approve orders. The as-built was verified
clean on 2026-07-27 (ForecastingView has no approve buttons and no execution endpoints;
its only link is /pricing), but "clean today" without a lock is a wall by convention:
one well-meaning PR wiring an approve button into the zoo would silently cross the
surface boundary that approval-gates.md reserves for /dashboard (Vote 2, admin-only).

Static, fail-closed per file: every forecasting component is scanned for order verbs
and approval/execution endpoints. Comments are NOT stripped on purpose — a commented-out
`fetch('/api/execution/...')` in this surface is already a design smell worth a red build.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "usdcop-trading-dashboard"

# The BL-06 scope: the GM forecasting view + everything under components/forecasting/.
FORECASTING_VIEW = DASH / "components" / "gm" / "views" / "ForecastingView.tsx"
FORECASTING_DIR = DASH / "components" / "forecasting"

# Forbidden in a diagnostic surface (pattern, why).
FORBIDDEN: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"/api/production/approve"),
     "Vote 2 endpoint — approval lives ONLY on /dashboard (approval-gates.md)"),
    (re.compile(r"/api/execution"),
     "execution endpoints — a diagnostic surface never reaches the OMS"),
    (re.compile(r"\bonApprove\b"),
     "approve handler — no approval affordances outside /dashboard"),
    (re.compile(r"\bCOMPRAR\b"),
     "order verb — forecasting shows forecasts, it does not solicit orders"),
    (re.compile(r"\bVENDER\b"),
     "order verb — forecasting shows forecasts, it does not solicit orders"),
]


def _scoped_files() -> list[Path]:
    files = [FORECASTING_VIEW] if FORECASTING_VIEW.is_file() else []
    if FORECASTING_DIR.is_dir():
        files += sorted(p for p in FORECASTING_DIR.rglob("*")
                        if p.suffix in {".ts", ".tsx"})
    return files


SCOPED = _scoped_files()


def test_scope_is_not_vacuous():
    """If the view moves/renames, this test must fail loudly — not pass over nothing."""
    assert FORECASTING_VIEW.is_file(), (
        f"{FORECASTING_VIEW} missing — if ForecastingView moved, update BL-06's lock "
        "in the same commit so the muralla follows the surface"
    )
    assert len(SCOPED) >= 2, (
        f"expected the forecasting surface to have several components, found {SCOPED}"
    )


@pytest.mark.parametrize("path", SCOPED, ids=[p.name for p in SCOPED])
def test_forecasting_surface_has_no_order_or_approval_verbs(path: Path):
    src = path.read_text(encoding="utf-8", errors="replace")
    hits: list[str] = []
    for pattern, why in FORBIDDEN:
        for m in pattern.finditer(src):
            line = src.count("\n", 0, m.start()) + 1
            hits.append(f"  line {line}: {m.group(0)!r} — {why}")
    assert not hits, (
        f"{path.relative_to(ROOT)}: forecasting is a DIAGNOSTIC surface (BL-06/BL-13) "
        "and must never approve or execute:\n" + "\n".join(hits)
    )
