"""A risk-controlled book must disclose what makes it look better than it is.

Contract: CTR-PRODUCT-CLASS-001 (ADR-0020)

ADR-0020 creates a product class with a lower evidence bar than an alpha claim: a book that
says only "diversified, vol-targeted beta with a drawdown brake" does not need DSR > 0.95.
That is defensible — but it is also exactly the kind of concession that decays into "we
promoted beta and called it skill" unless the disclosures are mechanical.

So the ADR's prohibitions are enforced here rather than trusted:

1. An unconditional correlation may never be published without its co-active counterpart.
   Measured on this book: unconditional max |rho| 0.086, CO-ACTIVE max 0.196, and all three
   legs are simultaneously active only 12.9% of days. Most of the apparent diversification is
   non-overlapping presence, not offsetting risk. Publishing the 0.086 alone is the specific
   mistake this test exists to prevent — and it is a mistake I made before writing the ADR.

2. Cash accounting must be present and sourced. A book flat ~43% of the time crediting 0% on
   the idle balance understates itself, and the understatement points in the direction that
   looks conservative, which is how it survives review.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / ".claude" / "evidence" / "portfolio"
ADR = ROOT / ".claude" / "specs" / "adr" / "ADR-0020-risk-controlled-book.md"


def _latest_artifact() -> dict:
    if not EVIDENCE.is_dir():
        pytest.skip("no portfolio evidence yet")
    files = sorted(EVIDENCE.glob("*/portfolio_daily.json"))
    if not files:
        pytest.skip("no portfolio_daily.json yet")
    return json.loads(files[-1].read_text(encoding="utf-8"))


def test_adr_exists():
    assert ADR.is_file(), (
        "ADR-0020 defines the product class. Without it, a book has no promotion path and "
        "`promotion_eligible: false` is the only honest value."
    )


def test_correlation_is_reported_conditionally():
    art = _latest_artifact()
    corr = art.get("correlations")
    assert isinstance(corr, dict), (
        "artifact has no `correlations` block. A single max|rho| number is the misleading form."
    )
    for key in ("unconditional", "co_active", "fraction_days_all_legs_active"):
        assert key in corr, f"`correlations.{key}` missing — required by ADR-0020 prohibition 3"

    assert corr.get("max_abs_co_active") is not None, (
        "co-active correlation is null. If the legs never overlap enough to measure it, that "
        "IS the finding and must be stated, not omitted."
    )
    # The co-active number is the one that governs. If it is ever LOWER than unconditional,
    # something is wrong with the conditioning, not with the market.
    assert corr["max_abs_co_active"] >= corr["max_abs_unconditional"] - 1e-9, (
        f"co-active rho ({corr['max_abs_co_active']}) below unconditional "
        f"({corr['max_abs_unconditional']}) — conditioning on both legs being active should "
        "not reduce measured dependence; check the masks."
    )


def test_cash_accounting_is_present_and_sourced():
    art = _latest_artifact()
    cash = art.get("cash")
    assert isinstance(cash, dict), (
        "no `cash` block. A book that is flat ~43% of the time and credits 0% on the idle "
        "balance is mis-stating itself in the flattering-by-looking-conservative direction."
    )
    for key in ("rate_column", "source", "lag", "mean_gross_exposure"):
        assert cash.get(key), f"`cash.{key}` missing — the rate must be traceable and lagged"
    assert "shift(1)" in str(cash["lag"]), (
        "the cash rate must be lagged: a day's published rate is not knowable at that day's open"
    )
    assert art.get("portfolio_with_cash_yield"), (
        "both series are required — the no-cash one compares against historical backtests, the "
        "with-cash one is the economically honest number. Publishing only one invites picking."
    )


def test_book_is_not_promoted_without_forward_evidence():
    """The ADR creates a promotion PATH, not a promotion."""
    art = _latest_artifact()
    assert art.get("promotion_eligible") is False, (
        "the book is promotion_eligible without a signed withdrawal protocol and accumulated "
        "forward evidence. ADR-0020 criterion 7 is not satisfied by a good backtest."
    )
