"""Carry, defined consistently across asset classes.

This is the idea the engine is built on. Chart patterns do not transfer between
a 24h FX pair, a perpetual swap, an index future and a gold contract - but
*carry* does. Every one of them has a well-defined return-if-nothing-moves:

    FX             interest rate differential (equivalently, forward points)
    Crypto perp    funding rate paid between longs and shorts
    Equity index   dividend yield minus financing rate
    Commodity      roll yield from the futures curve shape

All four are annualised decimal returns to a LONG position, so they are
directly comparable and can be ranked in one cross-sectional book. That is what
makes a single engine over FX + crypto + equities + gold coherent rather than
four screeners glued together.

Sign convention throughout: positive carry means a long position earns if spot
is unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "CarryEstimate",
    "fx_carry",
    "fx_carry_from_forward",
    "crypto_perp_carry",
    "equity_index_carry",
    "commodity_roll_carry",
    "carry_to_zscore",
]


@dataclass(frozen=True)
class CarryEstimate:
    """Annualised carry to a long position, with provenance.

    `reliability` flags carry whose realised value routinely diverges from the
    quoted figure - EM currencies with managed regimes being the main case.
    Ranking EM carry alongside G10 carry without this flag is how carry books
    accumulate hidden tail risk.
    """

    symbol: str
    annualised: float
    source: str
    reliability: str = "normal"  # normal | suspect
    note: str = ""

    @property
    def pct(self) -> float:
        return self.annualised * 100.0

    def __str__(self) -> str:
        s = f"{self.symbol}: {self.pct:+.2f}%/yr carry ({self.source})"
        if self.reliability == "suspect":
            s += f" [SUSPECT: {self.note}]"
        return s


def fx_carry(
    symbol: str,
    rate_base: float,
    rate_quote: float,
    *,
    is_em: bool = False,
    annual_inflation_base: float | None = None,
) -> CarryEstimate:
    """Carry for a long BASE/QUOTE position, e.g. long EURUSD earns EUR, pays USD.

    Rates are annualised decimals (0.0525 for 5.25%). Use comparable tenors -
    mixing an overnight rate against a 1-year rate silently embeds a curve view.

    For EM, nominal carry is the number that lures people in and real carry is
    what they keep. When `annual_inflation_base` is supplied and the base
    currency's real rate is deeply negative, the estimate is marked suspect:
    that is the signature of a currency being held up by nominal rates while
    losing value, which is how carry trades in TRY-like regimes end.
    """
    carry = rate_base - rate_quote
    reliability, note = "normal", ""

    if is_em:
        if annual_inflation_base is not None:
            real = rate_base - annual_inflation_base
            if real < 0:
                reliability = "suspect"
                note = (
                    f"base real rate {real:+.2%} - nominal carry {carry:+.2%} is "
                    "compensation for expected depreciation, not free money"
                )
        elif carry > 0.08:
            reliability = "suspect"
            note = (
                f"carry {carry:+.2%} exceeds 8%/yr with no inflation input - "
                "supply annual_inflation_base to distinguish edge from crash risk"
            )

    return CarryEstimate(symbol, carry, "rate differential", reliability, note)


def fx_carry_from_forward(
    symbol: str, spot: float, forward: float, days: int
) -> CarryEstimate:
    """Carry implied by the forward points - what you can actually transact on.

    Preferred over rate differentials for EM: where capital controls or NDF
    markets bind, covered interest parity breaks and the forward is the honest
    price. The rate differential can show +12% while the tradeable forward
    shows +6%; the forward is right.
    """
    if spot <= 0 or forward <= 0:
        raise ValueError("spot and forward must be positive")
    if days <= 0:
        raise ValueError("days must be positive")
    carry = (spot / forward - 1.0) * (365.0 / days)
    return CarryEstimate(symbol, carry, f"{days}d forward points")


def crypto_perp_carry(
    symbol: str, funding_rate: float, intervals_per_day: int = 3
) -> CarryEstimate:
    """Annualised carry to a LONG perpetual swap.

    Funding is quoted per interval (8h on most venues, so 3/day). When funding
    is positive, longs pay shorts - so carry to a long is NEGATIVE. Persistent
    positive funding is crowded long positioning, and shorting it is a carry
    trade with the same crash profile as FX carry: it pays steadily and then
    gaps against you in a squeeze.
    """
    if intervals_per_day <= 0:
        raise ValueError("intervals_per_day must be positive")
    annualised = -funding_rate * intervals_per_day * 365.0
    reliability, note = "normal", ""
    if abs(annualised) > 0.50:
        reliability = "suspect"
        note = (
            f"|carry| {abs(annualised):.0%}/yr - extreme funding mean-reverts "
            "fast; do not extrapolate a year from one print"
        )
    return CarryEstimate(
        symbol, annualised, f"funding x{intervals_per_day}/day", reliability, note
    )


def equity_index_carry(
    symbol: str, dividend_yield: float, financing_rate: float
) -> CarryEstimate:
    """Carry to a long index future: dividends collected minus financing paid.

    Usually negative when short rates exceed the dividend yield - which is the
    normal state for the S&P 500 in a positive-rate regime, and a cost that
    equity-only tooling tends to ignore entirely.
    """
    return CarryEstimate(
        symbol, dividend_yield - financing_rate, "div yield - financing"
    )


def commodity_roll_carry(
    symbol: str, front_price: float, back_price: float, days_between: int
) -> CarryEstimate:
    """Roll yield from the futures curve, annualised, for a long position.

    Backwardation (front > back) pays a long to roll: positive carry, and the
    structural reason trend-following in commodities has historically worked.
    Contango (front < back) bleeds - the mechanism that destroyed retail long
    positions in oil ETFs.

    For gold specifically, expect persistent mild contango: gold is a
    store-of-value with ample above-ground supply, so it carries a financing
    cost rather than a convenience yield. Negative carry there is normal, not
    a signal.
    """
    if front_price <= 0 or back_price <= 0:
        raise ValueError("prices must be positive")
    if days_between <= 0:
        raise ValueError("days_between must be positive")
    carry = (front_price / back_price - 1.0) * (365.0 / days_between)
    return CarryEstimate(symbol, carry, f"roll over {days_between}d")


def carry_to_zscore(
    estimates: list[CarryEstimate], drop_suspect: bool = True
) -> dict[str, float]:
    """Cross-sectionally standardise carry so assets are rankable together.

    Raw carry is not comparable across classes - a 6%/yr FX carry and a 6%/yr
    roll yield carry wildly different risk. Z-scoring puts them on one scale
    for portfolio construction; the volatility scaling in `sizing` then handles
    the risk side.

    Suspect estimates are dropped by default. Including a 40%/yr TRY carry in
    the cross-section would dominate the z-score and drag the whole book toward
    the one position most likely to gap.
    """
    pool = [e for e in estimates if not (drop_suspect and e.reliability == "suspect")]
    if len(pool) < 2:
        raise ValueError(
            f"need >= 2 usable estimates to standardise, got {len(pool)}"
            + (" (suspect ones were dropped)" if drop_suspect else "")
        )
    vals = [e.annualised for e in pool]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return {e.symbol: 0.0 for e in pool}
    return {e.symbol: (e.annualised - mean) / sd for e in pool}
