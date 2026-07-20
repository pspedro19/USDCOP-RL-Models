"""Trend and value signals that transfer across asset classes.

Together with `carry`, these are the three return premia with cross-asset
evidence spanning decades and dozens of markets:

  Trend  Moskowitz, Ooi & Pedersen (2012), "Time Series Momentum" - 58 futures
         across equities, FX, commodities and rates
  Value  Asness, Moskowitz & Pedersen (2013), "Value and Momentum Everywhere"
  Carry  Koijen, Moskowitz, Pedersen & Vrugt (2018), "Carry"

They are chosen over chart patterns for one reason: they were tested on the
asset classes this engine covers, rather than on US equities and then assumed
to generalise. Value in particular has to be redefined per class - there is no
single formula - and this module is explicit about where that definition is
weak rather than inventing one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Signal",
    "tsmom",
    "blended_tsmom",
    "vol_scaled_signal",
    "fx_value_ppp",
    "commodity_value_reversal",
    "equity_value_earnings_yield",
    "combine_signals",
]


@dataclass(frozen=True)
class Signal:
    symbol: str
    value: float          # signed, nominally [-1, 1]
    kind: str             # trend | carry | value
    confidence: str = "normal"   # normal | weak
    note: str = ""

    def __str__(self) -> str:
        s = f"{self.symbol} {self.kind}: {self.value:+.3f}"
        if self.confidence == "weak":
            s += f" [WEAK: {self.note}]"
        return s


# --------------------------------------------------------------------------
# Trend
# --------------------------------------------------------------------------

def tsmom(prices, lookback: int = 252, skip_recent: int = 0) -> float:
    """Time-series momentum: the sign of the return over `lookback` periods.

    `skip_recent` omits the most recent observations. Standard for equities,
    where the last month tends to reverse; usually left at 0 for FX and
    commodities, where no comparable short-term reversal is documented.

    Returns +1, -1, or 0.
    """
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p)]
    need = lookback + skip_recent + 1
    if p.size < need:
        raise ValueError(f"need >= {need} prices for lookback={lookback}, got {p.size}")
    if np.any(p <= 0):
        raise ValueError("prices must be positive (use prices, not returns)")

    end = p.size - 1 - skip_recent
    start = end - lookback
    ret = p[end] / p[start] - 1.0
    return float(np.sign(ret))


def blended_tsmom(prices, lookbacks: tuple[int, ...] = (63, 126, 252)) -> Signal:
    """Average TSMOM across horizons - the practitioner default.

    A single lookback is a free parameter begging to be overfit, and its
    performance is unstable across regimes. Averaging 3-, 6- and 12-month
    signals removes the choice and is what most managed-futures programmes
    actually run. The output is continuous in [-1, 1]: partial agreement
    across horizons produces a smaller position, which is the intended
    behaviour rather than a rounding artefact.
    """
    p = np.asarray(prices, dtype=float)
    usable = [lb for lb in lookbacks if p.size >= lb + 1]
    if not usable:
        raise ValueError(
            f"need >= {min(lookbacks) + 1} prices for the shortest lookback, "
            f"got {p.size}"
        )
    val = float(np.mean([tsmom(p, lb) for lb in usable]))
    if len(usable) < len(lookbacks):
        return Signal(
            "", val, "trend", "weak",
            f"only {len(usable)}/{len(lookbacks)} lookbacks had enough history",
        )
    return Signal("", val, "trend")


def vol_scaled_signal(signal_value: float, vol: float, target_vol: float = 0.10) -> float:
    """Scale a raw signal by target/realised vol - the TSMOM position rule.

    Without this, a +1 trend signal on bitcoin and a +1 on EURUSD produce the
    same notional and wildly different risk.
    """
    if vol <= 0:
        raise ValueError("vol must be > 0")
    return float(signal_value * target_vol / vol)


# --------------------------------------------------------------------------
# Value - redefined per asset class
# --------------------------------------------------------------------------

def fx_value_ppp(
    symbol: str, spot: float, ppp_rate: float, is_em: bool = False
) -> Signal:
    """FX value from deviation against purchasing power parity.

    Signal is positive when the base currency is CHEAP versus PPP. PPP is a
    5-10 year anchor with no timing content whatsoever - a currency can sit 30%
    from fair value for years - so this belongs in a slow, diversified book and
    nowhere near a stop-loss.

    For EM the estimate is marked weak: the Balassa-Samuelson effect means
    faster-growing economies sustain genuinely stronger real exchange rates, so
    a persistent PPP "overvaluation" in EM is often equilibrium rather than
    mispricing.
    """
    if spot <= 0 or ppp_rate <= 0:
        raise ValueError("spot and ppp_rate must be positive")
    deviation = math.log(ppp_rate / spot)
    val = float(np.clip(deviation, -1.0, 1.0))
    if is_em:
        return Signal(
            symbol, val, "value", "weak",
            "Balassa-Samuelson: EM real appreciation can be equilibrium, not mispricing",
        )
    return Signal(symbol, val, "value")


def commodity_value_reversal(symbol: str, prices, lookback_years: int = 5) -> Signal:
    """Commodity value: the 5-year real-price reversal of Asness et al.

    Commodities have no cash flows, so value is defined as the deviation of the
    current price from its own long-run level. Cheap versus five years ago is
    the value signal.
    """
    p = np.asarray(prices, dtype=float)
    p = p[np.isfinite(p)]
    need = int(lookback_years * 252)
    if p.size < need:
        raise ValueError(f"need >= {need} observations ({lookback_years}y), got {p.size}")
    if np.any(p <= 0):
        raise ValueError("prices must be positive")

    log_p = np.log(p[-need:])
    z = (log_p[-1] - log_p.mean()) / log_p.std(ddof=1)
    return Signal(symbol, float(np.clip(-z / 2.0, -1.0, 1.0)), "value")


def equity_value_earnings_yield(
    symbol: str, earnings_yield: float, real_rate: float
) -> Signal:
    """Equity value as the equity risk premium: earnings yield minus the real rate.

    Raw earnings yield is not a valuation signal across regimes - a 4% yield is
    expensive at a 3% real rate and cheap at a -1% real rate. Comparing against
    the real rate is what makes the signal usable at all, and what makes it
    comparable to carry on other assets.
    """
    erp = earnings_yield - real_rate
    return Signal(symbol, float(np.clip(erp / 0.06, -1.0, 1.0)), "value")


# --------------------------------------------------------------------------
# Combination
# --------------------------------------------------------------------------

def combine_signals(
    signals: list[Signal],
    weights: dict[str, float] | None = None,
    weak_penalty: float = 0.5,
) -> dict[str, float]:
    """Combine trend/carry/value into one conviction per symbol.

    Defaults to equal weight across the three premia. Equal weighting is not a
    lack of ambition - the correlations and relative Sharpes between these
    premia are unstable enough that optimised weights are usually fitted to the
    sample. Signals flagged weak are down-weighted rather than dropped.

    Returns symbol -> conviction in [-1, 1].
    """
    if not signals:
        raise ValueError("no signals to combine")
    w = weights or {"trend": 1 / 3, "carry": 1 / 3, "value": 1 / 3}

    unknown = {s.kind for s in signals} - set(w)
    if unknown:
        raise ValueError(f"no weight given for signal kind(s): {sorted(unknown)}")

    acc: dict[str, list[tuple[float, float]]] = {}
    for s in signals:
        if not s.symbol:
            raise ValueError(f"signal of kind {s.kind!r} has no symbol set")
        mult = weak_penalty if s.confidence == "weak" else 1.0
        acc.setdefault(s.symbol, []).append((s.value * w[s.kind] * mult, w[s.kind] * mult))

    out = {}
    for sym, parts in acc.items():
        total_w = sum(p[1] for p in parts)
        out[sym] = float(np.clip(sum(p[0] for p in parts) / total_w, -1.0, 1.0)) if total_w else 0.0
    return out
