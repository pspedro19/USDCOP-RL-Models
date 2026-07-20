"""Position-unit abstraction across asset classes.

The collection's existing `position-sizer` returns *shares*. Pointed at EURUSD
that answer is silently wrong: FX trades in units of base currency (100,000 per
standard lot), gold futures in 100oz contracts, crypto in coins. Getting a
number back that looks plausible but means nothing is worse than an error.

This module makes the unit explicit. Every sizing call returns a `Position`
that names its own unit, so a wrong unit is a visible wrong unit.

Risk arithmetic is uniform across classes:

    risk_per_unit(account_ccy) = |entry - stop| x point_value x fx_conversion
    quantity                   = risk_budget / risk_per_unit

What changes per class is `point_value` (what one unit of price movement is
worth) and whether quantity is continuous or must be floored to whole contracts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "AssetClass",
    "Unit",
    "Instrument",
    "Position",
    "REGISTRY",
    "get_instrument",
]


class AssetClass(str, Enum):
    FX_SPOT = "fx_spot"
    FX_EM = "fx_em"
    CRYPTO_SPOT = "crypto_spot"
    CRYPTO_PERP = "crypto_perp"
    EQUITY = "equity"
    EQUITY_INDEX_FUT = "equity_index_fut"
    COMMODITY_FUT = "commodity_fut"


class Unit(str, Enum):
    BASE_CCY_UNITS = "base_ccy_units"  # FX spot: 100,000 = 1 standard lot
    CONTRACTS = "contracts"            # futures: whole numbers only
    SHARES = "shares"
    COINS = "coins"


# Units that cannot be fractionally traded.
_DISCRETE = {Unit.CONTRACTS}


@dataclass(frozen=True)
class Instrument:
    """A tradeable instrument with the specs needed to size a position.

    point_value: account-currency value of a 1.00 move in quoted price, per
        unit. FX spot and equities/crypto are 1.0 (one unit moves one quote
        unit). Futures carry the contract multiplier.
    tick_size: minimum price increment. `pip_size` for FX is a display
        convention (0.0001, or 0.01 for JPY crosses) and is separate.
    """

    symbol: str
    asset_class: AssetClass
    quote_ccy: str
    unit: Unit
    point_value: float = 1.0
    tick_size: float = 0.01
    pip_size: float | None = None
    base_ccy: str | None = None
    description: str = ""
    # EM-specific flags. Default False keeps majors unaffected.
    nondeliverable: bool = False
    capital_controls: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.point_value <= 0:
            raise ValueError(f"{self.symbol}: point_value must be > 0")
        if self.tick_size <= 0:
            raise ValueError(f"{self.symbol}: tick_size must be > 0")

    @property
    def is_discrete(self) -> bool:
        return self.unit in _DISCRETE

    @property
    def is_em(self) -> bool:
        return self.asset_class is AssetClass.FX_EM

    def pip_value(self, quantity: float, fx_rate_to_account: float = 1.0) -> float:
        """Account-currency value of a one-pip move. FX only."""
        if self.pip_size is None:
            raise ValueError(f"{self.symbol}: pip_value is FX-only (no pip_size)")
        return self.pip_size * quantity * self.point_value * fx_rate_to_account

    def size_from_risk(
        self,
        entry: float,
        stop: float,
        risk_budget: float,
        fx_rate_to_account: float = 1.0,
        max_notional: float | None = None,
    ) -> "Position":
        """Size a position so that entry->stop loses exactly `risk_budget`.

        fx_rate_to_account converts one unit of `quote_ccy` into the account
        currency. A USD account trading EURUSD needs 1.0; trading USDJPY needs
        1/USDJPY. Getting this wrong scales the position, so it is explicit
        rather than inferred.
        """
        if risk_budget <= 0:
            raise ValueError("risk_budget must be > 0")
        if fx_rate_to_account <= 0:
            raise ValueError("fx_rate_to_account must be > 0")
        stop_distance = abs(entry - stop)
        if stop_distance == 0:
            raise ValueError("entry and stop are equal - stop distance is zero")
        if stop_distance < self.tick_size:
            raise ValueError(
                f"{self.symbol}: stop distance {stop_distance:g} is smaller than "
                f"one tick ({self.tick_size:g})"
            )

        risk_per_unit = stop_distance * self.point_value * fx_rate_to_account
        raw_qty = risk_budget / risk_per_unit

        qty = math.floor(raw_qty) if self.is_discrete else raw_qty
        capped_by = None

        if max_notional is not None:
            notional_per_unit = entry * self.point_value * fx_rate_to_account
            max_qty = max_notional / notional_per_unit
            if self.is_discrete:
                max_qty = math.floor(max_qty)
            if max_qty < qty:
                qty = max_qty
                capped_by = "max_notional"

        return Position(
            instrument=self,
            quantity=qty,
            raw_quantity=raw_qty,
            entry=entry,
            stop=stop,
            risk_budget=risk_budget,
            risk_at_stop=qty * risk_per_unit,
            notional=qty * entry * self.point_value * fx_rate_to_account,
            fx_rate_to_account=fx_rate_to_account,
            capped_by=capped_by,
        )


@dataclass(frozen=True)
class Position:
    """A sized position that knows its own unit.

    `risk_at_stop` is the realised risk after flooring/capping and is what you
    should check - it can be materially below `risk_budget` for futures (a
    single contract is indivisible) and is 0 when the budget cannot afford one.
    """

    instrument: Instrument
    quantity: float
    raw_quantity: float
    entry: float
    stop: float
    risk_budget: float
    risk_at_stop: float
    notional: float
    fx_rate_to_account: float = 1.0
    capped_by: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def unit(self) -> Unit:
        return self.instrument.unit

    @property
    def lots(self) -> float | None:
        """Standard lots, FX spot only."""
        if self.instrument.unit is not Unit.BASE_CCY_UNITS:
            return None
        return self.quantity / 100_000.0

    @property
    def is_tradeable(self) -> bool:
        return self.quantity > 0

    def describe(self) -> str:
        i = self.instrument
        if not self.is_tradeable:
            return (
                f"{i.symbol}: NOT TRADEABLE - risk budget {self.risk_budget:,.2f} "
                f"is below one {i.unit.value[:-1]} "
                f"(needs {self.risk_budget / self.raw_quantity:,.2f})"
            )
        head = f"{i.symbol}: {self.quantity:,.4f} {i.unit.value}"
        if self.lots is not None:
            head += f" ({self.lots:.3f} lots)"
        head += (
            f" | notional {self.notional:,.2f}"
            f" | risk at stop {self.risk_at_stop:,.2f}"
        )
        if self.capped_by:
            head += f" | CAPPED by {self.capped_by}"
        if i.is_em:
            head += " | EM: gap risk, widen slippage assumptions"
        if i.nondeliverable:
            head += " | NDF: cash-settled, no physical delivery"
        return head


def _fx(symbol: str, base: str, quote: str, *, em: bool = False, **kw) -> Instrument:
    jpy = quote == "JPY"
    return Instrument(
        symbol=symbol,
        asset_class=AssetClass.FX_EM if em else AssetClass.FX_SPOT,
        base_ccy=base,
        quote_ccy=quote,
        unit=Unit.BASE_CCY_UNITS,
        point_value=1.0,
        tick_size=0.001 if jpy else 0.00001,
        pip_size=0.01 if jpy else 0.0001,
        **kw,
    )


# Contract specs below are the standard published values. Verify against the
# exchange before trading - multipliers do get revised.
REGISTRY: dict[str, Instrument] = {
    # --- FX majors ---
    "EURUSD": _fx("EURUSD", "EUR", "USD", description="Euro / US Dollar"),
    "GBPUSD": _fx("GBPUSD", "GBP", "USD", description="Cable"),
    "USDJPY": _fx("USDJPY", "USD", "JPY", description="Dollar / Yen"),
    "USDCHF": _fx("USDCHF", "USD", "CHF", description="Swissie"),
    "AUDUSD": _fx("AUDUSD", "AUD", "USD", description="Aussie - China/commodity proxy"),
    "USDCAD": _fx("USDCAD", "USD", "CAD", description="Loonie - oil correlated"),
    "NZDUSD": _fx("NZDUSD", "NZD", "USD", description="Kiwi"),
    # --- FX emerging ---
    "USDMXN": _fx(
        "USDMXN", "USD", "MXN", em=True,
        description="Mexican peso - highest-liquidity EM cross, classic carry funder",
        notes="Deliverable, no capital controls. Most tradeable EM pair.",
    ),
    "USDBRL": _fx(
        "USDBRL", "USD", "BRL", em=True, nondeliverable=True,
        description="Brazilian real",
        notes="Offshore trades as NDF. BCB intervenes via FX swaps.",
    ),
    "USDZAR": _fx(
        "USDZAR", "USD", "ZAR", em=True,
        description="South African rand - high beta to global risk",
        notes="Deliverable but thin. Widest spreads of the liquid EM set.",
    ),
    "USDTRY": _fx(
        "USDTRY", "USD", "TRY", em=True, capital_controls=True,
        description="Turkish lira",
        notes="CARRY TRAP: nominal carry has repeatedly been erased by step "
              "devaluations. Policy risk dominates rate differential.",
    ),
    "USDINR": _fx(
        "USDINR", "USD", "INR", em=True, nondeliverable=True, capital_controls=True,
        description="Indian rupee",
        notes="Offshore NDF. RBI manages the rate; realised vol understates risk.",
    ),
    "USDCNH": _fx(
        "USDCNH", "USD", "CNH", em=True,
        description="Offshore Chinese yuan",
        notes="CNH (offshore) != CNY (onshore). PBoC fixes the onshore rate daily.",
    ),
    # --- Crypto ---
    "BTCUSD": Instrument(
        "BTCUSD", AssetClass.CRYPTO_SPOT, "USD", Unit.COINS,
        base_ccy="BTC", tick_size=0.01, description="Bitcoin spot",
    ),
    "ETHUSD": Instrument(
        "ETHUSD", AssetClass.CRYPTO_SPOT, "USD", Unit.COINS,
        base_ccy="ETH", tick_size=0.01, description="Ether spot",
    ),
    "BTCUSDT.P": Instrument(
        "BTCUSDT.P", AssetClass.CRYPTO_PERP, "USDT", Unit.COINS,
        base_ccy="BTC", tick_size=0.1,
        description="Bitcoin perpetual swap",
        notes="Funding settles every 8h on most venues - carry, not a fee.",
    ),
    # --- Equities / index futures ---
    "SPY": Instrument(
        "SPY", AssetClass.EQUITY, "USD", Unit.SHARES,
        tick_size=0.01, description="S&P 500 ETF",
    ),
    "ES": Instrument(
        "ES", AssetClass.EQUITY_INDEX_FUT, "USD", Unit.CONTRACTS,
        point_value=50.0, tick_size=0.25,
        description="E-mini S&P 500 - $50 x index",
    ),
    "MES": Instrument(
        "MES", AssetClass.EQUITY_INDEX_FUT, "USD", Unit.CONTRACTS,
        point_value=5.0, tick_size=0.25,
        description="Micro E-mini S&P 500 - $5 x index",
    ),
    # --- Commodities ---
    "GC": Instrument(
        "GC", AssetClass.COMMODITY_FUT, "USD", Unit.CONTRACTS,
        point_value=100.0, tick_size=0.10,
        description="COMEX Gold - 100 troy oz",
    ),
    "MGC": Instrument(
        "MGC", AssetClass.COMMODITY_FUT, "USD", Unit.CONTRACTS,
        point_value=10.0, tick_size=0.10,
        description="Micro Gold - 10 troy oz",
    ),
    "SI": Instrument(
        "SI", AssetClass.COMMODITY_FUT, "USD", Unit.CONTRACTS,
        point_value=5000.0, tick_size=0.005,
        description="COMEX Silver - 5,000 troy oz",
    ),
    "CL": Instrument(
        "CL", AssetClass.COMMODITY_FUT, "USD", Unit.CONTRACTS,
        point_value=1000.0, tick_size=0.01,
        description="WTI Crude - 1,000 barrels",
    ),
    "XAUUSD": Instrument(
        "XAUUSD", AssetClass.COMMODITY_FUT, "USD", Unit.COINS,
        point_value=1.0, tick_size=0.01,
        description="Spot gold (OTC), quoted per troy oz",
        notes="Unit is ounces, not contracts - spot gold is not exchange-listed.",
    ),
    # --- CME EM FX futures (the institutional route into EM) ---
    "6M": Instrument(
        "6M", AssetClass.FX_EM, "USD", Unit.CONTRACTS,
        base_ccy="MXN", point_value=500_000.0, tick_size=0.00001,
        description="CME Mexican Peso future - 500,000 MXN",
        notes="Quoted USD per MXN, inverse of the USDMXN spot convention.",
    ),
    "6L": Instrument(
        "6L", AssetClass.FX_EM, "USD", Unit.CONTRACTS,
        base_ccy="BRL", point_value=100_000.0, tick_size=0.00001,
        description="CME Brazilian Real future - 100,000 BRL",
        notes="Cash-settled. Quoted USD per BRL.",
    ),
}


def get_instrument(symbol: str) -> Instrument:
    """Look up an instrument, case-insensitively.

    Raises with the available symbols rather than returning a default - a
    silent fallback to equity conventions is exactly the bug this module exists
    to prevent.
    """
    key = symbol.upper().replace("/", "").replace("-", "")
    if key in REGISTRY:
        return REGISTRY[key]
    raise KeyError(
        f"Unknown instrument {symbol!r}. Known: {', '.join(sorted(REGISTRY))}. "
        "Add it to REGISTRY with verified specs rather than guessing a multiplier."
    )
