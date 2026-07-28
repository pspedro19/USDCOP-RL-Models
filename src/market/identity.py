"""Closed identities for provider symbols and bar intervals (BL-37)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable
import unicodedata


class IdentityError(ValueError):
    pass


class BarInterval(StrEnum):
    M1 = "PT1M"
    M5 = "PT5M"
    H1 = "PT1H"
    H4 = "PT4H"
    D1 = "P1D"
    W1 = "P1W"
    MO1 = "P1M"


_INTERVAL_ALIASES = {
    "1m": BarInterval.M1,
    "1min": BarInterval.M1,
    "m1": BarInterval.M1,
    "5m": BarInterval.M5,
    "5min": BarInterval.M5,
    "m5": BarInterval.M5,
    "1h": BarInterval.H1,
    "h1": BarInterval.H1,
    "4h": BarInterval.H4,
    "h4": BarInterval.H4,
    "1d": BarInterval.D1,
    "1day": BarInterval.D1,
    "d1": BarInterval.D1,
    "1w": BarInterval.W1,
    "w1": BarInterval.W1,
    "1mo": BarInterval.MO1,
    "1month": BarInterval.MO1,
}


def normalize_interval(value: str | BarInterval) -> BarInterval:
    if isinstance(value, BarInterval):
        return value
    if not isinstance(value, str):
        raise IdentityError("bar interval must be a string")
    try:
        return BarInterval(value)
    except ValueError:
        try:
            return _INTERVAL_ALIASES[value.strip().lower()]
        except KeyError as exc:
            raise IdentityError(f"unknown bar interval {value!r}") from exc


@dataclass(frozen=True, slots=True)
class ProviderSymbol:
    provider_id: str
    provider_symbol: str
    instrument_id: str

    def __post_init__(self) -> None:
        for field_name in ("provider_id", "provider_symbol", "instrument_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise IdentityError(f"{field_name} must be a non-empty string")
        object.__setattr__(
            self,
            "provider_id",
            unicodedata.normalize("NFC", self.provider_id.strip()).lower(),
        )
        object.__setattr__(
            self,
            "provider_symbol",
            unicodedata.normalize("NFC", self.provider_symbol.strip()),
        )
        object.__setattr__(
            self,
            "instrument_id",
            unicodedata.normalize("NFC", self.instrument_id.strip()).lower(),
        )


class ProviderSymbolRegistry:
    """Bijection guard for provider aliases.

    Provider symbols are compared case-insensitively after NFC/whitespace
    normalization.  An alias can resolve to exactly one canonical instrument.
    """

    def __init__(self, entries: Iterable[ProviderSymbol]) -> None:
        aliases: dict[tuple[str, str], str] = {}
        for entry in entries:
            if not isinstance(entry, ProviderSymbol):
                raise IdentityError("registry entries must be ProviderSymbol objects")
            key = (entry.provider_id, entry.provider_symbol.casefold())
            previous = aliases.get(key)
            if previous is not None and previous != entry.instrument_id:
                raise IdentityError(
                    f"provider alias {entry.provider_id}:{entry.provider_symbol!r} "
                    f"maps to both {previous!r} and {entry.instrument_id!r}"
                )
            aliases[key] = entry.instrument_id
        self._aliases = aliases

    def resolve(self, provider_id: str, provider_symbol: str) -> str:
        if not isinstance(provider_id, str) or not isinstance(provider_symbol, str):
            raise IdentityError("provider_id and provider_symbol must be strings")
        key = (
            unicodedata.normalize("NFC", provider_id.strip()).lower(),
            unicodedata.normalize("NFC", provider_symbol.strip()).casefold(),
        )
        try:
            return self._aliases[key]
        except KeyError as exc:
            raise IdentityError(
                f"unknown provider alias {provider_id}:{provider_symbol!r}"
            ) from exc
