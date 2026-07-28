"""Fail-closed, non-clipping quality rules for canonical market data (BL-40)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.market.identity import ProviderSymbolRegistry


@dataclass(frozen=True, slots=True)
class QualityDecision:
    accepted: bool
    status: str
    rule_id: str | None = None
    observed_value: Any = None
    reason: str | None = None


class QualityRuleSet:
    """Evaluate broad economic ranges; never mutates or clips the source row."""

    VERSION = "1.0.0"

    def __init__(
        self,
        price_ranges: Mapping[str, tuple[Decimal, Decimal]] | None = None,
        *,
        identity_registry: ProviderSymbolRegistry | None = None,
        version: str | None = None,
    ) -> None:
        self.version = version or self.VERSION
        self._price_ranges: dict[str, tuple[Decimal, Decimal]] = {}
        for instrument_id, bounds in (price_ranges or {}).items():
            if (
                not isinstance(instrument_id, str)
                or len(bounds) != 2
                or any(not isinstance(value, Decimal) or not value.is_finite() for value in bounds)
                or bounds[0] <= 0
                or bounds[0] >= bounds[1]
            ):
                raise ValueError(f"invalid price range for {instrument_id!r}")
            self._price_ranges[instrument_id.strip().lower()] = bounds
        self.identity_registry = identity_registry

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        identity_registry: ProviderSymbolRegistry | None = None,
    ) -> "QualityRuleSet":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        version = raw.get("version")
        ranges = raw.get("price_ranges")
        if not isinstance(version, str) or not version or not isinstance(ranges, dict):
            raise ValueError("quality range config requires version and price_ranges")
        parsed: dict[str, tuple[Decimal, Decimal]] = {}
        for instrument_id, bounds in ranges.items():
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError(f"{instrument_id}: price range must contain [low, high]")
            parsed[instrument_id] = (Decimal(str(bounds[0])), Decimal(str(bounds[1])))
        return cls(parsed, identity_registry=identity_registry, version=version)

    def evaluate_provider_bar(
        self,
        provider_id: str,
        provider_symbol: str,
        row: Mapping[str, Any],
    ) -> QualityDecision:
        if self.identity_registry is None:
            return QualityDecision(
                False,
                "QUARANTINED",
                "bar.identity_registry",
                {"provider_id": provider_id, "provider_symbol": provider_symbol},
                "provider alias registry is required",
            )
        try:
            instrument_id = self.identity_registry.resolve(provider_id, provider_symbol)
        except ValueError as exc:
            return QualityDecision(
                False,
                "QUARANTINED",
                "bar.unknown_alias",
                {"provider_id": provider_id, "provider_symbol": provider_symbol},
                str(exc),
            )
        return self.evaluate_bar(instrument_id, row)

    def evaluate_bar(self, instrument_id: str, row: Mapping[str, Any]) -> QualityDecision:
        required = ("open", "high", "low", "close")
        missing = [name for name in required if row.get(name) is None]
        if missing:
            return QualityDecision(
                False, "QUARANTINED", "bar.required", missing, "required OHLC missing"
            )
        try:
            values = {name: Decimal(str(row[name])) for name in required}
        except Exception as exc:
            return QualityDecision(
                False, "QUARANTINED", "bar.numeric", dict(row), f"invalid numeric: {exc}"
            )
        if any(not value.is_finite() for value in values.values()):
            return QualityDecision(
                False,
                "QUARANTINED",
                "bar.numeric",
                {name: str(value) for name, value in values.items()},
                "OHLC values must be finite",
            )
        if any(value <= 0 for value in values.values()):
            return QualityDecision(
                False,
                "QUARANTINED",
                "bar.positive",
                {name: str(value) for name, value in values.items()},
                "OHLC values must be strictly positive",
            )
        if values["high"] < max(values.values()) or values["low"] > min(values.values()):
            return QualityDecision(
                False, "QUARANTINED", "bar.ohlc_order", values, "OHLC ordering invalid"
            )
        if not isinstance(instrument_id, str) or not instrument_id.strip():
            return QualityDecision(
                False, "QUARANTINED", "bar.instrument", instrument_id, "instrument_id required"
            )
        canonical_id = instrument_id.strip().lower()
        limits = self._price_ranges.get(canonical_id)
        if limits is None:
            return QualityDecision(
                False,
                "QUARANTINED",
                "bar.unknown_instrument",
                instrument_id,
                "no versioned price range declared for canonical instrument",
            )
        low, high = limits
        for field_name, value in values.items():
            if value < low or value > high:
                return QualityDecision(
                    False,
                    "QUARANTINED",
                    f"bar.range.{canonical_id}",
                    {field_name: str(value)},
                    f"value outside declared broad range [{low}, {high}]",
                )
        return QualityDecision(True, "VALID")

    @staticmethod
    def feature_status(
        *,
        feature_id: str,
        measured: bool,
        all_values_identical: bool = False,
        source_enabled: bool = True,
    ) -> QualityDecision:
        if not source_enabled:
            return QualityDecision(
                False, "UNAVAILABLE", "feature.source_disabled", feature_id, "source disabled"
            )
        if not measured:
            return QualityDecision(
                False, "UNAVAILABLE", "feature.not_measured", feature_id, "no measurement"
            )
        if all_values_identical:
            return QualityDecision(
                False,
                "UNAVAILABLE",
                "feature.constant_placeholder",
                feature_id,
                "constant placeholder is not a measurement",
            )
        return QualityDecision(True, "AVAILABLE")
