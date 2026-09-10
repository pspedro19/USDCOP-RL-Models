"""Fail-closed, non-clipping quality rules for canonical market data (BL-40)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
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


@dataclass(frozen=True, slots=True)
class ScopedPriceRange:
    provider_id: str
    valid_from: datetime
    low: Decimal
    high: Decimal


class QualityRuleSet:
    """Evaluate broad economic ranges; never mutates or clips the source row."""

    VERSION = "1.0.0"

    def __init__(
        self,
        price_ranges: Mapping[str, tuple[Decimal, Decimal]] | None = None,
        *,
        identity_registry: ProviderSymbolRegistry | None = None,
        version: str | None = None,
        scoped_price_ranges: Mapping[str, tuple[ScopedPriceRange, ...]] | None = None,
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
        self._scoped_price_ranges: dict[str, tuple[ScopedPriceRange, ...]] = {}
        for instrument_id, rules in (scoped_price_ranges or {}).items():
            if not isinstance(instrument_id, str) or not instrument_id.strip() or not rules:
                raise ValueError(f"invalid scoped price ranges for {instrument_id!r}")
            regime_keys: set[tuple[str, datetime]] = set()
            for rule in rules:
                if not isinstance(rule, ScopedPriceRange):
                    raise ValueError(f"invalid scoped price range for {instrument_id!r}")
                regime_key = (rule.provider_id.strip().lower(), rule.valid_from)
                if regime_key in regime_keys:
                    raise ValueError(
                        f"duplicate scoped price range for {instrument_id!r}: "
                        f"{regime_key[0]} at {rule.valid_from.isoformat()}"
                    )
                regime_keys.add(regime_key)
            self._scoped_price_ranges[instrument_id.strip().lower()] = tuple(rules)
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
        scoped: dict[str, tuple[ScopedPriceRange, ...]] = {}
        for instrument_id, bounds in ranges.items():
            if isinstance(bounds, list) and len(bounds) == 2 and not any(
                isinstance(value, Mapping) for value in bounds
            ):
                parsed[instrument_id] = (Decimal(str(bounds[0])), Decimal(str(bounds[1])))
                continue
            if not isinstance(bounds, list) or not bounds:
                raise ValueError(
                    f"{instrument_id}: price range must contain [low, high] or scoped rules"
                )
            parsed_rules: list[ScopedPriceRange] = []
            for index, rule in enumerate(bounds):
                if not isinstance(rule, Mapping):
                    raise ValueError(f"{instrument_id}[{index}]: scoped rule must be a mapping")
                unknown = set(rule) - {"provider_id", "valid_from", "bounds"}
                if unknown:
                    raise ValueError(
                        f"{instrument_id}[{index}]: unknown fields {sorted(unknown)}"
                    )
                if set(rule) != {"provider_id", "valid_from", "bounds"}:
                    raise ValueError(
                        f"{instrument_id}[{index}]: provider_id, valid_from and bounds required"
                    )
                provider_id = rule["provider_id"]
                valid_from = rule["valid_from"]
                rule_bounds = rule["bounds"]
                if not isinstance(provider_id, str) or not provider_id.strip():
                    raise ValueError(f"{instrument_id}[{index}]: provider_id required")
                if not isinstance(valid_from, str):
                    raise ValueError(f"{instrument_id}[{index}]: valid_from must be ISO UTC")
                try:
                    parsed_from = datetime.fromisoformat(valid_from.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ValueError(
                        f"{instrument_id}[{index}]: valid_from must be ISO UTC"
                    ) from exc
                if parsed_from.tzinfo is None or parsed_from.utcoffset() != UTC.utcoffset(None):
                    raise ValueError(f"{instrument_id}[{index}]: valid_from must be UTC")
                if not isinstance(rule_bounds, list) or len(rule_bounds) != 2:
                    raise ValueError(f"{instrument_id}[{index}]: bounds must contain [low, high]")
                low, high = (Decimal(str(value)) for value in rule_bounds)
                if any(not value.is_finite() for value in (low, high)) or low <= 0 or low >= high:
                    raise ValueError(f"{instrument_id}[{index}]: invalid bounds")
                parsed_rules.append(
                    ScopedPriceRange(
                        provider_id=provider_id.strip().lower(),
                        valid_from=parsed_from.astimezone(UTC),
                        low=low,
                        high=high,
                    )
                )
            scoped[instrument_id] = tuple(parsed_rules)
        return cls(
            parsed,
            identity_registry=identity_registry,
            version=version,
            scoped_price_ranges=scoped,
        )

    def evaluate_provider_bar(
        self,
        provider_id: str,
        provider_symbol: str,
        row: Mapping[str, Any],
        *,
        observed_at: datetime | None = None,
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
        return self._evaluate_bar(
            instrument_id,
            row,
            provider_id=provider_id,
            observed_at=observed_at,
        )

    def evaluate_bar(self, instrument_id: str, row: Mapping[str, Any]) -> QualityDecision:
        return self._evaluate_bar(instrument_id, row, provider_id=None, observed_at=None)

    def _evaluate_bar(
        self,
        instrument_id: str,
        row: Mapping[str, Any],
        *,
        provider_id: str | None,
        observed_at: datetime | None,
    ) -> QualityDecision:
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
        scoped_rules = self._scoped_price_ranges.get(canonical_id)
        if scoped_rules is not None:
            if (
                not isinstance(provider_id, str)
                or not provider_id.strip()
                or not isinstance(observed_at, datetime)
                or observed_at.tzinfo is None
            ):
                return QualityDecision(
                    False,
                    "QUARANTINED",
                    "bar.range_context",
                    canonical_id,
                    "provider_id and timezone-aware observed_at required for scoped range",
                )
            normalized_provider = provider_id.strip().lower()
            instant = observed_at.astimezone(UTC)
            matches = [
                rule
                for rule in scoped_rules
                if rule.provider_id == normalized_provider and instant >= rule.valid_from
            ]
            if not matches:
                return QualityDecision(
                    False,
                    "QUARANTINED",
                    "bar.range_scope",
                    {
                        "instrument_id": canonical_id,
                        "provider_id": normalized_provider,
                        "observed_at": instant.isoformat(),
                    },
                    "a provider/time price range must apply",
                )
            active_rule = max(matches, key=lambda rule: rule.valid_from)
            limits = (active_rule.low, active_rule.high)
        else:
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
