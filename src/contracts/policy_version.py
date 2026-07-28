"""
Policy backend contract — control.policy_version + action.strategy_signal
=========================================================================
BL-46 R4. Normalized, engine-agnostic records for the two common tables of
spec §8. A rule-based strategy adds NO tables and NO columns: everything a
new rule needs to explain itself rides inside ``decision_components``
(versioned by ``decision_schema_version``) and the ``rule_trace_uri``.

Also holds the two NON-ECONOMIC schemas the schema-driven frontend (R5)
renders from:

- ``PresentationSpec`` (spec §9.1) — labels/format only, with its OWN
  ``presentation_hash``. Renaming a label must NEVER change ``policy_hash``:
  presentation is not an economic version of the strategy.
- ``ConfigFieldSpec`` / ``ConfigSchema`` — the per-deployment/per-user
  tuneables (``sb_trading_configs``) rendered from a schema instead of fixed
  form fields, and validated identically on both sides.

Every validator here is the Python HALF of a bilateral pair: the TypeScript
mirror is
``usdcop-trading-dashboard/lib/contracts/policy-version.contract.ts`` and it
must ACCEPT and REJECT exactly the same payloads (the shared fixture
``tests/fixtures/policy_backend_cases.v1.json`` is executed by both runners).

Spec:  .claude/specs/planes/05-rule-based-strategies.md §8, §9, §9.1
Rule:  .claude/rules/strategy-engines.md (invariants 1, 5, 7, 9)
Contract: CTR-POLICY-BACKEND-001 (BL-46 R4/R5)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.contracts.policy import (
    ENGINE_TYPES,
    VALID_DIRECTIONS,
    ISO_TIMESTAMP_PATTERN,
    StrategyDecision,
    require_hash,
    require_id,
    require_iso_timestamp,
)
from src.contracts.rule_trace import ensure_json_safe

# ---------------------------------------------------------------------------
# Whitelists (mirrored verbatim in policy-version.contract.ts)
# ---------------------------------------------------------------------------

#: How the policy logic is implemented (spec §3.2). Exactly two modes.
IMPLEMENTATION_MODES = ("coded_policy", "declarative")

#: Versioned schema of the ``decision_components`` JSONB payload — the reason
#: a new rule never adds a column (spec §8).
DECISION_SCHEMA_VERSIONS = ("decision_components_v1",)

#: Schema tag of a control.policy_version row.
POLICY_VERSION_SCHEMAS = ("policy_version_v1",)

#: Schema tag of an action.strategy_signal row.
STRATEGY_SIGNAL_SCHEMAS = ("strategy_signal_v1",)

#: Display formats a presentation component may declare (render-only).
PRESENTATION_FORMATS = ("price", "percent", "decimal", "integer", "text")

#: Field kinds of a schema-driven config form.
CONFIG_FIELD_TYPES = ("number", "boolean", "enum")

#: URI schemes accepted for manifests/traces. Relative paths are REJECTED on
#: both sides: a bare ``manifests/x.json`` is ambiguous and a traversal vector.
URI_SCHEMES = ("s3", "https", "file")

#: ``<scheme>://<rest>`` with a character set that excludes whitespace and
#: control characters (so a trailing '\n' can never sneak in). Mirrored as
#: URI_PATTERN in the TS contract.
URI_PATTERN = re.compile(
    r"^(?:s3|https|file)://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+$"
)

#: Max URI length accepted on both sides (defensive, symmetric).
URI_MAX_LENGTH = 512

#: ``package.module:ClassName`` — the only accepted coded_policy reference.
MODULE_REF_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*$"
)

#: Snake-case key of a presentation component / config field.
COMPONENT_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

#: Bounds so labels/descriptions cannot be used as unbounded payloads.
LABEL_MAX_LENGTH = 80
DESCRIPTION_MAX_LENGTH = 400


# ---------------------------------------------------------------------------
# Strict scalar helpers (each one has a TS twin with identical verdicts)
# ---------------------------------------------------------------------------

def require_uri(field_name: str, value: Any) -> str:
    """Absolute URI with a whitelisted scheme; anything else => ValueError."""
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a URI string with scheme in {URI_SCHEMES}, "
            f"got {value!r}"
        )
    if len(value) > URI_MAX_LENGTH:
        raise ValueError(
            f"{field_name} exceeds {URI_MAX_LENGTH} characters ({len(value)})"
        )
    if not URI_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must match {URI_PATTERN.pattern!r} "
            f"(absolute URI, schemes {URI_SCHEMES}, no whitespace), got {value!r}"
        )
    return value


def require_text(field_name: str, value: Any, max_length: int) -> str:
    """Non-empty single-line string within a length bound. bool => error."""
    if isinstance(value, bool) or not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
    if len(value) > max_length:
        raise ValueError(
            f"{field_name} exceeds {max_length} characters ({len(value)})"
        )
    if "\n" in value or "\r" in value:
        raise ValueError(f"{field_name} must be single-line, got {value!r}")
    return value


def require_finite_number(field_name: str, value: Any) -> float:
    """Real finite number — bool and numeric strings are typed errors."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be a real number (bool/string coercion "
            f"forbidden), got {value!r}"
        )
    if not math.isfinite(value):
        raise ValueError(
            f"{field_name} must be finite (NaN/Infinity forbidden), got {value!r}"
        )
    return float(value)


def require_instant(field_name: str, value: Any) -> str:
    """
    A point in TIME, not a date: full ISO-8601 datetime WITH an explicit UTC
    offset (``Z`` or ``±HH:MM``). ``valid_from``/``valid_until``/``frozen_at``/
    ``created_at`` are instants — a date-only or offset-less value is
    ambiguous and cannot be ordered, so both sides reject it.
    """
    require_iso_timestamp(field_name, value)          # real calendar + clock
    m = ISO_TIMESTAMP_PATTERN.fullmatch(value)
    if m is None:  # pragma: no cover — require_iso_timestamp already raised
        raise ValueError(f"{field_name} is not an ISO-8601 timestamp: {value!r}")
    if m.group(4) is None:
        raise ValueError(
            f"{field_name} must be a full datetime (date-only is not an "
            f"instant), got {value!r}"
        )
    if m.group(7) is None:
        raise ValueError(
            f"{field_name} must carry an explicit UTC offset (Z or ±HH:MM) — "
            f"an offset-less timestamp cannot be ordered, got {value!r}"
        )
    return value


def _days_from_civil(year: int, month: int, day: int) -> int:
    """
    Howard Hinnant's days-from-civil. Implemented with the SAME integer
    arithmetic in the TS mirror so instant ordering can never diverge
    (``Date.parse`` is deliberately not used on either side).
    """
    y = year - (1 if month <= 2 else 0)
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (month + (-3 if month > 2 else 9)) + 2) // 5 + day - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def instant_epoch_seconds(value: str) -> int:
    """Absolute seconds since the epoch for a validated instant string."""
    m = ISO_TIMESTAMP_PATTERN.fullmatch(value)
    if m is None or m.group(4) is None or m.group(7) is None:
        raise ValueError(f"not a comparable instant: {value!r}")
    days = _days_from_civil(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    seconds = (
        days * 86400
        + int(m.group(4)) * 3600
        + int(m.group(5)) * 60
        + (int(m.group(6)) if m.group(6) is not None else 0)
    )
    offset = m.group(7)
    if offset != "Z":
        offset_seconds = int(offset[1:3]) * 3600 + int(offset[4:6]) * 60
        seconds += -offset_seconds if offset[0] == "+" else offset_seconds
    return seconds


def _canonical_hash(payload: Any) -> str:
    """sha256 over canonical JSON — strict (no NaN, no ``default=`` fallback)."""
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# control.policy_version
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolicyVersionRecord:
    """
    One immutable row of ``control.policy_version`` (spec §8).

    Immutability is the point: changing a window, a threshold or a rule is a
    NEW row with a new ``policy_hash`` (CI §11), and that hash travels into
    every ``action.strategy_signal`` for exact attribution.
    """

    policy_version_id: str
    sleeve_id: str
    strategy_version: str
    engine_type: str
    implementation_mode: str
    params_hash: str
    policy_hash: str
    feature_set_hash: str
    resample_policy_hash: str
    frozen_at: str
    manifest_uri: str
    module_reference: str | None = None
    code_hash: str | None = None
    schema_version: str = POLICY_VERSION_SCHEMAS[0]

    def __post_init__(self) -> None:
        require_id("policy_version_id", self.policy_version_id)
        require_id("sleeve_id", self.sleeve_id)
        require_id("strategy_version", self.strategy_version)
        if self.engine_type not in ENGINE_TYPES:
            raise ValueError(
                f"engine_type must be one of {ENGINE_TYPES}, got {self.engine_type!r}"
            )
        if self.implementation_mode not in IMPLEMENTATION_MODES:
            raise ValueError(
                f"implementation_mode must be one of {IMPLEMENTATION_MODES}, "
                f"got {self.implementation_mode!r}"
            )
        if self.schema_version not in POLICY_VERSION_SCHEMAS:
            raise ValueError(
                f"schema_version must be one of {POLICY_VERSION_SCHEMAS}, "
                f"got {self.schema_version!r}"
            )
        for name in (
            "params_hash",
            "policy_hash",
            "feature_set_hash",
            "resample_policy_hash",
        ):
            require_hash(name, getattr(self, name))
        require_instant("frozen_at", self.frozen_at)
        require_uri("manifest_uri", self.manifest_uri)

        # coded_policy <=> module_reference. A declarative policy that points
        # at a module would be a hidden code path (invariant 6).
        if self.implementation_mode == "coded_policy":
            if self.module_reference is None:
                raise ValueError(
                    "coded_policy requires module_reference "
                    "('package.module:ClassName')"
                )
            if isinstance(self.module_reference, bool) or not isinstance(
                self.module_reference, str
            ) or not MODULE_REF_PATTERN.fullmatch(self.module_reference):
                raise ValueError(
                    "module_reference must match "
                    f"{MODULE_REF_PATTERN.pattern!r}, got {self.module_reference!r}"
                )
            require_hash("code_hash", self.code_hash)
        else:  # declarative
            if self.module_reference is not None:
                raise ValueError(
                    "declarative policies must NOT declare a module_reference "
                    "— the spec IS the logic (no arbitrary code from YAML)"
                )
            if self.code_hash is not None:
                require_hash("code_hash", self.code_hash)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version_id": self.policy_version_id,
            "sleeve_id": self.sleeve_id,
            "strategy_version": self.strategy_version,
            "engine_type": self.engine_type,
            "implementation_mode": self.implementation_mode,
            "module_reference": self.module_reference,
            "code_hash": self.code_hash,
            "params_hash": self.params_hash,
            "policy_hash": self.policy_hash,
            "feature_set_hash": self.feature_set_hash,
            "resample_policy_hash": self.resample_policy_hash,
            "frozen_at": self.frozen_at,
            "manifest_uri": self.manifest_uri,
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        payload = self.to_dict()
        ensure_json_safe(payload, "policy_version")
        return json.dumps(payload, allow_nan=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PolicyVersionRecord":
        if not isinstance(payload, Mapping):
            raise ValueError(f"policy_version payload must be a mapping, got {payload!r}")
        required = (
            "policy_version_id", "sleeve_id", "strategy_version", "engine_type",
            "implementation_mode", "params_hash", "policy_hash", "feature_set_hash",
            "resample_policy_hash", "frozen_at", "manifest_uri",
        )
        missing = [k for k in required if k not in payload]
        if missing:
            raise ValueError(f"policy_version payload missing fields: {missing}")
        unknown = set(payload) - set(required) - {
            "module_reference", "code_hash", "schema_version",
        }
        if unknown:
            raise ValueError(
                f"policy_version payload has unknown fields: {sorted(unknown)} "
                "(closed schema — never silently ignored)"
            )
        return cls(
            policy_version_id=payload["policy_version_id"],
            sleeve_id=payload["sleeve_id"],
            strategy_version=payload["strategy_version"],
            engine_type=payload["engine_type"],
            implementation_mode=payload["implementation_mode"],
            params_hash=payload["params_hash"],
            policy_hash=payload["policy_hash"],
            feature_set_hash=payload["feature_set_hash"],
            resample_policy_hash=payload["resample_policy_hash"],
            frozen_at=payload["frozen_at"],
            manifest_uri=payload["manifest_uri"],
            module_reference=payload.get("module_reference"),
            code_hash=payload.get("code_hash"),
            schema_version=payload.get("schema_version", POLICY_VERSION_SCHEMAS[0]),
        )


# ---------------------------------------------------------------------------
# action.strategy_signal
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategySignalRecord:
    """
    One row of ``action.strategy_signal`` (spec §8) — the SAME shape for every
    engine. Built from a ``StrategyDecision`` via :meth:`from_decision`, which
    is the only production path: backtest, paper and live publish through the
    same evaluation library, so the three cannot diverge (invariant 4).
    """

    signal_id: str
    sleeve_id: str
    strategy_version: str
    policy_version_id: str
    instrument_id: str
    as_of: str
    valid_from: str
    valid_until: str
    direction: str
    target_exposure: float
    decision_fingerprint: str
    created_at: str
    feature_snapshot_id: str | None = None
    reason_codes: tuple[str, ...] = ()
    decision_components: dict[str, Any] = field(default_factory=dict)
    decision_schema_version: str = DECISION_SCHEMA_VERSIONS[0]
    rule_trace_uri: str | None = None
    schema_version: str = STRATEGY_SIGNAL_SCHEMAS[0]

    def __post_init__(self) -> None:
        require_id("sleeve_id", self.sleeve_id)
        require_id("strategy_version", self.strategy_version)
        require_id("policy_version_id", self.policy_version_id)
        require_id("instrument_id", self.instrument_id)
        require_iso_timestamp("as_of", self.as_of)
        require_instant("valid_from", self.valid_from)
        require_instant("valid_until", self.valid_until)
        require_instant("created_at", self.created_at)
        if instant_epoch_seconds(self.valid_from) > instant_epoch_seconds(
            self.valid_until
        ):
            raise ValueError(
                f"valid_from ({self.valid_from}) must not be after valid_until "
                f"({self.valid_until}) — an inverted validity window is never valid"
            )
        if self.direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {VALID_DIRECTIONS}, got {self.direction!r}"
            )
        require_finite_number("target_exposure", self.target_exposure)
        require_hash("decision_fingerprint", self.decision_fingerprint)
        derived_signal_id = (
            f"{self.sleeve_id}:{self.as_of}:"
            f"{self.decision_fingerprint.removeprefix('sha256:')[:16]}"
        )
        if not isinstance(self.signal_id, str) or self.signal_id != derived_signal_id:
            raise ValueError(
                "signal_id must equal its derivation "
                f"'<sleeve_id>:<as_of>:<fingerprint-hex16>' = {derived_signal_id!r}, "
                f"got {self.signal_id!r}"
            )
        if self.feature_snapshot_id is not None:
            require_id("feature_snapshot_id", self.feature_snapshot_id)
        if not isinstance(self.reason_codes, (tuple, list)) or not all(
            isinstance(c, str) for c in self.reason_codes
        ):
            raise ValueError(
                f"reason_codes must be a sequence of strings, got {self.reason_codes!r}"
            )
        if not isinstance(self.decision_components, Mapping):
            raise ValueError(
                f"decision_components must be a mapping, got {self.decision_components!r}"
            )
        ensure_json_safe(self.decision_components, "decision_components")
        if self.decision_schema_version not in DECISION_SCHEMA_VERSIONS:
            raise ValueError(
                f"decision_schema_version must be one of {DECISION_SCHEMA_VERSIONS}, "
                f"got {self.decision_schema_version!r}"
            )
        if self.schema_version not in STRATEGY_SIGNAL_SCHEMAS:
            raise ValueError(
                f"schema_version must be one of {STRATEGY_SIGNAL_SCHEMAS}, "
                f"got {self.schema_version!r}"
            )
        if self.rule_trace_uri is not None:
            require_uri("rule_trace_uri", self.rule_trace_uri)

    @classmethod
    def from_decision(
        cls,
        decision: StrategyDecision,
        *,
        policy_version_id: str,
        instrument_id: str,
        valid_from: str,
        valid_until: str,
        created_at: str,
        rule_trace_uri: str | None = None,
        decision_schema_version: str = DECISION_SCHEMA_VERSIONS[0],
    ) -> "StrategySignalRecord":
        """Normalize a decision (ANY engine) into the common signal row."""
        if not isinstance(decision, StrategyDecision):
            raise ValueError(
                f"from_decision requires a StrategyDecision, got "
                f"{type(decision).__name__}"
            )
        return cls(
            signal_id=decision.signal_id,
            sleeve_id=decision.sleeve_id,
            strategy_version=decision.strategy_version,
            policy_version_id=policy_version_id,
            instrument_id=instrument_id,
            as_of=decision.as_of,
            valid_from=valid_from,
            valid_until=valid_until,
            direction=decision.direction,
            target_exposure=float(decision.target_exposure),
            decision_fingerprint=decision.decision_fingerprint,
            created_at=created_at,
            feature_snapshot_id=decision.feature_snapshot_id,
            reason_codes=tuple(decision.reason_codes),
            decision_components=dict(decision.decision_components),
            decision_schema_version=decision_schema_version,
            rule_trace_uri=rule_trace_uri,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "sleeve_id": self.sleeve_id,
            "strategy_version": self.strategy_version,
            "policy_version_id": self.policy_version_id,
            "instrument_id": self.instrument_id,
            "as_of": self.as_of,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "direction": self.direction,
            "target_exposure": self.target_exposure,
            "feature_snapshot_id": self.feature_snapshot_id,
            "decision_fingerprint": self.decision_fingerprint,
            "reason_codes": list(self.reason_codes),
            "decision_components": dict(self.decision_components),
            "decision_schema_version": self.decision_schema_version,
            "rule_trace_uri": self.rule_trace_uri,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        payload = self.to_dict()
        ensure_json_safe(payload, "strategy_signal")
        return json.dumps(payload, allow_nan=False)


# ---------------------------------------------------------------------------
# Presentation metadata (§9.1) — NON-executable, own hash
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PresentationComponent:
    key: str
    label: str
    format: str

    def __post_init__(self) -> None:
        if isinstance(self.key, bool) or not isinstance(self.key, str) or not (
            COMPONENT_KEY_PATTERN.fullmatch(self.key)
        ):
            raise ValueError(
                f"presentation component key must match "
                f"{COMPONENT_KEY_PATTERN.pattern!r}, got {self.key!r}"
            )
        require_text(f"presentation component {self.key} label", self.label,
                     LABEL_MAX_LENGTH)
        if self.format not in PRESENTATION_FORMATS:
            raise ValueError(
                f"presentation component {self.key} format must be one of "
                f"{PRESENTATION_FORMATS}, got {self.format!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "label": self.label, "format": self.format}


@dataclass(frozen=True)
class PresentationSpec:
    """
    Render-only metadata (spec §9.1). It does not decide, is never executed,
    and **never** feeds ``policy_hash``: changing a label produces a new
    ``presentation_hash`` and the SAME economic version.
    """

    engine_label: str
    description: str
    components: tuple[PresentationComponent, ...] = ()

    def __post_init__(self) -> None:
        require_text("presentation.engine_label", self.engine_label, LABEL_MAX_LENGTH)
        require_text("presentation.description", self.description,
                     DESCRIPTION_MAX_LENGTH)
        if not isinstance(self.components, (tuple, list)) or not all(
            isinstance(c, PresentationComponent) for c in self.components
        ):
            raise ValueError("presentation.components must be PresentationComponent[]")
        keys = [c.key for c in self.components]
        if len(set(keys)) != len(keys):
            raise ValueError(f"presentation.components keys must be unique, got {keys}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_label": self.engine_label,
            "description": self.description,
            "components": [c.to_dict() for c in self.components],
        }

    def presentation_hash(self) -> str:
        """Hash of the PRESENTATION only — orthogonal to ``policy_hash``."""
        return _canonical_hash(self.to_dict())


# ---------------------------------------------------------------------------
# Schema-driven config (sb_trading_configs rendered from a schema, §9)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigFieldSpec:
    """One tuneable parameter, rendered generically by the frontend."""

    key: str
    label: str
    type: str
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    options: tuple[str, ...] = ()
    unit: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.key, bool) or not isinstance(self.key, str) or not (
            COMPONENT_KEY_PATTERN.fullmatch(self.key)
        ):
            raise ValueError(
                f"config field key must match {COMPONENT_KEY_PATTERN.pattern!r}, "
                f"got {self.key!r}"
            )
        require_text(f"config field {self.key} label", self.label, LABEL_MAX_LENGTH)
        if self.type not in CONFIG_FIELD_TYPES:
            raise ValueError(
                f"config field {self.key} type must be one of {CONFIG_FIELD_TYPES}, "
                f"got {self.type!r}"
            )
        if self.type == "number":
            for name in ("minimum", "maximum", "step"):
                value = getattr(self, name)
                if value is not None:
                    require_finite_number(f"config field {self.key} {name}", value)
            if (
                self.minimum is not None
                and self.maximum is not None
                and self.minimum > self.maximum
            ):
                raise ValueError(
                    f"config field {self.key}: minimum > maximum "
                    f"({self.minimum} > {self.maximum})"
                )
            if self.step is not None and self.step <= 0:
                raise ValueError(f"config field {self.key}: step must be > 0")
            if self.options:
                raise ValueError(
                    f"config field {self.key}: options are only valid for type 'enum'"
                )
        elif self.type == "boolean":
            if any(
                v is not None for v in (self.minimum, self.maximum, self.step)
            ) or self.options:
                raise ValueError(
                    f"config field {self.key}: boolean fields take no "
                    "minimum/maximum/step/options"
                )
        else:  # enum
            if not isinstance(self.options, (tuple, list)) or not self.options:
                raise ValueError(
                    f"config field {self.key}: enum requires a non-empty options list"
                )
            if not all(
                isinstance(o, str) and o and not isinstance(o, bool)
                for o in self.options
            ):
                raise ValueError(
                    f"config field {self.key}: enum options must be non-empty strings"
                )
            if len(set(self.options)) != len(self.options):
                raise ValueError(
                    f"config field {self.key}: enum options must be unique"
                )
            if any(v is not None for v in (self.minimum, self.maximum, self.step)):
                raise ValueError(
                    f"config field {self.key}: enum fields take no minimum/maximum/step"
                )
        if self.unit is not None:
            require_text(f"config field {self.key} unit", self.unit, 16)
        if self.default is not None and self.validate_value(self.default):
            raise ValueError(
                f"config field {self.key}: default {self.default!r} violates its own "
                f"schema ({'; '.join(self.validate_value(self.default))})"
            )

    def validate_value(self, value: Any) -> list[str]:
        """Errors for a candidate value; empty list = valid (TS twin exists)."""
        if self.type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return [f"{self.key} must be a number"]
            if not math.isfinite(value):
                return [f"{self.key} must be finite (NaN/Infinity forbidden)"]
            errors = []
            if self.minimum is not None and value < self.minimum:
                errors.append(f"{self.key} must be >= {self.minimum}")
            if self.maximum is not None and value > self.maximum:
                errors.append(f"{self.key} must be <= {self.maximum}")
            return errors
        if self.type == "boolean":
            return [] if isinstance(value, bool) else [f"{self.key} must be a boolean"]
        if isinstance(value, bool) or not isinstance(value, str):
            return [f"{self.key} must be a string"]
        return (
            []
            if value in self.options
            else [f"{self.key} must be one of {list(self.options)}"]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "type": self.type,
            "default": self.default,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "options": list(self.options),
            "unit": self.unit,
        }


@dataclass(frozen=True)
class ConfigSchema:
    """The full set of tuneables of a deployment (rendered, not hardcoded)."""

    fields: tuple[ConfigFieldSpec, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.fields, (tuple, list)) or not all(
            isinstance(f, ConfigFieldSpec) for f in self.fields
        ):
            raise ValueError("config schema fields must be ConfigFieldSpec[]")
        keys = [f.key for f in self.fields]
        if len(set(keys)) != len(keys):
            raise ValueError(f"config schema keys must be unique, got {keys}")

    def validate_values(self, values: Mapping[str, Any]) -> list[str]:
        """Fail-closed: unknown keys are errors, never silently dropped."""
        if not isinstance(values, Mapping):
            return ["config values must be a mapping"]
        known = {f.key: f for f in self.fields}
        errors: list[str] = []
        for key in values:
            if key not in known:
                errors.append(f"unknown config key: {key}")
        for key, spec in known.items():
            if key in values:
                errors.extend(spec.validate_value(values[key]))
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {"fields": [f.to_dict() for f in self.fields]}


def build_config_schema(raw: Sequence[Mapping[str, Any]]) -> ConfigSchema:
    """Build a schema from parsed YAML/JSON (fail-closed on unknown keys)."""
    allowed = {
        "key", "label", "type", "default", "minimum", "maximum", "step",
        "options", "unit",
    }
    fields_out: list[ConfigFieldSpec] = []
    for i, raw_field in enumerate(raw):
        if not isinstance(raw_field, Mapping):
            raise ValueError(f"config field[{i}] must be a mapping, got {raw_field!r}")
        unknown = set(raw_field) - allowed
        if unknown:
            raise ValueError(f"config field[{i}] has unknown keys: {sorted(unknown)}")
        raw_options = raw_field.get("options", ())
        if not isinstance(raw_options, (list, tuple)):
            # tuple("abc") would silently explode a string into characters.
            raise ValueError(
                f"config field[{i}] options must be a list of strings, got {raw_options!r}"
            )
        fields_out.append(
            ConfigFieldSpec(
                key=raw_field.get("key"),
                label=raw_field.get("label"),
                type=raw_field.get("type"),
                default=raw_field.get("default"),
                minimum=raw_field.get("minimum"),
                maximum=raw_field.get("maximum"),
                step=raw_field.get("step"),
                options=tuple(raw_options),
                unit=raw_field.get("unit"),
            )
        )
    return ConfigSchema(fields=tuple(fields_out))
