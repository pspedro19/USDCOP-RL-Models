"""Canonical JSON writer used by ledgers, signals, targets and bundles (BL-16/17)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import unicodedata
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class CanonicalizationError(ValueError):
    """The value cannot be represented by the canonical contract."""


def _utc_z(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise CanonicalizationError("naive timestamps are forbidden")
    utc = value.astimezone(timezone.utc)
    rendered = utc.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return rendered.replace(".000000Z", "Z")


@dataclass(frozen=True, slots=True)
class _CanonicalNumber:
    """Internal raw JSON number.

    A private marker avoids the historic collision between the numeric value
    ``1.0`` and the user string ``"1"`` while still emitting valid JSON
    numbers.  It never escapes :func:`canonical_json_bytes`.
    """

    text: str


def _schema_path(path: str) -> str:
    """Remove array indexes so ``/legs/qty`` applies to every array member."""

    return "/" + "/".join(
        part for part in path.split("/") if part and not part.isdecimal()
    )


def _quantum(
    path: str,
    field_quantums: Mapping[str, str] | None,
    matched_quantums: set[str],
) -> Decimal | None:
    if not field_quantums:
        return None
    candidates = (path, _schema_path(path), "*")
    key = next((candidate for candidate in candidates if candidate in field_quantums), None)
    if key is None:
        return None
    matched_quantums.add(key)
    raw = field_quantums[key]
    try:
        quantum = Decimal(raw)
    except InvalidOperation as exc:
        raise CanonicalizationError(f"invalid quantum {raw!r} for {path or '/'}") from exc
    if not quantum.is_finite() or quantum <= 0:
        raise CanonicalizationError(f"quantum must be finite and positive for {path or '/'}")
    return quantum


def _decimal_text(value: Decimal, quantum: Decimal | None) -> str:
    if not value.is_finite():
        raise CanonicalizationError("NaN and Infinity are forbidden")
    if quantum is None:
        quantized = value
    else:
        try:
            quantized = value.quantize(quantum, rounding=ROUND_HALF_EVEN)
        except InvalidOperation as exc:
            raise CanonicalizationError(
                f"value {value!s} cannot be quantized with quantum {quantum!s}"
            ) from exc
    text = format(quantized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def canonicalize(
    value: Any,
    *,
    field_quantums: Mapping[str, str] | None = None,
    _path: str = "",
    _matched_quantums: set[str] | None = None,
) -> Any:
    """Return the normalized tree used by the canonical JSON encoder."""

    matched_quantums = _matched_quantums if _matched_quantums is not None else set()
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    elif is_dataclass(value):
        raise CanonicalizationError("dataclass types are not canonical values")
    if isinstance(value, Enum):
        value = value.value
    if type(value).__module__.split(".", 1)[0] == "numpy":
        raise CanonicalizationError("numpy scalars must be converted to Python primitives")
    if value is None or type(value) is bool:
        return value
    if isinstance(value, int):
        return _CanonicalNumber(
            _decimal_text(Decimal(value), _quantum(_path, field_quantums, matched_quantums))
        )
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    if isinstance(value, datetime):
        return _utc_z(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return _CanonicalNumber(
            _decimal_text(value, _quantum(_path, field_quantums, matched_quantums))
        )
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError("NaN and Infinity are forbidden")
        return _CanonicalNumber(
            _decimal_text(
                Decimal(str(value)), _quantum(_path, field_quantums, matched_quantums)
            )
        )
    if isinstance(value, Mapping):
        keys = list(value)
        if any(not isinstance(key, str) for key in keys):
            raise CanonicalizationError("canonical JSON object keys must be strings")
        normalized_keys = [
            (unicodedata.normalize("NFC", key), key)
            for key in keys
        ]
        result: dict[str, Any] = {}
        for normalized_key, original_key in sorted(normalized_keys, key=lambda item: item[0]):
            child_path = f"{_path}/{normalized_key}"
            if normalized_key in result:
                raise CanonicalizationError(f"NFC key collision at {child_path}")
            result[normalized_key] = canonicalize(
                value[original_key],
                field_quantums=field_quantums,
                _path=child_path,
                _matched_quantums=matched_quantums,
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            canonicalize(
                item,
                field_quantums=field_quantums,
                _path=f"{_path}/{idx}",
                _matched_quantums=matched_quantums,
            )
            for idx, item in enumerate(value)
        ]
    raise CanonicalizationError(f"unsupported canonical type: {type(value).__name__}")


def _encode_canonical(value: Any) -> str:
    if isinstance(value, _CanonicalNumber):
        return value.text
    if value is None:
        return "null"
    if type(value) is bool:
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ",".join(_encode_canonical(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(
            f"{json.dumps(key, ensure_ascii=False)}:{_encode_canonical(item)}"
            for key, item in value.items()
        ) + "}"
    raise CanonicalizationError(f"internal canonical type escaped: {type(value).__name__}")


def canonical_json_bytes(
    value: Any, *, field_quantums: Mapping[str, str] | None = None
) -> bytes:
    matched_quantums: set[str] = set()
    normalized = canonicalize(
        value,
        field_quantums=field_quantums,
        _matched_quantums=matched_quantums,
    )
    unmatched = sorted(set(field_quantums or {}) - matched_quantums - {"*"})
    if unmatched:
        raise CanonicalizationError(f"field quantum paths matched no numeric field: {unmatched}")
    return _encode_canonical(normalized).encode("utf-8")


def semantic_hash(value: Any, *, field_quantums: Mapping[str, str] | None = None) -> str:
    return "sha256:" + hashlib.sha256(
        canonical_json_bytes(value, field_quantums=field_quantums)
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalArtifact:
    payload: Any
    content: bytes
    semantic_hash: str
    bytes_hash: str

    @classmethod
    def build(
        cls, payload: Any, *, field_quantums: Mapping[str, str] | None = None
    ) -> "CanonicalArtifact":
        content = canonical_json_bytes(payload, field_quantums=field_quantums)
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        return cls(payload=payload, content=content, semantic_hash=digest, bytes_hash=digest)

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary.open("xb") as handle:
                handle.write(self.content)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                # A hard link publishes the fully-written inode only when the
                # destination does not exist.  Unlike exists()+replace(), this
                # is an atomic first-writer-wins claim across processes.
                os.link(temporary, destination)
            except FileExistsError:
                existing = destination.read_bytes()
                if existing != self.content:
                    raise CanonicalizationError(
                        "refusing divergent canonical artifact publication: "
                        f"{destination}"
                    )
        finally:
            temporary.unlink(missing_ok=True)
