"""Canonical serialization and hashing primitives.

Everything in this package that must be *provable after the fact* passes through
here. The guarantee we need is narrow but strict: two runs that produced the same
logical record must produce byte-identical output, so that a hash computed today
still verifies in a year.

That rules out `json.dumps(obj)` with defaults, which preserves insertion order
and emits non-ASCII verbatim. Key order and encoding must be pinned explicitly.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Chosen once and frozen. Changing any of these invalidates every hash already
# written to the ledger, which is the point: it makes the format a commitment.
_JSON_KWARGS: dict[str, Any] = {
    "sort_keys": True,        # key order must not depend on dict insertion order
    "ensure_ascii": True,     # escape non-ASCII so the bytes are locale-independent
    "separators": (",", ":"), # no incidental whitespace
    "allow_nan": False,       # NaN/Infinity are not valid JSON; fail loudly instead
}

GENESIS_HASH = "0" * 64
"""prev_hash of the first record in a chain. Not a real hash, just a fixed anchor."""


def canonical_json(payload: dict[str, Any]) -> str:
    """Serialize a mapping to its one canonical JSON string.

    Raises:
        ValueError: if the payload contains NaN or Infinity, which would
            serialize to invalid JSON and silently break re-verification.
    """
    return json.dumps(payload, **_JSON_KWARGS)


def sha256_text(text: str) -> str:
    """Hex SHA-256 of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_payload(payload: dict[str, Any]) -> str:
    """Hex SHA-256 of the canonical serialization of ``payload``."""
    return sha256_text(canonical_json(payload))


def chain_hash(prev_hash: str, payload: dict[str, Any]) -> str:
    """Hash a record *together with* its predecessor.

    This is what makes the ledger append-only in practice rather than by
    convention. Editing record N changes its hash, which breaks the ``prev_hash``
    stored in N+1, which breaks N+2, and so on to the tip. An auditor only needs
    the tip hash to detect any edit anywhere in the history.
    """
    return sha256_text(prev_hash + canonical_json(payload))
