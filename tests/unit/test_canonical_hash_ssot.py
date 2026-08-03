"""G-02 / INTEGRATION-CONTRACT F-02 — one logical object, one hash.

The policy family (``src/contracts/policy*.py`` + ``src/strategies/policies/
loader.py``) carried FOUR copies of the same ``json.dumps(sort_keys=True,
separators=(",",":"), allow_nan=False)`` + ``hashlib.sha256`` idiom. Four copies
of a hash rule is not a style problem: it is four opportunities for the identity
of a frozen strategy to drift, and the drift is silent because
``HASH_PATTERN`` used to accept any length from 8 to 64 hex — so a hash from a
different idiom still looked valid.

Two things are locked here:

1. **Structure** — exactly ONE hashing idiom exists in the family; every other
   site delegates. (Red before the fix: 4 sites.)
2. **Behaviour** — unifying changed NOTHING. The pinned digests below were
   computed from the pre-refactor code, so if a future "cleanup" swaps the
   family onto a different canonicalisation, the frozen ``policy_hash`` values
   in ``config/policies/*.yaml`` would move and these tests go red first.

SCOPE, stated honestly: this unifies the policy family with ITSELF. It does NOT
unify it with ``src/identity/canonical.py`` (CODEX, BL-17), which serialises
differently (``ensure_ascii=False``, Decimal quantisation, NFC, datetime
support) and would therefore change every frozen hash in the repo. That is a
deliberate re-freeze of published evidence -> **DECISIÓN DEL OPERADOR**, not a
refactor, and it is not taken here.

Contract: CTR-POLICY-001 · INTEGRATION-CONTRACT.md F-02 · TDD-GAPS.md G-02
"""

from __future__ import annotations

import ast
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.contracts.policy import HASH_PATTERN, policy_canonical_hash, require_hash

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Every module allowed to speak about the policy-family hash.
POLICY_FAMILY = (
    "src/contracts/policy.py",
    "src/contracts/policy_dsl.py",
    "src/contracts/policy_version.py",
    "src/strategies/policies/loader.py",
    "src/policy_engine/runner.py",
)

#: The ONE module that may implement it.
HASH_SSOT_MODULE = "src/contracts/policy.py"

#: Digests computed from the PRE-unification code. They must never move.
FROZEN_ADVERSARIAL_DIGESTS = {
    "nonascii_and_float": "sha256:bb3edc578dffe2b6d5077927cc9e6f71d13067b30c572cff6500bc3e30d1509a",
    "neg_zero": "sha256:2b7bcd622729931159e027106ce7b0a25a9c1fde3ccba4960f6be3e98df9b465",
    "unsorted_nested": "sha256:aa5acc11e864cec3b7c7462948bf081ba8843dbd798f17a53dbeff9c896dcc04",
    "unicode_nfc_vs_nfd": "sha256:9a95b0c0b9503f25c3027e8851a400b1927e0969e0747420d995f9dfd838f392",
    "empty": "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
}

ADVERSARIAL_PAYLOADS = {
    "nonascii_and_float": {"b": 1, "a": "café", "f": 0.1 + 0.2},
    "neg_zero": {"z": -0.0, "p": 0.0},
    "unsorted_nested": {"z": {"b": 2, "a": 1}, "a": [3, 1, 2]},
    "unicode_nfc_vs_nfd": {"k": "café"},
    "empty": {},
}


# --------------------------------------------------------------------------- structure


def _hash_idiom_sites(rel_path: str) -> list[tuple[int, str]]:
    """``(line, kind)`` for every canonical-JSON-hash construction in a file."""
    tree = ast.parse((REPO_ROOT / rel_path).read_text(encoding="utf-8"))
    sites: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        dotted = (
            f"{func.value.id}.{func.attr}"
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
            else None
        )
        if dotted == "json.dumps" and any(
            kw.arg == "sort_keys" for kw in node.keywords
        ):
            sites.append((node.lineno, "json.dumps(sort_keys=...)"))
        elif dotted == "hashlib.sha256":
            sites.append((node.lineno, "hashlib.sha256(...)"))
    return sites


def test_policy_family_has_exactly_one_hash_idiom() -> None:
    """Only ``policy.py`` may build the canonical JSON + sha256; the rest delegate.

    Mutation that turns this red: paste a second
    ``hashlib.sha256(json.dumps(payload, sort_keys=True, ...))`` anywhere in the
    policy family. (Before the fix it was red with 4 sites across
    policy.py / policy_dsl.py / policy_version.py / loader.py.)
    """
    offenders = {
        rel: sites
        for rel in POLICY_FAMILY
        if rel != HASH_SSOT_MODULE and (sites := _hash_idiom_sites(rel))
    }
    assert not offenders, (
        "the canonical policy hash must have ONE implementation "
        f"({HASH_SSOT_MODULE}); found copies in: "
        + "; ".join(f"{rel} at lines {[ln for ln, _ in s]}" for rel, s in offenders.items())
    )

    ssot_sites = _hash_idiom_sites(HASH_SSOT_MODULE)
    assert len(ssot_sites) == 2, (
        "the SSOT should contain exactly one json.dumps + one hashlib.sha256, "
        f"found {ssot_sites}"
    )


# --------------------------------------------------------------------------- behaviour


@pytest.mark.parametrize("name", sorted(ADVERSARIAL_PAYLOADS))
def test_unification_did_not_move_a_single_digest(name: str) -> None:
    """The SSOT reproduces the pre-refactor bytes on adversarial input.

    ``café`` (non-ASCII), ``0.1+0.2`` (float repr), ``-0.0`` (signed zero),
    NFD vs NFC and unsorted nested keys are the inputs on which two
    canonicalisations diverge silently.
    """
    assert policy_canonical_hash(ADVERSARIAL_PAYLOADS[name]) == FROZEN_ADVERSARIAL_DIGESTS[name]


def test_every_family_helper_agrees_with_the_ssot() -> None:
    """The delegating helpers return bit-identical results to the SSOT."""
    from src.contracts.policy_dsl import _canonical_policy_hash
    from src.contracts.policy_version import _canonical_hash

    for name, payload in ADVERSARIAL_PAYLOADS.items():
        expected = FROZEN_ADVERSARIAL_DIGESTS[name]
        assert _canonical_policy_hash(payload) == expected, name
        assert _canonical_hash(payload) == expected, name


@pytest.mark.parametrize(
    "value",
    [datetime(2026, 7, 28, tzinfo=timezone.utc), datetime(2026, 7, 28)],
    ids=["tz-aware", "naive"],
)
def test_datetime_is_a_typed_error_not_a_silent_stringification(value: datetime) -> None:
    """No ``default=`` fallback: a datetime raises rather than hashing its repr.

    Both aware and naive must behave identically — a canonicalisation that
    accepted one and not the other would make identity depend on tzinfo.
    """
    from src.contracts.policy_dsl import _canonical_policy_hash
    from src.contracts.policy_version import _canonical_hash

    for fn in (policy_canonical_hash, _canonical_policy_hash, _canonical_hash):
        with pytest.raises(TypeError):
            fn({"t": value})


def test_nan_and_infinity_are_rejected() -> None:
    """JSON-safety rule: ``allow_nan=False`` on every path into the hash."""
    for payload in ({"x": float("nan")}, {"x": float("inf")}, {"x": float("-inf")}):
        with pytest.raises(ValueError):
            policy_canonical_hash(payload)


# --------------------------------------------------------------------------- HASH_PATTERN


def test_hash_pattern_is_pinned_to_the_real_digest_length() -> None:
    """A hash of the wrong length is not "a valid short hash" — it is a foreign hash.

    ``^sha256:[0-9a-f]{8,64}$`` accepted 57 different lengths, so a digest
    produced by a different idiom (or a truncated hex16 id fragment) validated
    as if it were the canonical fingerprint. sha256 has exactly one length.

    Mutation that turns this red: widen the quantifier back to ``{8,64}``.
    """
    assert HASH_PATTERN.pattern == r"^sha256:[0-9a-f]{64}$"

    full = "sha256:" + "ab" * 32
    assert require_hash("policy_hash", full) == full

    for bad in (
        "sha256:deadbeef",  # 8 hex — used to pass
        "sha256:" + "ab" * 8,  # hex16, the truncated form
        "sha256:" + "ab" * 31,  # 62 hex
        "sha256:" + "ab" * 33,  # 66 hex
        "sha256:" + "AB" * 32,  # uppercase
        "sha256:",
        "",
    ):
        with pytest.raises(ValueError):
            require_hash("policy_hash", bad)


def test_hash_pattern_mirror_is_identical_in_typescript() -> None:
    """``policy.contract.ts`` must reject exactly what Python rejects.

    A one-sided relaxation is the classic mirror defect: the backend refuses a
    hash the dashboard happily renders as valid.
    """
    ts = (REPO_ROOT / "usdcop-trading-dashboard/lib/contracts/policy.contract.ts").read_text(
        encoding="utf-8"
    )
    match = re.search(r"export const HASH_PATTERN = /(?P<body>.+?)/;", ts)
    assert match is not None, "HASH_PATTERN not found in policy.contract.ts"
    assert match.group("body") == HASH_PATTERN.pattern


def test_frozen_policy_specs_still_validate_against_their_declared_hash() -> None:
    """The four frozen specs keep their published ``governance.policy_hash``.

    This is the real-world consequence check: if unifying the idiom had moved a
    digest, these YAMLs would need a deliberate re-freeze (operator decision).
    They did not.
    """
    from src.strategies.policies.loader import canonical_policy_hash, load_all_policy_specs

    specs = load_all_policy_specs()
    assert specs, "no policy specs found"
    for spec in specs:
        declared = (spec.get("governance") or {}).get("policy_hash")
        assert declared, f"{spec['id']}: spec is not frozen (no governance.policy_hash)"
        assert declared == canonical_policy_hash(spec), spec["id"]
        assert HASH_PATTERN.fullmatch(declared), f"{spec['id']}: {declared}"
