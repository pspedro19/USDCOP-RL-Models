"""Sealing and independent replay verification for the candidate paper ledger."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from src.identity.canonical import semantic_hash
from src.identity.fingerprints import decision_fingerprint, derivation_id


IDENTITY_SCHEMA_VERSION = "1.0.0"
IDENTITY_FIELDS = frozenset(
    {"schema_version", "semantic_hash", "decision_fingerprint", "derivation_id"}
)


class CandidateLedgerIdentityError(AssertionError):
    """The tracked ledger does not reproduce its sealed identity."""


def semantic_payload(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Return governed payload; identity and volatile generation date are excluded."""
    payload = deepcopy(dict(ledger))
    payload.pop("identity", None)
    payload.pop("generated_at", None)
    return payload


def _decision_inputs(payload: Mapping[str, Any]) -> dict[str, Any]:
    strategies = payload.get("strategies")
    governed_strategies: dict[str, Any] = {}
    if isinstance(strategies, Mapping):
        for strategy_id, record in sorted(strategies.items()):
            if not isinstance(record, Mapping):
                governed_strategies[str(strategy_id)] = record
                continue
            governed_strategies[str(strategy_id)] = {
                key: deepcopy(record.get(key))
                for key in ("ret_2026_ytd_pct", "n_trades", "judge_window")
            }
    return {
        "strategies": governed_strategies,
        "book": deepcopy(payload.get("book")),
    }


def expected_identity(
    ledger: Mapping[str, Any], *, producer_code_hash: str
) -> dict[str, str]:
    if not isinstance(producer_code_hash, str) or not producer_code_hash.startswith("sha256:"):
        raise CandidateLedgerIdentityError("producer_code_hash must be a sha256 fingerprint")
    payload = semantic_payload(ledger)
    payload_hash = semantic_hash(payload)
    specification_hash = semantic_hash(
        {
            "contract": payload.get("contract"),
            "anchor": payload.get("anchor"),
            "labels": payload.get("labels"),
            "judge_note": payload.get("judge_note"),
        }
    )
    decision_hash = decision_fingerprint(
        spec_fingerprint_value=specification_hash,
        as_of=str(payload.get("anchor") or ""),
        decision_inputs=_decision_inputs(payload),
    )
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "semantic_hash": payload_hash,
        "decision_fingerprint": decision_hash,
        "derivation_id": derivation_id(
            inputs={"semantic_hash": payload_hash},
            code_hash=producer_code_hash,
            params={
                "identity_schema_version": IDENTITY_SCHEMA_VERSION,
                "generated_at_excluded": True,
            },
        ),
    }


def seal_candidate_ledger(
    ledger: Mapping[str, Any], *, producer_code_hash: str
) -> dict[str, Any]:
    sealed = deepcopy(dict(ledger))
    sealed["identity"] = expected_identity(
        sealed, producer_code_hash=producer_code_hash
    )
    return sealed


def verify_candidate_ledger(
    ledger: Mapping[str, Any], *, producer_code_hash: str
) -> dict[str, str]:
    stored = ledger.get("identity")
    if not isinstance(stored, Mapping):
        raise CandidateLedgerIdentityError("candidate ledger has no identity envelope")
    if set(stored) != IDENTITY_FIELDS:
        raise CandidateLedgerIdentityError(
            "candidate ledger identity fields differ: "
            f"expected={sorted(IDENTITY_FIELDS)} obtained={sorted(stored)}"
        )
    expected = expected_identity(ledger, producer_code_hash=producer_code_hash)
    obtained = {str(key): str(value) for key, value in stored.items()}
    if obtained != expected:
        raise CandidateLedgerIdentityError(
            "candidate ledger identity mismatch\n"
            f"  expected semantic_hash: {expected['semantic_hash']}\n"
            f"  obtained semantic_hash: {obtained.get('semantic_hash')}\n"
            f"  expected decision_fingerprint: {expected['decision_fingerprint']}\n"
            f"  obtained decision_fingerprint: {obtained.get('decision_fingerprint')}\n"
            f"  expected derivation_id: {expected['derivation_id']}\n"
            f"  obtained derivation_id: {obtained.get('derivation_id')}"
        )
    return expected
