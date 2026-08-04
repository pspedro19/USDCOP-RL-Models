from __future__ import annotations

from copy import deepcopy

import pytest

from src.identity.candidate_ledger import (
    CandidateLedgerIdentityError,
    seal_candidate_ledger,
    verify_candidate_ledger,
)


CODE_HASH = "sha256:" + "a" * 64


def _ledger() -> dict:
    return {
        "contract": "CTR-QUANT-CONSTITUTION-001",
        "anchor": "2026-01-01",
        "labels": {"s1": "forward"},
        "judge_note": "post-freeze only",
        "generated_at": "2026-08-04",
        "strategies": {
            "s1": {
                "ret_2026_ytd_pct": 1.25,
                "n_trades": 8,
                "judge_window": None,
                "trades": [{"timestamp": "2026-01-05", "pnl_pct": 1.25}],
            }
        },
        "book": {"weeks": [{"iso_week": "2026-W02", "book_ret_pct": None}]},
    }


def test_seal_replays_and_generated_at_is_deliberately_volatile() -> None:
    sealed = seal_candidate_ledger(_ledger(), producer_code_hash=CODE_HASH)
    first = verify_candidate_ledger(sealed, producer_code_hash=CODE_HASH)
    regenerated = deepcopy(sealed)
    regenerated["generated_at"] = "2026-08-05"

    assert verify_candidate_ledger(regenerated, producer_code_hash=CODE_HASH) == first


def test_mutating_a_trade_breaks_replay_and_names_both_hashes() -> None:
    sealed = seal_candidate_ledger(_ledger(), producer_code_hash=CODE_HASH)
    mutated = deepcopy(sealed)
    mutated["strategies"]["s1"]["trades"][0]["pnl_pct"] = 99.0

    with pytest.raises(CandidateLedgerIdentityError) as exc:
        verify_candidate_ledger(mutated, producer_code_hash=CODE_HASH)
    assert "expected semantic_hash" in str(exc.value)
    assert "obtained semantic_hash" in str(exc.value)


def test_code_change_breaks_derivation_even_when_payload_is_unchanged() -> None:
    sealed = seal_candidate_ledger(_ledger(), producer_code_hash=CODE_HASH)

    with pytest.raises(CandidateLedgerIdentityError, match="derivation_id"):
        verify_candidate_ledger(
            sealed, producer_code_hash="sha256:" + "b" * 64
        )


def test_identity_envelope_is_exact_not_extensible_by_accident() -> None:
    sealed = seal_candidate_ledger(_ledger(), producer_code_hash=CODE_HASH)
    sealed["identity"]["generated_at"] = "forbidden"

    with pytest.raises(CandidateLedgerIdentityError, match="fields differ"):
        verify_candidate_ledger(sealed, producer_code_hash=CODE_HASH)
