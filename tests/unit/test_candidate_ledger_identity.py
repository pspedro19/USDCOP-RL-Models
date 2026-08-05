from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import pytest

from src.identity.candidate_ledger import (
    CandidateLedgerIdentityError,
    seal_candidate_ledger,
    verify_candidate_ledger,
)


CODE_HASH = "sha256:" + "a" * 64
ROOT = Path(__file__).resolve().parents[2]
PRODUCER = ROOT / "scripts" / "pipeline" / "candidates_paper_ledger.py"


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


def test_lineage_changes_semantic_identity_but_not_decision_fingerprint() -> None:
    before = seal_candidate_ledger(_ledger(), producer_code_hash=CODE_HASH)
    with_lineage = _ledger()
    with_lineage["strategies"]["s1"]["lineage"] = {
        "timestamp": "2026-01-05",
        "signal_node_id": "10000000-0000-0000-0000-000000000001",
        "snapshot_node_id": "20000000-0000-0000-0000-000000000002",
        "bar_l0_node_id": "30000000-0000-0000-0000-000000000003",
    }
    after = seal_candidate_ledger(with_lineage, producer_code_hash=CODE_HASH)

    assert before["identity"]["semantic_hash"] != after["identity"]["semantic_hash"]
    assert before["identity"]["decision_fingerprint"] == after["identity"]["decision_fingerprint"]


def test_identity_envelope_is_exact_not_extensible_by_accident() -> None:
    sealed = seal_candidate_ledger(_ledger(), producer_code_hash=CODE_HASH)
    sealed["identity"]["generated_at"] = "forbidden"

    with pytest.raises(CandidateLedgerIdentityError, match="fields differ"):
        verify_candidate_ledger(sealed, producer_code_hash=CODE_HASH)


def test_real_producer_seals_the_same_ledger_that_it_writes() -> None:
    source = PRODUCER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    seal_assignments = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "ledger"
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "seal_candidate_ledger"
        and node.value.args
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == "ledger"
    ]
    writes = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "safe_json_dump"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "ledger"
    ]

    assert len(seal_assignments) == 1
    assert len(writes) == 1
    assert seal_assignments[0].lineno < writes[0].lineno


def test_real_producer_commits_lineage_before_atomic_publication() -> None:
    source = PRODUCER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    def call_line(name: str, *, attribute: bool = False) -> int:
        matches = [
            node.lineno
            for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and (
                (attribute and isinstance(node.func, ast.Attribute) and node.func.attr == name)
                or (not attribute and isinstance(node.func, ast.Name) and node.func.id == name)
            )
        ]
        assert len(matches) == 1
        return matches[0]

    staged_write = call_line("safe_json_dump")
    commit = call_line("commit", attribute=True)
    publication = call_line("replace", attribute=True)
    rollback = call_line("rollback", attribute=True)

    assert staged_write < commit < publication
    assert rollback > publication
