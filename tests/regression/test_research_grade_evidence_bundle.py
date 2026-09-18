"""Mutation checks for archival evidence and truthful E2E classification."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.diagnostics.audit_thesis_e2e_status import _ledger_status, audit, verify_bundle
from scripts.diagnostics.freeze_thesis_evidence import freeze
from scripts.presentation.build_research_grade_thesis import Snapshot, independent_score


def test_snapshot_preserves_and_detects_mutation(tmp_path):
    source = tmp_path / "data.json"
    source.write_text('{"value":42}')
    out = tmp_path / "preserved"
    freeze(tmp_path, out, patterns=("data.json",))
    snap = Snapshot(out / "manifest.json")
    assert snap.read("data.json") == source.read_bytes()
    source.write_text('{"value":0}')
    assert snap.read("data.json") == b'{"value":42}'
    obj = out / snap.files["data.json"]["object"]
    obj.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="hash mismatch"):
        snap.read("data.json")
    with pytest.raises(FileExistsError):
        freeze(tmp_path, out, patterns=("data.json",))


@pytest.mark.parametrize(
    "name", [".env", ".env.test", "private.pem", "private.key", "credentials-test.json"]
)
def test_snapshot_refuses_sensitive_paths_without_reading(tmp_path, name, monkeypatch):
    # These are empty filename fixtures, not actual credentials.
    (tmp_path / name).touch()
    import scripts.diagnostics.freeze_thesis_evidence as module

    monkeypatch.setattr(module, "digest", lambda _: pytest.fail("must reject before reading bytes"))
    with pytest.raises(ValueError, match="path refused"):
        freeze(tmp_path, tmp_path / "out", patterns=(name,))


def test_snapshot_rejects_unaddressed_object(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "contract": "THESIS-EVIDENCE-SNAPSHOT-1",
                "files": [{"path": "fake", "object": "../anything", "sha256": "a" * 64}],
            }
        )
    )
    with pytest.raises(ValueError, match="content-addressed"):
        Snapshot(path).read("fake")


@pytest.mark.parametrize("name", [".env", ".env.test", "credentials-test.json", "private.key"])
def test_e2e_never_reads_sensitive_manifest_paths(tmp_path, name, monkeypatch):
    from pathlib import Path

    from scripts.diagnostics.audit_thesis_e2e_status import load_json, sha256

    path = tmp_path / name
    monkeypatch.setattr(
        Path, "read_text", lambda *a, **k: pytest.fail("must reject before reading")
    )
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("must reject before opening"))
    assert load_json(path) is None
    with pytest.raises(ValueError, match="sensitive evidence path"):
        sha256(path)


def test_independent_replay_prices_final_close_and_all_cost_sides():
    close = np.r_[np.full(59, 4000.0), 4400.0]
    spec = SimpleNamespace(close=close, spread_pips=3.0)
    g, c = independent_score(spec, np.ones(59), commission=0.5, slippage=0.0)
    assert g == pytest.approx(0.1)
    assert c == pytest.approx(2.0 / 4000.0 + 2.0 / 4400.0)
    assert independent_score(spec, np.zeros(59), 0.5, 0.1) == (0.0, 0.0)


def ledger_rows():
    return [
        {
            "session_date": "2023-01-03",
            "bar": b,
            "decision_id": f"d::{b}",
            "weight": 0.0,
            "previous_weight": 0.0,
            "valid_json": True,
            "unavailable": False,
            "dataset_block": "selection",
        }
        for b in range(59)
    ]


def ledger_file(tmp_path, rows):
    path = tmp_path / "decisions_deepseek_selection_diagnostic.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    return path


def test_complete_retrospective_ledger_is_not_prospective_evidence(tmp_path):
    # Parallel historical appends may be reversed; logical state must still match.
    ledger_file(tmp_path, list(reversed(ledger_rows())))
    result = _ledger_status(tmp_path, "deepseek", ["2023-01-03"])
    assert result["status"] == "RETROSPECTIVE_COMPLETE"
    assert not result["strict_provenance_verified"]
    assert not result["confirmatory"]


@pytest.mark.parametrize("mutation", ["duplicate", "unavailable", "missing", "wrong_state"])
def test_row_count_alone_cannot_certify_a_ledger(tmp_path, mutation):
    rows = ledger_rows()
    if mutation == "duplicate":
        rows[-1] = rows[-2].copy()
    elif mutation == "missing":
        rows.pop()
    elif mutation == "unavailable":
        rows[3]["unavailable"] = True
    else:
        rows[3]["previous_weight"] = 1.0
    ledger_file(tmp_path, rows)
    assert (
        _ledger_status(tmp_path, "deepseek", ["2023-01-03"])["status"] != "RETROSPECTIVE_COMPLETE"
    )


def test_arbitrary_figures_and_signed_text_never_certify_science(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "nice.png").write_bytes(b"not scientific evidence")
    assert verify_bundle(images)["status"] == "INVALID"
    prereg = tmp_path / ".claude/specs/planes/06-PRE-REGISTRATION-v3.md"
    prereg.parent.mkdir(parents=True)
    prereg.write_text("operator_signature: SIGNED\n")
    result = audit(tmp_path, bundle=images)
    assert not result["confirmatory_ready"]
    assert not result["engineering_ready"]
    assert not result["scientific_closure_ready"]
