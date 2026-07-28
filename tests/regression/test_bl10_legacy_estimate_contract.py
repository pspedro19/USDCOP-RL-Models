"""Fail-first mutation contract for BL-10 legacy backfill notes."""

from __future__ import annotations

import shutil

from scripts.validation import check_trial_ledger as ledger_contract


def test_backfilled_cell_without_legacy_estimate_note_is_rejected(
    tmp_path,
    monkeypatch,
):
    """A top-level label must not hide a missing note on a backfilled cell."""
    families_dir = tmp_path / "families"
    shutil.copytree(ledger_contract.FAMILIES_DIR, families_dir)

    family_path = families_dir / "trend_regime.yaml"
    original = family_path.read_text(encoding="utf-8")
    documented_note = "legacy_estimate — TRIALS_PROGRAM=74"
    assert documented_note in original, "fixture drift: expected a real backfilled cell"
    family_path.write_text(
        original.replace(documented_note, "TRIALS_PROGRAM=74", 1),
        encoding="utf-8",
    )

    monkeypatch.setattr(ledger_contract, "FAMILIES_DIR", families_dir)
    records = ledger_contract.load_ledger(ledger_contract.LEDGER_PATH)
    errors = ledger_contract.check_families(records, ledger_contract.FAMILIES_DIR)

    expected = (
        "trend_regime: celda 'legacy_trials_program' con backfill "
        "debe documentar 'legacy_estimate' en note"
    )
    assert expected in errors


def test_documented_direction_trials_missing_from_original_backfill_are_appended():
    """The 109->111 reconciliation must be represented as two named append-only rows."""
    records = ledger_contract.load_ledger(ledger_contract.LEDGER_PATH)
    by_variant = {record["variant"]: record for record in records}

    expected = {
        "h1_daily_shadow_v1": {
            "trial_id": "FT-0054",
            "result": "pending_forward",
        },
        "h1_latam_transport_v1": {
            "trial_id": "FT-0055",
            "result": "fail",
        },
    }
    for variant, fields in expected.items():
        assert variant in by_variant, f"missing documented ledger row: {variant}"
        record = by_variant[variant]
        assert record["trial_id"] == fields["trial_id"]
        assert record["result"] == fields["result"]
        assert record["asset"] == "usdcop"
        assert record["family"] == "usdcop_direction"
        assert record["cluster"] == "ml_meta"
        assert record["kind"] == "forecast"
        assert record["label"] == "documented"
        assert record["env"] == "legacy_backfill"
        assert record["source"].endswith("usdcop/HYPOTHESIS-REGISTRY.md")

    assert [record["trial_id"] for record in records[-2:]] == ["FT-0054", "FT-0055"]
    assert records[-2]["prev_hash"] == records[-3]["line_hash"]
    assert records[-1]["prev_hash"] == records[-2]["line_hash"]
