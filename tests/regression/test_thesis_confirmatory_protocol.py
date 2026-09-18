from pathlib import Path

import yaml

from scripts.diagnostics.validate_thesis_confirmatory_protocol import validate

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/research/thesis_confirmatory_v4.yaml"


def test_confirmatory_protocol_is_valid_after_explicit_operator_signature():
    assert validate(CONFIG) == []
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert data["status"] == "SIGNED"
    assert data["signed_by"]
    assert data["signed_at_utc"]


def test_confirmatory_protocol_does_not_reuse_historical_partition():
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    blocks = data["partitions"]
    assert blocks["confirmatory_holdout"]["start"] == "2024-01-01"
    assert blocks["confirmatory_holdout"]["end"] == "2025-12-31"
    assert blocks["forward"]["start"] == "2026-01-01"
    assert blocks["confirmatory_holdout"]["role"] == "one_look_confirmatory"


def test_confirmatory_protocol_keeps_llm_and_gold_exploratory():
    text = CONFIG.read_text(encoding="utf-8")
    assert "status: exploratory_only" in text
    assert "cannot_define_confirmatory_winner: true" in text


def test_confirmatory_protocol_forbids_silent_data_repairs():
    text = CONFIG.read_text(encoding="utf-8")
    assert "missing_file_policy: fail_closed" in text
    assert "same_day_macro_policy: forbidden" in text
    assert "backfill_policy: forbidden" in text
    assert "interpolation_policy: forbidden" in text
