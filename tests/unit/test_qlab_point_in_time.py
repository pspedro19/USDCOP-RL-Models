from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.research.point_in_time import (
    PointInTimeViolation,
    ResearchEnvironment,
    read_point_in_time,
)
from src.research.qlab import FamilyStore, TrialLedger


def _load_cli():
    path = Path("scripts/analysis/qlab.py")
    spec = importlib.util.spec_from_file_location("qlab_cli_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_screening_reader_forwards_cutoff_and_rejects_late_materialized_rows() -> None:
    seen: dict[str, object] = {}

    def reader(**kwargs):
        seen.update(kwargs)
        return [
            {"id": "past", "available_at": "2024-12-31T23:59:59Z"},
            {"id": "future", "available_at": "2025-06-01T00:00:00Z"},
        ]

    with pytest.raises(PointInTimeViolation, match="2025-06-01.*2025-01-01"):
        read_point_in_time(
            reader,
            cutoff="2025-01-01T00:00:00Z",
            environment=ResearchEnvironment.SCREENING,
        )

    assert seen["cutoff"] == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert seen["available_at_field"] == "available_at"


def test_qlab_screen_blocks_future_data_before_charging_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_cli()
    families = FamilyStore(tmp_path / "families")
    ledger = TrialLedger(tmp_path / "ledger.jsonl")
    families.declare(
        {
            "family_id": "pit_family",
            "kind": "forecast",
            "cluster_id": "ml_meta",
            "asset": "usdcop",
            "question": "does the signal survive?",
            "bar": "P1D",
        }
    )
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "FAMILIES", families)
    monkeypatch.setattr(cli, "LEDGER", ledger)

    evidence = tmp_path / "screening.jsonl"
    evidence.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"id": "past", "available_at": "2024-12-31T12:00:00Z"},
                {"id": "future", "available_at": "2025-06-01T00:00:00Z"},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    argv = [
        "screen",
        "pit_family",
        "--charge-trial",
        "--trial-id",
        "FT-9001",
        "--asset",
        "usdcop",
        "--variant",
        "ridge",
        "--cutoff",
        "2025-01-01",
        "--source",
        str(evidence),
    ]
    with pytest.raises(PointInTimeViolation, match="2025-06-01.*2025-01-01"):
        cli.main(argv)

    assert ledger.rows() == []
    assert families.load("pit_family")["state"] == "DECLARED"


def test_qlab_screen_charges_once_after_point_in_time_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_cli()
    families = FamilyStore(tmp_path / "families")
    ledger = TrialLedger(tmp_path / "ledger.jsonl")
    families.declare(
        {
            "family_id": "pit_family",
            "kind": "forecast",
            "cluster_id": "ml_meta",
            "asset": "usdcop",
            "question": "does the signal survive?",
            "bar": "P1D",
        }
    )
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "FAMILIES", families)
    monkeypatch.setattr(cli, "LEDGER", ledger)

    evidence = tmp_path / "screening.jsonl"
    evidence.write_text(
        json.dumps(
            {"id": "past", "available_at": "2024-12-31T12:00:00Z"}
        )
        + "\n",
        encoding="utf-8",
    )
    argv = [
        "screen",
        "pit_family",
        "--charge-trial",
        "--trial-id",
        "FT-9001",
        "--asset",
        "usdcop",
        "--variant",
        "ridge",
        "--cutoff",
        "2025-01-01",
        "--source",
        str(evidence),
    ]

    assert cli.main(argv) == 0
    assert cli.main(argv) == 0
    rows = ledger.rows()
    assert len(rows) == 1
    assert rows[0]["cutoff"] == "2025-01-01T23:59:59.999999Z"
    assert rows[0]["source"] == str(evidence)
    assert rows[0]["data_hash"].startswith("sha256:")
    assert families.load("pit_family")["state"] == "SCREENING"
