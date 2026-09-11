from __future__ import annotations

import asyncio
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.research.point_in_time import (
    PointInTimeViolation,
    ResearchEnvironment,
    bounded_select_sql,
    read_point_in_time,
    read_point_in_time_async,
)
from src.research.qlab import FamilyStore, QLabError, TrialCharge, TrialLedger


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


def test_async_screening_reader_forwards_cutoff_and_rejects_late_rows() -> None:
    seen: dict[str, object] = {}

    async def reader(**kwargs):
        seen.update(kwargs)
        return [{"id": "future", "available_at": "2025-01-01T00:00:00.000001Z"}]

    async def exercise() -> None:
        with pytest.raises(PointInTimeViolation, match="look-ahead blocked"):
            await read_point_in_time_async(
                reader,
                cutoff="2025-01-01T00:00:00Z",
                environment=ResearchEnvironment.SCREENING,
            )

    asyncio.run(exercise())
    assert seen["cutoff"] == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert seen["available_at_field"] == "available_at"


@pytest.mark.parametrize("environment", list(ResearchEnvironment))
def test_every_environment_is_point_in_time_bounded(environment: ResearchEnvironment) -> None:
    seen: dict[str, object] = {}

    def reader(**kwargs):
        seen.update(kwargs)
        return [{"id": "future", "available_at": "2025-01-01T00:00:00.000001Z"}]

    with pytest.raises(PointInTimeViolation):
        read_point_in_time(
            reader,
            cutoff="2025-01-01T00:00:00Z",
            environment=environment,
        )
    assert seen["cutoff"] == datetime(2025, 1, 1, tzinfo=timezone.utc)


def test_bounded_select_sql_has_no_raw_predicate_escape_hatch() -> None:
    query = bounded_select_sql(
        table="research.observations",
        cutoff="2025-01-01T00:00:00Z",
        columns=("asset_id", "available_at"),
    )

    assert query.sql == (
        "SELECT asset_id, available_at FROM research.observations "
        "WHERE available_at <= :pit_cutoff"
    )
    assert query.parameters == {
        "pit_cutoff": datetime(2025, 1, 1, tzinfo=timezone.utc)
    }
    with pytest.raises(TypeError, match="additional_where"):
        bounded_select_sql(
            table="research.observations",
            cutoff="2025-01-01T00:00:00Z",
            additional_where="1=1) OR (1=1",  # type: ignore[call-arg]
        )


def test_trial_id_is_validated_before_the_append_only_ledger_is_created(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = TrialLedger(path)

    with pytest.raises(QLabError, match=r"\^\(FT\|AT\)-\\d\{4\}\$"):
        ledger.charge(
            TrialCharge(
                trial_id="FT-99",
                family="pit_family",
                asset="usdcop",
                cluster="ml_meta",
                kind="forecast",
                variant="ridge",
                cutoff="2025-01-01T00:00:00Z",
            )
        )

    assert not path.exists()


def test_trial_ledger_rejects_subsecond_maximum_after_cutoff(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = TrialLedger(path)

    with pytest.raises(QLabError, match="max_available_at cannot exceed cutoff"):
        ledger.charge(
            TrialCharge(
                trial_id="FT-9001",
                family="pit_family",
                asset="usdcop",
                cluster="ml_meta",
                kind="forecast",
                variant="ridge",
                cutoff="2025-01-01T00:00:00Z",
                source="evidence.jsonl",
                data_hash="sha256:" + "0" * 64,
                available_at_field="available_at",
                n_rows=1,
                max_available_at="2025-01-01T00:00:00.000001Z",
            )
        )

    assert not path.exists()


def test_text_source_digest_is_canonical_across_lf_and_crlf(tmp_path: Path) -> None:
    cli = _load_cli()
    lf = tmp_path / "lf.jsonl"
    crlf = tmp_path / "crlf.jsonl"
    payload = (
        b'{"id":"one","available_at":"2024-12-30T00:00:00Z"}\n'
        b'{"id":"two","available_at":"2024-12-31T00:00:00Z"}\n'
    )
    lf.write_bytes(payload)
    crlf.write_bytes(payload.replace(b"\n", b"\r\n"))

    assert cli._canonical_source_digest(lf) == cli._canonical_source_digest(crlf)


def test_date_only_cutoff_uses_the_asset_session_timezone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_cli()
    profiles = tmp_path / "config" / "assets"
    profiles.mkdir(parents=True)
    (profiles / "eastasset.yaml").write_text(
        "asset_id: eastasset\nsession:\n  timezone: Asia/Tokyo\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "ROOT", tmp_path)

    assert cli._normalize_cutoff("2025-01-01", asset="eastasset") == (
        "2025-01-01T14:59:59.999999Z"
    )


def test_screen_parser_rejects_operator_selected_audit_inputs() -> None:
    cli = _load_cli()
    base = [
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
        "evidence.jsonl",
    ]

    with pytest.raises(SystemExit):
        cli._parser().parse_args([*base, "--data-hash", "sha256:" + "0" * 64])
    with pytest.raises(SystemExit):
        cli._parser().parse_args([*base, "--available-at-field", "published_at"])


def test_screen_rejects_asset_that_differs_from_family_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_cli()
    families = FamilyStore(tmp_path / "families")
    ledger_path = tmp_path / "ledger.jsonl"
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
    monkeypatch.setattr(cli, "FAMILIES", families)
    monkeypatch.setattr(cli, "LEDGER", TrialLedger(ledger_path))

    with pytest.raises(
        SystemExit, match="screening asset spx500 does not match family asset usdcop"
    ):
        cli.main(
            [
                "screen",
                "pit_family",
                "--charge-trial",
                "--trial-id",
                "FT-9001",
                "--asset",
                "spx500",
                "--variant",
                "ridge",
                "--cutoff",
                "2025-01-01",
                "--source",
                "does-not-exist.jsonl",
            ]
        )

    assert not ledger_path.exists()


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
    with pytest.raises(PointInTimeViolation, match="2025-06-01.*2025-01-02"):
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
    assert rows[0]["cutoff"] == "2025-01-02T04:59:59.999999Z"
    assert rows[0]["source"] == str(evidence)
    assert rows[0]["data_hash"] == cli._canonical_source_digest(evidence)
    assert rows[0]["available_at_field"] == "available_at"
    assert rows[0]["n_rows"] == 1
    assert rows[0]["max_available_at"] == "2024-12-31T12:00:00Z"
    assert families.load("pit_family")["state"] == "SCREENING"
