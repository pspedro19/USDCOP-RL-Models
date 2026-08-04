from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
WRITER = ROOT / "scripts" / "data" / "ingest_asset_ohlcv.py"


def _function(source: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _calls(function: ast.FunctionDef) -> list[str]:
    names: list[str] = []
    for call in ast.walk(function):
        if not isinstance(call, ast.Call):
            continue
        if isinstance(call.func, ast.Name):
            names.append(call.func.id)
        elif isinstance(call.func, ast.Attribute):
            names.append(call.func.attr)
    return names


def _call_nodes(function: ast.FunctionDef, name: str) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
        and (
            (isinstance(call.func, ast.Name) and call.func.id == name)
            or (isinstance(call.func, ast.Attribute) and call.func.attr == name)
        )
    ]


def test_writer_calls_fabric_before_legacy_and_owns_one_transaction() -> None:
    source = WRITER.read_text(encoding="utf-8")
    run = _function(source, "run")
    calls = _calls(run)

    assert calls.count("_publish_fabric_frame") == 2
    assert calls.count("_upsert") == 2
    assert calls.count("commit") == 1
    assert calls.count("rollback") == 1
    publications = sorted(call.lineno for call in _call_nodes(run, "_publish_fabric_frame"))
    upserts = sorted(call.lineno for call in _call_nodes(run, "_upsert"))
    assert all(publication < upsert for publication, upsert in zip(publications, upserts))
    commit = _call_nodes(run, "commit")
    upsert_calls = _call_nodes(run, "_upsert")
    assert len(commit) == 1 and len(upsert_calls) == 2
    assert commit[0].lineno > max(call.lineno for call in upsert_calls)
    assert all(
        len(call.args) >= 3
        and isinstance(call.args[2], ast.Name)
        and call.args[2].id == "accepted"
        for call in upsert_calls
    )


def test_writer_no_longer_turns_database_failure_into_warning_or_summary() -> None:
    source = WRITER.read_text(encoding="utf-8")
    run = _function(source, "run")
    run_source = ast.get_source_segment(source, run) or ""
    db_handler = next(
        handler
        for node in ast.walk(run)
        if isinstance(node, ast.Try)
        for handler in node.handlers
        if any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "rollback"
            for call in ast.walk(handler)
        )
    )

    assert 'summary["db_error"]' not in run_source
    assert "DB step skipped/failed" not in run_source
    assert "Atomic DB publication failed" in run_source
    assert any(isinstance(node, ast.Raise) for node in ast.walk(db_handler))
    assert not any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "warning"
        for call in ast.walk(db_handler)
    )


def test_legacy_upsert_does_not_commit_or_rollback_behind_callers_back() -> None:
    source = WRITER.read_text(encoding="utf-8")
    upsert = _function(source, "_upsert")
    calls = _calls(upsert)

    assert "commit" not in calls
    assert "rollback" not in calls


def test_run_executes_fabric_and_sends_only_accepted_rows_to_legacy(monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location("bl40_ingest_writer", WRITER)
    assert spec and spec.loader
    writer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = writer
    spec.loader.exec_module(writer)

    frame = pd.DataFrame(
        [
            {
                "time": pd.Timestamp("2026-08-03T13:00:00Z"),
                "open": 4.0,
                "high": 4.1,
                "low": 3.9,
                "close": 4.05,
                "volume": 10.0,
            },
            {
                "time": pd.Timestamp("2026-08-03T13:05:00Z"),
                "open": 400.0,
                "high": 401.0,
                "low": 399.0,
                "close": 400.5,
                "volume": 10.0,
            },
        ]
    )
    accepted = frame.iloc[[0]].copy()
    profile = SimpleNamespace(
        symbol="USD/COP",
        display_name="USD/COP",
        safe_name="usdcop",
        session=SimpleNamespace(mode="exchange_hours", timezone="America/Bogota"),
        data_source=SimpleNamespace(
            interval="5min",
            provider="twelvedata",
            provider_symbol="USD/COP",
            seed_file="unused.parquet",
        ),
        raw={"data_source": {}},
    )

    class FakeConnection:
        committed = False
        rolled_back = False
        closed = False

        def commit(self) -> None:
            self.committed = True

        def rollback(self) -> None:
            self.rolled_back = True

        def close(self) -> None:
            self.closed = True

    connection = FakeConnection()
    publications: list[pd.DataFrame] = []
    legacy_inputs: list[pd.DataFrame] = []

    monkeypatch.setattr(writer, "_load_env", lambda: None)
    monkeypatch.setattr(writer, "_load_asset_profile", lambda _asset_id: profile)
    monkeypatch.setattr(writer, "_api_keys", lambda: ["test-key"])
    monkeypatch.setattr(writer, "_paginate_back", lambda *_args, **_kwargs: frame.copy())
    monkeypatch.setattr(writer, "_filter_session", lambda value, _profile: value)
    monkeypatch.setattr(writer, "_clean", lambda value: value)
    monkeypatch.setattr(writer, "_audit", lambda *_args: {"status": "test"})
    monkeypatch.setattr(writer, "_gate_seed", lambda *_args: None)
    monkeypatch.setattr(writer, "_write_seed", lambda *_args: None)
    monkeypatch.setattr(writer, "_to_seed_schema", lambda value, _symbol: value)
    monkeypatch.setattr(writer, "_db_conn", lambda: connection)

    def publish(_conn, value, **_kwargs):
        publications.append(value.copy())
        return accepted.copy(), {"raw": 2, "canonical": 1, "quarantined": 1}

    def upsert(_conn, _table, value, _symbol, _source, **_kwargs):
        legacy_inputs.append(value.copy())
        return len(value), 0

    monkeypatch.setattr(writer, "_publish_fabric_frame", publish)
    monkeypatch.setattr(writer, "_upsert", upsert)

    summary = writer.run(
        "usdcop",
        use_db=True,
        skip_intraday=False,
        skip_daily=True,
        daily_start="2020-01-01",
        intraday_calls=1,
        daily_calls=1,
    )

    assert len(publications) == 1
    pd.testing.assert_frame_equal(publications[0], frame)
    assert len(legacy_inputs) == 1
    pd.testing.assert_frame_equal(legacy_inputs[0], accepted)
    assert summary["db_m5_upserted"] == 1
    assert connection.committed and connection.closed and not connection.rolled_back
