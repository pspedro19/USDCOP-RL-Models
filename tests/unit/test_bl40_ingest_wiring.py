from __future__ import annotations

import ast
from pathlib import Path


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
