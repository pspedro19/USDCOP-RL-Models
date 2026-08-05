from __future__ import annotations

import importlib.util
import ast
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.lineage.graph import RevisionType
from src.lineage.macro_revision import MacroRevisionWriter


class RecordingCursor:
    def __init__(self, prior_rows: list[tuple[object, ...]]) -> None:
        self.prior_rows = prior_rows
        self.executions: list[tuple[str, object]] = []
        self._last_query = ""
        self._node_ids: dict[str, str] = {}

    def execute(self, query: str, params: object = None) -> None:
        self._last_query = " ".join(query.split())
        self.executions.append((self._last_query, params))

    def fetchall(self) -> list[tuple[object, ...]]:
        return self.prior_rows

    def fetchone(self) -> tuple[str]:
        assert "INSERT INTO lineage.node" in self._last_query
        params = self.executions[-1][1]
        assert isinstance(params, tuple)
        digest = str(params[1])
        node_id = self._node_ids.setdefault(digest, f"node-{len(self._node_ids) + 1}")
        return (node_id,)


def _writer() -> MacroRevisionWriter:
    return MacroRevisionWriter(schema="public")


def _event_time() -> datetime:
    return datetime(2026, 8, 5, 14, 30, tzinfo=timezone.utc)


def _load_upsert_module():
    path = Path(__file__).parents[2] / "airflow" / "dags" / "services" / "upsert_service.py"
    spec = importlib.util.spec_from_file_location("test_macro_upsert_service", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _macro_dag_tree() -> ast.Module:
    path = Path(__file__).parents[2] / "airflow" / "dags" / "l0_macro_update.py"
    return ast.parse(path.read_text(encoding="utf-8"))


def test_changed_existing_observation_reads_before_writing_and_records_correction() -> None:
    cursor = RecordingCursor([(date(2026, 8, 4), Decimal("100.25"))])

    result = _writer().record_before_upsert(
        cursor,
        table="macro_indicators_daily",
        date_column="fecha",
        rows=[(date(2026, 8, 4), Decimal("100.50"))],
        columns=["dxy"],
        revision_type=RevisionType.PROVIDER_CORRECTION,
        actor="core_l0_04_macro_update",
        run_id="scheduled__2026-08-05",
        event_time=_event_time(),
    )

    assert result.revisions_recorded == 1
    assert result.nodes_recorded == 2
    assert "FOR UPDATE" in cursor.executions[0][0]
    node = next(q for q in cursor.executions if "INSERT INTO lineage.node" in q[0])
    assert "last_verified_at" in node[0]
    assert "GREATEST" in node[0]
    assert _event_time() in node[1]
    event = next(q for q in cursor.executions if "lineage.revision_event" in q[0])
    assert event[1][2] == "PROVIDER_CORRECTION"
    assert event[1][3] == "latest_revised"
    edge = next(q for q in cursor.executions if "INSERT INTO lineage.edge" in q[0])
    assert edge[1][2] == "CORRECTED_BY"


def test_legitimate_release_creates_new_snapshot_without_correction_edge() -> None:
    cursor = RecordingCursor([(date(2026, 7, 1), Decimal("4.10"))])

    result = _writer().record_before_upsert(
        cursor,
        table="macro_indicators_monthly",
        date_column="fecha",
        rows=[(date(2026, 7, 1), Decimal("4.20"))],
        columns=["polr_fed_funds_usa_m_fedfunds"],
        revision_type=RevisionType.LEGITIMATE_RELEASE,
        actor="core_l0_04_macro_update",
        run_id="manual__alfred_vintage",
        event_time=_event_time(),
    )

    assert result.revisions_recorded == 1
    event = next(q for q in cursor.executions if "lineage.revision_event" in q[0])
    assert event[1][2] == "LEGITIMATE_RELEASE"
    edge = next(q for q in cursor.executions if "INSERT INTO lineage.edge" in q[0])
    assert edge[1][2] == "SUPERSEDES"


def test_unchanged_rerun_is_idempotent_and_emits_no_revision() -> None:
    cursor = RecordingCursor([(date(2026, 8, 4), Decimal("100.2500"))])

    result = _writer().record_before_upsert(
        cursor,
        table="macro_indicators_daily",
        date_column="fecha",
        rows=[(date(2026, 8, 4), 100.25)],
        columns=["dxy"],
        revision_type=RevisionType.PROVIDER_CORRECTION,
        actor="core_l0_04_macro_update",
        run_id="scheduled__2026-08-05",
        event_time=_event_time(),
    )

    assert result.revisions_recorded == 0
    assert result.nodes_recorded == 1
    assert not any("lineage.revision_event" in q for q, _ in cursor.executions)


def test_new_observation_creates_a_real_node_but_not_a_revision_event() -> None:
    cursor = RecordingCursor([])

    result = _writer().record_before_upsert(
        cursor,
        table="macro_indicators_daily",
        date_column="fecha",
        rows=[(date(2026, 8, 5), Decimal("101.0"))],
        columns=["dxy"],
        revision_type=RevisionType.PROVIDER_CORRECTION,
        actor="core_l0_04_macro_update",
        run_id="scheduled__2026-08-05",
        event_time=_event_time(),
    )

    assert result.revisions_recorded == 0
    assert result.nodes_recorded == 1
    assert not any("lineage.revision_event" in q for q, _ in cursor.executions)


@pytest.mark.parametrize("identifier", ["macro;DROP TABLE lineage.node", "dxy--"])
def test_sql_identifiers_are_fail_closed(identifier: str) -> None:
    with pytest.raises(ValueError, match="unsafe SQL identifier"):
        _writer().record_before_upsert(
            RecordingCursor([]),
            table=identifier,
            date_column="fecha",
            rows=[],
            columns=["dxy"],
            revision_type=RevisionType.PROVIDER_CORRECTION,
            actor="test",
            run_id="test",
            event_time=_event_time(),
        )


def test_event_time_must_be_timezone_aware() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _writer().record_before_upsert(
            RecordingCursor([]),
            table="macro_indicators_daily",
            date_column="fecha",
            rows=[],
            columns=["dxy"],
            revision_type=RevisionType.PROVIDER_CORRECTION,
            actor="test",
            run_id="test",
            event_time=datetime(2026, 8, 5, 9, 30),
        )


def test_upsert_service_commits_lineage_and_values_in_one_transaction(monkeypatch) -> None:
    module = _load_upsert_module()
    events: list[str] = []

    class Cursor:
        def close(self) -> None:
            events.append("close")

    class Connection:
        def cursor(self) -> Cursor:
            return Cursor()

        def commit(self) -> None:
            events.append("commit")

        def rollback(self) -> None:
            events.append("rollback")

    class Writer:
        def __init__(self, *, schema: str) -> None:
            assert schema == "public"

        def record_before_upsert(self, cursor, **kwargs):
            events.append("lineage")
            assert kwargs["revision_type"] is RevisionType.PROVIDER_CORRECTION
            return SimpleNamespace(nodes_recorded=2, revisions_recorded=1)

    import psycopg2.extras

    monkeypatch.setattr(module, "MacroRevisionWriter", Writer)
    monkeypatch.setattr(
        psycopg2.extras,
        "execute_batch",
        lambda cursor, query, data, page_size: events.append("values"),
    )
    service = module.UpsertService(Connection(), "macro_indicators_daily")
    result = service.upsert_last_n(
        pd.DataFrame({"fecha": [date(2026, 8, 4)], "dxy": [100.5]}),
        ["dxy"],
        revision_type=RevisionType.PROVIDER_CORRECTION,
        actor="core_l0_04_macro_update",
        run_id="scheduled__2026-08-05",
        event_time=_event_time(),
    )

    assert result["success"] is True
    assert result["lineage_nodes"] == 2
    assert result["revision_events"] == 1
    assert events == ["lineage", "values", "commit", "close"]


def test_upsert_service_rolls_back_values_when_lineage_fails(monkeypatch) -> None:
    module = _load_upsert_module()
    events: list[str] = []

    class Cursor:
        def close(self) -> None:
            events.append("close")

    class Connection:
        def cursor(self):
            return Cursor()

        def commit(self) -> None:
            events.append("commit")

        def rollback(self) -> None:
            events.append("rollback")

    class Writer:
        def __init__(self, *, schema: str) -> None:
            pass

        def record_before_upsert(self, cursor, **kwargs):
            raise RuntimeError("lineage unavailable")

    monkeypatch.setattr(module, "MacroRevisionWriter", Writer)
    service = module.UpsertService(Connection(), "macro_indicators_daily")
    result = service.upsert_last_n(
        pd.DataFrame({"fecha": [date(2026, 8, 4)], "dxy": [100.5]}),
        ["dxy"],
        revision_type=RevisionType.PROVIDER_CORRECTION,
        actor="core_l0_04_macro_update",
        run_id="scheduled__2026-08-05",
        event_time=_event_time(),
    )

    assert result["success"] is False
    assert "lineage unavailable" in result["error"]
    assert events == ["rollback", "close"]


def test_macro_dag_passes_declared_revision_context_to_every_variable_upsert() -> None:
    tree = _macro_dag_tree()
    upsert_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "upsert_all"
    )
    calls = [
        node
        for node in ast.walk(upsert_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "upsert_variable"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg for keyword in calls[0].keywords}
    assert {"revision_type", "actor", "run_id", "event_time"} <= keywords

    assignments = {
        node.targets[0].id: node.value
        for node in ast.walk(upsert_function)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    default_call = assignments["raw_revision_type"]
    assert isinstance(default_call, ast.Call)
    assert isinstance(default_call.func, ast.Attribute)
    assert default_call.func.attr == "get"
    assert isinstance(default_call.args[0], ast.Constant)
    assert default_call.args[0].value == "macro_revision_type"
