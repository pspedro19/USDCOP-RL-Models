from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
BL18 = ROOT / ".claude/specs/planes/backlog/BL-18-catalogo-motor-metricas.md"
EXPECTED_ANCHORS = {
    "config/metrics/catalog.yaml",
    "config/metrics/legacy_bypass_allowlist.yaml",
    "src/metrics/engine.py",
    "src/metrics/persistence.py",
    "database/migrations/070_fabric_control_plane.sql",
    "airflow/dags/forecast_h5_l6_weekly_monitor.py",
    "airflow/dags/control_system_health.py",
}


def _document() -> tuple[dict[str, object], str]:
    text = BL18.read_text(encoding="utf-8")
    _, raw_frontmatter, body = text.split("---", 2)
    return yaml.safe_load(raw_frontmatter), body


def test_bl18_current_state_names_live_callers_and_remaining_gaps() -> None:
    frontmatter, body = _document()
    current = body.split("## Estado PostgreSQL posterior a Fabric", 1)[1]

    assert frontmatter["status"] == "PARTIAL"
    assert set(frontmatter["code_anchors"]) == EXPECTED_ANCHORS
    assert all((ROOT / anchor).is_file() for anchor in EXPECTED_ANCHORS)
    assert "persist_governed_metric_events" in current
    assert "control_system_health.py" in current
    assert "FROM control.metric_event" in current
    assert "allowlist" in current
    assert "cobertura" in current
    assert "ON CONFLICT" in current
    assert "Todavía no existe un productor y consumidor productivos" not in current


def test_bl18_preserves_the_pre_fabric_blocker_as_history() -> None:
    _, body = _document()
    historical = body.split("## Bloqueo de cableado medido (2026-08-03)", 1)[1]
    historical = historical.split("## Estado PostgreSQL posterior a Fabric", 1)[0]

    normalized = " ".join(historical.split())
    assert "el esquema `control` no existe" in normalized
    assert "cero llamadores productivos" in normalized
