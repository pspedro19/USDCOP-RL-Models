from __future__ import annotations

from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
BL19 = REPO / ".claude/specs/planes/backlog/BL-19-schema-forecast-roles-db.md"


def _frontmatter() -> dict[str, object]:
    text = BL19.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    raw = text.split("---\n", 2)[1]
    parsed = yaml.safe_load(raw)
    assert isinstance(parsed, dict)
    return parsed


def test_bl19_anchors_the_actual_forecast_wall_and_migrator() -> None:
    """The BL must not drift back to unrelated pre-FABRIC migrations 067/068."""
    frontmatter = _frontmatter()
    anchors = set(frontmatter.get("code_anchors") or [])

    assert anchors == {
        "database/migrations/071_forecast_schema_roles.sql",
        "scripts/ops/db_migrate.py",
        "Makefile",
    }
    assert frontmatter["status"] == "PARTIAL"
    for relative in anchors:
        assert (REPO / relative).is_file(), f"missing BL-19 anchor: {relative}"
