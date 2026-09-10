from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
BL28 = ROOT / ".claude/specs/planes/backlog/BL-28-factories-diff-semantico.md"
BL17 = ROOT / ".claude/specs/planes/backlog/BL-17-fingerprints-canonical-writer.md"
EXPECTED_ANCHORS = {
    "airflow/dags/asset_pipeline_factory.py",
    "airflow/dags/fabric_factories.py",
    "config/assets/pipelines.yaml",
    "config/assets/fabric_factories.yaml",
    "src/orchestration/factories.py",
    "src/orchestration/semantic_diff.py",
}


def _document(path: Path) -> tuple[dict[str, object], str]:
    text = path.read_text(encoding="utf-8")
    _, raw_frontmatter, body = text.split("---", 2)
    return yaml.safe_load(raw_frontmatter), body


def _dependency_and_own_anchors_are_present(
    dependency_frontmatter: dict[str, object], anchors: set[str], root: Path = ROOT
) -> bool:
    return dependency_frontmatter.get("status") == "IMPLEMENTED" and all(
        (root / anchor).is_file() for anchor in anchors
    )


def test_bl28_separates_closed_bl17_from_own_remaining_work() -> None:
    frontmatter, body = _document(BL28)
    dependency_frontmatter, _ = _document(BL17)

    assert frontmatter["status"] == "PARTIAL"
    assert set(frontmatter["code_anchors"]) == EXPECTED_ANCHORS
    assert _dependency_and_own_anchors_are_present(dependency_frontmatter, EXPECTED_ANCHORS)
    assert "Resolver la dependencia BL-17" not in body
    assert "conectar los productores propios" in body
    assert "candidate_generator: null" in body
    assert "dos semanas" in body


def test_bl28_dependency_gate_rejects_stale_or_missing_inputs() -> None:
    assert not _dependency_and_own_anchors_are_present(
        {"status": "PARTIAL"}, EXPECTED_ANCHORS
    )
    assert not _dependency_and_own_anchors_are_present(
        {"status": "IMPLEMENTED"}, EXPECTED_ANCHORS - {"src/orchestration/factories.py"},
        root=ROOT / "missing-root",
    )
