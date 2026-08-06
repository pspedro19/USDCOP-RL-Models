from __future__ import annotations

from datetime import date
from pathlib import Path

from scripts.diagnostics.generate_doc_indexes import DirectoryPlan, desired_text


def _document(path: Path, title: str) -> Path:
    path.write_text(
        f"---\nkind: guide\nstatus: PARTIAL\nversion: 1.0.0\n"
        f"last_verified: 2026-08-05\nsupersedes: []\ncode_anchors: []\n---\n\n# {title}\n",
        encoding="utf-8",
    )
    return path


def _plan(directory: Path, *documents: Path) -> DirectoryPlan:
    return DirectoryPlan(
        directory=directory,
        index=directory / "README.md",
        direct_documents=tuple(documents),
        child_anchors=(),
        generated_file=True,
    )


def test_generated_index_date_changes_only_with_semantic_content(tmp_path: Path) -> None:
    knowledge = tmp_path / "docs"
    knowledge.mkdir()
    first = _document(knowledge / "first.md", "First")

    initial = desired_text(
        _plan(knowledge, first), tmp_path, verified_on=date(2026, 8, 5)
    )
    index = knowledge / "README.md"
    index.write_text(initial, encoding="utf-8")

    unchanged_next_day = desired_text(
        _plan(knowledge, first), tmp_path, verified_on=date(2026, 8, 6)
    )
    assert unchanged_next_day == initial
    assert "last_verified: 2026-08-05" in unchanged_next_day

    second = _document(knowledge / "second.md", "Second")
    changed_next_day = desired_text(
        _plan(knowledge, first, second), tmp_path, verified_on=date(2026, 8, 6)
    )
    assert changed_next_day != initial
    assert "[Second](second.md)" in changed_next_day
    assert "last_verified: 2026-08-06" in changed_next_day
