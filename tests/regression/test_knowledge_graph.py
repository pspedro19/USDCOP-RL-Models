"""The governed knowledge base is reachable and the Obsidian vault stays scoped.

Contract: CTR-KNOWLEDGE-GRAPH-001
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.validation import check_knowledge_graph as graph

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _minimal_obsidian(root: Path, *, ignored: set[str] | None = None) -> None:
    required = set(graph.REQUIRED_IGNORE_FILTERS)
    if ignored is not None:
        required = ignored
    _write(
        root / ".obsidian" / "app.json",
        json.dumps(
            {
                "alwaysUpdateLinks": True,
                "newLinkFormat": "relative",
                "useMarkdownLinks": True,
                "showUnsupportedFiles": False,
                "userIgnoreFilters": sorted(required),
            }
        ),
    )
    _write(
        root / ".obsidian" / "graph.json",
        json.dumps({"showOrphans": True, "hideUnresolved": True}),
    )


def test_markdown_links_connect_notes_but_code_spans_do_not(tmp_path: Path) -> None:
    _minimal_obsidian(tmp_path)
    _write(
        tmp_path / ".claude" / "README.md",
        "[range [0, 1]](specs/linked.md)\n`specs/not-linked.md`\n",
    )
    _write(tmp_path / ".claude" / "specs" / "linked.md", "# Linked\n")
    _write(tmp_path / ".claude" / "specs" / "not-linked.md", "# Not linked\n")

    report = graph.analyze(tmp_path)

    assert report.orphans == (".claude/specs/not-linked.md",)
    assert report.unreachable == (".claude/specs/not-linked.md",)


def test_reachability_rejects_a_connected_island(tmp_path: Path) -> None:
    _minimal_obsidian(tmp_path)
    _write(tmp_path / ".claude" / "README.md", "[main](specs/main.md)\n")
    _write(tmp_path / ".claude" / "specs" / "main.md", "# Main\n")
    _write(tmp_path / ".claude" / "specs" / "island-a.md", "[B](island-b.md)\n")
    _write(tmp_path / ".claude" / "specs" / "island-b.md", "[A](island-a.md)\n")

    report = graph.analyze(tmp_path)

    assert report.orphans == ()
    assert report.unreachable == (
        ".claude/specs/island-a.md",
        ".claude/specs/island-b.md",
    )


def test_runtime_archive_and_raw_evidence_are_outside_live_graph(tmp_path: Path) -> None:
    _minimal_obsidian(tmp_path)
    _write(tmp_path / ".claude" / "README.md", "# Entry\n")
    excluded = (
        ".claude/coordination/CODEX-STATUS.md",
        ".claude/coordination/reviews/BL-01.md",
        ".claude/coordination/database/live-check.md",
        ".claude/coordination/.claude/agents/reviewer.md",
        ".claude/coordination/.pytest-isolated/.claude/specs/copied.md",
        ".claude/specs/archive/2026-07/old.md",
        ".claude/codex/evidence/run/error-context.md",
        ".claude/evidence/run/report.md",
    )
    for rel in excluded:
        _write(tmp_path / rel, "# Runtime\n")

    docs = {p.relative_to(tmp_path).as_posix() for p in graph.discover_documents(tmp_path)}

    assert docs == {".claude/README.md"}


def test_obsidian_config_fails_closed_on_scope_and_graph_hygiene(tmp_path: Path) -> None:
    _minimal_obsidian(tmp_path, ignored={"node_modules/"})

    problems = graph.check_obsidian_config(tmp_path)

    assert any("userIgnoreFilters" in item for item in problems)
    assert any("src/" in item for item in problems)


def test_repository_knowledge_graph_is_green() -> None:
    report = graph.analyze(graph.ROOT)
    config_problems = graph.check_obsidian_config(graph.ROOT)

    assert not config_problems, "\n".join(config_problems)
    assert not report.orphans, "orphan notes:\n  - " + "\n  - ".join(report.orphans)
    assert not report.unreachable, (
        "notes unreachable from a knowledge entry point:\n  - "
        + "\n  - ".join(report.unreachable)
    )
