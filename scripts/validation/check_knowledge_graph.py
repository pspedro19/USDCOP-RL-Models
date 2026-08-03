"""Validate the live knowledge graph and its Obsidian vault boundary.

Contract: CTR-KNOWLEDGE-GRAPH-001

`check_knowledge_links.py` proves that links which already exist resolve. It cannot detect
the opposite failure mode: a valid note that nobody links, a small connected island with no
route from an entry point, or code-adjacent READMEs leaking into the vault. This gate covers
those three cases without treating runtime logs, raw evidence, or archives as live memory.

Usage:
    python scripts/validation/check_knowledge_graph.py
    python scripts/validation/check_knowledge_graph.py --verbose
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts.validation.knowledge_markdown import iter_relative_links
except ModuleNotFoundError:  # Direct execution: python scripts/validation/<script>.py
    from knowledge_markdown import iter_relative_links

ROOT = Path(__file__).resolve().parents[2]

ROOT_DOCUMENTS = ("README.md", "CLAUDE.md", "AGENTS.md")
ENTRY_POINTS = frozenset(
    {
        "README.md",
        "CLAUDE.md",
        "AGENTS.md",
        ".claude/README.md",
        "docs/INDEX.md",
    }
)

# These are implementation/runtime trees, not the governed Markdown memory surface.
# Paths are exactly the forms stored by Obsidian in app.json.
REQUIRED_IGNORE_FILTERS = frozenset(
    {
        "node_modules/",
        "vendor/",
        "usdcop-trading-dashboard/",
        "services/",
        "tests/",
        "src/",
        "scripts/",
        "airflow/",
        "config/",
        ".github/",
        ".git/",
        "database/",
        "data/",
        "models/",
        "outputs/",
        "reports/",
        "registries/",
        "results/",
        "seeds/",
        "secrets/",
        "presentation/",
        "design-system/",
        ".claude/coordination/tmp/",
        ".claude/coordination/monitor/",
        ".claude/coordination/database/",
        ".claude/coordination/briefs/",
        ".claude/coordination/reviews/",
        ".claude/coordination/integration/",
        ".claude/coordination/.agents/",
        ".claude/coordination/.claude/",
        ".claude/coordination/.codex/",
        ".claude/coordination/.git/",
        ".claude/coordination/.pytest-",
        ".claude/codex/_runtime/",
        ".claude/codex/evidence/",
        ".claude/evidence/",
        ".claude/specs/archive/",
        ".claude/coordination/CLAUDE-STATUS.md",
        ".claude/coordination/CODEX-STATUS.md",
        ".claude/coordination/INBOX-CLAUDE.md",
        ".claude/coordination/INBOX-CODEX.md",
        ".claude/coordination/LEASES.md",
        ".claude/coordination/CONTRACTS.md",
        ".claude/coordination/KNOWLEDGE.md",
        ".claude/coordination/PROGRESS.md",
        ".claude/coordination/BASELINE.md",
        ".claude/coordination/INTEGRATION-AUDIT.md",
        ".claude/coordination/PROTOCOL-COMMS-v2.md",
    }
)

EXCLUDED_PREFIXES = tuple(
    sorted(
        {
            value
            for value in REQUIRED_IGNORE_FILTERS
            if value.startswith(".claude/") and value != ".claude/codex/_runtime/"
        }
        | {".claude/codex/_runtime/"}
    )
)

EPHEMERAL_PARTS = frozenset(
    {"tmp", "node_modules", ".next", "__pycache__", ".pytest_cache", "_runtime"}
)
EPHEMERAL_PREFIXES = (".pytest-",)

@dataclass(frozen=True)
class GraphReport:
    documents: tuple[str, ...]
    edges: int
    orphans: tuple[str, ...]
    unreachable: tuple[str, ...]


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_excluded(root: Path, path: Path) -> bool:
    rel = _relative(root, path)
    if any(rel == prefix.rstrip("/") or rel.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return True
    return any(
        part in EPHEMERAL_PARTS or part.startswith(EPHEMERAL_PREFIXES)
        for part in path.relative_to(root).parts
    )


def discover_documents(root: Path = ROOT) -> list[Path]:
    """Return the Markdown notes that form live, governed memory."""
    documents: list[Path] = []
    for base in (root / ".claude", root / "docs"):
        if not base.is_dir():
            continue
        # Prune runtime worktrees before descending. Filtering the result of
        # ``Path.rglob`` is too late: coordination/tmp may contain whole repository
        # copies, making a static knowledge gate walk thousands of irrelevant notes.
        for directory, dirnames, filenames in os.walk(base, topdown=True):
            parent = Path(directory)
            dirnames[:] = [
                name
                for name in dirnames
                if not _is_excluded(root, parent / name)
            ]
            documents.extend(
                parent / name
                for name in filenames
                if name.lower().endswith(".md")
                and not _is_excluded(root, parent / name)
            )
    documents.extend(
        root / name for name in ROOT_DOCUMENTS if (root / name).is_file()
    )
    return sorted(set(documents))


def _targets(path: Path) -> list[str]:
    """Extract real Markdown-link targets, ignoring examples in code."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return [target for _, target in iter_relative_links(text)]


def _resolve_target(source: Path, target: str, documents: dict[Path, str]) -> Path | None:
    try:
        candidate = (source.parent / target).resolve()
    except (OSError, RuntimeError):
        return None
    if candidate in documents:
        return candidate
    # Obsidian accepts extensionless Markdown destinations.
    if not candidate.suffix:
        markdown = candidate.with_suffix(".md")
        if markdown in documents:
            return markdown
    return None


def analyze(root: Path = ROOT) -> GraphReport:
    paths = discover_documents(root)
    rel_by_path = {path.resolve(): _relative(root, path) for path in paths}
    adjacency: dict[Path, set[Path]] = {path: set() for path in rel_by_path}

    for source in sorted(adjacency):
        for raw_target in _targets(source):
            target = _resolve_target(source, raw_target, rel_by_path)
            if target is None or target == source:
                continue
            adjacency[source].add(target)
            adjacency[target].add(source)

    roots = [
        path
        for path, rel in rel_by_path.items()
        if rel in ENTRY_POINTS
    ]
    reached: set[Path] = set(roots)
    queue: deque[Path] = deque(roots)
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current] - reached:
            reached.add(neighbor)
            queue.append(neighbor)

    orphans = tuple(
        sorted(rel_by_path[path] for path, neighbors in adjacency.items() if not neighbors)
    )
    unreachable = tuple(
        sorted(rel_by_path[path] for path in adjacency if path not in reached)
    )
    edges = sum(len(neighbors) for neighbors in adjacency.values()) // 2
    return GraphReport(
        documents=tuple(sorted(rel_by_path.values())),
        edges=edges,
        orphans=orphans,
        unreachable=unreachable,
    )


def _load_json(path: Path, label: str, problems: list[str]) -> dict:
    if not path.is_file():
        problems.append(f"missing {label}: {_relative(path.parents[1], path)}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"invalid {label}: {exc}")
        return {}
    if not isinstance(value, dict):
        problems.append(f"{label} must contain a JSON object")
        return {}
    return value


def check_obsidian_config(root: Path = ROOT) -> tuple[str, ...]:
    """Fail closed when Obsidian can leak code/runtime files into the graph."""
    problems: list[str] = []
    app = _load_json(root / ".obsidian" / "app.json", "Obsidian app config", problems)
    graph = _load_json(root / ".obsidian" / "graph.json", "Obsidian graph config", problems)

    expected = {
        "alwaysUpdateLinks": True,
        "newLinkFormat": "relative",
        "useMarkdownLinks": True,
        "showUnsupportedFiles": False,
    }
    for key, value in expected.items():
        if app.get(key) != value:
            problems.append(f"app.json: {key} must be {value!r}")

    configured = {
        str(value).replace("\\", "/")
        for value in app.get("userIgnoreFilters", [])
        if isinstance(value, str)
    }
    missing = sorted(REQUIRED_IGNORE_FILTERS - configured)
    if missing:
        problems.append(
            "app.json userIgnoreFilters missing governed exclusions: " + ", ".join(missing)
        )

    # Keep this visible: concealing isolated notes makes the graph look healthy without
    # making the memory reachable. The gate should drive the count to zero.
    if graph.get("showOrphans") is not True:
        problems.append("graph.json: showOrphans must remain true; do not hide graph debt")
    if graph.get("hideUnresolved") is not True:
        problems.append("graph.json: hideUnresolved must be true (existing notes only)")

    return tuple(problems)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config_problems = check_obsidian_config(ROOT)
    report = analyze(ROOT)

    if args.verbose:
        print(
            f"live graph: {len(report.documents)} notes, {report.edges} edges, "
            f"{len(report.orphans)} orphans, {len(report.unreachable)} unreachable"
        )

    if config_problems:
        print(f"OBSIDIAN CONFIG ({len(config_problems)}):", file=sys.stderr)
        for problem in config_problems:
            print(f"  - {problem}", file=sys.stderr)
    if report.orphans:
        print(f"ORPHAN NOTES ({len(report.orphans)}):", file=sys.stderr)
        for path in report.orphans:
            print(f"  - {path}", file=sys.stderr)
    if report.unreachable:
        print(
            f"UNREACHABLE FROM ENTRY POINTS ({len(report.unreachable)}):",
            file=sys.stderr,
        )
        for path in report.unreachable:
            print(f"  - {path}", file=sys.stderr)

    if config_problems or report.orphans or report.unreachable:
        return 1
    print(f"knowledge graph OK ({len(report.documents)} notes, {report.edges} edges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
