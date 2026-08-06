"""Generate minimal, connected directory indexes for the live knowledge graph.

Contract: CTR-KNOWLEDGE-INDEX-001

The repository is also an Obsidian vault, but source/runtime trees are not memory. This
generator walks only the governed Markdown surface, prunes excluded directories before
descending, and creates an index only when a directory has multiple navigation entries.
A singleton is linked directly from its nearest indexed ancestor, avoiding README sprawl.

Usage:
    python scripts/diagnostics/generate_doc_indexes.py --write
    python scripts/diagnostics/generate_doc_indexes.py --check
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

try:
    from scripts.validation.knowledge_markdown import iter_relative_links
except ModuleNotFoundError:  # Direct execution: python scripts/diagnostics/<script>.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
    from knowledge_markdown import iter_relative_links

ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_ROOT_NAMES = (".claude", "docs")
INDEX_NAMES = ("README.md", "INDEX.md", "00-INDEX.md", "00_INDEX.md")

# These prefixes contain runtime, archives, executable definitions, or raw evidence.
# Skills/agents are connected by the generated capability catalog in `.claude/README.md`;
# generating README files inside those directories would break their definition harness.
EXCLUDED_PATH_PREFIXES = (
    ".claude/agents",
    ".claude/archive",
    ".claude/codex/_runtime",
    ".claude/codex/evidence",
    ".claude/coordination",
    ".claude/evidence",
    ".claude/generated",
    ".claude/skills",
    ".claude/specs/archive",
)
EXCLUDED_PARTS = frozenset(
    {"tmp", "node_modules", ".next", ".git", "__pycache__", ".pytest_cache", "_runtime"}
)
EXCLUDED_PART_PREFIXES = (".pytest-",)

OPEN = "<!-- idx:auto -->"
CLOSE = "<!-- /idx -->"
FILE_MARKER = "<!-- idx:file-generated -->"
BLOCK_RE = re.compile(re.escape(OPEN) + r".*?" + re.escape(CLOSE), re.S)
SECTION_RE = re.compile(
    r"\n*## Documentos de este directorio\s*\n+"
    + re.escape(OPEN)
    + r".*?"
    + re.escape(CLOSE)
    + r"\s*",
    re.S,
)
FM_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n", re.S)
H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.M)

RUNTIME_BLOCK_FILES = (
    ".claude/coordination/README.md",
    ".claude/coordination/integration/README.md",
)
RUNTIME_GENERATED_FILES = (
    ".claude/coordination/briefs/README.md",
    ".claude/coordination/reviews/README.md",
)

FRONTMATTER = """---
kind: as-built
status: IMPLEMENTED
version: 1.0.0
last_verified: {today}
supersedes: []
code_anchors:
  - scripts/diagnostics/generate_doc_indexes.py
---
"""

HEADER = """{marker}
# Índice — `{rel}`

> Índice generado desde el árbol vivo. No editar este fichero a mano:
> `python scripts/diagnostics/generate_doc_indexes.py --write`.

"""


@dataclass(frozen=True)
class DirectoryPlan:
    directory: Path
    index: Path
    direct_documents: tuple[Path, ...]
    child_anchors: tuple[tuple[Path, Path], ...]
    generated_file: bool


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _excluded(root: Path, path: Path) -> bool:
    rel = _relative(root, path)
    if any(
        rel == prefix or rel.startswith(prefix + "/")
        for prefix in EXCLUDED_PATH_PREFIXES
    ):
        return True
    return any(
        part in EXCLUDED_PARTS or part.startswith(EXCLUDED_PART_PREFIXES)
        for part in path.relative_to(root).parts
    )


def discover_documents(root: Path = ROOT) -> tuple[Path, ...]:
    """Discover live Markdown once, pruning runtime before recursion."""
    documents: list[Path] = []
    for name in KNOWLEDGE_ROOT_NAMES:
        base = root / name
        if not base.is_dir():
            continue
        for directory, dirnames, filenames in os.walk(base, topdown=True):
            parent = Path(directory)
            dirnames[:] = [
                child
                for child in dirnames
                if not _excluded(root, parent / child)
            ]
            documents.extend(
                parent / filename
                for filename in filenames
                if filename.lower().endswith(".md")
                and not _excluded(root, parent / filename)
            )
    return tuple(sorted(set(documents)))


def _is_generated_file(path: Path) -> bool:
    """Recognize only files whose entire useful body belongs to this generator."""
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    if FILE_MARKER in text:
        return True
    if (
        "scripts/diagnostics/generate_doc_indexes.py" not in text
        or "Índice **generado**" not in text
        or OPEN not in text
    ):
        return False
    body = FM_RE.sub("", text, count=1)
    body = BLOCK_RE.sub("", body, count=1)
    remaining = [
        line
        for line in body.splitlines()
        if line.strip()
        and not line.startswith("# Índice —")
        and not line.startswith("> ")
    ]
    return not remaining


def _meta(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    status = ""
    frontmatter = FM_RE.match(text)
    if frontmatter:
        match = re.search(r"^status:\s*(\S+)\s*$", frontmatter.group(1), re.M)
        if match:
            status = match.group(1)
        text = text[frontmatter.end() :]
    heading = H1_RE.search(text)
    title = heading.group(1) if heading else path.stem.replace("-", " ").replace("_", " ")
    title = re.sub(r"[`*]", "", title)
    return title.replace("|", r"\|"), status


def _curated_index(
    directory: Path,
    semantic_documents: set[Path],
) -> Path | None:
    for name in INDEX_NAMES:
        candidate = directory / name
        if candidate in semantic_documents:
            return candidate
    return None


def build_plan(
    root: Path = ROOT,
) -> tuple[tuple[DirectoryPlan, ...], tuple[Path, ...]]:
    """Plan all targets before rendering, so the first write is idempotent."""
    live_documents = set(discover_documents(root))
    owned_indexes = {path for path in live_documents if _is_generated_file(path)}
    semantic_documents = live_documents - owned_indexes

    knowledge_roots = tuple(
        root / name for name in KNOWLEDGE_ROOT_NAMES if (root / name).is_dir()
    )
    directories: set[Path] = set(knowledge_roots)
    for document in semantic_documents:
        parent = document.parent
        base = next((item for item in knowledge_roots if item == parent or item in parent.parents), None)
        if base is None:
            continue
        while True:
            directories.add(parent)
            if parent == base:
                break
            parent = parent.parent

    curated = {
        directory: index
        for directory in directories
        if (index := _curated_index(directory, semantic_documents)) is not None
    }
    index_documents = {
        path for path in semantic_documents if path.name in INDEX_NAMES
    }
    direct_documents: dict[Path, list[Path]] = defaultdict(list)
    for document in semantic_documents - index_documents:
        direct_documents[document.parent].append(document)

    children: dict[Path, list[Path]] = defaultdict(list)
    for directory in directories:
        if directory.parent in directories:
            children[directory.parent].append(directory)

    anchor_by_directory: dict[Path, Path] = {}
    entries_by_directory: dict[
        Path,
        tuple[tuple[Path, ...], tuple[tuple[Path, Path], ...]],
    ] = {}
    generated_directories: set[Path] = set()

    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        docs = tuple(sorted(direct_documents.get(directory, [])))
        child_anchors = tuple(
            sorted(
                (
                    (child, anchor_by_directory[child])
                    for child in children.get(directory, [])
                    if child in anchor_by_directory
                ),
                key=lambda item: item[0].name.lower(),
            )
        )
        entries_by_directory[directory] = (docs, child_anchors)

        if directory in curated:
            anchor_by_directory[directory] = curated[directory]
        elif len(docs) + len(child_anchors) >= 2:
            generated_directories.add(directory)
            anchor_by_directory[directory] = directory / "README.md"
        elif docs:
            anchor_by_directory[directory] = docs[0]
        elif child_anchors:
            anchor_by_directory[directory] = child_anchors[0][1]

    plans: list[DirectoryPlan] = []
    for directory in sorted(curated.keys() | generated_directories):
        docs, child_anchors = entries_by_directory[directory]
        if not docs and not child_anchors:
            continue
        plans.append(
            DirectoryPlan(
                directory=directory,
                index=(
                    directory / "README.md"
                    if directory in generated_directories
                    else curated[directory]
                ),
                direct_documents=docs,
                child_anchors=child_anchors,
                generated_file=directory in generated_directories,
            )
        )

    desired_generated = {plan.index for plan in plans if plan.generated_file}
    obsolete = set(owned_indexes - desired_generated)
    for rel in RUNTIME_GENERATED_FILES:
        candidate = root / rel
        if _is_generated_file(candidate):
            obsolete.add(candidate)
    return tuple(plans), tuple(sorted(obsolete))


def render_block(plan: DirectoryPlan) -> str:
    lines = [OPEN, ""]
    if plan.direct_documents:
        lines.extend(["| Documento | Estado |", "|---|---|"])
        for document in plan.direct_documents:
            title, status = _meta(document)
            target = document.relative_to(plan.directory).as_posix()
            lines.append(f"| [{title}]({target}) | {status or '—'} |")
        lines.append("")

    if plan.child_anchors:
        links = [
            f"[`{child.name}/`]({anchor.relative_to(plan.directory).as_posix()})"
            for child, anchor in plan.child_anchors
        ]
        lines.extend(["**Subdirectorios:** " + " · ".join(links), ""])

    lines.append(CLOSE)
    return "\n".join(lines)


def _linked_paths(index: Path, text: str) -> set[Path]:
    linked: set[Path] = set()
    for _, target in iter_relative_links(text):
        try:
            candidate = (index.parent / target).resolve()
        except (OSError, RuntimeError):
            continue
        linked.add(candidate)
        if not candidate.suffix:
            linked.add(candidate.with_suffix(".md"))
    return linked


def _already_covers(plan: DirectoryPlan, text: str) -> bool:
    expected = {
        path.resolve()
        for path in (
            *plan.direct_documents,
            *(anchor for _, anchor in plan.child_anchors),
        )
    }
    return expected <= _linked_paths(plan.index, text)


def desired_text(
    plan: DirectoryPlan,
    root: Path = ROOT,
    *,
    verified_on: date | None = None,
) -> str:
    block = render_block(plan)
    if plan.generated_file:
        rel = _relative(root, plan.directory)
        verification_date = verified_on or date.today()

        def render_generated(last_verified: str) -> str:
            return (
                FRONTMATTER.format(today=last_verified)
                + HEADER.format(marker=FILE_MARKER, rel=rel)
                + block
                + "\n"
            )

        if plan.index.is_file():
            original = plan.index.read_text(encoding="utf-8", errors="replace")
            frontmatter = FM_RE.match(original)
            if frontmatter:
                previous = re.search(
                    r"^last_verified:\s*(20\d{2}-\d{2}-\d{2})\s*$",
                    frontmatter.group(1),
                    re.M,
                )
                if previous and render_generated(previous.group(1)) == original:
                    return original
        return render_generated(verification_date.isoformat())

    original = plan.index.read_text(encoding="utf-8", errors="replace")
    if BLOCK_RE.search(original):
        return BLOCK_RE.sub(lambda _: block, original)
    if _already_covers(plan, original):
        return original
    return original.rstrip() + "\n\n## Documentos de este directorio\n\n" + block + "\n"


def strip_auto_block(text: str) -> str:
    updated = SECTION_RE.sub("\n", text)
    if updated == text:
        updated = BLOCK_RE.sub("", text)
    return updated.rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    plans, obsolete = build_plan(ROOT)
    updates: list[tuple[Path, str]] = []
    for plan in plans:
        desired = desired_text(plan, ROOT)
        current = (
            plan.index.read_text(encoding="utf-8", errors="replace")
            if plan.index.is_file()
            else ""
        )
        if current != desired:
            updates.append((plan.index, desired))

    cleanup_updates: list[tuple[Path, str]] = []
    for rel in RUNTIME_BLOCK_FILES:
        path = ROOT / rel
        if not path.is_file():
            continue
        current = path.read_text(encoding="utf-8", errors="replace")
        desired = strip_auto_block(current)
        if current != desired:
            cleanup_updates.append((path, desired))

    if args.check:
        drift = [
            *(f"stale index: {_relative(ROOT, path)}" for path, _ in updates),
            *(f"obsolete generated index: {_relative(ROOT, path)}" for path in obsolete),
            *(f"runtime contains generated block: {_relative(ROOT, path)}" for path, _ in cleanup_updates),
        ]
        if drift:
            print("DOCUMENT INDEX DRIFT", file=sys.stderr)
            for item in drift:
                print(f"  - {item}", file=sys.stderr)
            print(
                "\nFix: python scripts/diagnostics/generate_doc_indexes.py --write",
                file=sys.stderr,
            )
            return 1
        print(f"document indexes OK ({len(plans)} governed directories)")
        return 0

    for path, text in (*updates, *cleanup_updates):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"index updated: {_relative(ROOT, path)}")
    for path in obsolete:
        path.unlink()
        print(f"obsolete generated index removed: {_relative(ROOT, path)}")
    print(
        f"{len(updates) + len(cleanup_updates)} indexes updated; "
        f"{len(obsolete)} obsolete indexes removed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
