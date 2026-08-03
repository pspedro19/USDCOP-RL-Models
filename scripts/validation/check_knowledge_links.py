"""Validate markdown links and referenced paths across the knowledge base.

Contract: CTR-KNOWLEDGE-LINKS-001

Four broken links sat in `.claude/specs/assets/xauusd/` pointing at `../rules/` for files
that live in `specs/`, and `mlops-lifecycle.md` cross-referenced itself. Nobody noticed
because nothing checked. This is the checker.

Scope (2026-07-30): the live `.claude/**` and `docs/**` memory plus the root entry points
(`README.md`, `CLAUDE.md`, `AGENTS.md`). `docs/` was added after `docs/INDEX.md` carried 30+
dead links — a fossil from 2025-10 that nothing watched because the checker only ever
looked at `.claude/`.

Usage:
    python scripts/validation/check_knowledge_links.py
    python scripts/validation/check_knowledge_links.py --verbose
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from scripts.validation.check_knowledge_graph import discover_documents
    from scripts.validation.knowledge_markdown import iter_relative_links
except ModuleNotFoundError:  # Direct execution: python scripts/validation/<script>.py
    from check_knowledge_graph import discover_documents
    from knowledge_markdown import iter_relative_links

ROOT = Path(__file__).resolve().parents[2]  # scripts/validation/<this> -> repo root


def _documents(root: Path = ROOT) -> list[Path]:
    return discover_documents(root)


def check(root: Path = ROOT) -> tuple[list[str], list[str], int]:
    broken: list[str] = []
    self_refs: list[str] = []
    checked = 0

    for md in _documents(root):
        text = md.read_text(encoding="utf-8", errors="replace")
        for lineno, path_part in iter_relative_links(text):
            checked += 1
            try:
                resolved = (md.parent / path_part).resolve()
                exists = resolved.exists()
                if not exists and not resolved.suffix:
                    markdown = resolved.with_suffix(".md")
                    if markdown.exists():
                        resolved = markdown
                        exists = True
            except OSError:
                # A target the OS refuses to even stat is not a valid repo path.
                broken.append(
                    f"{md.relative_to(root).as_posix()}:{lineno} -> "
                    f"{path_part} (unresolvable)"
                )
                continue
            if not exists:
                broken.append(
                    f"{md.relative_to(root).as_posix()}:{lineno} -> {path_part}"
                )
            elif resolved == md.resolve():
                # A document that cross-references itself sends the reader in a circle.
                self_refs.append(
                    f"{md.relative_to(root).as_posix()}:{lineno} -> itself"
                )
    return broken, self_refs, checked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    broken, self_refs, checked = check()

    if args.verbose:
        print(f"checked {checked} internal links across {len(_documents())} documents")

    if broken:
        print(f"BROKEN LINKS ({len(broken)}):", file=sys.stderr)
        for b in broken:
            print(f"  - {b}", file=sys.stderr)
    if self_refs:
        print(f"\nSELF-REFERENCES ({len(self_refs)}):", file=sys.stderr)
        for s in self_refs:
            print(f"  - {s}", file=sys.stderr)

    if broken or self_refs:
        return 1

    print(f"links OK ({checked} internal links resolve)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
