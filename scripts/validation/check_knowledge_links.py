"""Validate markdown links and referenced paths across the knowledge base.

Contract: CTR-KNOWLEDGE-LINKS-001

Four broken links sat in `.claude/specs/assets/xauusd/` pointing at `../rules/` for files
that live in `specs/`, and `mlops-lifecycle.md` cross-referenced itself. Nobody noticed
because nothing checked. This is the checker.

Usage:
    python scripts/validation/check_knowledge_links.py
    python scripts/validation/check_knowledge_links.py --verbose
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/validation/<this> -> repo root
CLAUDE = ROOT / ".claude"

# [text](target) — skip external, anchors and mailto
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
SKIP_PREFIXES = ("http://", "https://", "#", "mailto:", "<")


def check() -> tuple[list[str], list[str], int]:
    broken: list[str] = []
    self_refs: list[str] = []
    checked = 0

    for md in sorted(CLAUDE.rglob("*.md")):
        text = md.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            for target in LINK_RE.findall(line):
                target = target.strip()
                if target.startswith(SKIP_PREFIXES) or not target:
                    continue
                path_part = target.split("#", 1)[0].strip()
                if not path_part:
                    continue
                checked += 1
                resolved = (md.parent / path_part).resolve()
                if not resolved.exists():
                    broken.append(
                        f"{md.relative_to(ROOT).as_posix()}:{lineno} -> {path_part}"
                    )
                elif resolved == md.resolve():
                    # A document that cross-references itself sends the reader in a circle.
                    self_refs.append(
                        f"{md.relative_to(ROOT).as_posix()}:{lineno} -> itself"
                    )
    return broken, self_refs, checked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    broken, self_refs, checked = check()

    if args.verbose:
        print(f"checked {checked} internal links across {len(list(CLAUDE.rglob('*.md')))} documents")

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
