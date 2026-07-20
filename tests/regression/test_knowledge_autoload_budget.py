"""`.claude/rules/` is injected into EVERY session — it must stay thin.

Contract: CTR-KNOWLEDGE-BUDGET-001

`.claude/README.md` has always said rules must be "thin", but the rules grew to ~10,150
words (~77 KB) of runbooks, schemas, CLI examples and DAG inventories — a fixed context
tax on every single session, before the user typed anything. Dense reference belongs in
`specs/` (on-demand). This test makes the boundary enforceable rather than aspirational.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / ".claude" / "rules"
CLAUDE_MD = ROOT / "CLAUDE.md"

MAX_RULES_WORDS = 3000
MAX_RULE_LINES = 120
MAX_CLAUDE_MD_LINES = 400

FM_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.S)


def _body(path: Path) -> str:
    """Content without front matter — metadata shouldn't count against the budget."""
    return FM_RE.sub("", path.read_text(encoding="utf-8", errors="replace"), count=1)


def _rule_files() -> list[Path]:
    return sorted(RULES.glob("*.md"))


def test_rules_total_word_budget():
    total = sum(len(_body(p).split()) for p in _rule_files())
    breakdown = "\n  ".join(
        f"{p.name}: {len(_body(p).split())}" for p in sorted(
            _rule_files(), key=lambda p: -len(_body(p).split())
        )
    )
    assert total <= MAX_RULES_WORDS, (
        f"auto-loaded rules are {total} words (budget {MAX_RULES_WORDS}).\n"
        f"  {breakdown}\n"
        "Move dense reference into .claude/specs/ and leave invariants + DO NOTs."
    )


@pytest.mark.parametrize("path", _rule_files(), ids=lambda p: p.name)
def test_individual_rule_stays_short(path: Path):
    lines = len(_body(path).splitlines())
    assert lines <= MAX_RULE_LINES, (
        f"{path.name} is {lines} lines (max {MAX_RULE_LINES}) — split the reference into specs/"
    )


def test_claude_md_is_navigation_not_encyclopedia():
    lines = len(CLAUDE_MD.read_text(encoding="utf-8", errors="replace").splitlines())
    assert lines <= MAX_CLAUDE_MD_LINES, (
        f"CLAUDE.md is {lines} lines (max {MAX_CLAUDE_MD_LINES}) — it is auto-loaded too"
    )


def test_every_rule_is_listed_in_the_index():
    index = (RULES / "00-INDEX.md").read_text(encoding="utf-8", errors="replace")
    missing = [
        p.name for p in _rule_files()
        if p.name != "00-INDEX.md" and p.name not in index
    ]
    assert not missing, f"rules absent from 00-INDEX.md: {missing}"
