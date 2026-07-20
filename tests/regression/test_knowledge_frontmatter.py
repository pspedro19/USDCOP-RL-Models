"""Every knowledge document is typed, current, and anchored to real code.

Contract: CTR-KNOWLEDGE-FRONTMATTER-001

Before this existed, `.claude/README.md` asserted that every document carried a
`Contract · Version · Status` header while only 7 of 97 did, and specs kept describing
work as pending long after it shipped (Gold's profile, the BTC derivatives extractor,
`/admin`). Typed front matter turns both failure modes into test failures.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CLAUDE = ROOT / ".claude"

KINDS = {"rule", "as-built", "roadmap", "adr", "audit", "historical"}
STATUSES = {
    "IMPLEMENTED", "PARTIAL", "PLANNED", "PAUSED",
    "DEPRECATED", "SUPERSEDED", "HISTORICAL", "ARCHIVED",
}
REQUIRED = {"kind", "status", "version", "last_verified", "supersedes", "code_anchors"}
STALE_AFTER_DAYS = 90

# `skills/` and `agents/` carry the Claude Code front matter (name + description) that the
# harness itself parses — they are executable definitions, not reference documents, so they
# are validated against their own schema below rather than the spec schema.
EXECUTABLE_DIRS = ("skills", "agents")


def _is_executable_def(path: Path) -> bool:
    rel = path.relative_to(CLAUDE)
    return bool(rel.parts) and rel.parts[0] in EXECUTABLE_DIRS


def _is_definition_file(path: Path) -> bool:
    """Only `skills/<name>/SKILL.md` and `agents/<name>.md` are harness definitions.

    A skill may ship `references/*.md` and `assets/*.md` as supporting material; those are
    content, not definitions, and demanding harness front matter on them would block every
    real-world skill from being imported.
    """
    rel = path.relative_to(CLAUDE)
    if not rel.parts:
        return False
    if rel.parts[0] == "agents":
        return len(rel.parts) == 2
    if rel.parts[0] == "skills":
        return len(rel.parts) == 3 and path.name == "SKILL.md"
    return False


DOCS = [p for p in sorted(CLAUDE.rglob("*.md")) if not _is_executable_def(p)]
EXECUTABLES = [p for p in sorted(CLAUDE.rglob("*.md")) if _is_definition_file(p)]
# Supporting material inside a skill: checked lightly, not against the harness schema.
SKILL_CONTENT = [
    p for p in sorted(CLAUDE.rglob("*.md"))
    if _is_executable_def(p) and not _is_definition_file(p)
]
FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


def _frontmatter(path: Path) -> dict | None:
    m = FM_RE.match(path.read_text(encoding="utf-8", errors="replace"))
    if not m:
        return None
    return yaml.safe_load(m.group(1)) or {}


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def test_documents_exist():
    assert DOCS, "no knowledge documents found under .claude/"


@pytest.mark.parametrize("path", DOCS, ids=_rel)
def test_document_has_valid_frontmatter(path: Path):
    fm = _frontmatter(path)
    assert fm is not None, f"{_rel(path)} has no YAML front matter — run scripts/ops/migrate_spec_frontmatter.py"

    missing = REQUIRED - set(fm)
    assert not missing, f"{_rel(path)} missing front-matter fields: {sorted(missing)}"

    assert fm["kind"] in KINDS, f"{_rel(path)}: unknown kind {fm['kind']!r}"
    assert fm["status"] in STATUSES, f"{_rel(path)}: unknown status {fm['status']!r}"
    assert re.fullmatch(r"\d+\.\d+\.\d+", str(fm["version"])), (
        f"{_rel(path)}: version must be semver, got {fm['version']!r}"
    )
    assert isinstance(fm["supersedes"], list)
    assert isinstance(fm["code_anchors"], list)


@pytest.mark.parametrize("path", DOCS, ids=_rel)
def test_code_anchors_point_at_real_files(path: Path):
    """A spec anchored to code that no longer exists is describing a system that
    no longer exists — the single highest-signal staleness check available."""
    fm = _frontmatter(path)
    if not fm:
        pytest.skip("no front matter (covered by the schema test)")
    dead = [a for a in fm.get("code_anchors") or [] if not (ROOT / a).exists()]
    assert not dead, (
        f"{_rel(path)} anchors to paths that do not exist: {dead}. "
        "Either the code moved (update the anchor) or the spec is stale (archive it)."
    )


@pytest.mark.parametrize("path", DOCS, ids=_rel)
def test_supersedes_targets_resolve(path: Path):
    fm = _frontmatter(path)
    if not fm:
        pytest.skip("no front matter")
    for target in fm.get("supersedes") or []:
        assert (CLAUDE / target).exists() or (ROOT / target).exists(), (
            f"{_rel(path)} supersedes a document that does not exist: {target}"
        )


def test_active_specs_are_not_stale():
    """`as-built` docs must have been verified against the code recently."""
    today = date.today()
    stale = []
    for path in DOCS:
        fm = _frontmatter(path)
        if not fm or fm.get("kind") != "as-built":
            continue
        if fm.get("status") in {"ARCHIVED", "HISTORICAL", "SUPERSEDED", "DEPRECATED"}:
            continue
        lv = fm.get("last_verified")
        if isinstance(lv, str):
            lv = datetime.strptime(lv, "%Y-%m-%d").date()
        if lv is None or (today - lv).days > STALE_AFTER_DAYS:
            stale.append(f"{_rel(path)} (last_verified={lv})")
    assert not stale, (
        f"as-built specs unverified for >{STALE_AFTER_DAYS} days:\n  " + "\n  ".join(stale)
    )


def test_archived_docs_are_marked_and_quarantined():
    """Nothing under specs/archive/ may claim to be current."""
    archive = CLAUDE / "specs" / "archive"
    if not archive.is_dir():
        pytest.skip("no archive yet")
    for path in sorted(archive.rglob("*.md")):
        fm = _frontmatter(path)
        assert fm, f"{_rel(path)} archived but untyped"
        assert fm["status"] == "ARCHIVED", (
            f"{_rel(path)} lives in specs/archive/ but status is {fm['status']!r}"
        )


def test_rules_declare_themselves_as_rules():
    for path in sorted((CLAUDE / "rules").glob("*.md")):
        fm = _frontmatter(path)
        assert fm and fm["kind"] == "rule", f"{_rel(path)} must be kind: rule"


# --------------------------------------------------------------- skills & agents

@pytest.mark.parametrize("path", EXECUTABLES, ids=_rel)
def test_executable_definition_frontmatter(path: Path):
    """Skills and agents are parsed by the harness — a malformed header means the
    definition silently never loads."""
    fm = _frontmatter(path)
    assert fm is not None, f"{_rel(path)} has no front matter"
    for field in ("name", "description"):
        assert fm.get(field), f"{_rel(path)} missing required field: {field}"
    assert re.fullmatch(r"[a-z0-9-]+", str(fm["name"])), (
        f"{_rel(path)}: name must be kebab-case, got {fm['name']!r}"
    )
    assert len(str(fm["description"])) >= 40, (
        f"{_rel(path)}: description too short to route on — say WHEN to use it"
    )


def test_skill_directory_name_matches_declared_name():
    for path in EXECUTABLES:
        fm = _frontmatter(path) or {}
        expected = path.stem if path.parent.name == "agents" else path.parent.name
        assert fm.get("name") == expected, (
            f"{_rel(path)}: declared name {fm.get('name')!r} != location {expected!r}"
        )


@pytest.mark.parametrize("path", SKILL_CONTENT, ids=_rel)
def test_skill_supporting_material_is_not_empty(path: Path):
    """`references/`, `assets/` etc. are content, not definitions — just require substance."""
    assert path.read_text(encoding="utf-8", errors="replace").strip(), f"{_rel(path)} is empty"


def test_skill_metadata_budget():
    """Skill frontmatter is preloaded into EVERY session — it is an invisible context tax.

    Nothing else measures it: the rules budget test globs `.claude/rules/*.md` only. The
    source library carries ~60 KB of frontmatter across 119 skills (~15k tokens, four times
    the entire rules budget), so bulk-importing it would cost every future session silently.
    """
    total = 0
    for path in EXECUTABLES:
        m = FM_RE.match(path.read_text(encoding="utf-8", errors="replace"))
        if m:
            total += len(m.group(1))
    assert total <= 20000, (
        f"skill+agent frontmatter is {total} chars (cap 20000). Every byte here is loaded "
        "in every session — promote fewer skills, or tighten their descriptions."
    )


def test_reviewer_agents_are_read_only():
    """A reviewer that can edit is not an independent reviewer."""
    for path in sorted((CLAUDE / "agents").glob("*.md")) if (CLAUDE / "agents").is_dir() else []:
        fm = _frontmatter(path) or {}
        tools = str(fm.get("tools", ""))
        for forbidden in ("Write", "Edit", "NotebookEdit"):
            assert forbidden not in tools, (
                f"{_rel(path)} grants {forbidden} — reviewer agents must be read-only"
            )
