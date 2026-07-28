"""The 47-item backlog board may not declare `PLANNED` over code that already shipped.

Contract: CTR-BACKLOG-HONESTY-001

WHY THIS EXISTS (measured, not hypothetical)
--------------------------------------------
`.claude/specs/planes/backlog/BL-*.md` became a *third* source of truth about what is
built, alongside the code and the coordination log — and it drifted. On 2026-07-28 the
board was measured against the tree: 8 items owned by one engineer were still stamped
`status: PLANNED` while their code was written, committed and tracked, and ~18 more of
the other engineer's were in the same state. Every number handed to the operator that
was derived from that board ("N of 47 done") was therefore invented.

`test_knowledge_frontmatter.py` already forbids anchoring a spec to a file that does not
exist. It cannot see the opposite failure — a document that under-claims — because a
`PLANNED` item legitimately anchors to the *pre-existing* files the work will land in
(BL-08 anchors `.gitignore`; BL-19 anchors migrations that predate it). Anchor existence
is therefore worthless as a delivery signal here: it is true for every item on the board.

THE SIGNAL THIS TEST USES, AND WHY IT IS DEFENSIBLE
---------------------------------------------------
Delivery is asserted by the *author*, in the header of the artifact they wrote:

    -- Migration 074: common event-sourced execution ledger (BL-21)
    \"\"\"Domain-separated fingerprints for the control-plane spine (BL-17).\"\"\"

A file whose first lines say "I am BL-NN" is a signed statement that BL-NN's code
landed. That statement contradicts `status: PLANNED` on its face, with no inference and
no heuristics about file mtimes, commit dates or diff sizes.

Deliberately NOT used as evidence, because each is noisy:
  * anchor existence            — true for every item, planned or not (see above);
  * a BL id anywhere in a file  — `owner: BL-17` in a plan table and
                                  "that is BL-33 territory, not this script's" are
                                  *forward* references, not deliveries;
  * commit subjects             — only one of the two engineers tags them, so the signal
                                  is present for one author and absent for the other.

The header window is small (`HEADER_LINES`) and lines that phrase the reference as an
absence ("with no candidate generator (BL-28)", "pending BL-NN") are excluded. The test
is tuned to UNDER-report: a lock that cries wolf gets switched off, and a lock that is
off protects nothing.

DO NOT "FIX" A FAILURE BY WEAKENING THIS TEST
---------------------------------------------
The only legitimate responses to a red result are:
  1. the code shipped  -> update `status:` in the BL document to PARTIAL/IMPLEMENTED; or
  2. the header lies   -> remove the false claim from the source file's header.
Shrinking the header window, adding the offending BL to an allowlist, or broadening the
deferral phrases until the failure disappears re-creates the exact drift this exists to
catch. Widening the deferral list requires the same scrutiny as changing a gate.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
BACKLOG = ROOT / ".claude" / "specs" / "planes" / "backlog"

# Statuses that assert nothing has been built yet.
NOT_BUILT = {"PLANNED"}
# Statuses that assert code exists.
BUILT = {"IMPLEMENTED", "PARTIAL"}

BL_ID = re.compile(r"\bBL-(\d{2})\b")
FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)

# How much of a file counts as its "header" — the docstring / comment banner an author
# writes to say what the file IS. Body mentions are references, not claims of delivery.
HEADER_LINES = 5

# A header line that frames the BL as *missing* is not a delivery claim. Keep this list
# short and literal; every entry is a hole in the lock.
DEFERRAL = re.compile(
    r"\b(no|not|without|sin|pending|pendiente|falta|faltan|todo|fixme|future|futuro|"
    r"territory|territorio|depends|depende|blocked|bloquea|awaiting)\b",
    re.I,
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    ).stdout


def _backlog_items() -> dict[str, dict]:
    items: dict[str, dict] = {}
    for path in sorted(BACKLOG.glob("BL-*.md")):
        m = FM_RE.match(path.read_text(encoding="utf-8", errors="replace"))
        if not m:  # schema is enforced by test_knowledge_frontmatter.py
            continue
        fm = yaml.safe_load(m.group(1)) or {}
        bl = path.name[:5]
        items[bl] = {"path": path, "status": fm.get("status"), "anchors": fm.get("code_anchors") or []}
    return items


ITEMS = _backlog_items()


def header_claims(lines: list[str]) -> set[str]:
    """BL ids a file's header *claims to implement*. Pure — see the self-tests below."""
    found: set[str] = set()
    for line in lines[:HEADER_LINES]:
        if DEFERRAL.search(line):
            continue
        found.update(f"BL-{n}" for n in BL_ID.findall(line))
    return found


def _delivery_claims() -> dict[str, list[str]]:
    """BL id -> tracked files whose header declares itself that item's deliverable.

    `.claude/**` is excluded wholesale: the board, the plans and the coordination log all
    talk about BL ids by definition, so they can never be independent evidence about the
    board.
    """
    listed = _git("grep", "-IloE", r"BL-[0-9]{2}", "--", ":!.claude/**")
    claims: dict[str, list[str]] = {}
    for rel in listed.splitlines():
        rel = rel.strip()
        if not rel:
            continue
        try:
            head = (ROOT / rel).read_text(encoding="utf-8", errors="replace").splitlines()[:HEADER_LINES]
        except OSError:
            continue
        for bl in header_claims(head):
            claims.setdefault(bl, []).append(rel)
    return {k: sorted(set(v)) for k, v in claims.items()}


CLAIMS = _delivery_claims()
TRACKED = {p for p in _git("ls-files").splitlines() if p}


def _tracked_dirs(paths: set[str]) -> set[str]:
    """Anchors may name a directory (`src/execution`); a directory is "tracked" when it
    contains tracked files."""
    dirs: set[str] = set()
    for path in paths:
        acc = path
        while "/" in acc:
            acc = acc.rsplit("/", 1)[0]
            dirs.add(acc)
    return dirs


TRACKED_DIRS = _tracked_dirs(TRACKED)


def test_backlog_is_not_empty():
    assert ITEMS, f"no BL-*.md found under {BACKLOG} — the board moved or the glob broke"


def test_evidence_scan_is_not_vacuous():
    """A silent `git grep` failure would make every check below pass for the wrong reason.

    Without this, running outside a checkout (or a git that errors) turns the whole lock
    green while measuring nothing — the most dangerous failure mode a gate can have.
    """
    assert TRACKED, "`git ls-files` returned nothing — not a checkout, or git is unusable"
    assert CLAIMS, (
        "no tracked file outside .claude/ declares a BL in its header. Either the repo "
        "convention changed (update this test deliberately) or the scan is broken."
    )


@pytest.mark.parametrize("bl", sorted(ITEMS), ids=lambda b: b)
def test_planned_item_has_no_shipped_deliverable(bl: str):
    """`PLANNED` over code whose own header says it is that item = the board is lying."""
    item = ITEMS[bl]
    if item["status"] not in NOT_BUILT:
        pytest.skip(f"{bl} declares {item['status']!r}, not a not-built status")
    shipped = CLAIMS.get(bl, [])
    assert not shipped, (
        f"{bl} declares status: PLANNED but {len(shipped)} tracked file(s) declare "
        f"themselves its deliverable in their header:\n  " + "\n  ".join(shipped) +
        f"\nEither the work shipped (set status: PARTIAL/IMPLEMENTED in "
        f"{item['path'].relative_to(ROOT).as_posix()}) or those headers claim a BL they "
        f"do not implement. Do NOT relax this test — see the module docstring."
    )


@pytest.mark.parametrize("bl", sorted(ITEMS), ids=lambda b: b)
def test_built_item_anchors_are_tracked_code(bl: str):
    """The mirror failure: claiming built while the anchor is not in the repo.

    Existence alone is checked by `test_knowledge_frontmatter.py`. Tracking is the part
    that matters here — a file that exists only in a working tree is not delivered code,
    and untracked artifacts have already slipped through this protocol once.
    """
    item = ITEMS[bl]
    if item["status"] not in BUILT:
        pytest.skip(f"{bl} declares {item['status']!r}, not a built status")
    missing = [a for a in item["anchors"] if not (ROOT / a).exists()]
    untracked = [
        a for a in item["anchors"]
        if (ROOT / a).exists() and a not in TRACKED and a not in TRACKED_DIRS
    ]
    assert not missing and not untracked, (
        f"{bl} declares status: {item['status']} but its evidence is not in the repo — "
        f"absent: {missing or 'none'}; present-but-untracked: {untracked or 'none'}. "
        f"Commit the artifact or lower the status; do not point the anchor elsewhere."
    )


@pytest.mark.parametrize("bl", sorted(ITEMS), ids=lambda b: b)
def test_item_declares_a_status_this_test_understands(bl: str):
    """A typo in `status:` would silently disable both checks above for that item."""
    status = ITEMS[bl]["status"]
    assert status in NOT_BUILT | BUILT | {
        "PAUSED", "DEPRECATED", "SUPERSEDED", "HISTORICAL", "ARCHIVED",
    }, f"{bl}: unknown status {status!r} — the honesty checks would skip it silently"


# --------------------------------------------------------------- self-tests
# The criterion above is the whole value of this file: too loose and it accuses shipped
# work of not existing, too tight and the board drifts again unnoticed. These cases pin
# the exact boundary using lines copied verbatim out of the tree, so that "fixing" a red
# result by widening DEFERRAL or shrinking HEADER_LINES breaks a test that says why.

DELIVERY_HEADERS = [
    "-- Migration 074: common event-sourced execution ledger (BL-21)",
    '"""Domain-separated fingerprints for the control-plane spine (BL-17)."""',
    "# BL-11 PILOTO — familia transversal ACTION (FABRIC §9.4, ADR-0022)",
    "-- Migration 080 / BL-38 + BL-44",
]

DEFERRAL_HEADERS = [
    # `scripts/validation/check_strangler_parity.py` — this file is BL-31's deliverable
    # and says BL-28 is ABSENT. Counting it would accuse BL-28 of having shipped.
    "path?* Fail-closed — with no candidate generator (BL-28) and no execution-readiness",
    "# TODO: BL-22 will persist this",
    "# depends on BL-17 (identity before facts)",
]


@pytest.mark.parametrize("line", DELIVERY_HEADERS)
def test_header_claim_is_recognised(line: str):
    assert header_claims([line]), f"delivery header not recognised: {line!r}"


@pytest.mark.parametrize("line", DEFERRAL_HEADERS)
def test_forward_reference_is_not_a_claim(line: str):
    assert not header_claims([line]), (
        f"{line!r} refers to a BL as absent/future — counting it would make this lock "
        "fire on work that genuinely has not started, which is how locks get disabled."
    )


def test_body_mentions_are_not_claims():
    """A BL id deep in a file is a cross-reference, not "this file is that item"."""
    body = ["header"] * HEADER_LINES + ["    # see BL-33 for the readiness matrix"]
    assert not header_claims(body)


def test_the_board_itself_is_never_evidence():
    """Excluding `.claude/**` is load-bearing: the backlog, the plans and the
    coordination log mention every BL id by construction, so including them would make
    every item self-corroborating and the lock vacuous."""
    for rel in CLAIMS.values():
        assert not any(p.startswith(".claude/") for p in rel), (
            "evidence leaked from .claude/ — the board would be proving itself"
        )
