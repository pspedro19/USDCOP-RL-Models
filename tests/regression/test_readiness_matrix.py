"""BL-33: the institutional-readiness matrix must carry auditable evidence.

The original matrix reduced seven domains to seven blanket
``IMPLEMENTED_UNVERIFIED`` rows.  That format could not distinguish a repository
control from an operational drill, an external blocker, or a missing capability.
These tests keep the register fail-closed without asserting that the institution is
ready.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / ".claude" / "specs" / "planes" / "04b-readiness-matrix.md"

REQUIRED_COLUMNS = [
    "Control ID",
    "Domain",
    "Control",
    "Expected evidence",
    "Observed evidence",
    "State",
    "Owner",
    "Verified",
]
REQUIRED_DOMAINS = {
    "Technology",
    "Risk",
    "Execution",
    "Security",
    "Compliance",
    "Operations",
    "Investors",
}
ALLOWED_STATES = {
    "VERIFIED_REPO",
    "PARTIAL",
    "BLOCKED_EXTERNAL",
    "NOT_EVIDENCED",
}
CONTROL_ID = re.compile(r"^(TECH|RISK|EXEC|SEC|COMP|OPS|INV)-\d{2}$")
DATE = re.compile(r"^20\d{2}-\d{2}-\d{2}$")
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _register() -> list[dict[str, str]]:
    lines = MATRIX.read_text(encoding="utf-8").splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if _cells(line) == REQUIRED_COLUMNS),
        None,
    )
    assert header_index is not None, (
        "readiness matrix is missing the auditable register columns: "
        f"{REQUIRED_COLUMNS}"
    )
    assert header_index + 1 < len(lines)
    separator = _cells(lines[header_index + 1])
    assert len(separator) == len(REQUIRED_COLUMNS)
    assert all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator)

    rows: list[dict[str, str]] = []
    for line in lines[header_index + 2 :]:
        if not line.lstrip().startswith("|"):
            if rows:
                break
            continue
        cells = _cells(line)
        assert len(cells) == len(REQUIRED_COLUMNS), f"malformed register row: {line}"
        rows.append(dict(zip(REQUIRED_COLUMNS, cells, strict=True)))
    assert rows, "readiness register has no controls"
    return rows


def test_register_covers_all_domains_with_stable_ids_and_fail_closed_states() -> None:
    rows = _register()
    assert [row["Control ID"] for row in rows[:2]] == ["SEC-01", "SEC-02"], (
        "BL-33 requires the leaked-secret incident and identity segregation first"
    )

    ids = [row["Control ID"] for row in rows]
    assert len(ids) == len(set(ids)), "control ids must be unique"
    assert all(CONTROL_ID.fullmatch(control_id) for control_id in ids)
    assert {row["Domain"] for row in rows} == REQUIRED_DOMAINS
    for domain in REQUIRED_DOMAINS:
        assert sum(row["Domain"] == domain for row in rows) >= 2, (
            f"{domain} is represented by a single decorative checkbox"
        )

    states = {row["State"] for row in rows}
    assert states <= ALLOWED_STATES
    assert "IMPLEMENTED_UNVERIFIED" not in MATRIX.read_text(encoding="utf-8")
    assert {"PARTIAL", "BLOCKED_EXTERNAL", "NOT_EVIDENCED"} <= states, (
        "the register must expose gaps instead of presenting blanket readiness"
    )


def test_every_control_has_owner_date_expectation_and_existing_relative_evidence() -> None:
    rows = _register()
    for row in rows:
        where = row["Control ID"]
        assert len(row["Control"]) >= 8, f"{where}: control is not specific"
        assert len(row["Expected evidence"]) >= 12, f"{where}: expected evidence is empty"
        assert row["Owner"] not in {"", "—", "TBD"}, f"{where}: owner is missing"
        assert DATE.fullmatch(row["Verified"]), f"{where}: verification date is not ISO"

        targets = LINK.findall(row["Observed evidence"])
        assert targets, f"{where}: observed evidence has no Markdown link"
        for raw_target in targets:
            target = raw_target.split("#", 1)[0]
            assert target and "://" not in target and not target.startswith("/")
            resolved = (MATRIX.parent / target).resolve()
            assert resolved.is_relative_to(ROOT.resolve()), f"{where}: evidence escapes repo"
            assert resolved.exists(), f"{where}: missing evidence target {raw_target}"

    assert "BL-08-incidente-env-historial.md" in rows[0]["Observed evidence"]
    assert "03-institutional-readiness.md" in rows[1]["Observed evidence"]


def test_matrix_defines_state_semantics_and_scope_limit() -> None:
    text = MATRIX.read_text(encoding="utf-8")
    for state in ALLOWED_STATES:
        assert re.search(rf"^\| `{state}` \|", text, flags=re.MULTILINE), (
            f"state {state} is used without a definition"
        )
    assert "no autoriza capital" in text.lower()
    assert "caso b" in text.lower()
    assert "✅" not in text and "❌" not in text
