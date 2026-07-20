"""The contract mirror map declared in the `contract-change` skill must stay true.

Contract: CTR-CONTRACT-MIRROR-001

Contracts in this repo are mirrored across Python and TypeScript **by convention**, and the
convention is not filename-based (`analysis_schema.py` ↔ `weekly-analysis.contract.ts`). That
makes the map itself load-bearing — and a hand-written map that nothing verifies rots exactly
like the architecture counts did.

So the skill's table is the source, and this test is its guard: every path it names must exist.

It also pins the two threshold families apart. `test_action_threshold_ssot.py` exists because
Python said 0.35 while the TS mirror said 0.33; the test it left behind guards the Python side
only, so the TS constant is asserted here.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / ".claude" / "skills" / "contract-change" / "SKILL.md"

# `path/to/file.ext` inside a markdown table cell
PATH_RE = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|ts|yaml|yml))`")


def _mirror_rows() -> list[tuple[str, list[str], list[str]]]:
    """Parse the mirror table: (raw_row, source_paths, mirror_paths)."""
    text = SKILL.read_text(encoding="utf-8", errors="replace")
    rows: list[tuple[str, list[str], list[str]]] = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("| Authoritative source"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            if set(line) <= set("|- "):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 2:
                continue
            rows.append((line, PATH_RE.findall(cells[0]), PATH_RE.findall(cells[1])))
    return rows


MIRROR_ROWS = _mirror_rows()


def test_skill_exists_and_declares_a_map():
    assert SKILL.is_file(), "the contract-change skill is the SSOT for the mirror map"
    assert len(MIRROR_ROWS) >= 10, (
        f"mirror table looks truncated: parsed {len(MIRROR_ROWS)} rows"
    )


@pytest.mark.parametrize(
    "row,sources,mirrors", MIRROR_ROWS, ids=[r[0][:60] for r in MIRROR_ROWS]
)
def test_declared_mirror_paths_exist(row: str, sources: list[str], mirrors: list[str]):
    assert sources, f"row declares no source path: {row}"
    assert mirrors, f"row declares no mirror path: {row}"
    missing = [p for p in sources + mirrors if not (ROOT / p).exists()]
    assert not missing, (
        f"mirror map names paths that no longer exist: {missing}. "
        "Either the file moved (update .claude/skills/contract-change/SKILL.md) "
        "or the contract was deleted (remove the row)."
    )


def test_threshold_families_stay_distinct():
    """Dashboard/pipeline actions are ±0.35; backtest/experiment are ±0.50.

    Both are correct — they are different configs. The failure mode is someone "fixing"
    one to match the other, so pin them.
    """
    ssot_ts = ROOT / "usdcop-trading-dashboard/lib/contracts/ssot.contract.ts"
    backtest_ts = ROOT / "usdcop-trading-dashboard/lib/contracts/backtest-ssot.contract.ts"
    pipeline_yaml = ROOT / "config/pipeline_ssot.yaml"

    for path in (ssot_ts, backtest_ts, pipeline_yaml):
        assert path.is_file(), f"missing contract file: {path.relative_to(ROOT).as_posix()}"

    ssot = ssot_ts.read_text(encoding="utf-8", errors="replace")
    assert re.search(r"THRESHOLD_LONG\s*=\s*0\.35", ssot), (
        "ssot.contract.ts THRESHOLD_LONG must stay 0.35, mirroring pipeline_ssot.yaml"
    )

    backtest = backtest_ts.read_text(encoding="utf-8", errors="replace")
    assert re.search(r"THRESHOLD_LONG\s*=\s*0\.50", backtest), (
        "backtest-ssot.contract.ts THRESHOLD_LONG must stay 0.50 — it is a different config"
    )

    yaml_text = pipeline_yaml.read_text(encoding="utf-8", errors="replace")
    assert re.search(r"threshold_long:\s*0\.35", yaml_text), (
        "pipeline_ssot.yaml threshold_long drifted from the TS mirror (this exact "
        "mismatch already happened once: 0.35 vs 0.33)"
    )


def test_execution_contracts_are_paired():
    """`lib/contracts/execution/` mirrors the SignalBridge Python contracts.

    This whole directory was missed by a filename-based analysis — hence the explicit map.
    """
    py_dir = ROOT / "services/signalbridge_api/app/contracts"
    ts_dir = ROOT / "usdcop-trading-dashboard/lib/contracts/execution"
    if not py_dir.is_dir() or not ts_dir.is_dir():
        pytest.skip("SignalBridge contracts not present in this checkout")

    skill_text = SKILL.read_text(encoding="utf-8", errors="replace")
    for ts in sorted(ts_dir.glob("*.contract.ts")):
        rel = ts.relative_to(ROOT).as_posix()
        assert rel in skill_text, (
            f"{rel} exists but is absent from the mirror map — add it to "
            ".claude/skills/contract-change/SKILL.md so its Python source is discoverable"
        )
