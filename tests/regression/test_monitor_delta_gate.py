"""
Regression/TDD: the CI delta comparator (`scripts/validation/check_monitor_delta.py`).

WHY THIS EXISTS
---------------
Three CI steps in `.github/workflows/fabric-contracts.yml` carry `continue-on-error: true`
because they drag measured pre-existing debt (tsc 336 diagnostics, Vitest 46 failed tests,
`test_knowledge_frontmatter` 47 failed). `continue-on-error` makes the step *never* fail, so a
NEW failure introduced tomorrow is exactly as silent as the known ones — the gate is decoration.

The comparator replaces it: it runs the monitor, extracts failures from a MACHINE-READABLE
artifact, and compares against a registered baseline. Up => red. Same => green. Down => warn.
Different failures at the same count => red (identity comparison, not a blind counter).

These tests cover: increase => red, equality => green, decrease => warning, same-count-other-
failures => red, and every fail-closed path.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "validation" / "check_monitor_delta.py"
REAL_CONFIG = ROOT / ".claude" / "coordination" / "BASELINE.monitors.json"

sys.path.insert(0, str(ROOT))

from scripts.validation.check_monitor_delta import (  # noqa: E402
    EXIT_FAIL_CLOSED,
    EXIT_OK,
    EXIT_REGRESSION,
    ExtractionError,
    parse_pytest_junit,
    parse_tsc_text,
    parse_vitest_json,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


JUNIT_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="{errors}" failures="{failures}" tests="{tests}">
{cases}
</testsuite></testsuites>
"""


def junit_xml(failed: list[str], errored: list[str] = (), passed: int = 3) -> str:
    cases = []
    for fid in failed:
        cls, _, name = fid.partition("::")
        cases.append(f'<testcase classname="{cls}" name="{name}"><failure message="boom"/></testcase>')
    for eid in errored:
        cls, _, name = eid.partition("::")
        cases.append(f'<testcase classname="{cls}" name="{name}"><error message="collect"/></testcase>')
    for i in range(passed):
        cases.append(f'<testcase classname="t" name="ok{i}"/>')
    return JUNIT_TEMPLATE.format(
        errors=len(errored),
        failures=len(failed),
        tests=len(failed) + len(errored) + passed,
        cases="\n".join(cases),
    )


def write_stub_config(
    tmp_path: Path,
    *,
    payload: str,
    count: int,
    failures: list[str] | None,
    secondary_count: int = 0,
    secondary_failures: list[str] | None = None,
    md_text: str = "monitor stub: 2 failed pre-existentes\n",
    number_pattern: str | None = r"monitor stub: (\d+) failed",
    exit_code: int = 1,
    command: list[str] | None = None,
) -> tuple[Path, Path]:
    """Build an isolated config whose 'runner command' is a python stub that writes a
    canned junit report and exits with `exit_code`."""
    md = tmp_path / "BASELINE.md"
    md.write_text(md_text, encoding="utf-8")
    payload_file = tmp_path / "payload.xml"
    payload_file.write_text(payload, encoding="utf-8")

    stub = tmp_path / "stub.py"
    stub.write_text(
        "import shutil, sys\n"
        "shutil.copyfile(sys.argv[1], sys.argv[2])\n"
        "sys.exit(int(sys.argv[3]))\n",
        encoding="utf-8",
    )
    cmd = command or [
        sys.executable,
        str(stub),
        str(payload_file),
        "{report}",
        str(exit_code),
    ]
    entry: dict = {
        "runner": "pytest",
        "cwd": ".",
        "command": cmd,
        "count": count,
        "secondary_count": secondary_count,
        "measured_at": "2026-07-28T00:00:00-05:00",
        "baseline_md": {
            "assertions": [r"monitor stub"],
            "number_pattern": number_pattern,
            "number_authoritative": number_pattern is not None,
        },
    }
    if failures is not None:
        entry["failures"] = failures
    if secondary_failures is not None:
        entry["secondary_failures"] = secondary_failures

    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "contract_id": "CTR-CI-DELTA-001",
                "baseline_md": str(md),
                "monitors": {"stub": entry},
            }
        ),
        encoding="utf-8",
    )
    return cfg, md


# ---------------------------------------------------------------------------
# 1. core delta semantics
# ---------------------------------------------------------------------------
def test_equal_count_and_identity_is_green(tmp_path):
    ids = ["m.py::test_a", "m.py::test_b"]
    cfg, _ = write_stub_config(tmp_path, payload=junit_xml(ids), count=2, failures=ids)
    r = run_cli("--config", str(cfg), "--monitor", "stub")
    assert r.returncode == EXIT_OK, r.stdout + r.stderr
    assert "DELTA 0" in r.stdout


def test_increase_is_red(tmp_path):
    baseline = ["m.py::test_a", "m.py::test_b"]
    now = baseline + ["m.py::test_NEW"]
    cfg, md = write_stub_config(tmp_path, payload=junit_xml(now), count=2, failures=baseline)
    md.write_text("monitor stub: 2 failed pre-existentes\n", encoding="utf-8")
    r = run_cli("--config", str(cfg), "--monitor", "stub")
    assert r.returncode == EXIT_REGRESSION, r.stdout + r.stderr
    assert "m.py::test_NEW" in r.stdout
    assert "+1" in r.stdout


def test_same_count_different_failures_is_red(tmp_path):
    """The headline requirement: a blind counter would approve this."""
    baseline = ["m.py::test_a", "m.py::test_b"]
    now = ["m.py::test_a", "m.py::test_SWAPPED"]
    cfg, _ = write_stub_config(tmp_path, payload=junit_xml(now), count=2, failures=baseline)
    r = run_cli("--config", str(cfg), "--monitor", "stub")
    assert r.returncode == EXIT_REGRESSION, r.stdout + r.stderr
    assert "test_SWAPPED" in r.stdout
    assert "test_b" in r.stdout  # the disappeared one is reported too


def test_decrease_warns_but_passes(tmp_path):
    baseline = ["m.py::test_a", "m.py::test_b"]
    now = ["m.py::test_a"]
    cfg, md = write_stub_config(tmp_path, payload=junit_xml(now), count=2, failures=baseline)
    md.write_text("monitor stub: 2 failed pre-existentes\n", encoding="utf-8")
    r = run_cli("--config", str(cfg), "--monitor", "stub")
    assert r.returncode == EXIT_OK, r.stdout + r.stderr
    assert "WARN" in r.stdout
    assert "--update-baseline" in r.stdout  # tells you the debt must be re-registered


def test_decrease_is_red_under_strict_flag(tmp_path):
    baseline = ["m.py::test_a", "m.py::test_b"]
    cfg, _ = write_stub_config(
        tmp_path, payload=junit_xml(["m.py::test_a"]), count=2, failures=baseline
    )
    r = run_cli("--config", str(cfg), "--monitor", "stub", "--fail-on-decrease")
    assert r.returncode == EXIT_REGRESSION, r.stdout + r.stderr


def test_count_only_baseline_declares_its_limitation(tmp_path):
    """No id list registered => count-only mode. It must PASS on equal counts but say,
    loudly and in the report, that same-count-other-failures is NOT covered."""
    cfg, _ = write_stub_config(
        tmp_path, payload=junit_xml(["m.py::x", "m.py::y"]), count=2, failures=None
    )
    r = run_cli("--config", str(cfg), "--monitor", "stub")
    assert r.returncode == EXIT_OK, r.stdout + r.stderr
    assert "count-only" in r.stdout.lower()
    assert "cannot detect" in r.stdout.lower()


def test_secondary_dimension_regression_is_red(tmp_path):
    """Collection/suite-level errors are tracked separately and also gate."""
    cfg, _ = write_stub_config(
        tmp_path,
        payload=junit_xml(["m.py::a"], errored=["m.py::boom"]),
        count=1,
        failures=["m.py::a"],
        secondary_count=0,
        secondary_failures=[],
        md_text="monitor stub: 1 failed pre-existentes\n",
    )
    r = run_cli("--config", str(cfg), "--monitor", "stub")
    assert r.returncode == EXIT_REGRESSION, r.stdout + r.stderr
    assert "m.py::boom" in r.stdout


def test_non_ascii_failure_ids_do_not_crash_the_report(tmp_path):
    """Real Vitest titles carry accents and glyphs ("artefacto valido => 200" is literally
    written with U+21D2 in this repo). A gate that crashes while PRINTING a regression is a
    fail-open by accident on a cp1252 console."""
    now = ["m.py::test_artefacto válido ⇒ 200"]
    cfg, _ = write_stub_config(
        tmp_path,
        payload=junit_xml(now),
        count=0,
        failures=[],
        md_text="monitor stub: 0 failed pre-existentes\n",
    )
    r = run_cli("--config", str(cfg), "--monitor", "stub")
    assert r.returncode == EXIT_REGRESSION, r.stdout + r.stderr
    assert "NEW FAILURE" in r.stdout


# ---------------------------------------------------------------------------
# 2. fail-closed paths (a comparator that passes when in doubt is worse than
#    continue-on-error, because it also looks rigorous)
# ---------------------------------------------------------------------------
def test_unknown_monitor_fails_closed(tmp_path):
    cfg, _ = write_stub_config(tmp_path, payload=junit_xml([]), count=0, failures=[])
    r = run_cli("--config", str(cfg), "--monitor", "does_not_exist")
    assert r.returncode == EXIT_FAIL_CLOSED
    assert "no baseline" in (r.stdout + r.stderr).lower()


def test_missing_config_fails_closed(tmp_path):
    r = run_cli("--config", str(tmp_path / "nope.json"), "--monitor", "stub")
    assert r.returncode == EXIT_FAIL_CLOSED


def test_command_not_runnable_fails_closed(tmp_path):
    cfg, _ = write_stub_config(
        tmp_path,
        payload=junit_xml([]),
        count=0,
        failures=[],
        command=["this-binary-does-not-exist-42", "{report}"],
    )
    r = run_cli("--config", str(cfg), "--monitor", "stub")
    assert r.returncode == EXIT_FAIL_CLOSED
    assert "could not run" in (r.stdout + r.stderr).lower()


def test_unexpected_exit_code_fails_closed(tmp_path):
    """pytest exit 3 = INTERNALERROR (the known `pytest tests/` abort). A report may still
    exist and look plausible; the comparator must NOT trust it."""
    cfg, _ = write_stub_config(
        tmp_path, payload=junit_xml([]), count=0, failures=[], exit_code=3
    )
    r = run_cli("--config", str(cfg), "--monitor", "stub")
    assert r.returncode == EXIT_FAIL_CLOSED
    assert "exit" in (r.stdout + r.stderr).lower()


def test_unparseable_report_fails_closed(tmp_path):
    cfg, _ = write_stub_config(tmp_path, payload="<not xml", count=0, failures=[])
    r = run_cli("--config", str(cfg), "--monitor", "stub")
    assert r.returncode == EXIT_FAIL_CLOSED


def test_report_self_inconsistency_fails_closed(tmp_path):
    """junit header says 5 failures but only 1 testcase carries one => do not guess."""
    bad = JUNIT_TEMPLATE.format(
        errors=0,
        failures=5,
        tests=1,
        cases='<testcase classname="m.py" name="a"><failure message="x"/></testcase>',
    )
    cfg, _ = write_stub_config(tmp_path, payload=bad, count=1, failures=["m.py::a"])
    r = run_cli("--config", str(cfg), "--monitor", "stub")
    assert r.returncode == EXIT_FAIL_CLOSED
    assert "inconsist" in (r.stdout + r.stderr).lower()


def test_baseline_md_out_of_sync_fails_closed(tmp_path):
    """Two truths must be impossible: JSON says 2, prose says 9 => stop."""
    cfg, md = write_stub_config(
        tmp_path, payload=junit_xml(["m.py::a", "m.py::b"]), count=2, failures=["m.py::a", "m.py::b"]
    )
    md.write_text("monitor stub: 9 failed pre-existentes\n", encoding="utf-8")
    r = run_cli("--config", str(cfg), "--monitor", "stub")
    assert r.returncode == EXIT_FAIL_CLOSED
    assert "BASELINE.md" in r.stdout + r.stderr


def test_baseline_md_assertion_vanished_fails_closed(tmp_path):
    cfg, md = write_stub_config(
        tmp_path, payload=junit_xml(["m.py::a", "m.py::b"]), count=2, failures=["m.py::a", "m.py::b"]
    )
    md.write_text("someone rewrote this file\n", encoding="utf-8")
    r = run_cli("--config", str(cfg), "--monitor", "stub")
    assert r.returncode == EXIT_FAIL_CLOSED


def test_missing_report_placeholder_fails_closed(tmp_path):
    cfg, _ = write_stub_config(
        tmp_path, payload=junit_xml([]), count=0, failures=[], command=[sys.executable, "-c", "pass"]
    )
    r = run_cli("--config", str(cfg), "--monitor", "stub")
    assert r.returncode == EXIT_FAIL_CLOSED
    assert "{report}" in r.stdout + r.stderr


# ---------------------------------------------------------------------------
# 3. per-runner extraction (structured where possible; declared where not)
# ---------------------------------------------------------------------------
def test_parse_pytest_junit_splits_failures_from_errors(tmp_path):
    p = tmp_path / "r.xml"
    p.write_text(junit_xml(["a.py::t1", "a.py::t2"], errored=["b.py::t3"]), encoding="utf-8")
    obs = parse_pytest_junit(p, "", "")
    assert obs.count == 2
    assert obs.secondary_count == 1
    assert set(obs.ids) == {"a.py::t1", "a.py::t2"}
    assert set(obs.secondary_ids) == {"b.py::t3"}


def test_parse_vitest_json_counts_tests_and_collection_failures(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(
        json.dumps(
            {
                "numFailedTests": 2,
                "numTotalTests": 10,
                "testResults": [
                    {
                        "name": "/repo/tests/unit/a.test.ts",
                        "status": "failed",
                        "assertionResults": [
                            {"fullName": "x fails", "status": "failed"},
                            {"fullName": "y fails", "status": "failed"},
                            {"fullName": "z ok", "status": "passed"},
                        ],
                    },
                    {
                        "name": "/repo/tests/unit/missing.test.ts",
                        "status": "failed",
                        "message": "Cannot find module",
                        "assertionResults": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    obs = parse_vitest_json(p, "", "")
    assert obs.count == 2
    assert obs.secondary_count == 1  # suite that failed to even collect
    assert any("x fails" in i for i in obs.ids)
    assert any("missing.test.ts" in i for i in obs.secondary_ids)


def test_parse_vitest_json_disagreeing_with_its_own_totals_fails_closed(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(
        json.dumps(
            {
                "numFailedTests": 7,
                "testResults": [
                    {
                        "name": "a.test.ts",
                        "status": "failed",
                        "assertionResults": [{"fullName": "x", "status": "failed"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ExtractionError):
        parse_vitest_json(p, "", "")


def test_parse_tsc_text_counts_diagnostics_not_lines():
    out = (
        "app/a.ts(95,72): error TS2322: Type 'string' is not assignable to type 'UserRole'.\n"
        "app/b.ts(43,61): error TS2339: Property 'name' does not exist.\n"
        "  Property 'name' does not exist on type 'SessionUser'.\n"  # continuation line
        "app/b.ts(50,61): error TS2339: Property 'name' does not exist.\n"
    )
    obs = parse_tsc_text(None, out, "")
    assert obs.count == 3          # 3 diagnostics across 4 lines
    assert obs.secondary_count == 0
    # identity is (file, code, message) — line/col deliberately dropped, so the two
    # identical b.ts errors are a multiset of 2, not a set of 1.
    assert obs.ids.count("app/b.ts|TS2339|Property 'name' does not exist.") == 2


def test_parse_tsc_text_ignores_noise_and_keeps_fileless_diagnostics():
    obs = parse_tsc_text(None, "error TS18003: No inputs were found in config file.\n", "")
    assert obs.count == 1


# ---------------------------------------------------------------------------
# 4. the real registered baselines (this is the gate CODEX will wire)
# ---------------------------------------------------------------------------
def test_real_config_is_valid_and_every_monitor_is_md_corroborated():
    cfg = json.loads(REAL_CONFIG.read_text(encoding="utf-8"))
    assert cfg["monitors"], "no monitors registered"
    for name, m in cfg["monitors"].items():
        assert m["runner"] in {"pytest", "vitest", "tsc"}, name
        assert isinstance(m["count"], int), name
        assert m["baseline_md"]["assertions"], f"{name}: no BASELINE.md corroboration"
        assert "measured_at" in m, name


@pytest.mark.slow
def test_real_frontmatter_monitor_has_zero_delta():
    """End-to-end against the live repo: the 47 known failures, no more, no other ones."""
    r = run_cli("--monitor", "pytest_knowledge_frontmatter")
    assert r.returncode == EXIT_OK, r.stdout + r.stderr
    assert "DELTA 0" in r.stdout
