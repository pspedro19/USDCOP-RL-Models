#!/usr/bin/env python3
"""
check_monitor_delta.py — executable DELTA gate for CI monitors.

THE PROBLEM
-----------
`.github/workflows/fabric-contracts.yml` marks three steps `continue-on-error: true`
(tsc, Vitest, `test_knowledge_frontmatter`) because each drags measured pre-existing debt.
`continue-on-error` is a *declared* guarantee, not an *enforced* one: the step can never fail,
so a NEW failure introduced tomorrow is exactly as silent as the known ones. The gate becomes
decoration. This script replaces it with an enforced comparison:

    failures now  >  registered baseline  ->  exit 1  (regression: someone broke something)
    failures now  == registered baseline  ->  exit 0  (known debt, nothing new)
    failures now  <  registered baseline  ->  exit 0 + WARNING (debt paid; re-register it,
                                                                or the gate rots into laxity)
    cannot run / cannot parse / no baseline / prose contradicts JSON -> exit 2 (FAIL-CLOSED)

IDENTITY, NOT JUST COUNTING
---------------------------
"46 failures yesterday, 46 today" is NOT proof of no regression: one may have been fixed and
another introduced. Where the runner emits stable test identities (pytest junit-xml, Vitest
json), the comparison is a MULTISET of failure ids, so a swap is red even at constant count.
Where it does not (tsc has no structured output), the limitation is stated explicitly in the
report — see LIMITATIONS below.

LIMITATIONS (declared, not hidden)
----------------------------------
1. tsc has NO machine-readable diagnostic output (no --format json as of TS 5.x). This is the
   one text-parsed runner. Parsing is anchored on the stable `file(line,col): error TSxxxx: msg`
   contract of `--pretty false`, cross-checked against the process exit code; identity is
   (file, code, message) with line/col deliberately dropped so that shifting code does not
   fabricate regressions. Consequence: an error that MOVES inside the same file is treated as
   the same error (intended), and two byte-identical errors in one file are distinguished only
   by multiplicity. If tsc ever changes that output contract the parse cross-check fails closed.
2. A monitor registered WITHOUT a `failures` id list runs in count-only mode. Count-only CANNOT
   detect "same number, different failures". The report says so on every run; it is a declared
   downgrade, never a silent one.
3. Baselines are machine-measured artifacts. The numbers committed here were measured on the
   dev machine; the FIRST run on a CI runner must be reconciled there (`--update-baseline` on
   the runner, or accept a one-off red and re-register) because node/@types resolution and
   filesystem case-sensitivity can legitimately shift tsc/Vitest counts across platforms.

SINGLE SOURCE OF TRUTH
----------------------
`.claude/coordination/BASELINE.md` is the human record (prose, ownership, causes).
`.claude/coordination/BASELINE.monitors.json` is the machine record (commands, counts, ids).
They are NOT allowed to become two truths: every monitor declares regexes that must still match
the prose, and — where the prose states an exact number — that number is compared against the
JSON count. Any divergence is FAIL-CLOSED, in either direction.

Usage
-----
    python scripts/validation/check_monitor_delta.py --all
    python scripts/validation/check_monitor_delta.py --monitor tsc_dashboard
    python scripts/validation/check_monitor_delta.py --monitor vitest_dashboard_unit --json
    python scripts/validation/check_monitor_delta.py --monitor X --update-baseline   # re-measure

Contract: CTR-CI-DELTA-001
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / ".claude" / "coordination" / "BASELINE.monitors.json"

EXIT_OK = 0
EXIT_REGRESSION = 1
EXIT_FAIL_CLOSED = 2


class ExtractionError(Exception):
    """The monitor ran but its result could not be extracted with confidence."""


class FailClosed(Exception):
    """Anything that must stop the gate instead of letting it pass by default."""


# ---------------------------------------------------------------------------
# observation model
# ---------------------------------------------------------------------------
@dataclass
class Observation:
    """What a monitor reported, in two dimensions.

    primary   = failing tests / error diagnostics
    secondary = collection-level failures (pytest errors, Vitest suites that never ran)
                — tracked apart because they are a different kind of breakage and a
                  collapsing suite would otherwise LOWER the primary count and look green.
    """

    ids: list[str] = field(default_factory=list)
    secondary_ids: list[str] = field(default_factory=list)
    exact_identity: bool = True

    @property
    def count(self) -> int:
        return len(self.ids)

    @property
    def secondary_count(self) -> int:
        return len(self.secondary_ids)


# ---------------------------------------------------------------------------
# runner adapters — structured artifacts wherever the tool offers one
# ---------------------------------------------------------------------------
def parse_pytest_junit(report: Path | None, stdout: str, stderr: str) -> Observation:
    """pytest --junitxml: a stable, versioned XML contract (not screen-scraped text)."""
    if report is None or not report.is_file():
        raise ExtractionError("pytest produced no junit-xml report")
    try:
        tree = ET.parse(report)
    except ET.ParseError as exc:
        raise ExtractionError(f"junit-xml is not parseable: {exc}") from exc

    root = tree.getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    failed: list[str] = []
    errored: list[str] = []
    for suite in suites:
        for case in suite.iter("testcase"):
            cid = f"{case.get('classname', '')}::{case.get('name', '')}"
            if case.find("failure") is not None:
                failed.append(cid)
            elif case.find("error") is not None:
                errored.append(cid)

    declared_f = sum(int(s.get("failures", 0) or 0) for s in suites)
    declared_e = sum(int(s.get("errors", 0) or 0) for s in suites)
    if (declared_f, declared_e) != (len(failed), len(errored)):
        raise ExtractionError(
            "junit-xml is inconsistent with itself: header declares "
            f"{declared_f} failures/{declared_e} errors but {len(failed)}/{len(errored)} "
            "testcases carry them"
        )
    return Observation(ids=sorted(failed), secondary_ids=sorted(errored))


def parse_vitest_json(report: Path | None, stdout: str, stderr: str) -> Observation:
    """vitest --reporter=json: structured, includes its own totals to cross-check against."""
    if report is None or not report.is_file():
        raise ExtractionError("vitest produced no json report")
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ExtractionError(f"vitest json is not parseable: {exc}") from exc

    failed: list[str] = []
    dead_suites: list[str] = []
    for suite in data.get("testResults", []):
        rel = _relativize(suite.get("name", "?"))
        asserts = suite.get("assertionResults") or []
        suite_failed = [a for a in asserts if a.get("status") == "failed"]
        for a in suite_failed:
            failed.append(f"{rel}::{a.get('fullName') or a.get('title')}")
        # a suite that failed without running a single test = collection/import failure
        if suite.get("status") == "failed" and not suite_failed:
            dead_suites.append(f"{rel}::<suite did not run>")

    declared = data.get("numFailedTests")
    if declared is None:
        raise ExtractionError("vitest json has no numFailedTests to cross-check against")
    if int(declared) != len(failed):
        raise ExtractionError(
            f"vitest json is inconsistent with itself: numFailedTests={declared} "
            f"but {len(failed)} failed assertions were enumerated"
        )
    return Observation(ids=sorted(failed), secondary_ids=sorted(dead_suites))


_TSC_DIAG = re.compile(
    r"^(?P<file>[^\s(][^(]*)\((?P<line>\d+),(?P<col>\d+)\):\s+error\s+(?P<code>TS\d+):\s+(?P<msg>.*)$"
)
_TSC_GLOBAL = re.compile(r"^error\s+(?P<code>TS\d+):\s+(?P<msg>.*)$")


def parse_tsc_text(report: Path | None, stdout: str, stderr: str) -> Observation:
    """tsc has no structured output (no --format json in TS 5.x).

    We anchor on the `--pretty false` diagnostic contract: one diagnostic starts a line and
    begins in column 0; explanatory continuation lines are indented. Identity drops line/col
    on purpose (see LIMITATIONS #1). Cross-checked against the exit code by the caller.
    """
    ids: list[str] = []
    for raw in (stdout + "\n" + stderr).splitlines():
        line = raw.rstrip("\r")
        if not line or line[0].isspace():
            continue  # continuation of the previous diagnostic
        m = _TSC_DIAG.match(line)
        if m:
            ids.append(f"{m['file'].replace(os.sep, '/')}|{m['code']}|{m['msg'].strip()}")
            continue
        g = _TSC_GLOBAL.match(line)
        if g:
            ids.append(f"<global>|{g['code']}|{g['msg'].strip()}")
    return Observation(ids=sorted(ids))


ADAPTERS: dict[str, dict] = {
    # needs_report: the runner must be told where to write a machine-readable artifact
    "pytest": {"parse": parse_pytest_junit, "needs_report": True, "suffix": ".xml",
               "ok_exits": [0, 1], "artifact": "junit-xml"},
    "vitest": {"parse": parse_vitest_json, "needs_report": True, "suffix": ".json",
               "ok_exits": [0, 1], "artifact": "json reporter"},
    # tsc: 0 = clean, 1 = diagnostics present / outputs skipped, 2 = diagnostics + outputs
    "tsc": {"parse": parse_tsc_text, "needs_report": False, "suffix": ".txt",
            "ok_exits": [0, 1, 2], "artifact": "stdout text (no structured output exists)"},
}


def _relativize(path: str) -> str:
    p = path.replace("\\", "/")
    for anchor in ("/usdcop-trading-dashboard/", "/tests/"):
        if anchor in p:
            return p[p.index(anchor) + 1:] if anchor == "/tests/" else p.split(anchor, 1)[1]
    return p


# ---------------------------------------------------------------------------
# running a monitor
# ---------------------------------------------------------------------------
def _resolve_exe(argv: list[str]) -> list[str]:
    exe = argv[0]
    if Path(exe).is_file():
        return argv
    found = shutil.which(exe)
    if found is None and sys.platform == "win32":
        for ext in (".cmd", ".exe", ".bat"):
            found = shutil.which(exe + ext)
            if found:
                break
    if found is None:
        raise FailClosed(f"could not run monitor: executable not found: {exe!r}")
    return [found, *argv[1:]]


def run_monitor(name: str, cfg: dict) -> Observation:
    runner = cfg.get("runner")
    adapter = ADAPTERS.get(runner or "")
    if adapter is None:
        raise FailClosed(f"{name}: unknown runner {runner!r} (known: {sorted(ADAPTERS)})")

    command = cfg.get("command")
    if not command or not isinstance(command, list):
        raise FailClosed(f"{name}: baseline entry has no runnable 'command' list")

    cwd = (PROJECT_ROOT / cfg.get("cwd", ".")).resolve()
    if not cwd.is_dir():
        raise FailClosed(f"{name}: could not run monitor: cwd does not exist: {cwd}")

    tmpdir = Path(tempfile.mkdtemp(prefix="monitor-delta-"))
    report = tmpdir / f"report{adapter['suffix']}"
    argv = [str(a).replace("{report}", str(report)) for a in command]
    if adapter["needs_report"] and not any("{report}" in str(a) for a in command):
        raise FailClosed(
            f"{name}: runner {runner!r} needs a machine-readable report but the command has no "
            "{report} placeholder — refusing to screen-scrape stdout"
        )

    try:
        argv = _resolve_exe(argv)
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=cfg.get("timeout_s", 2400),
        )
    except FailClosed:
        raise
    except FileNotFoundError as exc:
        raise FailClosed(f"{name}: could not run monitor: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise FailClosed(f"{name}: could not run monitor: timed out after {exc.timeout}s") from exc
    except OSError as exc:
        raise FailClosed(f"{name}: could not run monitor: {exc}") from exc

    ok_exits = cfg.get("ok_exits", adapter["ok_exits"])
    if proc.returncode not in ok_exits:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        raise FailClosed(
            f"{name}: monitor exited {proc.returncode} (expected one of {ok_exits}) — the run "
            "itself is broken, its numbers mean nothing. Last output:\n    "
            + "\n    ".join(tail)
        )

    try:
        obs = adapter["parse"](report if adapter["needs_report"] else None, proc.stdout, proc.stderr)
    except ExtractionError as exc:
        raise FailClosed(f"{name}: {exc}") from exc

    # exit code vs parsed content coherence (the only cross-check tsc can offer)
    if proc.returncode == 0 and (obs.count or obs.secondary_count):
        raise FailClosed(
            f"{name}: monitor exited 0 but {obs.count} failures were parsed — inconsistent, "
            "refusing to guess"
        )
    if proc.returncode != 0 and not (obs.count or obs.secondary_count):
        raise FailClosed(
            f"{name}: monitor exited {proc.returncode} but 0 failures could be extracted — "
            "the output format probably changed; refusing to report a green delta"
        )
    return obs


# ---------------------------------------------------------------------------
# BASELINE.md corroboration (one truth, enforced)
# ---------------------------------------------------------------------------
def check_md_sync(name: str, cfg: dict, md_text: str, md_path: Path) -> list[str]:
    spec = cfg.get("baseline_md")
    if not spec or not spec.get("assertions"):
        raise FailClosed(
            f"{name}: no BASELINE.md corroboration declared — a machine baseline nobody wrote "
            "down in prose is a second truth waiting to drift"
        )
    notes: list[str] = []
    for pattern in spec["assertions"]:
        if not re.search(pattern, md_text, re.MULTILINE):
            raise FailClosed(
                f"{name}: BASELINE.md no longer corroborates this monitor (pattern not found: "
                f"{pattern!r} in {md_path}). Re-measure and update both records."
            )
    if spec.get("number_authoritative"):
        pat = spec.get("number_pattern")
        if not pat:
            raise FailClosed(f"{name}: number_authoritative but no number_pattern declared")
        m = re.search(pat, md_text, re.MULTILINE)
        if not m:
            raise FailClosed(f"{name}: BASELINE.md number not found (pattern {pat!r})")
        if int(m.group(1)) != int(cfg["count"]):
            raise FailClosed(
                f"{name}: TWO TRUTHS — BASELINE.md prose says {m.group(1)}, "
                f"BASELINE.monitors.json says {cfg['count']}. Fix both before trusting the gate."
            )
    else:
        notes.append(
            f"{name}: BASELINE.md states this monitor in APPROXIMATE prose "
            f"({spec.get('approximate_reason', 'no exact number declared')}); only the "
            "monitor's presence is corroborated, the exact number lives in the JSON."
        )
    return notes


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
@dataclass
class Verdict:
    name: str
    status: str            # OK | REGRESSION | IMPROVED
    count: int
    baseline_count: int
    secondary_count: int
    baseline_secondary_count: int
    new_ids: list[str] = field(default_factory=list)
    fixed_ids: list[str] = field(default_factory=list)
    new_secondary: list[str] = field(default_factory=list)
    fixed_secondary: list[str] = field(default_factory=list)
    identity_mode: str = "identity"
    notes: list[str] = field(default_factory=list)

    @property
    def delta(self) -> int:
        return self.count - self.baseline_count

    @property
    def secondary_delta(self) -> int:
        return self.secondary_count - self.baseline_secondary_count


def _multiset_diff(now: Iterable[str], base: Iterable[str]) -> tuple[list[str], list[str]]:
    c_now, c_base = Counter(now), Counter(base)
    new = sorted((c_now - c_base).elements())
    fixed = sorted((c_base - c_now).elements())
    return new, fixed


def compare(name: str, cfg: dict, obs: Observation) -> Verdict:
    if "count" not in cfg or not isinstance(cfg["count"], int):
        raise FailClosed(f"{name}: no baseline count registered — refusing to pass by default")
    base_count = int(cfg["count"])
    base_secondary = int(cfg.get("secondary_count", 0))

    base_ids = cfg.get("failures")
    base_secondary_ids = cfg.get("secondary_failures")
    identity = isinstance(base_ids, list)

    v = Verdict(
        name=name,
        status="OK",
        count=obs.count,
        baseline_count=base_count,
        secondary_count=obs.secondary_count,
        baseline_secondary_count=base_secondary,
        identity_mode="identity" if identity else "count-only",
    )

    if identity:
        if len(base_ids) != base_count:
            raise FailClosed(
                f"{name}: baseline is self-inconsistent — count={base_count} but "
                f"{len(base_ids)} failure ids registered"
            )
        v.new_ids, v.fixed_ids = _multiset_diff(obs.ids, base_ids)
        if isinstance(base_secondary_ids, list):
            if len(base_secondary_ids) != base_secondary:
                raise FailClosed(
                    f"{name}: baseline is self-inconsistent in the secondary dimension"
                )
            v.new_secondary, v.fixed_secondary = _multiset_diff(
                obs.secondary_ids, base_secondary_ids
            )
    else:
        v.notes.append(
            "count-only baseline: this monitor compares TOTALS, so it CANNOT detect "
            "'same number, different failures' (a fixed test masking a newly broken one). "
            "Register a `failures` id list to close that hole."
        )

    if v.new_ids or v.new_secondary:
        v.status = "REGRESSION"
    elif v.delta > 0 or v.secondary_delta > 0:
        v.status = "REGRESSION"
    elif v.delta < 0 or v.secondary_delta < 0 or v.fixed_ids or v.fixed_secondary:
        v.status = "IMPROVED"
    return v


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def render(v: Verdict, fail_on_decrease: bool) -> str:
    lines = []
    head = {
        "OK": "PASS",
        "REGRESSION": "FAIL",
        "IMPROVED": "WARN",
    }[v.status]
    sign = f"{v.delta:+d}" if v.delta else "0"
    sign2 = f"{v.secondary_delta:+d}" if v.secondary_delta else "0"
    lines.append(
        f"[{head}] {v.name}: {v.count} failures vs baseline {v.baseline_count} "
        f"=> DELTA {sign}"
        + (
            f" | secondary {v.secondary_count} vs {v.baseline_secondary_count} "
            f"(DELTA {sign2})"
            if (v.secondary_count or v.baseline_secondary_count)
            else ""
        )
        + f" [{v.identity_mode}]"
    )
    for i in v.new_ids:
        lines.append(f"    NEW FAILURE      {i}")
    for i in v.new_secondary:
        lines.append(f"    NEW SUITE ERROR  {i}")
    for i in v.fixed_ids:
        lines.append(f"    no longer failing {i}")
    for i in v.fixed_secondary:
        lines.append(f"    suite recovered   {i}")
    for n in v.notes:
        lines.append(f"    note: {n}")
    if v.status == "REGRESSION":
        lines.append(
            "    => This is a NEW failure, not registered debt. It is exactly what "
            "`continue-on-error` would have hidden."
        )
    if v.status == "IMPROVED":
        lines.append(
            "    => Debt was paid. Re-register it (`--update-baseline`) and update "
            "BASELINE.md, or the gate silently accepts these failures coming back."
        )
        if fail_on_decrease:
            lines.append("    => --fail-on-decrease is set: treating stale baseline as red.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# baseline update
# ---------------------------------------------------------------------------
def update_baseline(config_path: Path, config: dict, name: str, obs: Observation) -> None:
    entry = config["monitors"][name]
    entry["count"] = obs.count
    entry["secondary_count"] = obs.secondary_count
    entry["failures"] = obs.ids
    entry["secondary_failures"] = obs.secondary_ids
    entry["measured_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CI wiring proposal (PROPOSAL for the workflow owner — this script never edits
# .github/workflows/**; printing it keeps the suggested block versioned with the tool)
# ---------------------------------------------------------------------------
CI_YAML_PROPOSAL = """\
# ---------------------------------------------------------------------------
# PROPOSAL for .github/workflows/fabric-contracts.yml (owner: CODEX).
# Replaces the three `continue-on-error: true` steps. Rationale: continue-on-error
# cannot fail, so a NEW failure is as silent as the 336/46/47 registered ones.
# The comparator fails ONLY on the delta, so the debt stays tolerated but frozen.
# Note: `python -m pip install -r requirements.txt` already runs in python-contracts;
# the dashboard job needs no extra install (the script is stdlib-only), just python.
# ---------------------------------------------------------------------------
  dashboard-contracts:
    steps:
      # ... npm ci / rbac:check / rbac:test unchanged ...
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: TypeScript (delta vs registered baseline)
        working-directory: .            # the comparator resolves cwd from the baseline entry
        run: python scripts/validation/check_monitor_delta.py --monitor tsc_dashboard

      - name: Vitest (delta vs registered baseline)
        working-directory: .
        run: python scripts/validation/check_monitor_delta.py --monitor vitest_dashboard_unit

  python-contracts:
    steps:
      # ... checkout / setup-python / pip install unchanged ...
      - name: Knowledge front-matter (delta vs registered baseline)
        run: python scripts/validation/check_monitor_delta.py --monitor pytest_knowledge_frontmatter

      # the other two monitors that were bundled into that step have NO debt and must
      # stay hard-red, exactly as they are today:
      - run: python -m pytest tests/regression/test_strategy_manifests.py tests/regression/test_scripts_layout.py -q

      - name: Publish baseline contract
        run: |
          test -f .claude/coordination/BASELINE.md
          test -f .claude/coordination/BASELINE.monitors.json
          python scripts/validation/check_monitor_delta.py --all --json > monitor-delta.json || true
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: monitor-delta
          path: monitor-delta.json
"""


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fail CI on NEW failures while tolerating registered pre-existing debt.",
    )
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--monitor", action="append", default=[], help="monitor name (repeatable)")
    ap.add_argument("--all", action="store_true", help="run every registered monitor")
    ap.add_argument("--fail-on-decrease", action="store_true",
                    help="treat a stale (too high) baseline as red instead of a warning")
    ap.add_argument("--update-baseline", action="store_true",
                    help="re-measure and rewrite the baseline for the selected monitors")
    ap.add_argument("--json", action="store_true", help="machine-readable verdict on stdout")
    ap.add_argument("--print-ci-yaml", action="store_true",
                    help="print the proposed workflow block (proposal only; this tool never "
                         "edits .github/workflows/**)")
    args = ap.parse_args(argv)

    # Failure ids are arbitrary text (Vitest test titles carry accents and "=>" glyphs,
    # file paths carry non-ASCII). A gate that crashes while PRINTING a regression would
    # be a fail-open by accident on a cp1252 console.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass

    if args.print_ci_yaml:
        print(CI_YAML_PROPOSAL)
        return EXIT_OK

    try:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = (PROJECT_ROOT / config_path).resolve()
        if not config_path.is_file():
            raise FailClosed(f"baseline config not found: {config_path}")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FailClosed(f"baseline config is not valid JSON: {exc}") from exc

        monitors = config.get("monitors") or {}
        if args.all:
            selected = sorted(monitors)
        elif args.monitor:
            selected = args.monitor
        else:
            raise FailClosed("nothing selected: pass --monitor NAME or --all")

        unknown = [m for m in selected if m not in monitors]
        if unknown:
            raise FailClosed(
                f"no baseline registered for monitor(s): {unknown}. A monitor without a measured "
                f"baseline cannot be judged — registered: {sorted(monitors)}"
            )

        md_path = Path(config.get("baseline_md", ".claude/coordination/BASELINE.md"))
        if not md_path.is_absolute():
            md_path = (PROJECT_ROOT / md_path).resolve()
        if not md_path.is_file():
            raise FailClosed(f"human baseline record not found: {md_path}")
        md_text = md_path.read_text(encoding="utf-8")

        verdicts: list[Verdict] = []
        notes: list[str] = []
        for name in selected:
            cfg = monitors[name]
            obs = run_monitor(name, cfg)
            if args.update_baseline:
                update_baseline(config_path, config, name, obs)
                print(f"[baseline] {name}: re-measured => {obs.count} failures "
                      f"({obs.secondary_count} secondary), ids recorded in {config_path.name}")
            notes += check_md_sync(name, monitors[name], md_text, md_path)
            verdicts.append(compare(name, monitors[name], obs))
    except FailClosed as exc:
        print(f"[FAIL-CLOSED] {exc}")
        print("[FAIL-CLOSED] The gate refuses to pass when it cannot judge. Exit 2.")
        return EXIT_FAIL_CLOSED

    for n in notes:
        print(f"[note] {n}")
    for v in verdicts:
        print(render(v, args.fail_on_decrease))
    if args.json:
        print(json.dumps([v.__dict__ for v in verdicts], indent=2, ensure_ascii=False))

    if any(v.status == "REGRESSION" for v in verdicts):
        return EXIT_REGRESSION
    if args.fail_on_decrease and any(v.status == "IMPROVED" for v in verdicts):
        return EXIT_REGRESSION
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
