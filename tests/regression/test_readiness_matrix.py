"""BL-33: the institutional-readiness matrix must carry auditable evidence.

The original matrix reduced seven domains to seven blanket
``IMPLEMENTED_UNVERIFIED`` rows.  That format could not distinguish a repository
control from an operational drill, an external blocker, or a missing capability.
These tests keep the register fail-closed without asserting that the institution is
ready.
"""

from __future__ import annotations

import ast
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
REVIEWED_EVIDENCE_TARGETS = {
    "SEC-01": frozenset(
        {
            "backlog/BL-08-incidente-env-historial.md",
            "../../../docs/SECURITY-env-leak-remediation.md",
        }
    ),
    "SEC-02": frozenset({"03-institutional-readiness.md"}),
    "TECH-01": frozenset({"../../../tests/regression/test_contract_mirrors.py"}),
    "TECH-02": frozenset({"backlog/BL-17-fingerprints-canonical-writer.md"}),
    "TECH-03": frozenset(
        {
            "backlog/BL-24-linaje-camino-dorado.md",
            "backlog/BL-29-qlab-cli-cutoff-lectura.md",
            "backlog/BL-38-market-canonical-resampleo.md",
        }
    ),
    "TECH-04": frozenset(
        {
            "../../../tests/regression/test_restore_resyncs_sequences.py",
            "../../../docs/operations/DISASTER_RECOVERY_PLAYBOOK.md",
        }
    ),
    "TECH-05": frozenset({"backlog/BL-25-monitoreo-tres-relojes.md"}),
    "TECH-06": frozenset({"../../../tests/unit/test_codex_safety_contracts.py"}),
    "RISK-01": frozenset(
        {
            "../platform/execution-bridge.md",
            "../../../tests/unit/test_codex_safety_contracts.py",
        }
    ),
    "RISK-02": frozenset(
        {
            "../../../tests/regression/test_trading_flags.py",
            "../../../tests/unit/test_command_pattern.py",
            "backlog/BL-31-strangler-cop.md",
        }
    ),
    "RISK-03": frozenset(
        {
            "../../../tests/regression/test_trial_ledger.py",
            "../../rules/quant-constitution.md",
        }
    ),
    "RISK-04": frozenset(
        {
            "backlog/BL-26-portfolio-snapshot.md",
            "backlog/BL-27-allocator-v1-novedad.md",
        }
    ),
    "RISK-05": frozenset({"03-institutional-readiness.md"}),
    "RISK-06": frozenset(
        {
            "backlog/BL-18-catalogo-motor-metricas.md",
            "../../../tests/unit/test_codex_safety_contracts.py",
        }
    ),
    "EXEC-01": frozenset(
        {
            "../../../tests/unit/test_codex_safety_contracts.py",
            "backlog/BL-30-execution-service-externo.md",
        }
    ),
    "EXEC-02": frozenset(
        {
            "backlog/BL-21-event-sourcing-exec.md",
            "backlog/BL-22-fact-position-pnl.md",
        }
    ),
    "EXEC-03": frozenset(
        {
            "../../rules/approval-gates.md",
            "../../../tests/regression/test_approval_store_private.py",
        }
    ),
    "EXEC-04": frozenset({"backlog/BL-31-strangler-cop.md"}),
    "EXEC-05": frozenset({"../assets/usdcop/WITHDRAWAL-PROTOCOL.md"}),
    "SEC-03": frozenset(
        {
            "../../rules/rbac.md",
            "../../../usdcop-trading-dashboard/lib/contracts/rbac.contract.ts",
            "../../../usdcop-trading-dashboard/scripts/check-rbac-coverage.mjs",
        }
    ),
    "SEC-04": frozenset({"backlog/BL-41-seguridad-db-p0.md"}),
    "SEC-05": frozenset({"../platform/authentication.md"}),
    "SEC-06": frozenset({"../../../docs/operations/INCIDENT_RESPONSE_PLAYBOOK.md"}),
    "COMP-01": frozenset(
        {
            "../../../tests/regression/test_bl09_bl11_bl12_governance.py",
            "../../../tests/regression/test_trial_ledger.py",
        }
    ),
    "COMP-02": frozenset({"../../rules/rbac.md"}),
    "COMP-03": frozenset({"03-institutional-readiness.md"}),
    "COMP-04": frozenset({"03-institutional-readiness.md"}),
    "OPS-01": frozenset({"../../../docs/operations/DISASTER_RECOVERY_PLAYBOOK.md"}),
    "OPS-02": frozenset(
        {
            "../../../docs/operations/INCIDENT_RESPONSE_PLAYBOOK.md",
            "03-institutional-readiness.md",
        }
    ),
    "OPS-03": frozenset({"backlog/BL-22-fact-position-pnl.md"}),
    "OPS-04": frozenset({"backlog/BL-25-monitoreo-tres-relojes.md"}),
    "OPS-05": frozenset({"03-institutional-readiness.md"}),
    "INV-01": frozenset(
        {
            "../../../tests/unit/test_codex_phase2_backlog.py",
            "../../../usdcop-trading-dashboard/tests/unit/api/synthetic-backtest-honesty.test.ts",
            "backlog/BL-43-demo-sintetica-aislada.md",
        }
    ),
    "INV-02": frozenset({"03-institutional-readiness.md"}),
    "INV-03": frozenset({"03-institutional-readiness.md"}),
    "INV-04": frozenset({"03-institutional-readiness.md"}),
}


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


def _evidence_targets(row: dict[str, str]) -> frozenset[str]:
    return frozenset(
        raw_target.split("#", 1)[0]
        for raw_target in LINK.findall(row["Observed evidence"])
    )


def _evidence_correspondence_errors(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    row_ids = {row["Control ID"] for row in rows}
    reviewed_ids = set(REVIEWED_EVIDENCE_TARGETS)
    if row_ids != reviewed_ids:
        errors.append(
            "reviewed evidence map differs from register ids: "
            f"missing={sorted(row_ids - reviewed_ids)}, extra={sorted(reviewed_ids - row_ids)}"
        )
    for row in rows:
        control_id = row["Control ID"]
        expected = REVIEWED_EVIDENCE_TARGETS.get(control_id)
        actual = _evidence_targets(row)
        if expected is not None and actual != expected:
            errors.append(
                f"{control_id}: evidence targets differ from reviewed correspondence; "
                f"expected={sorted(expected)}, actual={sorted(actual)}"
            )
    return errors


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


def test_every_control_uses_its_reviewed_evidence_targets() -> None:
    assert _evidence_correspondence_errors(_register()) == []


def test_existing_but_irrelevant_file_cannot_replace_reviewed_evidence() -> None:
    forged = [dict(row) for row in _register()]
    investor_reporting = next(row for row in forged if row["Control ID"] == "INV-04")
    investor_reporting["Observed evidence"] = "[licencia](../../../LICENSE)"

    errors = _evidence_correspondence_errors(forged)
    assert len(errors) == 1
    assert errors[0].startswith("INV-04: evidence targets differ")


def test_matrix_defines_state_semantics_and_scope_limit() -> None:
    text = MATRIX.read_text(encoding="utf-8")
    for state in ALLOWED_STATES:
        assert re.search(rf"^\| `{state}` \|", text, flags=re.MULTILINE), (
            f"state {state} is used without a definition"
        )
    assert "no autoriza capital" in text.lower()
    assert "caso b" in text.lower()
    assert "✅" not in text and "❌" not in text


def test_risk06_reports_current_metric_engine_gaps() -> None:
    risk06 = next(row for row in _register() if row["Control ID"] == "RISK-06")
    evidence = risk06["Observed evidence"].lower()

    assert "productor" in evidence and "consumidor" in evidence
    assert "allowlist" in evidence
    assert "falla hoy" not in evidence
    assert "annualization_by_asset" not in evidence


def test_removed_metric_engine_constructor_argument_is_not_reintroduced() -> None:
    offenders: list[str] = []
    for base in ("src", "services", "scripts", "airflow"):
        for path in (ROOT / base).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and any(
                    keyword.arg == "annualization_by_asset" for keyword in node.keywords
                ):
                    offenders.append(str(path.relative_to(ROOT)))

    assert offenders == [], (
        "MetricEngine callers must derive annualization from the asset registry; "
        f"obsolete annualization_by_asset callers: {sorted(set(offenders))}"
    )
