"""Multi-domain quality harness for the trading platform.

The engine is intentionally evidence-first: every domain returns PASS/FAIL/BLOCKED,
and the aggregate can never be green while a required artifact is missing. It is safe
to run in CI without credentials; external gates become BLOCKED with a reason.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CODEX = ROOT / ".claude" / "codex"
ASSETS = ("usdcop", "xauusd", "btcusdt", "spx500")


@dataclass
class Gate:
    name: str
    status: str
    evidence: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    command: str | None = None


def _file_gate(name: str, paths: list[str]) -> Gate:
    missing = [p for p in paths if not (ROOT / p).exists()]
    return Gate(name, "PASS" if not missing else "BLOCKED", paths if not missing else [],
                [] if not missing else [f"missing:{p}" for p in missing])


def _pytest_gate(name: str, paths: list[str]) -> Gate:
    cmd = [sys.executable, "-m", "pytest", *paths, "-q"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=180)
    return Gate(name, "PASS" if proc.returncode == 0 else "FAIL",
                ["stdout: " + proc.stdout[-1000:]],
                [] if proc.returncode == 0 else [proc.stderr[-1000:]], " ".join(cmd))


def _command_gate(name: str, command: list[str]) -> Gate:
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
    return Gate(name, "PASS" if proc.returncode == 0 else "FAIL",
                [proc.stdout[-1200:]], [] if proc.returncode == 0 else [proc.stderr[-1200:]],
                " ".join(command))


def _market_statistics_gate() -> Gate:
    """Require a fresh descriptive audit; REVIEW_REQUIRED is not promotable."""
    path = CODEX / "evidence" / "market-data-statistics.json"
    if not path.exists():
        return Gate("market-data-statistics", "BLOCKED", [], ["missing: .claude/codex/evidence/market-data-statistics.json"])
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return Gate("market-data-statistics", "FAIL", [], [f"invalid evidence: {exc}"])
    decision = report.get("decision")
    if decision != "PASS":
        return Gate("market-data-statistics", "BLOCKED", [str(path)],
                    [f"audit decision={decision}; PIT lineage and feature review remain open"])
    return Gate("market-data-statistics", "PASS", [str(path)], [])


def _acquisition_assets_gate() -> Gate:
    path = CODEX / "evidence" / "acquisition-assets-audit.json"
    if not path.exists():
        return Gate("acquisition-assets", "BLOCKED", [], ["missing acquisition asset audit"])
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return Gate("acquisition-assets", "FAIL", [], [f"invalid evidence: {exc}"])
    if report.get("decision") != "PASS":
        return Gate("acquisition-assets", "BLOCKED", [str(path)],
                    [f"audit decision={report.get('decision')}; reconcile seeds/backups and prove provider runs"])
    return Gate("acquisition-assets", "PASS", [str(path)], [])


def _acquisition_manifest_gate() -> Gate:
    path = CODEX / "evidence" / "acquisition-manifest-validation.json"
    if not path.exists():
        return Gate("acquisition-manifests", "BLOCKED", [], ["missing provider execution manifests"])
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("decision") != "PASS":
        return Gate("acquisition-manifests", "BLOCKED", [str(path)], ["provider run evidence incomplete"])
    return Gate("acquisition-manifests", "PASS", [str(path)], [])


def build_gates(run_tests: bool = True) -> list[Gate]:
    gates = [
        _file_gate("commerce-contracts", [
            "database/migrations/057_catalog_watchlist_cart.sql",
            "database/migrations/058_billing_webhook_idempotency.sql",
            "database/migrations/059_checkout_order_ledger.sql",
            "usdcop-trading-dashboard/app/api/cart/checkout/route.ts",
            "usdcop-trading-dashboard/app/api/billing/webhook/route.ts",
        ]),
        _file_gate("rbac-contracts", [
            "usdcop-trading-dashboard/lib/contracts/rbac.contract.ts",
            "usdcop-trading-dashboard/lib/api/relay.ts",
            ".claude/codex/RBAC-CART-AUDIT-2026-07-20.md",
        ]),
        _file_gate("news-safety", [
            "usdcop-trading-dashboard/lib/security/safe-url.ts",
            ".claude/codex/FRONTEND-NEWS-CART-AUDIT.md",
        ]),
        _file_gate("quantitative-contracts", [
            "config/assets/spx500.yaml",
            "src/validation/sp500_oos_gate.py",
            "config/forecast_experiments/spx500_regime_gated_v1.yaml",
        ]),
        _market_statistics_gate(),
        _acquisition_assets_gate(),
        _acquisition_manifest_gate(),
    ]
    if run_tests:
        gates.extend([
            _command_gate("commerce-tests", [sys.executable, "scripts/validation/commerce_rbac_harness.py", "--json"]),
            _pytest_gate("asset-tests", [
                "tests/regression/test_spx500_integration.py",
                "tests/unit/test_sp500_oos_gate.py",
                "tests/integration/test_spx500_pipeline_config.py",
            ]),
            _pytest_gate("quant-tests", ["tests/unit/test_quant_harness.py"]),
            _pytest_gate("production-tests", ["tests/unit/test_production_harness.py"]),
        ])
    return gates


def run(output: Path, run_tests: bool = True) -> dict[str, Any]:
    gates = build_gates(run_tests)
    # External production evidence is intentionally blocked until supplied.
    gates.append(Gate("real-data-oos", "BLOCKED", [], [
        "requires real PIT datasets and OOS evidence for all four assets",
    ]))
    gates.append(Gate("provider-e2e", "BLOCKED", [], [
        "requires sandbox provider credentials and disposable test tenant",
    ]))
    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "assets": list(ASSETS),
        "gates": [asdict(g) for g in gates],
        "passed": all(g.status == "PASS" for g in gates),
        "decision": "GO" if all(g.status == "PASS" for g in gates) else "NO-GO",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=CODEX / "evidence" / "harness-latest.json")
    parser.add_argument("--no-tests", action="store_true")
    args = parser.parse_args()
    result = run(args.output, run_tests=not args.no_tests)
    print(json.dumps({"decision": result["decision"], "output": str(args.output),
                      "gates": {g["name"]: g["status"] for g in result["gates"]}}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
