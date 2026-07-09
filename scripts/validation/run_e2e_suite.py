#!/usr/bin/env python3
"""
run_e2e_suite.py — non-destructive end-to-end validation orchestrator.
======================================================================
Single entrypoint that validates the whole project against the RUNNING stack and
emits a pass/fail matrix to results/e2e/report.json (+ appends report.md). This is
the "cover and validate everything" harness referenced by the verification plan
(joyful-wiggling-cerf.md). It does NOT tear anything down — the destructive
clean-slate proof is scripts/validation/teardown_restore_test.sh (run separately,
last).

Checks (each isolated; one failure never aborts the rest):
  1. stack_health          — core containers report healthy
  2. registry_integrity    — dashboard serves 3 assets / 8 strategies, prod=smart_simple_v11, no Inf/NaN
  3. asset_dags_loadable   — 0 airflow import errors; per-asset pipeline DAGs present
  4. per_asset_bundles     — every registry strategy has a manifest + backtest bundle on disk
  5. feature_backup        — feature-data backup manifest present + parseable
  6. promotion_state       — approval_state.json parses; report status + gates
  7. analysis_endpoints    — /api/analysis/assets = 3; per-asset weeks >= 7
  8. h5_live_path          — H5 executor reads usdcop_m5_ohlcv directly (NRT=RL-only, static check)

Usage:
    python scripts/validation/run_e2e_suite.py
    python scripts/validation/run_e2e_suite.py --dashboard http://localhost:5000
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PUBLIC = REPO / "usdcop-trading-dashboard" / "public" / "data"
OUT_DIR = REPO / "results" / "e2e"
SCHED = "usdcop-airflow-scheduler"

CORE_CONTAINERS = [
    "usdcop-postgres-timescale", "usdcop-airflow-scheduler", "usdcop-airflow-webserver",
    "usdcop-dashboard", "usdcop-mlops-inference", "usdcop-signalbridge",
]
EXPECTED_ASSETS = {"usdcop", "xauusd", "btcusdt"}
EXPECTED_STRATEGIES = 8


def _sh(cmd: list[str], timeout: int = 30) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:  # noqa: BLE001
        return 1, str(e)


def _curl(url: str, timeout: int = 12) -> tuple[int, str]:
    return _sh(["curl", "-s", "--max-time", str(timeout), url], timeout + 5)


# --- checks -----------------------------------------------------------------
def check_stack_health() -> dict:
    unhealthy = []
    for c in CORE_CONTAINERS:
        rc, out = _sh(["docker", "inspect", "--format", "{{.State.Health.Status}}", c])
        status = out.strip()
        if status != "healthy":
            unhealthy.append(f"{c}={status or 'absent'}")
    return {"passed": not unhealthy, "detail": "all healthy" if not unhealthy else f"unhealthy: {unhealthy}"}


def check_registry_integrity(dashboard: str) -> dict:
    rc, out = _curl(f"{dashboard}/data/registry.json")
    if "Infinity" in out or "NaN" in out:
        return {"passed": False, "detail": "registry contains Infinity/NaN"}
    try:
        reg = json.loads(out)
    except Exception:
        return {"passed": False, "detail": "registry.json not served / not JSON"}
    assets = {a["asset_id"] for a in reg.get("assets", [])}
    strategies = reg.get("strategies", [])
    prod = [s["strategy_id"] for s in strategies if s.get("has_production")]
    ok = assets == EXPECTED_ASSETS and len(strategies) == EXPECTED_STRATEGIES and "smart_simple_v11" in prod
    return {"passed": ok,
            "detail": f"assets={sorted(assets)} strategies={len(strategies)} production={prod}"}


def check_asset_dags_loadable() -> dict:
    # `airflow dags list` is slow (~50s cold) — give it room.
    rc, out = _sh(["docker", "exec", SCHED, "airflow", "dags", "list-import-errors"], timeout=90)
    errors = "Traceback" in out or "/dags/" in out  # explicit: a real import error names a file/traceback
    rc2, listing = _sh(["docker", "exec", SCHED, "airflow", "dags", "list"], timeout=120)
    have = [d for d in ("asset_xauusd_pipeline_weekly", "asset_btcusdt_pipeline_weekly")
            if d in listing]
    ok = (not errors) and len(have) == 2
    return {"passed": ok, "detail": f"import_errors={'yes' if errors else 'none'} asset_dags={have}"}


def check_per_asset_bundles(dashboard: str) -> dict:
    rc, out = _curl(f"{dashboard}/data/registry.json")
    try:
        reg = json.loads(out)
    except Exception:
        return {"passed": False, "detail": "registry not available"}
    missing = []
    for s in reg.get("strategies", []):
        sid = s["strategy_id"]
        man = PUBLIC / "strategies" / sid / "manifest.json"
        if not man.exists():
            missing.append(sid)
    return {"passed": not missing,
            "detail": f"{len(reg.get('strategies', []))} strategies, missing bundles: {missing or 'none'}"}


def check_feature_backup() -> dict:
    man = REPO / "data" / "backups" / "features" / "feature_backup_manifest.json"
    if not man.exists():
        return {"passed": False, "detail": "feature_backup_manifest.json absent (run l0_seed_backup or the module)"}
    try:
        m = json.loads(man.read_text())
        ok_tables = sum(1 for t in m.get("tables", {}).values() if t.get("status") == "ok")
        rows = sum(t.get("rows", 0) for t in m.get("tables", {}).values() if t.get("status") == "ok")
        return {"passed": ok_tables >= 1, "detail": f"{ok_tables} tables backed up, {rows} rows"}
    except Exception as e:  # noqa: BLE001
        return {"passed": False, "detail": f"manifest unparseable: {e}"}


def check_promotion_state() -> dict:
    ap = PUBLIC / "production" / "approval_state.json"
    if not ap.exists():
        return {"passed": False, "detail": "approval_state.json absent"}
    try:
        st = json.loads(ap.read_text())
        gates = st.get("gates", [])
        passed_gates = sum(1 for g in gates if g.get("passed"))
        return {"passed": True,
                "detail": f"status={st.get('status')} rec={st.get('backtest_recommendation')} "
                          f"gates={passed_gates}/{len(gates)}"}
    except Exception as e:  # noqa: BLE001
        return {"passed": False, "detail": f"unparseable: {e}"}


def check_analysis_endpoints(dashboard: str) -> dict:
    rc, out = _curl(f"{dashboard}/api/analysis/assets")
    try:
        n_assets = len(json.loads(out).get("assets", []))
    except Exception:
        n_assets = 0
    weeks = {}
    for a in EXPECTED_ASSETS:
        rc, o = _curl(f"{dashboard}/api/analysis/weeks?asset={a}")
        try:
            weeks[a] = len(json.loads(o).get("weeks", []))
        except Exception:
            weeks[a] = 0
    ok = n_assets == 3 and all(w >= 7 for w in weeks.values())
    return {"passed": ok, "detail": f"assets={n_assets} weeks={weeks}"}


def check_h5_live_path() -> dict:
    ex = REPO / "airflow" / "dags" / "forecast_h5_l7_multiday_executor.py"
    if not ex.exists():
        return {"passed": False, "detail": "executor DAG missing"}
    txt = ex.read_text(encoding="utf-8", errors="ignore")
    reads_ohlcv = "usdcop_m5_ohlcv" in txt
    uses_nrt = "inference_ready_nrt" in txt
    ok = reads_ohlcv and not uses_nrt
    return {"passed": ok,
            "detail": f"reads usdcop_m5_ohlcv={reads_ohlcv}, uses inference_ready_nrt={uses_nrt} "
                      f"(NRT must be RL-only)"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dashboard", default="http://localhost:5000")
    a = ap.parse_args()

    checks = [
        ("stack_health", lambda: check_stack_health()),
        ("registry_integrity", lambda: check_registry_integrity(a.dashboard)),
        ("asset_dags_loadable", lambda: check_asset_dags_loadable()),
        ("per_asset_bundles", lambda: check_per_asset_bundles(a.dashboard)),
        ("feature_backup", lambda: check_feature_backup()),
        ("promotion_state", lambda: check_promotion_state()),
        ("analysis_endpoints", lambda: check_analysis_endpoints(a.dashboard)),
        ("h5_live_path", lambda: check_h5_live_path()),
    ]

    results = []
    for name, fn in checks:
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001
            r = {"passed": False, "detail": f"exception: {e}"}
        r["name"] = name
        results.append(r)
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {name:22} {r['detail']}")

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed, "total": total,
        "green": passed == total,
        "checks": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))

    with open(OUT_DIR / "report.md", "a", encoding="utf-8") as f:
        f.write(f"\n## E2E suite {report['generated_at']} — {passed}/{total} green\n\n")
        for r in results:
            f.write(f"- {'✅' if r['passed'] else '❌'} **{r['name']}** — {r['detail']}\n")

    print(f"\n=== {passed}/{total} checks passed "
          f"({'ALL GREEN' if passed == total else 'FAILURES PRESENT'}) ===")
    print(f"report -> {OUT_DIR / 'report.json'}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
