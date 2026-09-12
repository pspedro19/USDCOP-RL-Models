#!/usr/bin/env python
"""Create an honest, local-only E2E status for EXP-TESIS-RL-02.

This report never loads dotenv, calls a provider, or treats a smoke run as a
confirmatory experiment.  It is intended as the handoff artifact for an operator
who has to sign the preregistration and execute the expensive PPO/LLM stages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# El portable guarda objetos de `src.research`, asi que despicklearlo necesita la raiz del
# repo en `sys.path`. Invocado como fichero (`python scripts/diagnostics/...`) no la tiene y
# el audit moria con `ModuleNotFoundError: No module named 'src'` justo en el control que
# debia demostrar que el portable es legible. Tercer script de esta serie con el mismo fallo.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _prereg_signed(path: Path) -> bool:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    return bool(re.search(r"^status:\s*SIGNED\s*$", text, re.MULTILINE))


def audit(root: Path = ROOT) -> dict:
    repair = root / "outputs" / "thesis-repair"
    macro = load_json(repair / "macro_identity_research_v2_latest.json")
    cross = load_json(repair / "usdcop_daily_cross_source_v2.json")
    integrity_paths = (
        repair / "integrity_after_dxy_refresh.json",
        repair / "integrity_after_rebuild_v2_current.json",
    )
    integrity_path = next((path for path in integrity_paths if path.is_file()), integrity_paths[-1])
    integrity = load_json(integrity_path)
    contract = load_json(repair / "research_data_contract_frequency_v2_latest.json")
    portable_path = root / "data" / "thesis" / "research_data_portable_v2.pkl"
    ppo_full = repair / "ppo_v2_full"
    smoke = repair / "ppo_smoke_v2_20k" / "ppo_regime_seed42.json"
    llm_dir = root / "data" / "thesis" / "llm"
    figures_v2 = repair / "results_v2" / "figuras"
    prereg = root / ".claude" / "specs" / "planes" / "06-PRE-REGISTRATION-v3.md"

    ppo_files = sorted(ppo_full.glob("*.json")) if ppo_full.is_dir() else []
    ppo_runs = []
    for path in ppo_files:
        payload = load_json(path)
        if payload and {"config", "seed", "timesteps"} <= payload.keys():
            ppo_runs.append(payload)
    portable_identity = None
    portable_identity_matches = False
    if portable_path.is_file():
        try:
            with portable_path.open("rb") as handle:
                portable_identity = pickle.load(handle).get("identity")
            from src.research.dataset import dataset_identity
            portable_identity_matches = portable_identity == dataset_identity()
        except (OSError, EOFError, pickle.PickleError, AttributeError, ValueError):
            portable_identity_matches = False

    controls = {
        "dxy_investing_942611": {
            "status": "PASS_WITH_RESERVATION" if macro and macro.get("all_declared_identities_honoured") else "FAIL",
            "artifact": str(repair / "macro_identity_research_v2_latest.json"),
            "independent": False,
        },
        "usdcop_daily_twelvedata_vs_investing": {
            "status": "PASS" if cross and cross.get("agreement_flag") == "OK" else "FAIL",
            "artifact": str(repair / "usdcop_daily_cross_source_v2.json"),
            "common_rows": cross.get("common_rows") if cross else None,
            "median_abs_diff_pct": cross.get("median_abs_diff_pct") if cross else None,
            "p95_abs_diff_pct": cross.get("p95_abs_diff_pct") if cross else None,
            "max_abs_diff_pct": cross.get("max_abs_diff_pct") if cross else None,
        },
        "macro_causality_t_minus_1": {
            "status": "PASS" if integrity and integrity.get("features", {}).get(
                "macro_same_day_perturbation", {}).get("causality_gate_pass") is True else "FAIL",
            "artifact": str(integrity_path),
        },
        "portable_v2": {
            "status": "PASS" if portable_identity_matches else "FAIL",
            "artifact": str(portable_path),
            "identity": portable_identity,
        },
        "macro_forward_freshness": {
            "status": "PASS" if contract and contract.get("verdict", {}).get(
                "macro_fresh_enough_for_forward") is True else "BLOCKED",
            "artifact": str(repair / "research_data_contract_frequency_v2_latest.json"),
            "series_lag": (contract or {}).get("macro", {}).get("series_business_lag_to_market"),
        },
        "ppo_10_confirmatory_runs": {
            "status": "PASS" if len(ppo_runs) == 10 and all(
                int(row.get("timesteps", 0)) >= 300_000 for row in ppo_runs
            ) else "NOT_EXECUTED",
            "count": len(ppo_runs),
            "artifact_dir": str(ppo_full),
        },
        "ppo_smoke": {
            "status": "DIAGNOSTIC_ONLY" if smoke.is_file() else "MISSING",
            "timesteps": (load_json(smoke) or {}).get("timesteps"),
            "artifact": str(smoke),
        },
        "deepseek_ledger": {
            "status": "PASS" if (llm_dir / "decisions_deepseek.jsonl").is_file() else "NOT_EXECUTED",
            "artifact": str(llm_dir / "decisions_deepseek.jsonl"),
        },
        "azure_ledger": {
            "status": "PASS" if (llm_dir / "decisions_azure.jsonl").is_file() else "NOT_EXECUTED",
            "artifact": str(llm_dir / "decisions_azure.jsonl"),
        },
        "hybrid": {"status": "NOT_EXECUTED"},
        "figures_v2": {
            "status": "PASS" if figures_v2.is_dir() and any(figures_v2.glob("*.png")) else "NOT_EXECUTED",
            "artifact_dir": str(figures_v2),
        },
        "preregistration": {
            "status": "PASS" if _prereg_signed(prereg) else "BLOCKED",
            "artifact": str(prereg),
        },
    }
    return {
        "contract": "CTR-RESEARCH-THESIS-E2E-STATUS-001",
        "network_called": False,
        "secrets_read": False,
        "controls": controls,
        "confirmatory_ready": bool(
            controls["preregistration"]["status"] == "PASS"
            and controls["portable_v2"]["status"] == "PASS"
            and controls["macro_forward_freshness"]["status"] == "PASS"
        ),
        "sha256": {
            "macro_identity": sha256(repair / "macro_identity_research_v2_latest.json")
            if (repair / "macro_identity_research_v2_latest.json").is_file() else None,
            "daily_crosscheck": sha256(repair / "usdcop_daily_cross_source_v2.json")
            if (repair / "usdcop_daily_cross_source_v2.json").is_file() else None,
            "portable_v2": sha256(portable_path) if portable_path.is_file() else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "confirmatory_ready": report["confirmatory_ready"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
