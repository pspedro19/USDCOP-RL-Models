#!/usr/bin/env python
"""Machine-readable readiness status for the EXP-TESIS-RL repair plan.

Read-only and fail-closed: it inspects published artifacts, never opens dotenv files,
never calls a model, and never treats a missing artifact as a pass.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _exists(path: Path) -> bool:
    return path.is_file()


def build_status(root: Path = ROOT) -> dict:
    macro_path = root / "outputs" / "thesis-repair" / "macro_identity_research_v2_latest.json"
    if not macro_path.is_file():
        macro_path = root / "outputs" / "thesis-repair" / "macro_identity_research_v2.json"
    source_lineage_path = root / "outputs" / "thesis-repair" / "source_lineage_v2.json"
    data_contract_candidates = (
        root / "outputs" / "thesis-repair" / "research_data_contract_frequency_v2_latest.json",
        root / "outputs" / "thesis-repair" / "research_data_contract_v2_latest.json",
        root / "outputs" / "thesis-repair" / "research_data_contract_v2.json",
    )
    data_contract_path = next((path for path in data_contract_candidates if path.is_file()),
                              data_contract_candidates[-1])
    sanity_path = root / "outputs" / "thesis-repair" / "sanity_protocol_v2.json"
    portable = root / "data" / "thesis" / "research_data_portable_v2.pkl"
    diagnostic_portable = root / "outputs" / "thesis-repair" / "research_data_portable_diagnostic_v2.pkl"
    prereg = root / ".claude" / "specs" / "planes" / "06-PRE-REGISTRATION-v3.md"
    llm_config = root / "config" / "research" / "llm_thesis.yaml"
    contexts = root / "data" / "thesis" / "llm" / "selection_contexts.jsonl"
    ledger = root / "data" / "thesis" / "llm" / "decisions.jsonl"
    figures = root / "outputs" / "thesis" / "figuras"
    stream_runner = root / "src" / "research" / "llm_forward" / "stream_runner.py"
    stream_cli = root / "scripts" / "analysis" / "run_ppo_stream_bar.py"
    stream_settlement = root / "src" / "research" / "llm_forward" / "settle_thesis.py"

    macro = None
    if macro_path.is_file():
        try:
            macro = json.loads(macro_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            macro = None
    source_lineage = None
    if source_lineage_path.is_file():
        try:
            source_lineage = json.loads(source_lineage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            source_lineage = None
    if source_lineage is None:
        # Keep the status command useful before the first persisted report.  The
        # auditor is local-only and never loads dotenv or calls a provider.
        try:
            from scripts.diagnostics.audit_research_source_lineage import audit
            source_lineage = audit(root=root)
        except (OSError, ValueError, TypeError, KeyError):
            source_lineage = None
    data_contract = None
    if data_contract_path.is_file():
        try:
            data_contract = json.loads(data_contract_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data_contract = None
    data_verdict = (data_contract or {}).get("verdict", data_contract or {})
    macro_gate = bool(macro and macro.get("all_declared_identities_honoured") is True)

    sanity = None
    if sanity_path.is_file():
        try:
            sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            sanity = None
    sanity_gate = bool(sanity and sanity.get("passed") is True)
    prereg_status = None
    prereg_signed = False
    if prereg.is_file():
        try:
            prereg_text = prereg.read_text(encoding="utf-8")
            match = re.search(r"^status:\s*([A-Z_]+)\s*$", prereg_text, re.MULTILINE)
            prereg_status = match.group(1) if match else None
            prereg_signed = bool(re.search(
                r"^operator_signature:\s*SIGNED\s*$", prereg_text, re.MULTILINE
            ))
        except OSError:
            prereg_status = None
    # Do not call load_portable here: status must remain cheap and must not rebuild data.
    portable_identity_present = False
    portable_identity_matches = None
    if portable.is_file():
        try:
            import pickle
            with portable.open("rb") as handle:
                actual_identity = pickle.load(handle).get("identity")
                portable_identity_present = bool(actual_identity)
            from src.research.dataset import dataset_identity
            portable_identity_matches = actual_identity == dataset_identity()
        except (OSError, EOFError, pickle.PickleError, AttributeError, ValueError):
            portable_identity_present = False
            portable_identity_matches = False

    diagnostic_identity_present = False
    diagnostic_identity_matches = None
    diagnostic_sizes = None
    if diagnostic_portable.is_file():
        try:
            import pickle
            with diagnostic_portable.open("rb") as handle:
                blob = pickle.load(handle)
            diagnostic_identity = blob.get("identity")
            diagnostic_identity_present = bool(diagnostic_identity)
            from src.research.dataset import dataset_identity
            diagnostic_identity_matches = diagnostic_identity == dataset_identity()
            diagnostic_sizes = {
                block: len(blob.get(block, []))
                for block in ("development", "selection", "holdout")
            }
        except (OSError, EOFError, pickle.PickleError, AttributeError, ValueError):
            diagnostic_identity_present = False
            diagnostic_identity_matches = False

    result = {
        "contract": "CTR-RESEARCH-THESIS-PLAN-STATUS-001",
        "macro_identity": {
            "artifact_present": _exists(macro_path),
            "all_declared_identities_honoured": macro_gate,
            "dxy_honoured": bool(macro and macro.get("series", {}).get("dxy", {}).get("honoured") is True),
        },
        "source_lineage": {
            "artifact_present": _exists(source_lineage_path),
            "confirmatory_ready": bool(source_lineage and source_lineage.get("confirmatory_ready") is True),
            "diagnostic_allowed": bool(source_lineage and source_lineage.get("diagnostic_allowed") is True),
            "errors": list((source_lineage or {}).get("errors", [])),
            "warnings": list((source_lineage or {}).get("warnings", [])),
            "network_called": bool(source_lineage and source_lineage.get("network_called") is True),
            "secrets_read": bool(source_lineage and source_lineage.get("secrets_read") is True),
        },
        "data_contract": {
            "artifact_present": _exists(data_contract_path),
            "structural_m5_clean": bool(data_contract and data_verdict.get("structural_m5_clean") is True),
            "market_numeric_clean": bool(data_contract and data_verdict.get("market_numeric_clean") is True),
            "macro_columns_complete": bool(data_contract and data_verdict.get("macro_columns_complete") is True),
            "macro_numeric_clean": bool(data_contract and data_verdict.get("macro_numeric_clean") is True),
            "macro_availability_declared": bool(data_contract and data_verdict.get("macro_availability_declared") is True),
            "macro_fresh_enough_for_forward": bool(
                data_contract and data_verdict.get("macro_fresh_enough_for_forward") is True
            ),
            "complete_session_grid": bool(data_contract and data_verdict.get("complete_session_grid") is True),
        },
        "ppo_sanity": {"artifact_present": _exists(sanity_path), "protocol_pass": sanity_gate},
        "preregistration": {"artifact_present": _exists(prereg), "status": prereg_status,
                             "signed": prereg_signed},
        "portable": {"artifact_present": _exists(portable), "identity_present": portable_identity_present,
                      "identity_matches_current_code": portable_identity_matches,
                      "diagnostic_artifact_present": _exists(diagnostic_portable),
                      "diagnostic_identity_present": diagnostic_identity_present,
                      "diagnostic_identity_matches_current_code": diagnostic_identity_matches,
                      "diagnostic_sizes": diagnostic_sizes},
        "llm": {"config_present": _exists(llm_config), "contexts_present": _exists(contexts),
                "ledger_present": _exists(ledger), "secrets_read": False, "network_called": False},
        "forward_stream": {
            "runner_present": _exists(stream_runner),
            "cli_present": _exists(stream_cli),
            "aggregate_settlement_present": _exists(stream_settlement),
            "activation_authorized": False,
        },
        "figures": {"directory_present": figures.is_dir(),
                    "png_count": len(list(figures.glob("*.png"))) if figures.is_dir() else 0},
    }
    # Historical v2 reconstruction only needs the research block to be structurally
    # valid.  Current-market freshness is a separate gate for live/forward execution:
    # the DXY feed can lag today's M5 without invalidating the frozen block ending at
    # the declared hold-out date.
    research_data_gate = bool(
        result["data_contract"]["structural_m5_clean"]
        and result["data_contract"]["market_numeric_clean"]
        and result["data_contract"]["macro_columns_complete"]
        and result["data_contract"]["macro_numeric_clean"]
        and result["data_contract"]["macro_availability_declared"]
    )
    forward_data_gate = research_data_gate and result["data_contract"]["macro_fresh_enough_for_forward"]
    result["ready_for_v2_rebuild"] = bool(
        macro_gate and research_data_gate and sanity_gate and prereg_signed
        and result["source_lineage"]["confirmatory_ready"]
    )
    result["ready_for_confirmatory_llm"] = bool(
        result["ready_for_v2_rebuild"] and forward_data_gate
        and _exists(llm_config) and _exists(contexts)
    )
    result["ready_for_forward"] = bool(
        result["ready_for_v2_rebuild"] and forward_data_gate
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_status()
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
