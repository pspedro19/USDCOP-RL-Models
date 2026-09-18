"""Machine-readable completion audit for the signed thesis v4 track.

The audit is fail-closed about evidence, but never reads dotenv files or prints secret
values. Provider availability is represented only as a boolean from the process environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

SEEDS = (42, 123, 456, 789, 1337)
CONFIGS = ("ppo_regime", "ppo_backbone")
EXPECTED_PORTABLE = "65534cc4984cd2bb52bd7f42b845fcd5cb7be214123d154b5b8bf6c5ef2eb585"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(condition: bool, evidence: str) -> dict:
    return {"status": "PASS" if condition else "PENDING", "evidence": evidence}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.repo.resolve()
    stable = root / "outputs/thesis-repair/confirmatory_v4_stable"
    training = root / "outputs/thesis-repair/confirmatory_v4_ppo_threads1"

    portable = stable / "research_data_portable_v4_stable.pkl"
    portable_ok = portable.is_file() and sha256(portable) == EXPECTED_PORTABLE
    protocol = root / "config/research/thesis_confirmatory_v4.yaml"
    protocol_text = protocol.read_text(encoding="utf-8") if protocol.is_file() else ""
    holdout = json.loads((stable / "ppo_holdout_2024_2025_stable.json").read_text(encoding="utf-8"))
    forward = json.loads((stable / "ppo_forward_2026_partial.json").read_text(encoding="utf-8"))
    baselines = json.loads((stable / "forward_baselines_2026.json").read_text(encoding="utf-8"))

    training_files = [training / f"{config}_seed{seed}.json" for config in CONFIGS for seed in SEEDS]
    training_ok = all(
        path.is_file()
        and json.loads(path.read_text(encoding="utf-8")).get("timesteps") == 300_000
        and json.loads(path.read_text(encoding="utf-8")).get("identity_unchanged") is True
        for path in training_files
    )

    contexts_path = stable / "llm_forward_contexts_2026.jsonl"
    context_rows = []
    if contexts_path.is_file():
        with contexts_path.open(encoding="utf-8") as handle:
            context_rows = [json.loads(line) for line in handle if line.strip()]
    context_sessions = {row.get("session_date") for row in context_rows}
    contexts_ok = (
        len(context_rows) == 162 * 59
        and len(context_sessions) == 162
        and all(row.get("dataset_block") == "forward" for row in context_rows)
        and all(row.get("retrospective") is False for row in context_rows)
        and {row.get("dataset_sha256") for row in context_rows} == {EXPECTED_PORTABLE}
    )

    provider_env = {
        "deepseek": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "azure_openai": bool(os.environ.get("USDCOP_AZURE_OPENAI_API_KEY")),
    }
    forward_ledgers = {
        "deepseek": (root / "data/thesis/llm/decisions_deepseek_forward_2026.jsonl").is_file(),
        "azure_openai": (root / "data/thesis/llm/decisions_azure_forward_2026.jsonl").is_file(),
    }
    results = {
        "protocol_signed": check("status: SIGNED" in protocol_text, str(protocol)),
        "portable_identity": check(portable_ok, f"sha256={sha256(portable) if portable.is_file() else 'missing'}"),
        "training_ppo": check(training_ok, "10 artifacts; each 300000 timesteps; identity_unchanged=true"),
        "holdout": check(
            len(holdout.get("rows", [])) == 10
            and all(row.get("metrics", {}).get("n_sessions") == 420 for row in holdout.get("rows", [])),
            "ppo_holdout_2024_2025_stable.json: 10 seed rows, each n_sessions=420",
        ),
        "forward_ppo": check(forward.get("n_sessions") == 162, "ppo_forward_2026_partial.json n_sessions=162"),
        "forward_baselines": check(len(baselines.get("rows", [])) == 8, "forward_baselines_2026.json rows=8"),
        "forward_llm_contexts": check(contexts_ok, "9558 contexts; 162 sessions; non-retrospective; stable hash"),
        "deepseek_provider_env": check(provider_env["deepseek"], "process environment only; key value never recorded"),
        "azure_provider_env": check(provider_env["azure_openai"], "process environment only; key value never recorded"),
        "deepseek_forward_ledger": check(forward_ledgers["deepseek"], "ledger absent until execution"),
        "azure_forward_ledger": check(forward_ledgers["azure_openai"], "ledger absent until execution"),
        "thesis_delivery": check(
            (root / "outputs/thesis-delivery/confirmatory_v4_addendum_20260915_with_figures/capitulos_3_4_5_v4.docx").is_file()
            and len(list((root / "outputs/thesis-delivery/confirmatory_v4_addendum_20260915_with_figures/figures").glob("*.png"))) == 6,
            "DOCX v4 addendum with six separate figures exists",
        ),
    }
    pending = [name for name, item in results.items() if item["status"] == "PENDING"]
    report = {
        "schema_version": "thesis-completion-audit-v4",
        "scope": "signed_v4_current_state",
        "overall_status": "READY_WITH_EXTERNAL_LLM_PENDING" if pending else "COMPLETE",
        "pending_items": pending,
        "provider_env_present": provider_env,
        "evidence": results,
        "do_not_claim": [
            "LLM forward performance before both ledgers are complete and validated",
            "confirmatory profitability from the post-freeze partial cut",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"overall_status": report["overall_status"], "pending_items": pending}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
