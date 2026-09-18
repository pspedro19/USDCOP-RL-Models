"""Fail-closed orchestration of the frozen 2026 LLM forward lane.

This command never reads dotenv files. The operator must expose provider variables in the
process environment. It refuses to start either provider if any required configuration is
missing, preventing a partial provider comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
CONTEXTS = ROOT / "outputs/thesis-repair/confirmatory_v4_stable/llm_forward_contexts_2026.jsonl"
PORTABLE = ROOT / "outputs/thesis-repair/confirmatory_v4_stable/research_data_portable_v4_stable.pkl"
SPECS = ROOT / "outputs/thesis-repair/confirmatory_v4_stable/forward_specs_2026.pkl"
PPO_ACTIONS = ROOT / "outputs/thesis-repair/confirmatory_v4_stable/ppo_forward_actions_2026.json"
HASH = "65534cc4984cd2bb52bd7f42b845fcd5cb7be214123d154b5b8bf6c5ef2eb585"


def _run(args: list[str]) -> None:
    subprocess.run([PYTHON, *args], cwd=ROOT, check=True)


def _required_env(provider: str) -> list[str]:
    if provider == "deepseek":
        return ["DEEPSEEK_API_KEY"]
    return ["USDCOP_AZURE_OPENAI_API_KEY", "USDCOP_AZURE_OPENAI_ENDPOINT",
            "USDCOP_AZURE_OPENAI_DEPLOYMENT"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepseek-model", required=True)
    parser.add_argument("--azure-model", required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/thesis-repair/confirmatory_v4_stable")
    args = parser.parse_args()
    required_files = (CONTEXTS, PORTABLE, SPECS, PPO_ACTIONS)
    missing_files = [str(path) for path in required_files if not path.is_file()]
    providers = ("deepseek", "azure_openai")
    missing_env = {provider: [name for name in _required_env(provider) if not os.environ.get(name)]
                   for provider in providers}
    digest = hashlib.sha256(PORTABLE.read_bytes()).hexdigest() if PORTABLE.is_file() else None
    report = {
        "schema_version": "forward-llm-e2e-readiness-v1",
        "portable_sha256": digest,
        "portable_expected": HASH,
        "missing_files": missing_files,
        "missing_env_names": missing_env,
        "ready": not missing_files and digest == HASH and not any(missing_env.values()),
    }
    if args.check_only or not report["ready"]:
        print(json.dumps(report, sort_keys=True))
        return 0 if args.check_only else 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger_paths = {
        "deepseek": ROOT / "data/thesis/llm/decisions_deepseek_forward_2026.jsonl",
        "azure_openai": ROOT / "data/thesis/llm/decisions_azure_forward_2026.jsonl",
    }
    for provider, model in (("deepseek", args.deepseek_model), ("azure_openai", args.azure_model)):
        _run([
            "scripts/analysis/run_thesis_llm.py",
            "--input-jsonl", str(CONTEXTS), "--provider", provider, "--model-id", model,
            "--ledger", str(ledger_paths[provider]), "--portable", str(PORTABLE),
            "--execute", "--resume",
        ])
        _run([
            "scripts/validation/validate_thesis_llm_ledger.py",
            "--ledger", str(ledger_paths[provider]), "--require-complete-sessions",
            "--expected-dataset-sha256", HASH, "--expected-dataset-block", "forward",
            "--forbid-retrospective",
        ])
        settlement = args.output_dir / f"llm_{provider}_forward_2026.json"
        _run([
            "scripts/analysis/settle_thesis_llm.py", "--ledger", str(ledger_paths[provider]),
            "--block", "forward", "--portable", str(PORTABLE), "--forward-specs", str(SPECS),
            "--strict-ledger", "--output", str(settlement),
        ])
        hybrid = args.output_dir / f"hybrid_{provider}_forward_2026.json"
        _run([
            "scripts/analysis/thesis_hybrid.py", "--ppo-weights", str(PPO_ACTIONS),
            "--ppo-config", "ppo_regime", "--ledger", str(ledger_paths[provider]),
            "--block", "forward", "--portable", str(PORTABLE), "--forward-specs", str(SPECS),
            "--output", str(hybrid),
        ])
    print(json.dumps({"status": "completed", "providers": list(providers)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
