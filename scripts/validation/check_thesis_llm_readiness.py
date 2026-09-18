#!/usr/bin/env python
"""Fail-closed preflight for the optional thesis LLM arm.

Reads process environment by default. ``--load-dotenv`` is an explicit operator
opt-in for local runs and only reports presence/absence, never values. It does not
make a network call and does not charge a trial.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "research" / "llm_thesis.yaml"


def check(provider: str) -> dict[str, object]:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    entry = cfg["provider_selection"]["primary" if provider == "deepseek" else "robustness"]
    env_names = [entry["api_key_env"]]
    # URLs/models with an explicit *_default are usable without an environment override;
    # credentials and Azure endpoint are never defaulted.
    for key in ("base_url_env", "endpoint_env", "deployment_env"):
        has_default = key == "base_url_env" and "base_url_default" in entry
        if key in entry and not has_default:
            env_names.append(entry[key])
    present = {name: bool(os.environ.get(name, "").strip()) for name in env_names}
    missing = [name for name, value in present.items() if not value]
    return {
        "provider": provider,
        "config_status": cfg.get("status"),
        "environment": present,
        "missing": missing,
        "network_called": False,
        "secrets_printed": False,
        "ready": not missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("deepseek", "azure_openai"), default="deepseek")
    parser.add_argument("--load-dotenv", action="store_true",
                        help="operator opt-in: load workspace .env without printing secrets")
    parser.add_argument("--dotenv-path", type=Path, default=ROOT / ".env",
                        help="operator-supplied dotenv path (used only with --load-dotenv)")
    args = parser.parse_args()
    if not CONFIG.is_file():
        print(f"missing config: {CONFIG}", file=sys.stderr)
        return 2
    if args.load_dotenv:
        from dotenv import load_dotenv
        try:
            load_dotenv(args.dotenv_path, override=False)
        except OSError as exc:
            print(json.dumps({"dotenv_error": type(exc).__name__,
                              "network_called": False,
                              "secrets_printed": False}), file=sys.stderr)
            return 2
    result = check(args.provider)
    print(json.dumps(result, indent=2))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
