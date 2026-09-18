"""Fail-closed validator for the thesis confirmatory protocol.

This script validates only declarations. It never loads market data, secrets,
models, or providers, and it cannot open a trial by itself.
"""

from __future__ import annotations

import argparse
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "config/research/thesis_confirmatory_v4.yaml"


def _d(value: Any) -> date:
    return date.fromisoformat(str(value))


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("contract") != "CTR-THESIS-CONFIRMATORY-004":
        errors.append("contract id is not the confirmatory thesis contract")
    if data.get("status") not in {"PENDING_OPERATOR_SIGNATURE", "SIGNED"}:
        errors.append("protocol status must be pending or signed")
    if data.get("status") == "SIGNED" and (
        not data.get("signed_by") or not data.get("signed_at_utc")
    ):
        errors.append("signed protocol requires signed_by and signed_at_utc")

    blocks = data.get("partitions", {})
    expected = ["training", "selection", "confirmatory_holdout", "forward"]
    if list(blocks)[:4] != expected:
        errors.append("partition order must be training, selection, holdout, forward")
    spans = []
    for name in expected:
        block = blocks.get(name, {})
        try:
            start, end = _d(block["start"]), _d(block["end"])
            if start > end:
                errors.append(f"{name} starts after it ends")
            spans.append((start, end, name))
        except (KeyError, TypeError, ValueError):
            errors.append(f"{name} has invalid date bounds")
    for previous, current in pairwise(spans):
        if current[0] <= previous[1]:
            errors.append(f"partition overlap: {previous[2]} and {current[2]}")

    if blocks.get("selection", {}).get("retrospective_for_current_repository") is not True:
        errors.append("selection must be marked retrospective")
    if blocks.get("confirmatory_holdout", {}).get("role") != "one_look_confirmatory":
        errors.append("confirmatory holdout must be one-look")
    if blocks.get("forward", {}).get("role") != "post_freeze_forward":
        errors.append("forward must be post-freeze")

    contract = data.get("data_contract", {})
    for field in (
        "same_day_macro_policy",
        "backfill_policy",
        "interpolation_policy",
        "missing_file_policy",
    ):
        if field not in contract:
            errors.append(f"missing data contract field: {field}")
    if contract.get("same_day_macro_policy") != "forbidden":
        errors.append("same-day macro must be forbidden")
    if contract.get("missing_file_policy") != "fail_closed":
        errors.append("missing data must fail closed")
    if contract.get("identity_hash_fields") is None:
        errors.append("dataset identity hash fields are required")

    models = data.get("models", {})
    if models.get("seeds_per_ppo_configuration") != 5:
        errors.append("each PPO configuration requires five seeds")
    if models.get("report_each_seed") is not True:
        errors.append("per-seed reporting is required")
    if data.get("llm", {}).get("status") != "exploratory_only":
        errors.append("LLM must remain exploratory in this protocol")
    if data.get("asset", {}).get("secondary_replication", {}).get("status") != "exploratory_only":
        errors.append("gold must remain exploratory")

    statistics = data.get("statistics", {})
    if statistics.get("primary_unit") != "session":
        errors.append("primary unit must be session")
    if statistics.get("bootstrap", {}).get("hierarchical_by_seed") is not True:
        errors.append("bootstrap must be hierarchical by seed")
    if statistics.get("dsr", {}).get("trial_count_source") != "reconciled_registry_before_run":
        errors.append("DSR requires a reconciled registry before the run")
    if data.get("gates", {}).get("signature_required_before_training") is not True:
        errors.append("operator signature gate is required")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT)
    args = parser.parse_args()
    errors = validate(args.config)
    if errors:
        print("THESIS CONFIRMATORY PROTOCOL: FAIL")
        for error in errors:
            print(f"- {error}")
        return 2
    state = (
        "signed"
        if yaml.safe_load(args.config.read_text(encoding="utf-8")).get("status") == "SIGNED"
        else "blocked pending signature"
    )
    print(f"THESIS CONFIRMATORY PROTOCOL: PASS ({state}): {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
