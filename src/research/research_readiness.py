"""Read-only preparation audit. Does not mistake a repaired schema for a thesis.

This increment prepares/validates an independent representation; the old trainer
refuses it. Training admission and the prospective pilot are distinct later gates.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import yaml

from src.research.observation_contract import REGIME5_VERSION, observation_contract
from src.research.regime5_bundle import ROOT, code_identity, digest, safe_path, strict_json

if TYPE_CHECKING:
    from pathlib import Path

CONTRACT = "CTR-RESEARCH-REGIME5-PREPARATION-001"


def checked_bytes(reference: dict) -> tuple[Path, bytes]:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("exact path and SHA reference required")
    path = safe_path(ROOT / reference["path"])
    raw = path.read_bytes()
    if digest(raw) != reference["sha256"]:
        raise ValueError(f"artifact SHA mismatch: {path.name}")
    return path, raw


def retrospective_totals(reference: dict) -> dict:
    """Compare published scalar metrics to the original pinned daily series.

    This is numerical reproduction, not a test of macro availability or a claim
    that old figures were generated under today's source code.
    """
    path, raw = checked_bytes(reference)
    manifest = strict_json(raw)
    from scripts.diagnostics.audit_thesis_e2e_status import verify_bundle

    evidence = verify_bundle(path.parent)
    if evidence["status"] != "REPRODUCED":
        raise ValueError(evidence.get("reason", "historical artifact verification failed"))
    parts = {}
    for name in ("daily_series.json", "metrics.csv"):
        target = safe_path(path.parent / name)
        if target.parent != path.parent:
            raise ValueError("retrospective artifact path escape")
        parts[name] = target.read_bytes()
        if digest(parts[name]) != manifest["artifacts_sha256"][name]:
            raise ValueError("historical numeric artifact SHA mismatch")
    daily = strict_json(parts["daily_series.json"])
    metrics = pd.read_csv(io.BytesIO(parts["metrics.csv"])).set_index("arm")
    for arm in metrics.index:
        rows = daily[arm]
        gross, cost, net = (
            np.array([r[k] for r in rows]) for k in ("gross_return", "cost_return", "net_return")
        )
        if (
            not np.allclose(gross - cost, net, atol=1e-12, rtol=0)
            or not np.isclose(
                metrics.loc[arm, "return_compounded_pct"],
                100 * (np.prod(1 + net) - 1),
                atol=1e-9,
                rtol=0,
            )
            or not np.isclose(
                metrics.loc[arm, "gross_compounded_pct"],
                100 * (np.prod(1 + gross) - 1),
                atol=1e-9,
                rtol=0,
            )
            or not np.isclose(metrics.loc[arm, "cost_sum_pct"], 100 * cost.sum(), atol=1e-9, rtol=0)
        ):
            raise ValueError(f"published metrics do not reproduce: {arm}")
    return {**evidence, "scalar_metrics_recomputed": True, "arms": len(metrics)}


def audit_preparation(config_path: Path) -> dict:
    raw = safe_path(config_path).read_bytes()
    cfg = yaml.safe_load(raw)
    if (
        cfg.get("contract") != CONTRACT
        or cfg.get("status") != "PREPARATION_ONLY"
        or cfg.get("observation_version") != REGIME5_VERSION
        or cfg.get("hmm")
        != {
            "candidates": [2, 3, 4, 5],
            "default_k": 3,
            "bic_hysteresis": 10.0,
            "fit_block": "development",
        }
        or cfg.get("training_intent", {}).get("seeds") != [42, 123, 456, 789, 1337]
        or cfg["training_intent"].get("timesteps") != 300000
        or cfg["training_intent"].get("variants") != ["ppo_regime", "ppo_backbone"]
    ):
        raise ValueError("preparation contract differs from the operator-approved plan")
    checks = {}
    try:
        schema = strict_json(safe_path(ROOT / cfg["schema_path"]).read_bytes())
        if schema != observation_contract(REGIME5_VERSION).to_dict():
            raise ValueError("schema file differs from source")
        checks["observation_schema"] = {"status": "VERIFIED", "dimensions": 38}
    except (OSError, ValueError) as exc:
        checks["observation_schema"] = {"status": "BLOCKED", "reason": str(exc)}
    try:
        _, source = checked_bytes(cfg["inputs"]["m5"])
        checks["m5_snapshot_identity"] = {
            "status": "VERIFIED",
            "sha256": digest(source),
            "independent_price_validation": False,
        }
    except (OSError, ValueError) as exc:
        checks["m5_snapshot_identity"] = {"status": "BLOCKED", "reason": str(exc)}
    try:
        checks["retrospective_delivery"] = retrospective_totals(cfg["retrospective_bundle"])
    except (OSError, ValueError, KeyError) as exc:
        checks["retrospective_delivery"] = {"status": "BLOCKED", "reason": str(exc)}
    try:
        path, _ = checked_bytes(cfg["historical_sanity"])
        from src.research.sanity_gate import require_sanity_pass

        require_sanity_pass(path)
        # Even current legacy controls do not license the enlarged input network.
        checks["sanity38"] = {
            "status": "BLOCKED",
            "reason": "historical controls are not a 38-dimensional run",
        }
    except (OSError, ValueError, RuntimeError) as exc:
        checks["sanity38"] = {"status": "BLOCKED", "reason": str(exc)}
    for name in ("observed_releases", "per_series_publication_policies", "frozen_regime5_bundle"):
        reference = cfg["inputs"].get(name)
        if reference is None:
            checks[name] = {"status": "NOT_SUPPLIED"}
        else:
            # Never interpret a checksum or a supplied JSON flag as PIT authentication.
            _, content = checked_bytes(reference)
            checks[name] = {
                "status": "HASH_VERIFIED_CONTENT_ADMISSION_PENDING",
                "sha256": digest(content),
            }
    return {
        "contract": CONTRACT,
        "config_sha256": digest(raw),
        "code_identity": code_identity(),
        "checks": checks,
        "training_ready": False,
        "pilot_executed": False,
        "profitability_established": False,
        "retrospective_delivery_ready": checks["retrospective_delivery"].get(
            "scalar_metrics_recomputed"
        )
        is True,
        "pending": cfg["pending"],
        "scope": "preparation audit; not a training authorization or complete E2E certificate",
    }
