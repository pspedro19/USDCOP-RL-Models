from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "experiments" / "thesis_ppo_v2.yaml"


def test_thesis_v2_config_points_to_existing_identity_artifacts() -> None:
    spec = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert spec["experiment_id"] == "EXP-TESIS-RL-02"
    assert spec["dataset_version"] == "v2"
    assert spec["dataset_identity_required"] is True
    for key in ("source_schema", "frozen_scaler", "frozen_regime",
                "portable_dataset", "sanity_gate_report", "macro_identity_report"):
        path = ROOT / spec[key]
        assert path.is_file(), f"{key} no apunta a un artefacto existente: {path}"


def test_thesis_v2_evaluation_declares_retrospective_and_forward_roles() -> None:
    spec = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    evaluation = spec["evaluation"]
    assert evaluation["delays_bars"] == [0, 1, 2]
    assert evaluation["report_per_seed"] is True
    assert evaluation["report_ensemble_separately"] is True
    assert evaluation["holdout_role"] == "retrospective_diagnostic_only"
    assert evaluation["confirmatory_role"] == "forward_after_freeze"
