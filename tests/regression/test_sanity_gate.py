import json

import pytest

from src.research.sanity_gate import require_sanity_pass


def test_sanity_gate_rejects_negative_s1_report(tmp_path):
    report = tmp_path / "s1.json"
    report.write_text(json.dumps({"synthetic_only": True,
                                  "market_trials_charged": 0,
                                  "selected_probe": None,
                                  "attempts": [{"fixtures": [{"fixture": "S1"}]}]}))
    with pytest.raises(RuntimeError, match="ninguna receta"):
        require_sanity_pass(report)


def test_sanity_gate_accepts_complete_protocol(tmp_path):
    report = tmp_path / "ok.json"
    report.write_text(json.dumps({"synthetic_only": True,
                                  "market_trials_charged": 0,
                                  "selected_probe": "ent_coef_zero",
                                  "attempts": [{"fixtures": [{"fixture": f} for f in ("S1", "S2", "S3", "S4")]}]}))
    assert require_sanity_pass(report)["selected_probe"] == "ent_coef_zero"
