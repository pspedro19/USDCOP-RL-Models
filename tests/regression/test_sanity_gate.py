import json
import hashlib
from pathlib import Path

import pytest

from src.research.sanity_gate import require_macro_identity, require_sanity_pass


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


def test_macro_identity_gate_rejects_unreconciled_sources(tmp_path):
    report = tmp_path / "macro.json"
    report.write_text(json.dumps({"all_declared_identities_honoured": False,
                                  "series": {"brent": {"status": "NO COINCIDE"}}}))
    with pytest.raises(RuntimeError, match="no coinciden"):
        require_macro_identity(report)


def test_macro_identity_gate_binds_positive_report_to_ssot(tmp_path):
    import yaml
    root = Path(__file__).resolve().parents[2]
    availability = root / "config" / "research" / "macro_availability.yaml"
    declared = yaml.safe_load(availability.read_text(encoding="utf-8"))["series"]
    series = {
        name: {"column": spec["column"], "declared_source": spec["source"],
               "fallback": spec.get("fallback"), "status": "COINCIDE", "honoured": True}
        for name, spec in declared.items()
    }
    clean = root / "data" / "pipeline" / "04_cleaning" / "output" / "MACRO_DAILY_CLEAN.parquet"
    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    report = tmp_path / "macro.json"
    report.write_text(json.dumps({"all_declared_identities_honoured": True, "series": series,
                                  "inputs": {"availability_sha256": digest(availability),
                                             "clean_sha256": digest(clean)}}))
    assert require_macro_identity(report, availability=availability, clean=clean)["series"] == series


def test_macro_identity_checker_exposes_input_hashes():
    from scripts.diagnostics.verify_macro_declared_identity import _digest
    root = Path(__file__).resolve().parents[2]
    availability = root / "config" / "research" / "macro_availability.yaml"
    assert len(_digest(availability)) == 64
