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
    with pytest.raises(RuntimeError, match="manifest"):
        require_sanity_pass(report)


def test_sanity_gate_rejects_legacy_protocol_even_with_all_fixture_names(tmp_path):
    report = tmp_path / "ok.json"
    report.write_text(json.dumps({"synthetic_only": True,
                                  "market_trials_charged": 0,
                                  "selected_probe": "ent_coef_zero",
                                  "attempts": [{"fixtures": [{"fixture": f} for f in ("S1", "S2", "S3", "S4")]}]}))
    with pytest.raises(RuntimeError, match="manifest"):
        require_sanity_pass(report)


def test_sanity_gate_rejects_legacy_aggregate_without_artifacts(tmp_path):
    report = tmp_path / "protocol.json"
    report.write_text(json.dumps({
        "protocol": "S1-S4", "probe": "flat_init_no_turn", "synthetic_only": True,
        "market_trials_charged": 0, "market_evidence": False, "passed": True,
        "fixtures": {f: {"passed": True} for f in ("S1", "S2", "S3", "S4")},
    }))
    with pytest.raises(RuntimeError, match="manifest"):
        require_sanity_pass(report)


def test_macro_identity_gate_rejects_unreconciled_sources(tmp_path):
    report = tmp_path / "macro.json"
    report.write_text(json.dumps({"all_declared_identities_honoured": False,
                                  "series": {"brent": {"status": "NO COINCIDE"}}}))
    with pytest.raises(RuntimeError, match="legacy"):
        require_macro_identity(report)


def test_macro_identity_gate_rejects_flags_even_when_input_hashes_match(tmp_path):
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
    with pytest.raises(RuntimeError, match="legacy"):
        require_macro_identity(report, availability=availability, clean=clean)


def test_macro_identity_checker_exposes_input_hashes():
    from scripts.diagnostics.verify_macro_declared_identity import CLEAN, _digest
    root = Path(__file__).resolve().parents[2]
    availability = root / "config" / "research" / "macro_availability.yaml"
    assert len(_digest(availability)) == 64
    assert CLEAN.name == "MACRO_RESEARCH_v2.parquet"


def test_macro_identity_unverified_series_cannot_claim_global_match(tmp_path):
    from scripts.diagnostics.verify_macro_declared_identity import _digest
    report = tmp_path / "macro_unverified.json"
    report.write_text(json.dumps({
        "all_declared_identities_honoured": False,
        "series": {"dxy": {"status": "NO COMPROBABLE AQUI", "honoured": False}},
        "inputs": {"availability_sha256": _digest(Path("config/research/macro_availability.yaml")),
                   "clean_sha256": _digest(Path("data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet"))},
    }))
    with pytest.raises(RuntimeError, match="legacy"):
        require_macro_identity(report)


def test_local_dxy_reference_requires_one_numeric_series(tmp_path):
    from scripts.diagnostics.verify_macro_declared_identity import _local_reference

    path = tmp_path / "ice_dxy_export.csv"
    path.write_text("Date,DXY\n2026-01-02,108.12\n2026-01-05,108.40\n", encoding="utf-8")
    series, digest = _local_reference(path)
    assert list(series.index.strftime("%Y-%m-%d")) == ["2026-01-02", "2026-01-05"]
    assert float(series.iloc[0]) == 108.12
    assert len(digest) == 64


def test_local_dxy_reference_ohlc_requires_explicit_value_column(tmp_path):
    from scripts.diagnostics.verify_macro_declared_identity import _local_reference_with_column

    path = tmp_path / "ice_dxy_ohlc.csv"
    path.write_text(
        "Date,Open,High,Low,Close\n2026-01-02,108,109,107,108.12\n"
        "2026-01-05,108,109,107,108.40\n",
        encoding="utf-8",
    )
    series, digest = _local_reference_with_column(path, "Close")
    assert list(series.index.strftime("%Y-%m-%d")) == ["2026-01-02", "2026-01-05"]
    assert float(series.iloc[0]) == 108.12
    assert len(digest) == 64
