import pandas as pd

from scripts.diagnostics.audit_research_data_contract import audit, audit_frequency_file


def test_data_contract_audit_reports_frequency_and_pit_requirements():
    report = audit()
    assert report["verdict"]["structural_m5_clean"] is True
    assert report["verdict"]["requires_pit_merge"] is True
    assert report["verdict"]["macro_columns_complete"] is True
    assert report["verdict"]["market_numeric_clean"] is True
    assert report["verdict"]["macro_numeric_clean"] is True
    assert report["verdict"]["macro_availability_declared"] is True
    assert report["m5"]["duplicate_symbol_time"] == 0
    assert report["m5"]["off_five_minute_grid"] == 0
    # Incomplete historical source sessions are explicitly reported, never silently padded.
    assert report["verdict"]["complete_session_grid"] is False


def test_frequency_audit_reports_effective_cadence_and_missing_artifact(tmp_path):
    path = tmp_path / "m5.parquet"
    frame = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC"),
                          "close": [1.0, 1.1, 1.2]})
    frame.to_parquet(path)
    result = audit_frequency_file(path, expected_frequency="5min")
    assert result["status"] == "ok"
    assert result["timezone_present"] is True
    assert result["delta_mismatch_count"] == 0
    assert audit_frequency_file(tmp_path / "missing.parquet")["status"] == "missing"


def test_frequency_lineage_declares_cross_frequency_dependence():
    report = audit()
    lineage = report["frequency_lineage"]
    assert lineage["independent_samples"] is False
    assert "usdcop_m5" in lineage["artifacts"]
