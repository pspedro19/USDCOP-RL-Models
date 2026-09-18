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
    assert result["delta_equal_expected_count"] == 2
    assert result["delta_shorter_than_expected_count"] == 0
    assert result["delta_longer_than_expected_count"] == 0
    assert audit_frequency_file(tmp_path / "missing.parquet")["status"] == "missing"


def test_frequency_lineage_declares_cross_frequency_dependence():
    report = audit()
    lineage = report["frequency_lineage"]
    assert lineage["independent_samples"] is False
    assert "usdcop_m5" in lineage["artifacts"]


def test_macro_freshness_is_checked_per_series_not_by_union_index(tmp_path):
    """A fresh DGS2 row must not hide a stale DXY row."""
    times = pd.date_range("2024-01-10 08:00", periods=60, freq="5min", tz="America/Bogota")
    m5 = pd.DataFrame({
        "time": times,
        "symbol": "USD/COP",
        "open": 4000.0,
        "high": 4001.0,
        "low": 3999.0,
        "close": 4000.0,
    })
    m5_path = tmp_path / "m5.parquet"
    m5.to_parquet(m5_path)
    idx = pd.date_range("2024-01-01", "2024-01-09", freq="D")
    macro = pd.DataFrame({
        "FXRT_INDEX_DXY_USA_D_DXY": [100.0] * len(idx),
        "COMM_OIL_BRENT_GLB_D_BRENT": [80.0] * len(idx),
        "FINC_RATE_IBR_OVERNIGHT_COL_D_IBR": [10.0] * len(idx),
        "FINC_BOND_YIELD2Y_USA_D_DGS2": [4.0] * len(idx),
    }, index=idx)
    # Make only DXY stale while the other series have an observation on 2024-01-09.
    macro.loc[idx[1:], "FXRT_INDEX_DXY_USA_D_DXY"] = float("nan")
    macro_path = tmp_path / "macro.parquet"
    macro.to_parquet(macro_path)
    report = audit(m5_path=m5_path, macro_path=macro_path)
    assert report["macro"]["series_business_lag_to_market"]["dxy"] > 5
    assert report["verdict"]["macro_fresh_enough_for_forward"] is False
