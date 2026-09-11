from scripts.diagnostics.audit_research_data_contract import audit


def test_data_contract_audit_reports_frequency_and_pit_requirements():
    report = audit()
    assert report["verdict"]["structural_m5_clean"] is True
    assert report["verdict"]["requires_pit_merge"] is True
    assert report["verdict"]["macro_columns_complete"] is True
    assert report["m5"]["duplicate_symbol_time"] == 0
    assert report["m5"]["off_five_minute_grid"] == 0
    # Incomplete historical source sessions are explicitly reported, never silently padded.
    assert report["verdict"]["complete_session_grid"] is False
