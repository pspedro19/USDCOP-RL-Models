def test_p0_3_adx_not_hardcoded():
    """ADX NO debe estar hardcoded."""
    with open("airflow/dags/l5_multi_model_inference.py") as f:
        source = f.read()
    
    # Should not have hardcoded ADX values
    assert "return 25.0" not in source
    assert "FEATURE_CONTRACT" in source or "technical_periods" in source
