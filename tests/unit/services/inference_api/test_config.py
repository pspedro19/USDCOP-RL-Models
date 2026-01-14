def test_p0_2_thresholds_symmetric():
    """Thresholds DEBEN ser simétricos."""
    from services.inference_api.config import Settings
    settings = Settings()
    assert settings.threshold_long == -settings.threshold_short
