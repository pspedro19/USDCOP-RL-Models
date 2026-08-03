from src.validation.sp500_oos_gate import validate_sp500_oos_evidence


def _evidence(**overrides):
    e = {
        "data_source_status": "real_pit_verified",
        "point_in_time": True,
        "available_at_required": True,
        "data_pit": True,
        "leakage": True,
        "purged_cv": True,
        "pbo": True,
        "dsr": True,
        "costs": True,
        "benchmark": True,
        "cost_model": {"commission_bps": 1, "slippage_bps": 2},
        "benchmarks": ["SPY_TR", "MA200", "VOL_TARGET_10", "SIXTY_FORTY"],
        "oos_sharpe": 0.8,
        "oos_return_pct": 4.0,
        "max_drawdown_pct": 20.0,
        "pbo": 0.2,
        "dsr": 0.97,
    }
    e.update(overrides)
    return e


def test_synthetic_data_always_rejected():
    result = validate_sp500_oos_evidence(_evidence(data_source_status="synthetic_scaffold_until_real_feed"))
    assert not result.passed
    assert "synthetic_or_unidentified_data" in result.reasons


def test_complete_real_evidence_passes():
    assert validate_sp500_oos_evidence(_evidence()).passed


def test_missing_costs_and_statistics_rejected():
    result = validate_sp500_oos_evidence(_evidence(cost_model=None, dsr=0.5, pbo=0.8))
    assert not result.passed
    assert "cost_model_missing" in result.reasons
    assert "dsr_below_threshold" in result.reasons
    assert "pbo_too_high" in result.reasons
