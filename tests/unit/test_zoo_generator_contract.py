"""BL-15 FASE-2 — contract gate in the zoo generator (CTR-FORECAST-OUTPUT-001).

Exercises scripts.pipeline.generate_weekly_forecasts._validate_row_contract in
isolation (no dataset load, no model training): a contract-clean row returns a
validated ForecastOutput; a broken row (NaN point, inverted timestamps) returns
None — fail-closed per row, never an exception that would abort the run.
"""

import math

from scripts.pipeline.generate_weekly_forecasts import _validate_row_contract
from src.contracts.forecast_output import ForecastOutput

AS_OF = "2026-07-17T00:00:00"
AVAILABLE_AT = "2026-07-27T10:00:00"
TARGET = "2026-07-24T00:00:00"


def _call(**overrides):
    kwargs = dict(
        asset="usdcop",
        model_id="ridge",
        horizon=5,
        pred_return=0.0042,
        as_of=AS_OF,
        available_at=AVAILABLE_AT,
        target_time=TARGET,
    )
    kwargs.update(overrides)
    return _validate_row_contract(**kwargs)


class TestValidRow:
    def test_valid_row_returns_validated_forecast_output(self):
        out = _call()
        assert isinstance(out, ForecastOutput)
        assert out.forecast_spec_id == "usdcop_forecast_zoo"
        assert out.asset == "usdcop"
        assert out.model_id == "ridge"
        assert out.horizon == "5d"
        assert out.as_of == AS_OF
        assert out.available_at == AVAILABLE_AT
        assert out.target_time == TARGET
        # Zoo produces point-only predictions: lower == upper == point (by design)
        assert out.prediction.type == "log_return"
        assert out.prediction.point == 0.0042
        assert out.prediction.lower == 0.0042
        assert out.prediction.upper == 0.0042
        # The wall: a forecast can never claim to be actionable
        assert out.diagnostic_only is True

    def test_spec_id_follows_asset(self):
        out = _call(asset="btcusdt")
        assert out is not None
        assert out.forecast_spec_id == "btcusdt_forecast_zoo"


class TestInvalidRowFailClosed:
    def test_nan_prediction_is_excluded(self, caplog):
        with caplog.at_level("ERROR"):
            out = _call(pred_return=float("nan"))
        assert out is None
        assert "CONTRACT-EXCLUDED" in caplog.text
        assert "ridge" in caplog.text

    def test_inf_prediction_is_excluded(self):
        assert _call(pred_return=math.inf) is None

    def test_none_prediction_is_excluded(self):
        assert _call(pred_return=None) is None

    def test_target_time_not_after_as_of_is_excluded(self):
        # Anti-look-ahead ordering: target_time must be > as_of
        assert _call(target_time=AS_OF) is None

    def test_available_at_before_as_of_is_excluded(self):
        assert _call(available_at="2026-07-10T00:00:00") is None
