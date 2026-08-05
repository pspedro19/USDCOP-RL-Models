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
        # El zoo predice PUNTO: los limites se OMITEN. Esta asercion decia antes
        # `lower == upper == point` con el comentario "by design" — o sea, fijaba
        # como correcto un intervalo de ANCHURA CERO. Eso no es "sin intervalo":
        # es publicar incertidumbre nula. Se invierte la asercion, no se relaja.
        assert out.prediction.type == "log_return"
        assert out.prediction.point == 0.0042
        assert out.prediction.lower is None
        assert out.prediction.upper is None
        # The wall: a forecast can never claim to be actionable
        assert out.diagnostic_only is True

    def test_zoo_never_publishes_a_zero_width_interval(self):
        """El zoo no puede publicar un intervalo degenerado, para NINGUN valor.

        POR QUE ES UN CANDADO Y NO UN DETALLE. `lower == upper == point` pasa el
        contrato sin rechistar (`lower <= point <= upper` se cumple con igualdad),
        asi que ningun muro existente lo veia; y el generador lo emitia con el
        comentario "by design". Un consumidor que dibuje la banda —BL-19 migra
        `ForecastingView` a `parseForecastOutput`— pintaria una cinta de certeza
        alrededor de un numero cuya DA ronda 0.46. Publicar anchura cero es una
        afirmacion de incertidumbre nula; omitir los limites es un hecho.

        Rojo con: devolver `"lower": pred_return, "upper": pred_return` en
        `_validate_row_contract` (que es exactamente lo que hacia).
        """
        for valor in (0.0042, -0.031, 0.0, 1e-9):
            out = _call(pred_return=valor)
            assert out is not None, f"fila excluida inesperadamente para {valor}"
            lower, upper = out.prediction.lower, out.prediction.upper
            assert (lower is None) and (upper is None), (
                f"pred={valor}: el zoo declaro un intervalo [{lower}, {upper}]. No "
                f"produce intervalos: si ambos limites valen el punto, la anchura es "
                f"CERO y eso se lee como incertidumbre nula, no como ausencia."
            )

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


# ---------------------------------------------------------------------------
# (3) The PUBLISHED artifact must BE the validated one — publishing a row with
#     no validated contract record behind it must be impossible (fail-closed).
# ---------------------------------------------------------------------------

import csv as _csv

import pytest

from scripts.pipeline.generate_weekly_forecasts import write_csv
from src.contracts.forecast_output import (
    CONTRACT_ID,
    ForecastOutputError,
    ingest_forecast_outputs,
)


def _row(fo=None, **over):
    """A published row, carrying the provenance of the validated record."""
    row = {
        "record_id": "FF_ridge_h5_2026_W30",
        "view_type": "forward_forecast",
        "model_id": "ridge",
        "horizon_days": 5,
        "inference_date": "2026-07-17",
        "forecast_id": fo.forecast_id if fo else "usdcop_forecast_zoo:ridge:h5:2026-07-17",
        "prediction_point": fo.prediction.point if fo else 0.0042,
        "contract_id": CONTRACT_ID,
    }
    row.update(over)
    return row


class TestPublicationWall:
    def test_write_csv_refuses_rows_without_validated_provenance(self, tmp_path):
        out = tmp_path / "bi_dashboard_unified.csv"
        with pytest.raises(ForecastOutputError, match="never validated"):
            write_csv([_row()], out, validated={})
        assert not out.exists(), "nothing may be published when the gate rejects"

    def test_write_csv_refuses_a_row_with_no_forecast_id(self, tmp_path):
        out = tmp_path / "bi_dashboard_unified.csv"
        with pytest.raises(ForecastOutputError, match="no forecast_id"):
            write_csv([_row(forecast_id=None)], out, validated={})
        assert not out.exists()

    def test_write_csv_refuses_a_tampered_prediction(self, tmp_path):
        """The published number must BE the validated number."""
        fo = _call()
        out = tmp_path / "bi_dashboard_unified.csv"
        with pytest.raises(ForecastOutputError, match="must BE the"):
            write_csv([_row(fo, prediction_point=9.99)], out,
                      validated={fo.forecast_id: fo})
        assert not out.exists()

    def test_write_csv_is_all_or_nothing(self, tmp_path):
        """One bad row publishes NOTHING (no partial artifact on disk)."""
        fo = _call()
        out = tmp_path / "bi_dashboard_unified.csv"
        with pytest.raises(ForecastOutputError):
            write_csv([_row(fo), _row(fo, record_id="FF_bad", forecast_id="ghost")],
                      out, validated={fo.forecast_id: fo})
        assert not out.exists()
        assert not (tmp_path / "bi_dashboard_unified.csv.tmp").exists()

    def test_write_csv_publishes_the_validated_record(self, tmp_path):
        fo = _call()
        assert fo is not None
        out = tmp_path / "bi_dashboard_unified.csv"
        write_csv([_row(fo)], out, validated={fo.forecast_id: fo})
        published = list(_csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert len(published) == 1
        assert published[0]["forecast_id"] == fo.forecast_id
        assert published[0]["contract_id"] == CONTRACT_ID
        assert float(published[0]["prediction_point"]) == fo.prediction.point


class TestPublishedArtifactSurvivesTheIngestWall:
    """The artifact we publish must be re-ingestable by the contract itself:
    what leaves the generator is exactly what the wall accepts."""

    def test_round_trip_through_the_ingest_wall(self, tmp_path):
        fo = _call()
        out = tmp_path / "bi_dashboard_unified.csv"
        write_csv([_row(fo)], out, validated={fo.forecast_id: fo})
        published = list(_csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        # Rebuild the contract record from the published provenance columns.
        rebuilt = fo.to_dict()
        rebuilt["prediction"] = dict(
            rebuilt["prediction"], point=float(published[0]["prediction_point"])
        )
        assert published[0]["forecast_id"] == rebuilt["forecast_id"]
        assert ingest_forecast_outputs([rebuilt])[0].prediction.point == fo.prediction.point
