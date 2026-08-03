from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    with (ROOT / "config" / "assets" / name).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_model_zoo_is_not_an_executable_directional_signal():
    for name in ("btcusdt_forecasting.yaml", "xauusd_forecasting.yaml"):
        meta = _load(name)["_meta"]
        assert meta["signal_authorized"] is False
        assert meta["direction_usage"] == "display_only"
        assert meta["status"] != "production"


def test_btc_declares_crypto_native_data_unlock():
    meta = _load("btcusdt_forecasting.yaml")["_meta"]
    assert "funding" in meta["data_unlock"]
    assert "open_interest" in meta["data_unlock"]
    assert "basis" in meta["data_unlock"]
