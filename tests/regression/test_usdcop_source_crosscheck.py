from pathlib import Path

import yaml

from scripts.data.ingest_asset_ohlcv import _KeyRotator

ROOT = Path(__file__).resolve().parents[2]


def test_usdcop_profile_requires_investing_daily_crosscheck():
    config = yaml.safe_load((ROOT / "config/assets/usdcop.yaml").read_text(encoding="utf-8"))
    source = config["data_source"]
    assert source["provider"] == "twelvedata"
    assert source["interval"] == "5min"
    assert source["investing_pair_id"] == 2112
    assert source["daily_crosscheck_required"] is True


def test_twelvedata_key_rotator_cycles_without_exposing_values():
    rotator = _KeyRotator(["a", "b"])
    assert [rotator.next() for _ in range(5)] == ["a", "b", "a", "b", "a"]
