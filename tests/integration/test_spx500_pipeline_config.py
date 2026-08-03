from pathlib import Path
import yaml

ROOT = Path(__file__).parents[2]


def test_spx500_pipeline_is_profile_driven():
    pipelines = yaml.safe_load((ROOT / "config/assets/pipelines.yaml").read_text())
    spec = pipelines["assets"]["spx500"]
    assert spec["enabled"] is True
    assert spec["stages"][0]["args"][:2] == ["--asset", "spx500"]
    assert spec["stages"][-1]["script"].endswith("run_spx500_pipeline.py")
    assert spec["verify"]["registry_asset"] == "spx500"


def test_spx500_asset_and_experiment_contract_match():
    asset = yaml.safe_load((ROOT / "config/assets/spx500.yaml").read_text())
    exp = yaml.safe_load((ROOT / "config/forecast_experiments/spx500_regime_gated_v1.yaml").read_text())
    assert asset["asset_id"] == exp["asset_id"] == "spx500"
    assert asset["strategy_id"] == exp["strategy_id"]
    assert exp["data"]["point_in_time"] is True
    assert exp["promotion"]["status"] == "blocked_until_real_data"


def test_spx500_price_feed_is_investing_daily_only_and_fail_closed():
    asset = yaml.safe_load((ROOT / "config/assets/spx500.yaml").read_text())
    source = asset["data_source"]
    assert source["provider"] == source["daily_provider"] == "investing"
    assert source["provider_symbol"] == "SPX"
    assert source["investing_pair_id"] == 166
    assert source["intraday_enabled"] is False
    assert source["seed_file"] is None
    assert source["authoritative_daily"] is True
    assert source["fail_closed"] is True
    assert "SPY" not in source
