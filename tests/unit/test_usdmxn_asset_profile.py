from __future__ import annotations

from pathlib import Path

import yaml

from src.contracts.asset_profile import load_asset_profile
from src.data_quality.ingest_guard import declared_ranges_by_canonical_symbol


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "config" / "assets"
QUALITY_RANGES = ROOT / "config" / "quality" / "market_price_ranges.yaml"


def test_usdmxn_is_auxiliary_identity_not_an_execution_strategy() -> None:
    profile = load_asset_profile("usdmxn")

    assert profile.symbol == "USD/MXN"
    assert profile.base_ccy == "USD" and profile.quote_ccy == "MXN"
    assert profile.strategy_id is None and profile.base_strategy is None
    assert profile.pipeline_type == "auxiliary_market_data"
    assert profile.data_source.provider == "twelvedata"
    assert profile.data_source.provider_symbol == "USD/MXN"
    assert profile.session.trading_days_per_year == 261


def test_usdmxn_declares_scoped_quality_policy_and_never_flattens_it() -> None:
    profile_raw = yaml.safe_load((ASSETS / "usdmxn.yaml").read_text(encoding="utf-8"))
    quality_raw = yaml.safe_load(QUALITY_RANGES.read_text(encoding="utf-8"))
    scoped = quality_raw["price_ranges"]["usdmxn"]

    assert profile_raw["quality_range_policy"] == "scoped_provider_time"
    assert scoped == [
        {
            "provider_id": "twelvedata",
            "valid_from": "1993-01-01T00:00:00Z",
            "bounds": [2.5, 100],
        }
    ]
    assert "USD/MXN" not in declared_ranges_by_canonical_symbol(ASSETS, QUALITY_RANGES)
