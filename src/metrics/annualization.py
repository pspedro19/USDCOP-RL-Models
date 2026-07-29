"""Asset-profile-backed annualization for governed metrics (BL-18)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from src.contracts.asset_profile import AssetProfile, load_asset_profile


_INTRADAY_SOURCE_SECONDS = 300
_INTRADAY_SECONDS = {"PT5M": 300, "PT1H": 3_600, "PT4H": 14_400}


@dataclass(frozen=True, slots=True)
class AnnualizationRegistry:
    """Resolve observations/year without caller-provided magic numbers."""

    profiles: Mapping[str, AssetProfile]

    @classmethod
    def load(cls, assets_dir: str | Path) -> "AnnualizationRegistry":
        base = Path(assets_dir)
        profiles: dict[str, AssetProfile] = {}
        for path in sorted(base.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or "asset_id" not in raw:
                continue
            profile = load_asset_profile(path.stem, assets_dir=base)
            if profile.asset_id in profiles:
                raise ValueError(f"duplicate asset_id {profile.asset_id!r}")
            profiles[profile.asset_id] = profile
        if not profiles:
            raise ValueError(f"no AssetProfile documents found under {base}")
        return cls(profiles=profiles)

    def periods_per_year(self, asset_id: str, interval_id: str) -> int:
        try:
            profile = self.profiles[asset_id]
        except KeyError as exc:
            raise ValueError(f"unknown asset_id {asset_id!r}") from exc

        if interval_id == "P1D":
            value = profile.session.trading_days_per_year
        elif interval_id == "P1W":
            value = 52
        elif interval_id == "P1M":
            value = 12
        elif interval_id in _INTRADAY_SECONDS:
            if interval_id == "PT5M":
                value = profile.session.bars_per_year
            else:
                bars_per_day = profile.session.bars_per_day
                source_bars = _INTRADAY_SECONDS[interval_id] // _INTRADAY_SOURCE_SECONDS
                value = (
                    None
                    if bars_per_day is None
                    else (bars_per_day // source_bars)
                    * profile.session.trading_days_per_year
                )
        else:
            raise ValueError(f"unsupported return interval {interval_id!r}")

        if type(value) is not int or value <= 0:
            raise ValueError(
                f"{asset_id}/{interval_id} has no positive annualization in AssetProfile"
            )
        return value
