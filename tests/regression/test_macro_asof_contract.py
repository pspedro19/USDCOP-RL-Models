from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.research.macro_asof import strict_asof


def test_strict_asof_excludes_same_day_observation() -> None:
    targets = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    series = pd.Series([10.0, 20.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    result = strict_asof(targets, series, name="x")
    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == 10.0


def test_strict_asof_rejects_stale_observation_from_contract(tmp_path: Path) -> None:
    availability = tmp_path / "macro_availability.yaml"
    availability.write_text("max_staleness_business_days: 1\n", encoding="utf-8")
    targets = pd.DatetimeIndex(["2024-01-08"])
    series = pd.Series([10.0], index=pd.to_datetime(["2024-01-02"]))
    result = strict_asof(
        targets, series, name="x", availability_path=availability
    )
    assert pd.isna(result.iloc[0])


def test_strict_asof_never_interpolates_or_zero_fills() -> None:
    targets = pd.DatetimeIndex(["2024-01-04"])
    series = pd.Series([10.0], index=pd.to_datetime(["2024-01-02"]))
    result = strict_asof(targets, series, name="x")
    assert result.iloc[0] == 10.0
    assert result.iloc[0] != 0.0


def test_research_feature_and_regime_lanes_share_v2_macro_default(monkeypatch) -> None:
    monkeypatch.delenv("THESIS_MACRO_CLEAN", raising=False)
    import src.research.features as features
    import src.research.regime_hmm as regime
    assert features.MACRO_CLEAN.name == "MACRO_RESEARCH_v2.parquet"
    assert regime.MACRO_CLEAN.name == "MACRO_RESEARCH_v2.parquet"
