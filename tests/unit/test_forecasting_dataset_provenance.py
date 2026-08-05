from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.forecasting.dataset_loader import ForecastingDatasetLoader


class Config:
    def get_data_source(self, kind: str) -> dict:
        if kind == "ohlcv":
            return {"db_table": "daily", "fallback_parquet": "daily.parquet"}
        return {"db_table": "macro", "fallback_parquet": "macro.parquet"}

    def get_macro_column_mapping(self) -> dict[str, str]:
        return {"raw_macro": "macro"}

    def get_feature_columns(self) -> list[str]:
        return ["macro"]


def _ohlcv() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    return pd.DataFrame({
        "date": dates,
        "open": [1.0, 2.0, 3.0],
        "high": [2.0, 3.0, 4.0],
        "low": [0.5, 1.5, 2.5],
        "close": [1.5, 2.5, 3.5],
    })


def _macro() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=3, freq="D"),
        "macro": [10.0, 11.0, 12.0],
    })


def test_provenance_records_the_actual_db_winners_and_consumed_columns(monkeypatch, tmp_path: Path) -> None:
    loader = ForecastingDatasetLoader(Config(), db_url="postgresql://configured", project_root=tmp_path)
    monkeypatch.setattr(loader, "_load_ohlcv_from_db", _ohlcv)
    monkeypatch.setattr(loader, "_load_macro_from_db", _macro)
    monkeypatch.setattr(loader, "_build_features", lambda frame: frame)

    _, features = loader.load_dataset(target_horizon=1)
    provenance = loader.provenance

    assert features == ["macro"]
    assert provenance.ohlcv.kind == provenance.macro.kind == "postgresql"
    assert provenance.ohlcv.storage_uri == "db://daily"
    assert provenance.macro.storage_uri == "db://macro"
    assert provenance.snapshot_columns == ("date", "open", "high", "low", "close", "macro")
    assert "target_return_1d" not in provenance.snapshot_columns
    assert provenance.snapshot_semantic_hash.startswith("sha256:")


def test_source_content_mutation_changes_snapshot_identity(monkeypatch, tmp_path: Path) -> None:
    def load(close: float) -> str:
        loader = ForecastingDatasetLoader(Config(), db_url="postgresql://configured", project_root=tmp_path)
        frame = _ohlcv()
        frame.loc[1, "close"] = close
        monkeypatch.setattr(loader, "_load_ohlcv_from_db", lambda: frame)
        monkeypatch.setattr(loader, "_load_macro_from_db", _macro)
        monkeypatch.setattr(loader, "_build_features", lambda value: value)
        loader.load_dataset()
        return loader.provenance.snapshot_semantic_hash

    assert load(2.5) != load(2.6)


def test_provenance_is_unavailable_before_a_successful_load(tmp_path: Path) -> None:
    loader = ForecastingDatasetLoader(Config(), project_root=tmp_path)
    try:
        loader.provenance
    except RuntimeError as exc:
        assert "before load_dataset" in str(exc)
    else:
        raise AssertionError("provenance unexpectedly existed before loading data")


def test_parquet_provenance_uri_is_stable_across_project_roots(monkeypatch, tmp_path: Path) -> None:
    uris: list[tuple[str, str]] = []
    for project_root in (tmp_path / "checkout-a", tmp_path / "checkout-b"):
        loader = ForecastingDatasetLoader(Config(), project_root=project_root)
        monkeypatch.setattr(loader, "_load_ohlcv_from_db", lambda: None)
        monkeypatch.setattr(loader, "_load_macro_from_db", lambda: None)
        monkeypatch.setattr(loader, "_load_ohlcv_from_parquet", _ohlcv)
        monkeypatch.setattr(loader, "_load_macro_from_parquet", _macro)
        monkeypatch.setattr(loader, "_build_features", lambda frame: frame)

        loader.load_dataset()
        uris.append((loader.provenance.ohlcv.storage_uri, loader.provenance.macro.storage_uri))

    assert uris == [("repo://daily.parquet", "repo://macro.parquet")] * 2
