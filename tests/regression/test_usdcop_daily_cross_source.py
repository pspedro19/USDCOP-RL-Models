from pathlib import Path

import pandas as pd

from scripts.diagnostics.verify_usdcop_daily_cross_source import _daily_close


def test_daily_cross_source_parser_accepts_twelvedata_schema(tmp_path):
    path = tmp_path / "daily.parquet"
    pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC"),
        "close": [4000.0, 4001.0, 3999.0],
    }).to_parquet(path)
    series = _daily_close(path)
    assert len(series) == 3
    assert float(series.iloc[0]) == 4000.0
