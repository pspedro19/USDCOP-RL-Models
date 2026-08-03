from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = ROOT / "scripts" / "data" / "ingest_asset_ohlcv.py"
    spec = importlib.util.spec_from_file_location("spx_investing_ingest", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, status_code=200, data=None):
        self.status_code = status_code
        self._data = data or []

    def json(self):
        return {"data": self._data}


def test_authoritative_investing_daily_parses_raw_ohlc(monkeypatch):
    module = _module()
    payload = [{
        "rowDateTimestamp": "2026-07-20T00:00:00Z",
        "last_openRaw": "7489.18",
        "last_maxRaw": "7513.23",
        "last_minRaw": "7440.53",
        "last_closeRaw": "7443.28",
        "volumeRaw": 0,
    }]
    monkeypatch.setattr(module.requests.Session, "get",
                        lambda *args, **kwargs: _Response(data=payload))
    monkeypatch.setattr(module._time, "sleep", lambda *_: None)
    frame = module._investing_daily(
        166, "SPX/500", date(2026, 7, 1), date(2026, 7, 21),
        fail_closed=True, max_chunks=None,
    )
    assert len(frame) == 1
    assert frame.iloc[0][["open", "high", "low", "close"]].tolist() == [
        7489.18, 7513.23, 7440.53, 7443.28,
    ]
    assert str(frame["time"].dt.tz) == "UTC"


def test_authoritative_investing_daily_rejects_http_fallback(monkeypatch):
    module = _module()
    monkeypatch.setattr(module.requests.Session, "get",
                        lambda *args, **kwargs: _Response(status_code=403))
    with pytest.raises(RuntimeError, match="Authoritative Investing daily fetch failed"):
        module._investing_daily(
            166, "SPX/500", date(2026, 7, 1), date(2026, 7, 21),
            fail_closed=True, max_chunks=None,
        )


def test_spx_daily_labels_anchor_at_1600_new_york():
    module = _module()
    profile = module._load_asset_profile("spx500")
    frame = pd.DataFrame({
        "time": pd.to_datetime(["2026-01-05", "2026-07-20"], utc=True),
        "open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0],
        "close": [1.0, 1.0], "volume": [0.0, 0.0],
    })
    anchored = module._daily_to_session_close(frame, profile)
    local = anchored["time"].dt.tz_convert("America/New_York")
    assert local.dt.hour.tolist() == [16, 16]
    assert local.dt.date.astype(str).tolist() == ["2026-01-05", "2026-07-20"]
