"""Causal-boundary tripwires for the Codex directional research backtests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd


def _market_dates(pre_oos: int, post_oos: int) -> pd.DatetimeIndex:
    before = pd.bdate_range(end="2024-12-31", periods=pre_oos)
    after = pd.bdate_range(start="2025-01-01", periods=post_oos)
    return before.append(after)


def test_canonical_selection_excludes_label_ending_on_first_oos_day(monkeypatch):
    from scripts.analysis import weekly_forecasting_oos_canonical as subject

    dates = _market_dates(300, 40)
    frame = pd.DataFrame(
        {
            "date": dates,
            "close": np.linspace(3_500.0, 4_100.0, len(dates)),
            "candidate": np.linspace(-1.0, 1.0, len(dates)),
        }
    )
    observed: dict[str, int] = {}

    def spy_mutual_information(x, y, random_state):
        observed["rows"] = len(x)
        observed["labels"] = len(y)
        return np.ones(x.shape[1])

    monkeypatch.setattr(subject, "mutual_info_classif", spy_mutual_information)
    subject.select_features(frame, ["candidate"], horizon=5, k=1)

    cutoff = int(np.searchsorted(dates, pd.Timestamp("2025-01-01")))
    expected = cutoff - 5
    assert observed == {"rows": expected, "labels": expected}, (
        "Feature selection observed a label whose target lands on or after "
        "the first OOS date."
    )


def test_two_stage_selection_embargoes_twenty_day_target(monkeypatch, tmp_path):
    from scripts.analysis import evaluate_2025_feature_selection_2026 as subject

    dates = _market_dates(300, 40)
    frame = pd.DataFrame(
        {
            "date": dates,
            "close": np.linspace(3_500.0, 4_100.0, len(dates)),
            "feature_a": np.sin(np.arange(len(dates)) / 17.0),
            "feature_b": np.cos(np.arange(len(dates)) / 23.0),
        }
    )
    observed: dict[str, int] = {}

    class FakeLoader:
        def __init__(self, _config, project_root):
            self.project_root = project_root

        def load_dataset(self):
            return frame.copy(), ("feature_a", "feature_b")

    def spy_mutual_information(x, y, random_state):
        observed["rows"] = len(x)
        observed["labels"] = len(y)
        return np.arange(x.shape[1], dtype=float)

    (tmp_path / "reports").mkdir()
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    monkeypatch.setattr(
        subject,
        "ForecastingSSOTConfig",
        SimpleNamespace(load=lambda: object()),
    )
    monkeypatch.setattr(subject, "ForecastingDatasetLoader", FakeLoader)
    monkeypatch.setattr(subject, "mutual_info_classif", spy_mutual_information)
    monkeypatch.setattr(subject, "fit_score", lambda *args, **kwargs: None)
    subject.main()

    cutoff = int(np.searchsorted(dates, pd.Timestamp("2025-01-01")))
    expected = cutoff - 20
    assert observed == {"rows": expected, "labels": expected}, (
        "The pre-2025 feature selector consumed returns that mature in 2025."
    )


def test_colombia_macro_selector_embargoes_each_horizon(monkeypatch, tmp_path):
    from scripts.analysis import evaluate_colombia_macro_candidates as subject

    dates = _market_dates(390, 50)
    price = pd.DataFrame(
        {
            "date": dates,
            "open": np.linspace(3_500.0, 4_100.0, len(dates)),
            "high": np.linspace(3_510.0, 4_110.0, len(dates)),
            "low": np.linspace(3_490.0, 4_090.0, len(dates)),
            "close": np.linspace(3_505.0, 4_105.0, len(dates)),
        }
    )
    macro_columns = [
        "FXRT_SPOT_USDMXN_MEX_D_USDMXN",
        "FXRT_SPOT_USDCLP_CHL_D_USDCLP",
        "VOLT_VIX_USA_D_VIX",
        "CRSK_SPREAD_EMBI_COL_D_EMBI",
        "COMM_OIL_BRENT_GLB_D_BRENT",
        "COMM_AGRI_COFFEE_GLB_D_COFFEE",
        "COMM_METAL_GOLD_GLB_D_GOLD",
        "FINC_RATE_IBR_OVERNIGHT_COL_D_IBR",
        "POLR_POLICY_RATE_COL_M_TPM",
        "FINC_BOND_YIELD10Y_COL_D_COL10Y",
        "FINC_BOND_YIELD5Y_COL_D_COL5Y",
        "EQTY_INDEX_COLCAP_COL_D_COLCAP",
    ]
    macro = pd.DataFrame(index=pd.DatetimeIndex(dates, name="fecha"))
    for offset, column in enumerate(macro_columns):
        macro[column] = 100.0 + offset + np.arange(len(dates)) / 100.0

    class FakeLoader:
        def __init__(self, _config, project_root):
            self.project_root = project_root

        def load_dataset(self):
            return price.copy(), ()

    class FakeModel:
        def fit(self, x, y):
            return self

        def predict(self, x):
            return np.zeros(len(x), dtype=int)

    observed_rows: list[int] = []

    def spy_mutual_information(x, y, random_state):
        observed_rows.append(len(x))
        return np.arange(x.shape[1], dtype=float)

    (tmp_path / "reports").mkdir()
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    monkeypatch.setattr(subject, "HORIZONS", (1, 5, 10))
    monkeypatch.setattr(
        subject,
        "ForecastingSSOTConfig",
        SimpleNamespace(load=lambda: object()),
    )
    monkeypatch.setattr(subject, "ForecastingDatasetLoader", FakeLoader)
    monkeypatch.setattr(subject.pd, "read_parquet", lambda _path: macro.copy())
    monkeypatch.setattr(subject, "mutual_info_classif", spy_mutual_information)
    monkeypatch.setattr(subject, "make_pipeline", lambda *args: FakeModel())
    subject.main()

    # T+1 availability removes the first row; 20-day changes remove 20 more.
    post_warmup_dates = dates[21:]
    cutoff = int(
        np.searchsorted(post_warmup_dates, pd.Timestamp("2025-01-01"))
    )
    assert observed_rows == [cutoff - 1, cutoff - 5, cutoff - 10], (
        "Macro feature selection crossed the OOS boundary for at least one horizon."
    )
