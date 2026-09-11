import numpy as np
import pandas as pd
import pytest


def test_hmm_macro_attachment_does_not_zero_fill_missing_or_stale(monkeypatch, tmp_path):
    import src.research.regime_hmm as subject

    macro = pd.DataFrame({
        "FXRT_INDEX_DXY_USA_D_DXY": [100.0, 101.0],
        "COMM_OIL_BRENT_GLB_D_BRENT": [70.0, 71.0],
    }, index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
    path = tmp_path / "macro.parquet"
    macro.to_parquet(path)
    monkeypatch.setattr(subject, "MACRO_CLEAN", path)
    daily = pd.DataFrame(index=pd.to_datetime(["2024-01-03", "2024-01-12"]))
    out = subject._attach_macro(daily)
    assert np.isfinite(out.loc[pd.Timestamp("2024-01-03"), "dxy_ret"])
    assert pd.isna(out.loc[pd.Timestamp("2024-01-12"), "dxy_ret"])
    assert not (out[["dxy_ret", "brent_ret"]] == 0.0).any().any()


def test_hmm_macro_attachment_rejects_missing_required_column(monkeypatch, tmp_path):
    import src.research.regime_hmm as subject

    path = tmp_path / "macro.parquet"
    pd.DataFrame({"FXRT_INDEX_DXY_USA_D_DXY": [100.0]}, index=pd.to_datetime(["2024-01-01"])).to_parquet(path)
    monkeypatch.setattr(subject, "MACRO_CLEAN", path)
    with pytest.raises(ValueError, match="required HMM columns"):
        subject._attach_macro(pd.DataFrame(index=pd.to_datetime(["2024-01-02"])))
