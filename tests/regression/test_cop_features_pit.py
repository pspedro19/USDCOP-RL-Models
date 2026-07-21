"""PIT audit of the COP forecasting features (plan COP Fase A3 / U0.4).

Contract: CTR-QUANT-CONSTITUTION-001 §4 (anti-look-ahead, capa de datos)

The T-1 rule for macro features has always been claimed (merge_asof backward + shift);
this test VERIFIES it row by row against the CLEAN macro source: the macro value the
model sees on decision date d must equal the value of a date STRICTLY EARLIER than d.
If someone ever swaps the shift for a same-day join (the classic silent upgrade that
inflates DA), this turns red.

Slow (loads the real dataset once, ~10-30s) — that is the point: it audits what the
TRACK actually consumes, not a synthetic fixture.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MACRO_CLEAN = ROOT / "data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet"
# feature -> columna CLEAN (mapeo del SSOT data_sources.macro.column_mapping)
PAIRS = {
    "dxy_close_lag1": "FXRT_INDEX_DXY_USA_D_DXY",
    "vix_close_lag1": "VOLT_VIX_USA_D_VIX",
    "embi_close_lag1": "CRSK_SPREAD_EMBI_COL_D_EMBI",
}


@pytest.fixture(scope="module")
def dataset():
    from src.forecasting.ssot_config import ForecastingSSOTConfig
    from src.forecasting.dataset_loader import ForecastingDatasetLoader
    cfg = ForecastingSSOTConfig.load()
    df, feats = ForecastingDatasetLoader(cfg, project_root=ROOT).load_dataset(target_horizon=5)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df, feats


def test_macro_features_are_strictly_t_minus_1(dataset):
    df, _ = dataset
    if not MACRO_CLEAN.is_file():
        pytest.skip("MACRO_DAILY_CLEAN absent")
    clean = pd.read_parquet(MACRO_CLEAN)
    clean.index = pd.to_datetime(clean.index)
    cols = {c.upper(): c for c in clean.columns}

    sample = df.tail(400)  # ventana reciente: cubre el año vivo + el empalme
    for feat, clean_col_u in PAIRS.items():
        if feat not in df.columns:
            pytest.fail(f"feature {feat} desapareció del dataset")
        col = cols.get(clean_col_u)
        assert col is not None, f"columna {clean_col_u} ausente del CLEAN"
        serie = clean[col].dropna()
        bad = 0
        checked = 0
        for _, row in sample.iterrows():
            fv = row[feat]
            if pd.isna(fv):
                continue
            d = row["date"]
            prior = serie.loc[serie.index < d]
            if prior.empty:
                continue
            checked += 1
            # el valor visto debe ser el ultimo ESTRICTAMENTE anterior a d
            if not np.isclose(float(fv), float(prior.iloc[-1]), rtol=1e-6):
                # tolerancia: puede ser el anterior-del-anterior si el T-1 cayo festivo
                if len(prior) >= 2 and np.isclose(float(fv), float(prior.iloc[-2]), rtol=1e-6):
                    continue
                # pero JAMAS el valor del mismo dia d (look-ahead)
                same_day = serie.loc[serie.index == d]
                if len(same_day) and np.isclose(float(fv), float(same_day.iloc[0]), rtol=1e-6) \
                        and not np.isclose(float(same_day.iloc[0]), float(prior.iloc[-1]), rtol=1e-6):
                    pytest.fail(f"{feat} @ {d.date()}: valor del MISMO dia (look-ahead)")
                bad += 1
        assert checked > 100, f"{feat}: muestra insuficiente ({checked})"
        assert bad / checked < 0.02, (
            f"{feat}: {bad}/{checked} filas no casan con T-1 ni T-2 del CLEAN — "
            "o la fuente divergio del CLEAN o el lag se rompio")


def test_no_future_target_leak_in_features(dataset):
    """Ninguna feature correlaciona ~1 con el target futuro (el smoke más barato de leak)."""
    df, feats = dataset
    tgt = "target_return_5d"
    if tgt not in df.columns:
        pytest.skip("target ausente")
    d = df.dropna(subset=[tgt])
    for f in feats:
        if f not in d.columns:
            continue
        x = d[f].astype(float)
        if x.std() == 0 or x.isna().all():
            continue
        r = abs(np.corrcoef(x.fillna(x.median()), d[tgt].astype(float))[0, 1])
        assert r < 0.95, f"{f}: |corr| con el target futuro = {r:.3f} — huele a leak"
