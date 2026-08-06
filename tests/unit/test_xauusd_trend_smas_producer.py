# -*- coding: utf-8 -*-
"""Las CUATRO features de `gold_trend_simple`, por la vía del catálogo (slice Gold).

Gold es el caso **mixto** de los tres:

* `realized_vol_20` **sí** tenía productor congelado (`src/gold_rl/indicators.py`), así
  que el catálogo apunta ahí — shape de BTC;
* `sma_63/126/252` **no existían en ninguna parte**: se calculaban dentro del voto de
  `gold_trend_simple.py:51`, sin materializarse, y otra vez a mano en el harness de
  paridad. Dos copias por casualidad y ninguna declarada — shape de SPX.

Lo que estos candados prueban es que la vía del catálogo reproduce **bit a bit** la
referencia legacy sobre la **serie completa**, que es lo único que permite declarar
0 trials. La referencia se reescribe aquí a mano a propósito: importar el productor
para calcular ambos lados sería compararlo consigo mismo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.features.observations import (  # noqa: E402
    _catalog_index,
    _resolve_producer,
    resolve_feature_series,
)
from src.features.xauusd_trend_smas import SMA_WINDOWS, build_trend_smas  # noqa: E402
from src.gold_rl.indicators import build_daily_features  # noqa: E402

SEED = REPO / "seeds" / "latest" / "xauusd_daily_ohlcv.parquet"
CATALOGO = REPO / "config" / "features" / "feature_catalog.yaml"


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    if not SEED.is_file():
        pytest.skip(f"falta el seed del oro: {SEED}")
    df = pd.read_parquet(SEED).sort_values("time").reset_index(drop=True)
    return df[["time", "open", "high", "low", "close"]].reset_index(drop=True)


def _por_catalogo(feature_id: str, frame: pd.DataFrame) -> pd.Series:
    """La MISMA función que usa producción (`resolve_feature_series`)."""
    entrada = _catalog_index()[("xauusd", feature_id)]
    return resolve_feature_series(
        entrada, _resolve_producer(entrada), frame, frame["close"].astype(float)
    )


@pytest.mark.parametrize("window", SMA_WINDOWS)
def test_each_sma_matches_the_legacy_formula_over_the_whole_series(window, frame) -> None:
    """Serie COMPLETA contra la fórmula legacy, reescrita a mano aquí.

    La referencia es literalmente lo que hace `gold_trend_simple.py:51` dentro del voto
    (`d["close"].rolling(w).mean()`) y lo que el harness recalculaba. Se reescribe en
    el test en vez de importar el productor: si ambos lados salieran de él, la
    comparación pasaría con cualquier fórmula.

    Igualdad EXACTA en float64, sin `approx`: es la misma operación por dos caminos.
    """
    legacy = frame["close"].astype(float).rolling(window).mean()
    catalogo = _por_catalogo(f"sma_{window}", frame)

    assert len(catalogo) == len(frame)
    assert catalogo.isna().equals(legacy.isna()), f"sma_{window}: el warm-up no coincide"
    validos = ~legacy.isna()
    assert validos.sum() == len(frame) - (window - 1), (
        f"sma_{window}: válidas={validos.sum()} no cuadra con len-{window - 1}"
    )
    assert np.array_equal(catalogo[validos].to_numpy(), legacy[validos].to_numpy()), (
        f"sma_{window}: la vía del catálogo NO reproduce la fórmula legacy; no sería "
        f"reparación de representación y no podría declararse 0 trials"
    )


def test_realized_vol_matches_the_frozen_gold_builder_over_the_whole_series(frame) -> None:
    """La vol sale del builder CONGELADO de Gold, serie entera."""
    congelado = build_daily_features(frame)["realized_vol_20"].astype(float)
    catalogo = _por_catalogo("realized_vol_20", frame)

    validos = ~congelado.isna()
    assert validos.sum() == len(frame) - 20, (
        f"válidas={validos.sum()} no cuadra con len-20: el warm-up no es la ventana "
        f"dura de 20 que declara el catálogo"
    )
    assert np.array_equal(catalogo[validos].to_numpy(), congelado[validos].to_numpy())


def test_gold_vol_uses_the_equity_clock_and_btc_the_crypto_one(frame) -> None:
    """√252 para el oro, √365 para BTC — **la misma `feature_id`, distinta receta**.

    Es el punto donde una identidad mal puesta haría daño en silencio: si Gold heredara
    el reloj de cripto, su vol subiría ~20% y —como el sizing es
    `TARGET_VOL / realized_vol_20`— **la exposición caería ~20% sin que nada fallara**.
    Por eso el `transformation` del catálogo los separa (`_ann252` vs `_ann365`) en vez
    de llamarse igual: dos recetas con el mismo nombre invitan a copiar la equivocada.

    Se comprueba sobre los DATOS, no sobre la etiqueta: la razón entre la vol declarada
    y la desviación diaria tiene que ser √252 y no √365.
    """
    doc = yaml.safe_load(CATALOGO.read_text(encoding="utf-8"))
    por_id = {
        (e["asset_id"], e["feature_id"]): e for e in doc["features"]
    }
    assert por_id[("xauusd", "realized_vol_20")]["transformation"] == "realized_vol_20d_ann252"
    assert por_id[("btcusdt", "realized_vol_20")]["transformation"] == "realized_vol_20d_ann365"

    vol = _por_catalogo("realized_vol_20", frame)
    logret = np.log(frame["close"].astype(float) / frame["close"].astype(float).shift(1))
    diaria = logret.rolling(20, min_periods=20).std()
    ratio = (vol / diaria).dropna()
    assert np.allclose(ratio.to_numpy(), np.sqrt(252.0)), (
        f"la anualización efectiva es {ratio.iloc[-1]:.4f}, no √252={np.sqrt(252.0):.4f}"
    )


def test_the_producer_has_no_runtime_window_parameter() -> None:
    """Ninguna forma de publicar la SMA de 63 bajo la identidad de la de 126.

    Es el defecto de `window` (CXD-618) aplicado a tres features a la vez: una
    `sma(close, window=w)` habría permitido exactamente eso, con el mismo `series_id` y
    el mismo hash de código. Se comprueba por FIRMA, no por comportamiento.
    """
    import inspect

    assert list(inspect.signature(build_trend_smas).parameters) == ["df"], (
        "el productor acepta parámetros además del frame: cualquiera permite publicar "
        "una ventana bajo la identidad de otra"
    )
    assert SMA_WINDOWS == (63, 126, 252)


def test_the_three_entries_share_the_producer_and_differ_only_in_output_column() -> None:
    """Un `code_reference`, tres `output_column`. La identidad la fija el catálogo."""
    from src.identity.source_hash import file_code_hash

    doc = yaml.safe_load(CATALOGO.read_text(encoding="utf-8"))
    filas = [
        e for e in doc["features"]
        if e.get("asset_id") == "xauusd" and str(e.get("feature_id", "")).startswith("sma_")
    ]
    assert len(filas) == 3
    refs = {f["code_reference"]["file"] for f in filas}
    assert refs == {"src/features/xauusd_trend_smas.py"}
    assert {f["output_column"] for f in filas} == {"sma_63", "sma_126", "sma_252"}
    esperado = file_code_hash(REPO / "src" / "features" / "xauusd_trend_smas.py")
    for f in filas:
        assert f["code_reference"]["sha256_16"] == esperado, (
            f"{f['feature_id']}: hash desalineado; re-registra en el MISMO commit"
        )
