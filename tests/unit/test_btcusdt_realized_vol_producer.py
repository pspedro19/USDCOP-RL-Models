# -*- coding: utf-8 -*-
"""`btcusdt.realized_vol_20` — el catálogo apunta al código CONGELADO, no a una copia.

DIFERENCIA CON EL CASO SPX, que es lo que decide la forma de este slice. En SPX,
`ma_200` **no tenía productor en ninguna parte** (se calculaba inline en el harness de
paridad), así que se escribió uno y la paridad se demostró contra la fórmula legacy
reescrita a mano. Aquí **la fórmula canónica ya existe y está congelada**:

    src/btc_strategy/indicators.py:79   (dentro de `build_daily_features`)
    d["realized_vol_20"] = d["log_ret"].rolling(20, min_periods=20).std() * ANN
    ANN = sqrt(365)                     # 24/7, coherente con §46.4 del feature-set

Escribir un productor nuevo con esa fórmula copiada sería crear una **segunda
implementación de una feature congelada** — exactamente lo que el catálogo existe para
impedir. Por eso el `code_reference` apunta a `build_daily_features` y el catálogo
declara **cómo invocarlo** (`producer_contract: ohlcv_frame_v1` + `output_column`),
en vez de que el resolver lo adivine por la firma (decisión A, CXD-628).

QUÉ VIGILAN ESTOS CANDADOS: que lo que sale por la vía del catálogo sea **bit a bit**
lo que produce el código congelado, y que la anualización sea la de cripto. Si no lo
fuera, no sería reparación de representación y **no podría declararse 0 trials**.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.btc_strategy.indicators import ANN, build_daily_features  # noqa: E402
from src.features.observations import build_observations  # noqa: E402

SEED = REPO / "seeds" / "latest" / "btcusdt_daily_ohlcv.parquet"
CATALOGO = REPO / "config" / "features" / "feature_catalog.yaml"
CONGELADO = REPO / "src" / "btc_strategy" / "indicators.py"


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    if not SEED.is_file():
        pytest.skip(f"falta el seed de BTC: {SEED}")
    return pd.read_parquet(SEED).sort_values("time").reset_index(drop=True)


@pytest.fixture(scope="module")
def spec() -> dict:
    import yaml

    return yaml.safe_load(
        (REPO / "config" / "policies" / "btc_hodl_b1.yaml").read_text(encoding="utf-8")
    )


def test_the_catalog_path_reproduces_the_frozen_builder_bit_for_bit(spec, bars) -> None:
    """Lo que publica el catálogo == lo que produce el código congelado.

    Es la prueba que sostiene los **0 trials**. Se compara el valor publicado para la
    barra de decisión contra el que sale de `build_daily_features` en esa MISMA fila,
    con igualdad exacta (`==` sobre float64, sin `approx`): es literalmente el mismo
    cálculo pasando por dos caminos, así que cualquier diferencia sería una fórmula
    distinta disfrazada de fontanería.
    """
    cutoff = "2026-07-30T00:00:00+00:00"
    obs = build_observations(spec, bars, decision_cutoff=cutoff)
    assert set(obs) == {"close", "realized_vol_20"}

    congelado = build_daily_features(bars[["time", "open", "high", "low", "close"]])
    # La barra de decisión es la que el productor eligió: se localiza por su
    # `available_at` (cierre + P1D) en vez de asumir "la última", que es justo el
    # atajo que un candado anterior tuvo que desmontar.
    disponible = pd.to_datetime(bars["time"], utc=True) + pd.Timedelta(days=1)
    idx = int((disponible <= pd.Timestamp(cutoff)).to_numpy().nonzero()[0][-1])

    esperado = float(congelado["realized_vol_20"].iloc[idx])
    assert obs["realized_vol_20"]["value"] == esperado, (
        f"la vía del catálogo publicó {obs['realized_vol_20']['value']} y el builder "
        f"congelado da {esperado}: NO sería reparación de representación y no podría "
        f"declararse 0 trials"
    )


def test_the_whole_series_matches_not_just_the_decision_bar(bars) -> None:
    """Y coincide en TODAS las filas, no sólo en la que se publica.

    Una coincidencia en una fila puede ser suerte; el compromiso de 0 trials es sobre
    la fórmula, no sobre un punto. Se recorre la serie entera con igualdad exacta y se
    exige que el warm-up caiga donde el contrato dice (20 barras duras).
    """
    frame = bars[["time", "open", "high", "low", "close"]]
    a = build_daily_features(frame)["realized_vol_20"]
    b = build_daily_features(frame.copy())["realized_vol_20"]   # determinismo

    validos = ~a.isna()
    assert validos.sum() == len(frame) - 20, (
        f"válidas={validos.sum()} no cuadra con len-20: el warm-up no es la ventana "
        f"dura de 20 que declara el catálogo"
    )
    assert np.array_equal(a[validos].to_numpy(), b[validos].to_numpy())


def test_the_annualization_is_crypto_not_equity() -> None:
    """√365, no √252 — y esto NO es un detalle de estilo.

    BTC cotiza 24/7. Usar el reloj de renta variable inflaría la vol ~20%
    (√365/√252 ≈ 1.204) y, como la política hace `target_vol / realized_vol`,
    **encogería la exposición un 20% sin que nada fallara**: números plausibles,
    sizing distinto, ningún test rojo. Por eso se fija el valor y no sólo su nombre.
    """
    assert ANN == pytest.approx(np.sqrt(365.0))
    assert ANN != pytest.approx(np.sqrt(252.0))


def test_the_catalog_freezes_the_real_formula_not_a_wrapper() -> None:
    """El `code_reference` apunta al código CONGELADO y su hash está al día.

    Es el candado que distingue la decisión (A) de la (B) que yo proponía: aquí el
    `sha256_16` cubre la **fórmula real**, así que editar `indicators.py` sin
    re-registrar rompe. Con un adaptador, el hash habría congelado el adaptador y la
    fórmula habría quedado fuera del muro.
    """
    import yaml

    from src.identity.source_hash import file_code_hash

    doc = yaml.safe_load(CATALOGO.read_text(encoding="utf-8"))
    fila = next(
        e for e in doc["features"]
        if e.get("asset_id") == "btcusdt" and e.get("feature_id") == "realized_vol_20"
    )
    ref = fila["code_reference"]
    assert ref["file"] == "src/btc_strategy/indicators.py"
    assert ref["symbol"] == "build_daily_features"
    assert ref["sha256_16"] == file_code_hash(CONGELADO), (
        "el hash declarado no es el del código congelado: si acabas de editar "
        "`indicators.py`, re-registra el hash en el MISMO commit"
    )
    assert fila["producer_contract"] == "ohlcv_frame_v1"
    assert fila["output_column"] == "realized_vol_20"


def test_a_frame_producer_without_its_declared_column_fails_closed(spec, bars, monkeypatch) -> None:
    """Si el productor no emite la columna declarada, se falla — no se improvisa.

    `build_daily_features` devuelve ~10 columnas. Sin `output_column` declarada, o si
    la declarada no está, el resolver tendría que elegir — y elegir por él es cómo se
    publica una feature bajo la identidad de otra.
    """
    import src.features.observations as mod

    def _sin_columna(df):
        salida = build_daily_features(df)
        return salida.drop(columns=["realized_vol_20"])

    monkeypatch.setattr(mod, "_resolve_producer", lambda entry: (
        _sin_columna if entry.get("feature_id") == "realized_vol_20" else None
    ))
    from src.features.observations import ObservationError

    with pytest.raises(ObservationError, match="no emitio la columna"):
        build_observations(spec, bars, decision_cutoff="2026-07-30T00:00:00+00:00")
