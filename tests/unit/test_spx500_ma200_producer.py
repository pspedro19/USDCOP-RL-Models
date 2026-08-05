# -*- coding: utf-8 -*-
"""`spx500.ma_200` — el productor único, y la prueba de que no cambió nada.

Decisión C (CXD-608): `ma_200` pasa de calcularse **inline** en el harness de paridad
—sin catálogo, sin productor declarado, y con el feature-set de la policy declarando
sólo `close`— a ser una feature del sistema con un solo productor.

Ese movimiento sólo puede ser **0 trials** si es reparación de REPRESENTACIÓN: si la
serie que sale de aquí no reprodujera bit a bit la que ya se publicaba, sería un
cambio económico disfrazado de fontanería y habría que cobrarlo como trial y
revalidarlo. Por eso la paridad se demuestra sobre la **serie completa** (7943 barras
reales del índice oficial, 1995→2026), no por muestreo: un muestreo que pasa no
distingue "idéntico" de "casi idéntico en los puntos que miré".
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

from src.features.spx500_ma200 import MA_WINDOW, compute_ma_200  # noqa: E402

SEED = REPO / "seeds" / "latest" / "spx500_daily_ohlcv.parquet"
CATALOGO = REPO / "config" / "features" / "feature_catalog.yaml"
PRODUCTOR = REPO / "src" / "features" / "spx500_ma200.py"


@pytest.fixture(scope="module")
def close_real() -> pd.Series:
    if not SEED.is_file():
        pytest.skip(f"falta el seed del índice oficial: {SEED}")
    df = pd.read_parquet(SEED).sort_values("time").reset_index(drop=True)
    return df["close"].astype(float)


def test_full_series_parity_against_the_legacy_inline_formula(close_real: pd.Series) -> None:
    """Paridad EXACTA sobre las 7943 barras contra la fórmula que había inline.

    La referencia es literalmente lo que hacía `check_policy_parity.py:101` antes de
    este cambio —`close.rolling(200, min_periods=200).mean()`— reescrita aquí a mano
    a propósito: si el test importara el productor para calcular *ambos* lados, se
    estaría comparando una función consigo misma y pasaría con cualquier fórmula.

    Igualdad EXACTA, no `approx`: es la misma operación sobre los mismos float64.
    Una tolerancia aquí escondería justo lo que se quiere descartar.
    """
    legacy = close_real.rolling(200, min_periods=200).mean()
    nuevo = compute_ma_200(close_real)

    assert len(nuevo) == len(legacy) == len(close_real)
    # NaN en las mismas posiciones (el warm-up es parte del contrato, no un borde).
    assert nuevo.isna().equals(legacy.isna()), "el warm-up no coincide"
    validos = ~legacy.isna()
    assert validos.sum() > 7000, (
        f"sólo {validos.sum()} barras válidas: la serie no es la completa y la "
        f"paridad no probaría lo que dice probar"
    )
    assert np.array_equal(nuevo[validos].to_numpy(), legacy[validos].to_numpy()), (
        "la media difiere del cálculo legacy en al menos una barra: esto NO sería "
        "reparación de representación y no podría declararse 0 trials"
    )


def test_the_warmup_is_a_hard_window_not_a_soft_start(close_real: pd.Series) -> None:
    """Las primeras 199 barras son NaN, y la 200 ya no.

    `min_periods=200` es contrato: sin 200 sesiones no hay media, y ahí NO se inventa
    un valor. El spec lo declara como `warmup_bars: 200`; una media "parcial" en la
    barra 50 daría una señal que el legacy nunca dio.
    """
    ma = compute_ma_200(close_real)
    assert ma.iloc[: MA_WINDOW - 1].isna().all(), "hay media antes de tener 200 sesiones"
    assert not np.isnan(ma.iloc[MA_WINDOW - 1]), "falta la media en la barra 200 exacta"


def test_the_window_is_the_one_the_frozen_strategy_declares() -> None:
    """200, y viene del manifiesto congelado — no es un default cómodo.

    Se cruza contra `policy.params.ma_window` del spec: si alguien cambiara uno de
    los dos, la feature y la regla estarían hablando de ventanas distintas y la
    señal cambiaría sin que ningún hash lo notara (el `params` sí entra en el
    payload canónico; la constante de este módulo no).
    """
    import yaml

    spec = yaml.safe_load(
        (REPO / "config" / "policies" / "spx500_daily_ma200_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    declarada = ((spec.get("policy") or {}).get("params") or {}).get("ma_window")
    assert MA_WINDOW == declarada == 200, (
        f"la ventana del productor ({MA_WINDOW}) y la del spec ({declarada}) divergen"
    )


def test_the_catalog_code_reference_still_points_at_this_file() -> None:
    """El `sha256_16` del catálogo sigue siendo el de este fichero.

    Es el candado que impide que el productor se edite y el catálogo siga afirmando
    que apunta a lo de antes: un `code_reference` obsoleto es una firma sobre un
    documento que ya cambió. Editar el productor obliga a re-registrar el hash en el
    MISMO commit — que es exactamente el punto de tener catálogo.
    """
    import yaml

    from src.identity.source_hash import file_code_hash

    # El catalogo es un MAPPING con `features:`, no una lista en raiz (medido).
    doc = yaml.safe_load(CATALOGO.read_text(encoding="utf-8"))
    entradas = doc["features"]
    fila = next(
        e for e in entradas
        if e.get("asset_id") == "spx500" and e.get("feature_id") == "ma_200"
    )
    ref = fila["code_reference"]
    assert ref["file"] == "src/features/spx500_ma200.py"
    assert ref["symbol"] == "compute_ma_200"
    assert ref["sha256_16"] == file_code_hash(PRODUCTOR), (
        "el hash declarado en el catálogo no es el del productor. Si acabas de "
        "editarlo, re-registra el hash en el MISMO commit"
    )


def test_a_non_series_input_is_rejected_instead_of_silently_coerced() -> None:
    """Una lista o un array no se aceptan a la callada.

    `rolling` sobre un tipo inesperado podría fallar más adelante, o peor, funcionar
    con semántica distinta. El productor de una feature catalogada tiene que ser
    estricto con su entrada.
    """
    with pytest.raises(TypeError):
        compute_ma_200([1.0, 2.0, 3.0])  # type: ignore[arg-type]
