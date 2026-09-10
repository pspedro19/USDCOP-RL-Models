# -*- coding: utf-8 -*-
"""`xauusd.sma_63 / sma_126 / sma_252` — el voto de tendencia del oro, materializado.

POR QUÉ EXISTE
--------------
`gold_trend_simple` decide con un voto 2-de-3 sobre esas tres medias y su policy las
declara `required`. Su propio código lo dice —`gold.py`: *"the policy only consumes
them"*—, o sea que las **consume**, no las deriva. Pero **no existían como feature en
ninguna parte**:

* `src/gold_rl/indicators.py::build_daily_features` emite `sma_20/50/100/200`,
  **ninguna** de las tres (medido);
* `scripts/analysis/gold_trend_simple.py:51` las calcula **dentro** de la expresión del
  voto, sin materializarlas nunca como columna;
* `scripts/validation/check_policy_parity.py` las recalculaba **otra vez**, a mano.

Dos copias por casualidad y ninguna declarada — exactamente el estado de `ma_200` antes
de la decisión C. Un productor construido desde el feature-set habría entregado sólo
`close` y la policy habría fallado por `missing` en toda corrida.

POR QUÉ UN PRODUCTOR DE **FRAME** Y NO TRES FUNCIONES
-----------------------------------------------------
Una `sma(close, window=63)` permitiría publicar la media de 63 bajo la identidad de la
de 126: mismo `series_id`, mismo hash de código, valor distinto. Es literalmente el
defecto del parámetro `window` que costó el rechazo de `compute_ma_200` (CXD-618) —
**el catálogo congela el código, no el argumento en runtime**.

Aquí las ventanas están **hard-coded** y la identidad de cada feature la fija el
catálogo con su `output_column`, no un llamador. Quien quiera otra ventana declara otra
feature, con su entrada y su hash.

Contract: CTR-FEATURE-CATALOG-001 (`producer_contract: ohlcv_frame_v1`)
Referencia legacy READ-ONLY: `scripts/analysis/gold_trend_simple.py::VOTE_WINDOWS`
"""
from __future__ import annotations

import pandas as pd

#: Las tres ventanas del voto 2-de-3, hard-coded. Espejo de `VOTE_WINDOWS` en la
#: referencia legacy; NO son configurables — cambiarlas es cambiar la estrategia.
SMA_WINDOWS = (63, 126, 252)


def build_trend_smas(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve el frame con `sma_63`, `sma_126` y `sma_252` añadidas.

    `min_periods = window` en las tres: sin la ventana completa no hay media y el valor
    es `NaN`. Es lo mismo que hace la referencia legacy —`rolling(w)` sin `min_periods`
    usa `min_periods=w` por defecto— y se escribe explícito porque un contrato que
    depende de un default de pandas es un contrato que cambia cuando cambie pandas.

    Sin parámetros: ver el docstring del módulo. La firma es parte del candado.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df debe ser un DataFrame, no {type(df).__name__}")
    if "close" not in df.columns:
        raise ValueError(f"df no trae `close`; columnas: {sorted(df.columns)}")

    out = df.copy()
    close = out["close"].astype(float)
    for w in SMA_WINDOWS:
        out[f"sma_{w}"] = close.rolling(w, min_periods=w).mean()
    return out
