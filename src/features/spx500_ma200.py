# -*- coding: utf-8 -*-
"""`spx500.ma_200` — la ÚNICA definición de la media de 200 sesiones del S&P 500.

POR QUÉ EXISTE ESTE FICHERO
---------------------------
`spx500_daily_ma200_v1` decide con `close > ma_200`, y `src/contracts/policy_dsl.py`
**no tiene ningún operador de ventana**: una policy declarativa no puede derivar una
media móvil, la necesita materializada. Pero hasta ahora `ma_200` no existía como
feature del sistema: se calculaba **inline** en
`scripts/validation/check_policy_parity.py`, no estaba en el catálogo, y el
feature-set al que la policy apuntaba (`spx500_regime_gated_v1_action_v1`) sólo
declaraba `close`, listando `ma200` como `derived_in_policy` — prosa de otra
estrategia, la *coded*, que sí la deriva dentro.

Resultado: **la feature que decide la señal no tenía productor declarado**, así que
un productor construido desde el feature-set habría entregado sólo `close` y la
policy habría fallado por `missing` en toda corrida (decisión C, CXD-608).

QUÉ ES Y QUÉ NO ES ESTE CAMBIO
------------------------------
Es **reparación de representación**, no un cambio económico: misma ventana (200),
mismo `min_periods`, mismo operador, misma exposición. **0 trials**
(`quant-constitution.md` §2 — "no cobra trial reejecutar una política congelada").
Lo que cambia es *quién declara* la feature, no *qué vale*.

Y por eso mismo la paridad se demuestra sobre la **serie completa**, no por muestreo:
si esto no reprodujera bit a bit lo que ya se publicaba, sería un cambio económico
disfrazado de fontanería.

CAUSALIDAD
----------
`same_bar`: la media de 200 sesiones **incluye** el cierre de la barra de decisión,
igual que el legacy. No es look-ahead — el cierre está disponible en el
`decision_point: session_close` declarado por la policy —, pero se declara explícito
porque un `same_bar` no declarado es indistinguible de una fuga.

`min_periods=200` es parte del contrato, no un detalle: sin 200 sesiones no hay
media, y el valor es `NaN`. El legacy expresaba lo mismo (`warmup_bars: 200` en el
spec) y esa ausencia se traducía en comparación `False` → exposición 0. **Aquí no se
inventa un valor para las primeras 199 barras**: se devuelve `NaN` y quien consuma
decide, que es lo que hace el `missing_input_policy` declarado.

Contract: CTR-FEATURE-CATALOG-001 (entrada `spx500.ma_200`)
Spec: `.claude/specs/planes/backlog/BL-39-*.md` · decisión C: CXD-608
"""
from __future__ import annotations

import pandas as pd

#: Ventana declarada por la estrategia congelada (`config/strategy_manifests/spx500.yaml`,
#: code_hash `ea76413e60621521`) y por el spec de la policy (`policy.params.ma_window`).
#: NO es un parámetro ajustable aquí: cambiarlo es cambiar la estrategia, no la feature.
MA_WINDOW = 200

FEATURE_ID = "ma_200"
SERIES_ID = "spx500.ma_200"


def compute_ma_200(close: pd.Series, *, window: int = MA_WINDOW) -> pd.Series:
    """SMA de `window` sesiones sobre el cierre, con `min_periods=window`.

    Una sola línea de aritmética, y aun así merece ser función: lo que se está
    congelando no es el cálculo sino **que haya UNO**. Antes existían dos fórmulas
    idénticas por casualidad (harness de paridad y publisher legacy) y ninguna
    declarada; dos definiciones que coinciden hoy son dos definiciones que pueden
    divergir mañana sin que nada lo note.

    `window` es keyword-only y por defecto la declarada: se puede *inspeccionar* en
    un test, no *ajustar* desde un llamador de producción sin que se vea.
    """
    if not isinstance(close, pd.Series):
        raise TypeError(f"close debe ser una pd.Series, no {type(close).__name__}")
    if window < 1:
        raise ValueError(f"window debe ser >= 1, no {window}")
    return close.astype(float).rolling(window, min_periods=window).mean()
