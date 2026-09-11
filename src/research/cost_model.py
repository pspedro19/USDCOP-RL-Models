"""Contrato de costos de la tesis (§9.3): pips, no basis points.

Contract: CTR-RESEARCH-COST-001 · Date: 2026-08-24

## Por qué un módulo propio y no el de producción

`src/training/environments/trading_env.py` cobra `transaction_cost_bps: 2.5` plano. Eso es
correcto para MEXC —comisión fija sobre un par cripto líquido— y **falso** para USD/COP
intradía, donde el costo dominante es el spread y el spread depende del régimen de
volatilidad del día. Un costo plano abarata los días malos y encarece los buenos: exactamente
al revés de la realidad, y en la dirección que infla el Sharpe.

## La fórmula, tal cual §9.3 (no hay nada que elegir aquí)

    sigma12_pips = C_b · rv_12
    cost_pips    = |Δw_b| · (spread_d/2 + 0.5) + 0.1 · |Δw_b| · sigma12_pips
    cost_ret     = cost_pips / C_b

con 1 pip = 1,00 COP por USD, comisión 0,5 pips por lado, y `spread_d` el **spread esperado**
que sale del posterior filtrado del HMM (§8.4), no de su `argmax`.

## Las dos reglas que evitan el doble conteo

1. **Se usan retornos de MID y se resta `cost_ret`.** No se mueve además el precio de
   ejecución. Hacer ambas cosas cobraría el spread dos veces — un error que no da error, solo
   un backtest pesimista de forma inconsistente entre estrategias.
2. **`|Δw|` ya cuenta los lados.** Un flip `+1 → −1` da `|Δw| = 2` y por tanto cobra dos
   lados **una** vez. No hay que multiplicar por 2 en ningún sitio: hacerlo sería el mismo
   doble conteo por la puerta de atrás.

## Anclaje verificable

Round-trip mínimo sin slippage (abrir 1× y cerrar) = `2 · (spread/2 + 0.5)`:

    spread 2 pips -> 3 pips  ·  spread 3 -> 4  ·  spread 6 -> 7

que son exactamente los **3 / 4 / 7 pips** que §9.3 declara para vol baja/media/alta. Si esa
igualdad se rompe, o cambió el contrato o cambió el código; el test lo caza.

## El cierre terminal no es opcional

§9.1: *"El cierre terminal es una regla del entorno, no una acción evitable"*. `Δw_close =
0 − w_58` se cobra como cualquier otro cambio. Omitirlo regala al agente una posición gratis
al final de cada sesión — y sobre 584 sesiones eso no es un detalle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# §9.3. 1 pip = 1,00 COP por USD.
COMMISSION_PIPS_PER_SIDE = 0.5
SLIPPAGE_COEF = 0.1          # `0.1 · |Δw| · sigma12_pips`
RV_WINDOW_BARS = 12          # rv_12: 12 barras de 5 min = 1 hora


@dataclass(frozen=True)
class CostBreakdown:
    """Desglose por barra, para que el test 9 pueda mirar cada componente por separado."""

    dw: float
    spread_pips: float
    sigma12_pips: float
    spread_component_pips: float
    slippage_component_pips: float
    cost_pips: float
    cost_ret: float


def realized_vol_pips(close: np.ndarray, window: int = RV_WINDOW_BARS) -> np.ndarray:
    """`sigma12_pips = C_b · rv_12` — la vol realizada expresada en pips.

    `rv_12` es la desviación de los log-retornos de las últimas `window` barras. Al
    multiplicarla por el precio queda en unidades de COP, que es lo mismo que pips por la
    convención de §9.3 (1 pip = 1 COP).

    Causal por construcción: la ventana termina en `b`, nunca la cruza.

    LIMITACION DECLARADA: en la barra 0 de cada sesión la ventana tiene un solo elemento, así
    que `rv_12 = 0` y la apertura no paga slippage. La ventana NO cruza el cierre de la sesión
    anterior — hacerlo mezclaría el gap overnight con la vol intradía, que son cosas distintas.
    Medido sobre el hold-out: el slippage es ~0,16 de ~3,2 pips por round-trip, así que esto
    subestima el costo en torno a un 5%. Va en la dirección OPTIMISTA, que es la que hay que
    declarar; el stress de costos ×2/×3 de la constitución §3.4 es la defensa real contra
    haberse quedado corto aquí.
    """
    c = np.asarray(close, dtype=float)
    log_ret = np.diff(np.log(c), prepend=np.log(c[0]))
    n = len(c)

    # Vectorizado (2026-08-25). La version en bucle recalculaba `np.std` por barra y este
    # helper se invoca una vez por `run_session`; el re-scoring de frecuencia lo llama
    # 29.200 veces sobre los MISMOS cierres y tardaba mas que todo el resto junto.
    #
    # Se usa la identidad `var = E[x^2] - E[x]^2` sobre sumas acumuladas, con la correccion
    # de Bessel (`ddof=1`) aplicada a mano. `test_realized_vol_vectorization_is_exact`
    # compara contra la implementacion en bucle: si divergieran, TODOS los costos de la
    # tesis cambiarian y ninguna tabla lo delataria.
    cs = np.concatenate([[0.0], np.cumsum(log_ret)])
    cs2 = np.concatenate([[0.0], np.cumsum(log_ret ** 2)])
    idx = np.arange(n)
    lo = np.maximum(0, idx - window + 1)
    hi = idx + 1
    cnt = (hi - lo).astype(float)
    ssum = cs[hi] - cs[lo]
    ssq = cs2[hi] - cs2[lo]

    with np.errstate(invalid="ignore", divide="ignore"):
        var = (ssq - ssum ** 2 / cnt) / (cnt - 1.0)
    var = np.where(cnt > 1, np.maximum(var, 0.0), 0.0)   # cnt==1 -> sin dispersion
    return c * np.sqrt(var)


def bar_cost(dw: float, spread_pips: float, sigma12_pips: float,
             close: float) -> CostBreakdown:
    """Costo de UN cambio de exposición, en pips y en retorno.

    `dw` es `w_b − w_{b−1}` con signo; solo importa su magnitud. Se devuelve el desglose
    entero en vez de un escalar porque el test 9 verifica cada componente: un total correcto
    puede esconder un spread mal cobrado compensado por un slippage mal cobrado.
    """
    a = abs(float(dw))
    spread_component = a * (spread_pips / 2.0 + COMMISSION_PIPS_PER_SIDE)
    slippage_component = SLIPPAGE_COEF * a * sigma12_pips
    cost_pips = spread_component + slippage_component
    return CostBreakdown(
        dw=float(dw),
        spread_pips=float(spread_pips),
        sigma12_pips=float(sigma12_pips),
        spread_component_pips=float(spread_component),
        slippage_component_pips=float(slippage_component),
        cost_pips=float(cost_pips),
        cost_ret=float(cost_pips / close) if close else 0.0,
    )


def session_costs(weights: np.ndarray, close: np.ndarray, spread_pips: float,
                  include_terminal: bool = True) -> tuple[np.ndarray, list[CostBreakdown]]:
    """Costos de una sesión completa, incluido el cierre terminal.

    `weights[b]` es la exposición decidida al cierre de la barra `b` y mantenida durante
    `b+1`. `w_{-1} = 0` (§9.1). Con `include_terminal`, la última entrada es
    `Δw = 0 − w_last`: el cierre forzado, cobrado como cualquier otro cambio.

    Devuelve `(cost_ret_por_paso, desglose)`. `cost_ret` tiene un elemento más que `weights`
    cuando hay cierre terminal — y esa asimetría es intencional: el paso terminal tiene costo
    pero **no** retorno (§9.1).
    """
    w = np.asarray(weights, dtype=float)
    c = np.asarray(close, dtype=float)
    if include_terminal and len(c) != len(w) + 1:
        raise ValueError(
            "con cierre terminal se requieren len(close) == len(weights) + 1; "
            "el último cierre liquida w_last"
        )
    sigma = realized_vol_pips(c)

    prev = 0.0
    costs, breakdown = [], []
    for b in range(len(w)):
        bd = bar_cost(w[b] - prev, spread_pips, sigma[b], c[b])
        costs.append(bd.cost_ret)
        breakdown.append(bd)
        prev = w[b]

    if include_terminal:
        bd = bar_cost(0.0 - prev, spread_pips, sigma[-1], c[-1])
        costs.append(bd.cost_ret)
        breakdown.append(bd)

    return np.asarray(costs, dtype=float), breakdown


def min_round_trip_pips(spread_pips: float) -> float:
    """Round-trip mínimo sin slippage: abrir 1× y cerrar. Debe dar 3/4/7 para 2/3/6."""
    return 2.0 * (spread_pips / 2.0 + COMMISSION_PIPS_PER_SIDE)
