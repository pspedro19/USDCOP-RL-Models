"""Los cuatro análisis que explican POR QUÉ la tesis salió negativa.

Contract: CTR-RESEARCH-DECOMP-001 · Date: 2026-08-25

El rechazo no cambia: PPO no bate a `always_flat`, ΔSharpe −6,215 en hold-out, p < 0,0001,
0/10 semillas. Lo que este módulo añade es el **mecanismo**, y con él la conclusión pasa de
descriptiva a explicativa:

> El agente **sí aprende algo predictivo** —bruto positivo en 10/10 corridas— y el **costo de
> ejecución de la frecuencia que elige** se lo come cuatro veces. La restricción que ata es la
> ejecución, no la predicción.

Todo sale de `outputs/thesis/decomposition_<bloque>.json`, que a su vez sale de un replay
verificado contra la evaluación original (cuadre a 1e-17). Nada aquí reentrena ni reabre el
universo: **0 trials**.

## Advertencia que acompaña a cada número de bruto

El bruto exige **costo cero**, que no existe. Es una **cota superior contrafactual**, no un
retorno alcanzable, y jamás un claim de edge. Se etiqueta así en toda tabla.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.research.cost_model import COMMISSION_PIPS_PER_SIDE, SLIPPAGE_COEF
from src.research.session_env import EXPOSURE_LEVELS, OPERABLE_RETURNS, run_session

# Supuesto del track de PRODUCCION de este mismo repo, para contrastar el de la tesis.
# `src/config/backtest_ssot.py:34-35`: spread 2,5 bps + slippage 1,0 bps.
PRODUCTION_SPREAD_BPS = 2.5
PRODUCTION_SLIPPAGE_BPS = 1.0


def production_cost_per_side_pips(price: float) -> float:
    """El costo por lado que asume producción, en pips, al precio dado.

    Se expresa en pips para que sea comparable con el `spread/2 + 0.5` de la tesis. A 4.000
    COP, 2,5 bps de spread son 1,0 pip (medio spread 0,5) y 1 bps de slippage 0,4 pips: unos
    **0,9 pips por lado**, frente a los ~2,24 que asume la tesis con su spread medio de 3,48.
    """
    half_spread = price * (PRODUCTION_SPREAD_BPS / 10_000.0) / 2.0
    slip = price * (PRODUCTION_SLIPPAGE_BPS / 10_000.0)
    return half_spread + slip


# ---------------------------------------------------------------------------
# 2b. Break-even: ¿cuánto costo aguanta el alfa?
# ---------------------------------------------------------------------------

@dataclass
class BreakEven:
    """Spread al que el bruto y el costo se igualan."""

    sum_gross: float
    sum_abs_dw: float
    sum_abs_dw_sigma: float
    mean_close: float
    spread_star_pips: float | None
    spread_assumed_pips: float
    alpha_per_unit_dw_pips: float
    cost_per_unit_dw_pips: float

    @property
    def viable_at_assumed(self) -> bool:
        return self.spread_star_pips is not None and \
            self.spread_star_pips >= self.spread_assumed_pips

    def to_dict(self) -> dict:
        return {**self.__dict__, "viable_at_assumed": self.viable_at_assumed}


def break_even_spread(sum_gross: float, sum_abs_dw: float, sum_abs_dw_sigma: float,
                      mean_close: float, spread_assumed: float) -> BreakEven:
    """Despeja el spread `s*` que hace `costo(s*) = bruto`. Forma cerrada, no búsqueda.

    El contrato de §9.3 es lineal en el spread:

        costo_pips(s) = Σ|Δw|·(s/2 + 0.5) + 0.1·Σ|Δw|·σ12

    y el costo en retorno es `costo_pips / precio`. Igualando al bruto y despejando:

        s* = 2·[ (bruto·precio − 0.1·Σ|Δw|·σ12) / Σ|Δw| − 0.5 ]

    **Por qué este número es el más útil de la tesis**: convierte «perdió» en «necesitaría un
    spread por debajo de `s*` para no perder». Lo primero no es falsable ni comparable; lo
    segundo se contrasta contra cualquier fuente de spread que aparezca —incluido el supuesto
    del propio track de producción— sin volver a entrenar nada.

    `s*` negativo significa que no hay spread admisible: ni con spread cero el alfa cubre la
    comisión de 0,5 pips por lado.
    """
    if sum_abs_dw <= 0:
        return BreakEven(sum_gross, sum_abs_dw, sum_abs_dw_sigma, mean_close, None,
                         spread_assumed, 0.0, 0.0)

    gross_pips = sum_gross * mean_close
    s_star = 2.0 * ((gross_pips - SLIPPAGE_COEF * sum_abs_dw_sigma) / sum_abs_dw
                    - COMMISSION_PIPS_PER_SIDE)
    cost_pips = sum_abs_dw * (spread_assumed / 2.0 + COMMISSION_PIPS_PER_SIDE) \
        + SLIPPAGE_COEF * sum_abs_dw_sigma
    return BreakEven(
        sum_gross=sum_gross, sum_abs_dw=sum_abs_dw, sum_abs_dw_sigma=sum_abs_dw_sigma,
        mean_close=mean_close, spread_star_pips=float(s_star),
        spread_assumed_pips=spread_assumed,
        alpha_per_unit_dw_pips=float(gross_pips / sum_abs_dw),
        cost_per_unit_dw_pips=float(cost_pips / sum_abs_dw),
    )


# ---------------------------------------------------------------------------
# 2c. ¿Ata la frecuencia? Re-scoring sin reentrenar
# ---------------------------------------------------------------------------

def hold_k_bars(weights, k: int) -> np.ndarray:
    """Decide solo cada `k` barras y arrastra la exposición.

    Con `k = 1` devuelve la senda original — es la identidad, y un test lo fija: si no lo
    fuese, toda la curva de frecuencia estaría comparando contra algo que no es el resultado
    publicado.
    """
    w = np.asarray(weights, dtype=float)
    if k <= 1:
        return w.copy()
    out = np.empty_like(w)
    for i in range(len(w)):
        out[i] = w[(i // k) * k]
    return out


def frequency_curve(sessions, specs_by_date, ks=(1, 5, 15, 30, 59)) -> list[dict]:
    """Re-puntúa las MISMAS decisiones muestreadas más gruesas.

    No es una simulación de otro agente: son las decisiones que el agente tomó, tomadas cada
    `k` barras en vez de cada una. Responde con una curva a «¿la frecuencia fue el problema?».

    **Limitación que se declara y no se puede saltar**: submuestrear la senda de un agente
    entrenado a 59 decisiones NO es lo mismo que entrenar uno a `k` decisiones. La curva
    ACOTA el efecto de la frecuencia sobre el costo; no demuestra qué haría un agente
    entrenado a esa frecuencia, que sería una hipótesis nueva con su propio trial (BL-48).
    """
    out = []
    for k in ks:
        gross = cost = net = 0.0
        n_changes = 0
        for row in sessions:
            spec = specs_by_date[row["date"]]
            w = hold_k_bars(row["weights"], k)
            res = run_session(spec.close, w, spec.spread_pips, date=spec.date)
            gross += res.gross_return
            cost += res.total_cost
            net += res.daily_return
            n_changes += res.n_changes
        out.append({"k": k, "decisions_per_session": int(np.ceil(OPERABLE_RETURNS / k)),
                    "sum_gross": gross, "sum_cost": cost, "sum_net": net,
                    "n_changes": n_changes})
    return out


# ---------------------------------------------------------------------------
# 2d. La paradoja del always-flat
# ---------------------------------------------------------------------------

def flat_paradox(sessions) -> dict:
    """`w = 0` estaba disponible y el agente nunca lo eligió de forma sostenida.

    La política óptima del bloque —no operar, 0,00%— **era alcanzable dentro del espacio de
    acción congelado** (§2, decisión 4: `{−1, −0.5, 0, +0.5, +1}`), y el agente terminó en
    −55%.

    Y el plan temía justo lo contrario: §10.1 lista *«Colapso a flat»* entre los criterios de
    trial degenerado y fija `ent_coef = 0.01` para evitarlo, con el comentario *"con costos
    penalizados el agente colapsa a 'siempre neutral' si la exploración es baja"*. **Ocurrió
    la inversión del riesgo previsto**: exploró de más, no de menos.

    Es un hallazgo sobre RL con recompensa densa y negativa, no sobre el mercado.
    """
    all_w = np.concatenate([np.asarray(r["weights"], dtype=float) for r in sessions])
    counts = {f"{lvl:+.1f}": float(np.mean(np.isclose(all_w, lvl)))
              for lvl in EXPOSURE_LEVELS}
    flat_sessions = sum(1 for r in sessions
                        if np.allclose(np.asarray(r["weights"], dtype=float), 0.0))
    return {
        "action_distribution": counts,
        "fraction_bars_flat": counts["+0.0"],
        "fully_flat_sessions": flat_sessions,
        "total_sessions": len(sessions),
        "mean_abs_exposure": float(np.mean(np.abs(all_w))),
        "mean_changes_per_session": float(np.mean([r["n_changes"] for r in sessions])),
        "note": ("`always_flat` = 0,00% era alcanzable dentro del espacio de accion y el "
                 "agente no lo encontro. El plan (§10.1) temia el colapso A flat y fijo "
                 "`ent_coef=0.01` para evitarlo; ocurrio la inversion del riesgo previsto."),
    }
