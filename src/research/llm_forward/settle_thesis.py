"""Liquidación de los brazos forward con el contrato de costos de la tesis.

Contract: CTR-RESEARCH-FORWARD-001 · Date: 2026-08-25

## Por qué no vale el `settle.py` del arnés

El original calcula `(close − open)/open` y lo multiplica por el signo. Es correcto para lo
que era —un brazo, una decisión, bruto de costos— pero aquí se comparan cuatro brazos, uno de
ellos con 59 decisiones por sesión, y **la comparación solo significa algo si los cuatro se
puntúan con la misma aritmética**.

Esa aritmética ya existe y ya está verificada: `session_env::run_session`, el motor que produjo
las tablas del hold-out, con `cost_model` (§9.3). Reimplementarla aquí sería crear una segunda
definición de «cuánto costó», y la primera vez que discrepen nadie sabrá cuál mirar.

## La consecuencia práctica

Como el motor es el mismo, las cifras forward son comparables **línea a línea** con las tablas
4.8 / 4.11 / 4.12 del hold-out. El `s*` de break-even se calcula con la misma función
(`decomposition::break_even_spread`), así que la pregunta «¿cuánto costo aguantaría?» se
responde igual en los dos lados.

## Bruto y neto, separados

`run_session` devuelve `gross_return` y `total_cost` por separado, así que la liquidación los
guarda por separado. Es lo que permitió descubrir que el agente de la tesis **sí tenía señal** y
que el costo se la comía — un dato que estuvo meses invisible por estar solo el neto.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.research.llm_forward.ledger import Ledger, LedgerError  # noqa: E402
from src.research.llm_forward.paths import DECISIONS_PATH, SETTLEMENTS_PATH  # noqa: E402
from src.research.llm_forward.schema import SettlementRecord, utc_now_iso  # noqa: E402
from src.research.session_env import (EXPOSURE_LEVELS, OPERABLE_RETURNS,  # noqa: E402
                                      run_session)

MIN_BARS = 60          # una sesion completa; menos no se liquida, se espera


def snap_to_action_space(score: float) -> float:
    """Lleva un score continuo al nivel de exposición más cercano del espacio congelado.

    El LLM emite un score en `[-1, 1]` continuo; el espacio de acción de la tesis es
    `{−1, −0.5, 0, +0.5, +1}` (§2, decisión 4). Sin esta proyección el LLM podría tomar
    posiciones que el RL tiene prohibidas, y la comparación mediría también esa libertad
    extra en vez de solo la calidad de la señal.
    """
    levels = np.asarray(EXPOSURE_LEVELS, dtype=float)
    return float(levels[int(np.argmin(np.abs(levels - float(score))))])


def weights_from_record(record: dict) -> np.ndarray:
    """Senda de exposición de un registro sellado.

    Si el registro trae `decision_path` se usa tal cual —es el brazo de 59 decisiones, donde
    la decisión ES la senda—. Si no, el score se proyecta al espacio de acción y se sostiene
    toda la sesión, que es lo que hace un brazo de una sola decisión.
    """
    path = record.get("decision_path")
    if path:
        return np.asarray(path, dtype=float)
    score = record["decision"]["score"]
    return np.full(OPERABLE_RETURNS, snap_to_action_space(score))


def settle_session(record: dict, closes: np.ndarray) -> dict:
    """Puntúa una decisión con el motor de la tesis. Devuelve bruto, costo y neto."""
    weights = weights_from_record(record)
    spread = record.get("spread_pips")
    if spread is None:
        raise ValueError(
            f"{record['decision_id']}: sin `spread_pips` sellado. El contrato de costos "
            "tiene que fijarse ANTES del resultado, no al liquidar."
        )

    res = run_session(closes, weights, float(spread), date=record["session_date"])
    dw = np.abs(np.diff(np.concatenate([[0.0], weights, [0.0]])))
    return {
        "gross_return": float(res.gross_return),
        "total_cost": float(res.total_cost),
        "daily_return": float(res.daily_return),
        "terminal_cost": float(res.terminal_cost),
        "n_changes": int(res.n_changes),
        "mean_abs_exposure": float(res.mean_abs_exposure),
        "sum_abs_dw": float(dw.sum()),
        "spread_pips": float(spread),
    }


def run(closes_by_session, min_bars: int = MIN_BARS) -> int:
    """Liquida toda decisión sellada y aún no liquidada.

    Args:
        closes_by_session: `{'YYYY-MM-DD': np.ndarray de cierres}`. Sale de la serie m5 del
            repo, no de un CSV aparte: la tesis se midió sobre `usdcop_m5_ohlcv` y usar otra
            fuente introduciria una diferencia que ninguna tabla mostraria.
    """
    decisions = Ledger(DECISIONS_PATH, "decision_id")
    settlements = Ledger(SETTLEMENTS_PATH, "decision_id")
    already = settlements.keys()
    n = 0

    for record in decisions:
        decision_id = record["decision_id"]
        if decision_id in already:
            continue
        if not record["sealed_before_open"]:
            print(f"  [excluida] {decision_id}: no se sello antes de la apertura")
            continue
        if record["abstained"]:
            print(f"  [omitida]  {decision_id}: abstencion, no hay nada que liquidar")
            continue

        closes = closes_by_session.get(record["session_date"])
        if closes is None or len(closes) < min_bars:
            got = 0 if closes is None else len(closes)
            print(f"  [espera]   {decision_id}: {got} barras de {min_bars}")
            continue

        scored = settle_session(record, np.asarray(closes, dtype=float))
        settlement = SettlementRecord(
            seq=-1,
            decision_id=decision_id,
            session_date=record["session_date"],
            settled_at_utc=utc_now_iso(),
            open_price=float(closes[0]),
            close_price=float(closes[-1]),
            realized_return=round(float(closes[-1] / closes[0] - 1.0), 8),
            # `signed_return` se conserva con el significado del arnes —BRUTO— y el neto
            # viaja aparte. Mezclarlos haria irrecuperable la descomposicion que dio el
            # resultado principal de la tesis.
            signed_return=round(scored["gross_return"], 8),
            bars_observed=len(closes),
        )
        try:
            written = settlements.append(settlement)
        except LedgerError as exc:
            print(f"  [rechazada] {exc}")
            continue

        print(f"  liquidada {decision_id}: bruto {scored['gross_return']:+.5f} "
              f"costo {scored['total_cost']:.5f} neto {scored['daily_return']:+.5f} "
              f"hash={written['record_hash'][:12]}...")
        n += 1

    print(f"\n{n} liquidacion(es) escritas")
    return 0
