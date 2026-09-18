"""Liquidación de los brazos forward con el contrato de costos de la tesis.

Contract: CTR-RESEARCH-FORWARD-001 · Date: 2026-08-25

## Por qué no vale el `settle.py` del arnés

El original calcula `(close - open)/open` y lo multiplica por el signo. Es correcto para lo
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

`run_session` devuelve bruto y costos por separado. C048 los conserva en `accounting`.
El bruto positivo no establece alfa; el registro tampoco acredita quotes ejecutables,
fills, disponibilidad point-in-time ni congelamiento previo de todas las tarifas.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.research.llm_forward.ledger import Ledger, LedgerError  # noqa: E402
from src.research.llm_forward.paths import DECISIONS_PATH, SETTLEMENTS_PATH  # noqa: E402
from src.research.llm_forward.schema import SettlementRecord, utc_now_iso  # noqa: E402
from src.research.llm_forward.settlement_accounting import (  # noqa: E402
    aggregate,
    build_accounting,
    check_chain,
    check_decision_schedules,
    number,
    read_ledger,
    sealed,
    session_open,
    timestamp,
    validate_accounting,
    vector,
)
from src.research.session_env import EXPOSURE_LEVELS, OPERABLE_RETURNS, run_session  # noqa: E402

MIN_BARS = 60          # una sesion completa; menos no se liquida, se espera


def snap_to_action_space(score: float) -> float:
    """Lleva un score continuo al nivel de exposición más cercano del espacio congelado.

    El LLM emite un score en `[-1, 1]` continuo; el espacio de acción de la tesis es
    `{-1, -0.5, 0, +0.5, +1}` (§2, decisión 4). Sin esta proyección el LLM podría tomar
    posiciones que el RL tiene prohibidas, y la comparación mediría también esa libertad
    extra en vez de solo la calidad de la señal.
    """
    score = number(score, "score", minimum=-1, maximum=1)
    levels = np.asarray(EXPOSURE_LEVELS, dtype=float)
    return float(levels[int(np.argmin(np.abs(levels - score)))])


def weights_from_record(record: dict) -> np.ndarray:
    """Senda de exposición de un registro sellado.

    Si el registro trae `decision_path` se usa tal cual —es el brazo de 59 decisiones, donde
    la decisión ES la senda—. Si no, el score se proyecta al espacio de acción y se sostiene
    toda la sesión, que es lo que hace un brazo de una sola decisión.
    """
    path = record.get("decision_path")
    if path is not None:
        return vector(path, OPERABLE_RETURNS, "decision_path", weights=True)
    score = record["decision"]["score"]
    return np.full(OPERABLE_RETURNS, snap_to_action_space(score))


def aggregate_bar_records(records: list[dict]) -> dict | None:
    """Aggregate a complete 59-bar stream into one settlement candidate.

    Incomplete, duplicated, late, or mixed-spread streams return ``None`` and are
    deliberately not treated as zero returns.  The returned synthetic record is an
    in-memory view only; source bar records remain immutable in the decisions ledger.
    """
    return aggregate(records)


def settle_session(record: dict, closes: np.ndarray) -> dict:
    """Puntúa una decisión con el motor de la tesis. Devuelve bruto, costo y neto."""
    closes = vector(closes, MIN_BARS, "closes", positive=True)
    weights = weights_from_record(record)
    spread = record.get("spread_pips")
    if spread is None:
        raise ValueError(
            f"{record['decision_id']}: sin `spread_pips` sellado. El contrato de costos "
            "tiene que fijarse ANTES del resultado, no al liquidar."
        )

    spread = number(spread, "spread_pips", minimum=0)
    res = run_session(closes, weights, spread, date=record["session_date"])
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


@contextmanager
def _settlement_lock(path: Path):
    """Serialize cooperating settlement jobs; stale locks require operator review."""
    lock = path.with_suffix(path.suffix + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise LedgerError("settlement writer already locked") from exc
    try:
        yield
    finally:
        os.close(descriptor)
        lock.unlink()


def run(closes_by_session, min_bars: int = MIN_BARS) -> int:
    """Preflight complete paper sessions before appending; never overwrite history."""
    if type(min_bars) is not int or min_bars != MIN_BARS:
        raise ValueError("frozen settlement requires exactly 60 bars")
    ledger = Ledger(SETTLEMENTS_PATH, "decision_id")
    with _settlement_lock(ledger.path):
        return _run_locked(closes_by_session, ledger)


def _run_locked(closes_by_session, settlements) -> int:
    """Liquida toda decisión sellada y aún no liquidada.

    Args:
        closes_by_session: `{'YYYY-MM-DD': np.ndarray de cierres}`. Sale de la serie m5 del
            repo, no de un CSV aparte: la tesis se midió sobre `usdcop_m5_ohlcv` y usar otra
            fuente introduciria una diferencia que ninguna tabla mostraria.
    """
    decisions = Ledger(DECISIONS_PATH, "decision_id")
    rows = read_ledger(decisions.path)
    old_settlements = read_ledger(settlements.path)
    check_chain(rows)
    check_chain(old_settlements)
    check_decision_schedules(rows)
    for old in old_settlements:
        validate_accounting(old, rows)
    already = {row["decision_id"] for row in old_settlements}
    by_id = {row["decision_id"]: row for row in rows}
    stream_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    candidates: list[dict] = []
    for record in rows:
        if "source_decisions" in record:
            raise ValueError("source_decisions is reserved for internal aggregation")
        if record.get("bar_index") is None or record.get("decision_schedule") == "first_bar_hold":
            candidates.append(record)
        else:
            arm_key = str(record["decision_id"]).rsplit("::b", 1)[0]
            stream_groups[(record["session_date"], arm_key)].append(record)
    for group in stream_groups.values():
        aggregate = aggregate_bar_records(group)
        if aggregate is not None:
            candidates.append(aggregate)
        else:
            print(f"  [excluida/espera] stream invalido o incompleto: {group[0].get('session_date')}"
                  f" ({len(group)}/{OPERABLE_RETURNS} barras)")

    pending = []
    for record in candidates:
        decision_id = record["decision_id"]
        if decision_id in already:
            continue
        if not record.get("source_decisions") and not sealed(record):
            print(f"  [excluida] {decision_id}: no se sello antes de la apertura")
            continue
        if record.get("abstained") is not False or record.get("provider", "").lower() == "stub":
            print(f"  [omitida]  {decision_id}: abstencion, no hay nada que liquidar")
            continue

        closes = closes_by_session.get(record["session_date"])
        if closes is None or len(closes) < MIN_BARS:
            got = 0 if closes is None else len(closes)
            print(f"  [espera]   {decision_id}: {got} barras de {MIN_BARS}")
            continue
        settled_at = utc_now_iso()
        if timestamp(settled_at, "settled_at") < session_open(record["session_date"]) + timedelta(hours=5):
            print(f"  [espera] {decision_id}: sesion aun no cerrada")
            continue
        closes = vector(closes, MIN_BARS, "closes", positive=True)
        references = record.get("source_decisions")
        sources = [by_id[ref["decision_id"]] for ref in references] if references else [record]
        scored = build_accounting(record, closes, sources)
        settlement = SettlementRecord(
            seq=-1,
            decision_id=decision_id,
            session_date=record["session_date"],
            settled_at_utc=settled_at,
            open_price=float(closes[0]),
            close_price=float(closes[-1]),
            realized_return=round(float(closes[-1] / closes[0] - 1.0), 8),
            # `signed_return` se conserva con el significado del arnes —BRUTO— y el neto
            # viaja aparte. Mezclarlos haria irrecuperable la descomposicion que dio el
            # resultado principal de la tesis.
            signed_return=round(scored["gross_return"], 8),
            bars_observed=len(closes),
            accounting=scored,
        )
        validate_accounting(asdict(settlement), rows)
        pending.append(settlement)

    if len({row.decision_id for row in pending}) != len(pending):
        raise ValueError("duplicate pending settlement candidates")
    for settlement in pending:
        written = settlements.append(settlement)
        scored = settlement.accounting
        print(f"  liquidada {settlement.decision_id}: bruto {scored['gross_return']:+.5f} "
              f"costo {scored['total_cost']:.5f} neto {scored['daily_return']:+.5f} "
              f"hash={written['record_hash'][:12]}...")

    print(f"\n{len(pending)} liquidacion(es) escritas; PAPER, costos supuestos")
    return 0
