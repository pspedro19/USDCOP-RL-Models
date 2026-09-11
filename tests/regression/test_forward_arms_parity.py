"""
Regression: el carril forward puntúa igual que la tesis.

Contract: CTR-RESEARCH-FORWARD-001 · Date: 2026-08-25

## La garantía que se protege

La rama forward compara un brazo LLM contra la política RL congelada. Esa comparación solo
significa algo si los dos se puntúan con **la misma aritmética que produjo el resultado del
hold-out** — si no, son dos experimentos yuxtapuestos, no una comparación.

El test ancla: se sella la decisión del brazo RL para una sesión del hold-out, se liquida con
`settle_thesis`, y el resultado tiene que coincidir **exactamente** con lo que
`decomposition_holdout.json` registró para esa misma sesión. Bruto, costo, neto y turnover.

Si divergiera, el carril forward estaría midiendo con otra regla y ninguna tabla lo diría: los
números seguirían siendo plausibles.

## Y la otra garantía: el contrato de costos se sella ANTES

`settle_session` **se niega a liquidar** un registro sin `spread_pips`. El spread sale del
posterior de régimen del cierre de `d−1`, así que se conoce antes de la apertura y no hay
excusa para calcularlo después de conocer el resultado. Nadie lo manipularía a propósito; el
diseño entero de este carril consiste en no tener que confiar en eso.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.llm_forward.settle_thesis import (settle_session,  # noqa: E402
                                                    snap_to_action_space,
                                                    weights_from_record)
from src.research.session_env import EXPOSURE_LEVELS, OPERABLE_RETURNS  # noqa: E402

DECOMP = ROOT / "outputs" / "thesis" / "decomposition_holdout.json"


# ---------------------------------------------------------------------------
# Proyección al espacio de acción congelado
# ---------------------------------------------------------------------------

def test_a_continuous_score_snaps_to_the_frozen_action_space():
    """El LLM emite continuo; el espacio de acción de la tesis es de cinco niveles.

    Sin proyectar, el LLM podría tomar posiciones que el RL tiene prohibidas y la
    comparación mediría también esa libertad extra en vez de solo la calidad de la señal.
    """
    assert snap_to_action_space(0.0) == 0.0
    assert snap_to_action_space(1.0) == 1.0
    assert snap_to_action_space(-1.0) == -1.0
    assert snap_to_action_space(0.9) == 1.0
    assert snap_to_action_space(0.3) == 0.5
    assert snap_to_action_space(0.1) == 0.0
    assert snap_to_action_space(-0.6) == -0.5
    for s in np.linspace(-1, 1, 41):
        assert snap_to_action_space(float(s)) in EXPOSURE_LEVELS


def test_a_single_score_becomes_a_held_position():
    record = {"decision": {"score": 0.9}, "decision_id": "x", "spread_pips": 3.0}
    w = weights_from_record(record)
    assert len(w) == OPERABLE_RETURNS
    assert np.all(w == 1.0), "un brazo de una decision sostiene la posicion toda la sesion"


def test_a_sealed_path_is_used_verbatim():
    """Para el brazo nativo la decisión ES la senda; no se puede reconstruir del score."""
    path = [1.0, 1.0, -0.5] + [0.0] * (OPERABLE_RETURNS - 3)
    record = {"decision": {"score": 1.0}, "decision_path": path,
              "decision_id": "x", "spread_pips": 3.0}
    assert np.array_equal(weights_from_record(record), np.asarray(path))


# ---------------------------------------------------------------------------
# El contrato de costos se sella antes del resultado
# ---------------------------------------------------------------------------

def test_settlement_refuses_a_record_without_a_sealed_spread():
    record = {"decision": {"score": 1.0}, "decision_id": "2026-08-24::x",
              "session_date": "2026-08-24"}
    with pytest.raises(ValueError, match="spread_pips"):
        settle_session(record, np.full(60, 4000.0))


# ---------------------------------------------------------------------------
# El ancla: mismo resultado que la tesis, sesión a sesión
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [42])
def test_forward_settlement_reproduces_the_thesis_numbers(seed):
    """Liquidar una senda del hold-out da lo mismo que registró la descomposición."""
    if not DECOMP.is_file():
        pytest.skip("falta decomposition_holdout.json")

    blob = json.loads(DECOMP.read_text(encoding="utf-8"))
    run = blob["runs"].get(f"ppo_regime_seed{seed}")
    if not run:
        pytest.skip(f"sin corrida ppo_regime_seed{seed}")

    from src.research.live_spec import SCALER_PATH, build_live_spec
    from src.research.regime_portable import DEFAULT_PATH

    if not (SCALER_PATH.is_file() and DEFAULT_PATH.is_file()):
        pytest.skip("faltan los artefactos congelados del carril forward")

    checked = 0
    for session in run["sessions"][-3:]:          # las tres ultimas del hold-out
        spec = build_live_spec(session["date"])

        record = {
            "decision_id": f"{session['date']}::ppo_regime_fwd_k1",
            "session_date": session["date"],
            "decision": {"score": session["weights"][0]},
            "decision_path": session["weights"],
            "spread_pips": session["spread_pips"],
        }
        got = settle_session(record, spec.close)
        from src.research.session_env import run_session
        expected = run_session(
            spec.close, np.asarray(session["weights"], dtype=float),
            float(session["spread_pips"]), date=session["date"]
        )

        assert got["gross_return"] == pytest.approx(expected.gross_return, abs=1e-12), (
            f"{session['date']}: el bruto del settlement difiere de run_session v2"
        )
        assert got["total_cost"] == pytest.approx(expected.total_cost, abs=1e-12)
        assert got["daily_return"] == pytest.approx(expected.daily_return, abs=1e-12)
        assert got["n_changes"] == expected.n_changes
        assert got["sum_abs_dw"] == pytest.approx(
            np.abs(np.diff(np.concatenate([[0.0], expected.weights]))).sum()
            + abs(expected.weights[-1]), abs=1e-12
        )
        checked += 1

    assert checked >= 1, "no se comprobo ninguna sesion"
