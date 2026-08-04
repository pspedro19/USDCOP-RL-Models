"""BL-45 §15.2 — la policy stateful reproduce la DECISION del simulador congelado.

`gold_dynamic_exit` es la estrategia que la spec cita como la que exige estado. Vivia
solo como simulador de investigacion, y mientras tanto `PolicyContext.state` no tenia
ningun consumidor.

Estos candados exigen lo unico que hace util el port: que la policy sostenga la MISMA
serie de exposicion que el bucle congelado, barra a barra. No se compara el PnL --eso es
ejecucion y el contrato la separa de la señal a proposito--, sino la posicion, que es lo
que la policy decide.

El estado se ejercita de verdad porque las tres propiedades que lo requieren no son
derivables de la barra actual: el trailing solo sube, el sizing se fija en la entrada, y
la salida se evalua antes de mover el stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts.policy import PolicyContext  # noqa: E402

SPEC = {
    "id": "xauusd_dynamic_exit_v1",
    "version": "1.0.0",
    "engine": {"type": "rule_based", "implementation": {"mode": "coded_policy"}},
    "inputs": {
        "feature_set_id": "xauusd_dynamic_exit_action_v1",
        "resample_policy_id": "gold_daily_official_v1",
        "required_features": ["open", "high", "low", "close", "atr_14", "atr_prev",
                              "signal_prev", "size_prev"],
        "optional_features": [],
        "decision_point": "session_close",
    },
    "policy": {
        "params": {"trail_mult": 2.0},
        "resolution": {"mode": "first_match", "default_target_exposure": 0.0,
                       "default_direction": "FLAT", "default_reason_code": "NO_SIGNAL"},
        "missing_input_policy": "FAIL_CLOSED",
        "stale_input_policy": "FLAT",
    },
}


def _politica():
    from src.strategies.policies.gold_dynamic_exit import GoldDynamicExitPolicy

    return GoldDynamicExitPolicy(SPEC)


def _barra(**kw):
    base = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
            "atr_14": 1.0, "atr_prev": 1.0, "signal_prev": 1, "size_prev": 0.5}
    base.update(kw)
    return base


def test_the_trailing_stop_only_ever_rises() -> None:
    """La propiedad que OBLIGA a tener memoria: con la barra de hoy no se sabe el stop."""
    politica = _politica()
    ctx = PolicyContext(as_of="2026-01-05", mode="DECISION")

    politica.evaluate(_barra(open=100.0, close=100.0), ctx)
    stop_entrada = ctx.state["trail_px"]

    politica.evaluate(_barra(close=110.0, low=105.0), ctx)
    stop_alto = ctx.state["trail_px"]
    assert stop_alto > stop_entrada, "el trailing no subio con el precio"

    # El precio retrocede SIN perforar el stop: el stop debe quedarse donde estaba.
    # (La primera version de este test usaba low=103 con el stop en 108 y la policy
    # salia correctamente; el escenario estaba mal construido, no el codigo.)
    assert stop_alto == 108.0
    politica.evaluate(_barra(close=109.0, low=108.5), ctx)
    assert ctx.state["trail_px"] == stop_alto, (
        "el trailing bajo al retroceder el precio: dejaria de ser un trailing"
    )
    assert ctx.state["in_trade"] is True, "salio sin que el minimo perforara el stop"


def test_size_is_fixed_at_entry_and_not_recomputed_each_bar() -> None:
    """Recalcular el sizing cada barra seria OTRA estrategia, no esta."""
    politica = _politica()
    ctx = PolicyContext(as_of="2026-01-05", mode="DECISION")

    entrada = politica.evaluate(_barra(size_prev=0.5), ctx)
    assert float(entrada.target_exposure) == 0.5

    # size_prev cambia, pero el trade sigue abierto: la exposicion no debe moverse.
    dentro = politica.evaluate(_barra(size_prev=1.4, close=101.0, low=100.5), ctx)
    assert float(dentro.target_exposure) == 0.5, (
        "el sizing se recalculo dentro del trade: cambia la estrategia sin declararlo"
    )


def test_a_dead_signal_exits_and_clears_the_state() -> None:
    """Salir debe dejar el store limpio: un residuo reabriria con datos del trade viejo."""
    politica = _politica()
    ctx = PolicyContext(as_of="2026-01-05", mode="DECISION")
    politica.evaluate(_barra(), ctx)
    assert ctx.state["in_trade"] is True

    salida = politica.evaluate(_barra(signal_prev=0), ctx)
    assert salida.direction == "FLAT"
    assert salida.reason_codes == ("EXIT_SIGNAL_OFF",)
    assert ctx.state["in_trade"] is False and ctx.state["trail_px"] == 0.0


def test_the_trailing_exit_uses_the_low_not_the_close() -> None:
    """El stop es intra-dia: comparar contra el cierre lo haria disparar tarde."""
    politica = _politica()
    ctx = PolicyContext(as_of="2026-01-05", mode="DECISION")
    politica.evaluate(_barra(open=100.0, close=100.0), ctx)
    stop = ctx.state["trail_px"]

    # El minimo perfora el stop aunque el cierre quede por encima.
    salida = politica.evaluate(_barra(low=stop - 0.5, close=stop + 5.0), ctx)
    assert salida.direction == "FLAT"
    assert salida.reason_codes == ("EXIT_TRAILING_STOP",)


def test_entry_uses_yesterdays_signal_not_todays() -> None:
    """Causalidad: entrar con la señal de hoy adelantaria la informacion un dia."""
    politica = _politica()
    ctx = PolicyContext(as_of="2026-01-05", mode="DECISION")

    plano = politica.evaluate(_barra(signal_prev=0), ctx)
    assert plano.direction == "FLAT" and ctx.state["in_trade"] is False

    from src.strategies.policies.gold_dynamic_exit import GoldDynamicExitPolicy

    assert "signal_prev" in GoldDynamicExitPolicy.required
    assert "signal" not in GoldDynamicExitPolicy.required, (
        "la policy no puede consumir la señal de hoy: seria look-ahead de un dia"
    )


def test_two_contexts_do_not_share_the_open_trade() -> None:
    """Aislamiento real: dos activos no pueden heredar el trade del otro."""
    politica = _politica()
    uno = PolicyContext(as_of="2026-01-05", mode="DECISION")
    otro = PolicyContext(as_of="2026-01-05", mode="DECISION")

    politica.evaluate(_barra(), uno)
    assert uno.state["in_trade"] is True
    assert otro.state == {}, "el segundo contexto heredo el trade abierto del primero"


def test_the_entry_stop_uses_yesterdays_atr_not_todays() -> None:
    """Defecto que midio Codex (CXD-490) y por el que rechazo la promocion: era real.

    El simulador fija el stop de entrada con `atr_14.iloc[i-1]` y solo DESPUES lo
    actualiza con el de hoy. Mi port usaba el de hoy en ambos sitios:

        atr_prev=1, atr=10  ->  simulador 98.0   ·   mi policy 80.0

    No es un detalle de precision: usar el ATR de HOY para el stop de entrada consume
    informacion de la misma barra en la que se entra, que es la clase de fuga que esta
    familia de estrategias tiene mas a mano.
    """
    politica = _politica()
    ctx = PolicyContext(as_of="2026-01-05", mode="DECISION")

    politica.evaluate(_barra(open=100.0, close=100.0, atr_14=10.0, atr_prev=1.0), ctx)

    assert ctx.state["trail_px"] == 98.0, (
        f"stop de entrada {ctx.state['trail_px']} en vez de 98.0: el ATR de hoy "
        "volvio a fijar la entrada"
    )


def test_partial_state_fails_closed_instead_of_defaulting_to_zero() -> None:
    """Un `in_trade=True` sin memoria no es "un trade de tamaño cero": es corrupcion.

    Mi primera version leia `estado.get("size", 0.0)`, asi que un store truncado --por
    un reinicio a medias, por una restauracion parcial-- habria decidido sobre memoria
    inventada. Es el mismo defecto que perseguimos en los datos (una ausencia
    renderizada como valor), aplicado al estado.
    """
    politica = _politica()
    ctx = PolicyContext(as_of="2026-01-05", mode="DECISION")
    ctx.state = {"in_trade": True}   # truncado: sin size ni trail_px

    with pytest.raises(ValueError, match="estado incompleto"):
        politica.evaluate(_barra(), ctx)
