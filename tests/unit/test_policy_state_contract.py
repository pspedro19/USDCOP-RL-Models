"""BL-45 §15.2 — el contrato del estado, y su primer consumidor real.

Cuando escribí estos candados, `PolicyContext.state` existía y **no lo usaba nadie**:
cero referencias en `src/strategies/`. Era el mismo "mecanismo correcto sin llamador"
del que salió BL-16, escondido en un campo de dataclass. `gold_dynamic_exit` —la
estrategia que la spec cita como la que exige estado— vivía sólo como simulador de
investigación en `scripts/analysis/`.

El orden importó: primero se fijó **qué garantías debe dar el store** (aislamiento entre
contextos, no herencia entre modos), y sólo después nació la policy que lo usa. Al revés,
esas propiedades se habrían descubierto por accidente, que es como se cuelan los fallos
de aislamiento entre estrategias.

El canario que exigía "nadie usa `state` todavía" **se puso rojo** en cuanto la policy
existió — su trabajo — y fue sustituido por su sucesor: quien use el store debe declarar
`STATE_KEYS`, para que el runner sepa qué persistir.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.contracts.policy import PolicyContext


def test_state_exists_and_defaults_to_an_isolated_store() -> None:
    """Cada contexto nace con su propio `state`: nunca un dict compartido de clase.

    Un `state` por defecto compartido —el clásico argumento mutable por defecto— haría
    que dos estrategias distintas se pisaran los contadores sin que nada lo revelara, y
    el síntoma aparecería como "la señal de Oro depende de si antes corrió BTC".
    """
    campos = {f.name for f in dataclasses.fields(PolicyContext)}
    assert "state" in campos, "PolicyContext perdió el store de estado de §15.2"

    uno = PolicyContext(as_of="2026-01-05")
    otro = PolicyContext(as_of="2026-01-05")

    uno.state["racha"] = 3
    assert otro.state == {}, (
        "dos contextos comparten el mismo dict de estado: una estrategia leería los "
        "contadores de otra"
    )


def test_state_survives_within_a_context_and_does_not_leak_across_modes() -> None:
    """El estado persiste entre evaluaciones del MISMO contexto, no entre modos.

    `PolicyContext.mode` distingue DECISION de FREEZE/REVALIDATE/BACKFILL. Reusar el
    mismo store entre una decisión real y un backfill mezclaría una racha histórica con
    la viva — y el backfill, que existe para reconstruir el pasado, acabaría alterando
    el presente.
    """
    vivo = PolicyContext(as_of="2026-01-05", mode="DECISION")
    vivo.state["cierres_sobre_ma"] = 2

    reconstruccion = PolicyContext(as_of="2020-01-05", mode="BACKFILL")
    assert reconstruccion.state == {}, (
        "el backfill arrancó con el estado de la decisión viva: reconstruir el pasado "
        "no puede heredar el presente"
    )


def test_every_stateful_policy_declares_the_keys_it_persists() -> None:
    """Toda policy que use el store debe DECLARAR que guarda y por que.

    Este candado sucede al canario que exigia que nadie usara `state` todavia: se puso
    rojo en cuanto nacio `gold_dynamic_exit` como policy stateful, que era exactamente
    su trabajo -- obligar a que ese incremento llegara con su contrato en vez de
    aparecer como efecto colateral.

    Ahora exige lo que hace operable el estado: que el modulo publique `STATE_KEYS`. Sin
    esa lista, el runner no sabe QUE persistir entre corridas, y un trade abierto se
    perderia en silencio al reiniciar -- la policy volveria a entrar creyendo estar plana.
    """
    import importlib
    import inspect
    from pathlib import Path

    raiz = Path(inspect.getfile(PolicyContext)).resolve().parents[2]
    for ruta in (raiz / "src" / "strategies").rglob("*.py"):
        texto = ruta.read_text(encoding="utf-8", errors="ignore")
        if "context.state" not in texto and "ctx.state" not in texto:
            continue
        modulo = importlib.import_module(
            str(ruta.relative_to(raiz).with_suffix("")).replace("\\", ".").replace("/", ".")
        )
        claves = getattr(modulo, "STATE_KEYS", None)
        assert claves, (
            f"{ruta.name} usa el store de estado sin declarar STATE_KEYS: el runner no "
            "sabria que persistir, y un trade abierto se perderia al reiniciar"
        )
        assert all(isinstance(k, str) and k for k in claves), (
            f"{ruta.name}: STATE_KEYS debe ser una lista de nombres"
        )


@pytest.mark.parametrize("modo", ["DECISION", "FREEZE", "REVALIDATE", "BACKFILL"])
def test_every_declared_mode_accepts_a_state_store(modo) -> None:
    """Los cuatro modos admiten estado: ninguno queda fuera del contrato por descuido."""
    contexto = PolicyContext(as_of="2026-01-05", mode=modo)
    contexto.state["x"] = 1
    assert contexto.state == {"x": 1}
