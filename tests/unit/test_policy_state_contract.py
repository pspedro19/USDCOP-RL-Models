"""BL-45 §15.2 — el contrato del estado, antes de que exista la primera policy stateful.

Medido al auditar el hueco que la ficha nombra:

* `PolicyContext.state` **existe** (`src/contracts/policy.py`) y **no lo usa nadie**: cero
  referencias a `context.state` en `src/strategies/` y `src/contracts/`. Es el mismo
  patrón "mecanismo correcto sin llamador" del que salió BL-16.
* `gold_dynamic_exit` —la estrategia que la spec cita como la que exige estado— **no es
  una `Policy`**: vive en `scripts/analysis/` como un simulador de investigación cuyo
  estado son variables locales del bucle (`in_trade`, `entry_i`, `hi_close`, `trail_px`,
  `prev_px`).

Portarla es modelado, no migración, y no se hace de paso. Lo que sí se puede hacer hoy
—y es lo que falta para que el port no herede un bug— es **fijar el contrato del estado**:
qué garantías debe dar el store antes de que alguien confíe en él. Sin esto, la primera
policy stateful descubriría estas propiedades por accidente, que es como se cuelan los
fallos de aislamiento entre estrategias.
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


def test_no_production_policy_silently_depends_on_state_yet() -> None:
    """Hoy NINGUNA policy productiva usa `state`, y eso debe seguir siendo explícito.

    Este candado se pondrá **rojo** cuando alguien escriba la primera policy stateful —
    que es su propósito: obliga a que ese incremento venga con su propia decisión sobre
    persistencia (quién guarda el store entre corridas, qué pasa si se pierde) en vez de
    aparecer como un efecto colateral. Portar `gold_dynamic_exit` es modelado, y el
    modelado se declara.
    """
    import inspect
    from pathlib import Path

    raiz = Path(inspect.getfile(PolicyContext)).resolve().parents[2]
    usuarios = []
    for ruta in (raiz / "src" / "strategies").rglob("*.py"):
        texto = ruta.read_text(encoding="utf-8", errors="ignore")
        if "context.state" in texto or "ctx.state" in texto:
            usuarios.append(ruta.name)

    assert not usuarios, (
        f"{usuarios} ya usa el store de estado: este incremento necesita declarar su "
        "política de persistencia (§15.2) — quién lo guarda entre corridas y qué decide "
        "la policy si el estado se perdió"
    )


@pytest.mark.parametrize("modo", ["DECISION", "FREEZE", "REVALIDATE", "BACKFILL"])
def test_every_declared_mode_accepts_a_state_store(modo) -> None:
    """Los cuatro modos admiten estado: ninguno queda fuera del contrato por descuido."""
    contexto = PolicyContext(as_of="2026-01-05", mode=modo)
    contexto.state["x"] = 1
    assert contexto.state == {"x": 1}
