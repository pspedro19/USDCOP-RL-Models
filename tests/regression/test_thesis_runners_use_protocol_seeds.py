"""Todo runner de la tesis usa las cinco semillas del protocolo, sin excepciones.

`.claude/rules/experiment-protocol.md` regla 2 fija `[42, 123, 456, 789, 1337]` y anade
"sin excepciones". La regla existe por una razon medida, citada en la propia regla: la
semilla 456 perdio 20,6 % y la 1337 gano 9,6 % en el mismo experimento, asi que la eleccion
de semillas mueve el resultado y dejarla suelta convierte cualquier comparacion en otra cosa.

El 2026-09-11, `thesis_ppo_sanity.py` declaraba `(42, 123, 456, 789, 2024)`. Eso contradecia
a la vez la regla, la tabla de identidad congelada del pre-registro v3 -- que publica las
cinco correctas -- y a `thesis_train_ppo.py`, que si las usaba. El efecto practico es que la
evidencia de sanidad del optimizador no era comparable por semilla con las corridas de
mercado que pretendia justificar, y nada lo senalaba.

Se comprueba por AST y no por texto: una lista de semillas escrita como `range`, importada o
recompuesta seguiria siendo una desviacion, y un `grep` de los cinco numeros la daria por
buena.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SEEDS = (42, 123, 456, 789, 1337)

RUNNERS = (
    "scripts/analysis/thesis_train_ppo.py",
    "scripts/analysis/thesis_ppo_sanity.py",
)


def _seeds_literal(path: Path) -> tuple[int, ...] | None:
    """Devuelve la tupla asignada a SEEDS a nivel de modulo, o None si no es literal."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "SEEDS" not in names:
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            return None
        return tuple(value) if isinstance(value, (tuple, list)) else None
    return None


@pytest.mark.parametrize("relative", RUNNERS)
def test_runner_declares_the_protocol_seeds(relative: str) -> None:
    path = ROOT / relative
    if not path.is_file():
        pytest.skip(f"{relative} no existe en este checkout")
    seeds = _seeds_literal(path)
    assert seeds is not None, (
        f"{relative} no asigna SEEDS a un literal evaluable; la regla exige una lista "
        "declarada, no construida en tiempo de ejecucion."
    )
    assert seeds == PROTOCOL_SEEDS, (
        f"{relative} declara {seeds} y el protocolo fija {PROTOCOL_SEEDS} sin excepciones "
        "(`.claude/rules/experiment-protocol.md` regla 2). Cambiar una semilla rompe la "
        "comparabilidad con el resto de la evidencia del experimento."
    )


def test_frozen_prereg_publishes_the_same_seeds() -> None:
    """La identidad congelada y el codigo no pueden discrepar en las semillas."""
    prereg = ROOT / ".claude" / "specs" / "planes" / "06-PRE-REGISTRATION-v3.md"
    if not prereg.is_file():
        pytest.skip("el pre-registro v3 no existe en este checkout")
    text = prereg.read_text(encoding="utf-8")
    for seed in PROTOCOL_SEEDS:
        assert str(seed) in text, (
            f"el pre-registro v3 no publica la semilla {seed}; su tabla de identidad debe "
            "coincidir con la que ejecutan los runners."
        )
