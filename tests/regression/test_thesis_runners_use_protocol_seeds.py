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

Se comprueba el valor efectivo del runner. Importar la lista desde un SSOT no es
una desviacion: la igualdad exacta protege contra sustituir o reordenar semillas.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SEEDS = (42, 123, 456, 789, 1337)

RUNNERS = (
    "scripts/analysis/thesis_train_ppo.py",
    "scripts/analysis/thesis_ppo_sanity.py",
)


@pytest.mark.parametrize("relative", RUNNERS)
def test_runner_declares_the_protocol_seeds(relative: str) -> None:
    path = ROOT / relative
    if not path.is_file():
        pytest.skip(f"{relative} no existe en este checkout")
    module = importlib.import_module(relative[:-3].replace('/', '.'))
    seeds = tuple(module.SEEDS)
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
