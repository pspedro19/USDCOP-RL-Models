"""La regla del hibrido esta congelada, y una de sus propiedades desmiente su justificacion.

El pre-registro v3 congelo el 2026-09-11 una regla de **acuerdo de signo**:

    w_hib(b) = w_ppo(b)  si signo(w_ppo(b)) == signo(w_llm(b)) y ninguno es 0
    w_hib(b) = 0         en cualquier otro caso

La justificacion escrita entonces decia: *"un veto solo puede reducir rotacion; no puede
inventar bruto"*. **La primera mitad de esa frase es falsa, y se midio el 2026-09-11 en cuanto
hubo datos**: sobre 33 sesiones, el hibrido hizo **11,58 cambios por sesion contra 5,85 del PPO
solo**. El veto casi duplico la rotacion.

La razon es geometrica y deberia haberse visto antes de escribirla: un veto no recorta una
posicion, la **interrumpe**. Donde el PPO mantenia +0,5 durante veinte barras seguidas, el veto
la corta cada vez que el LLM discrepa y la restaura cuando vuelve a coincidir, convirtiendo un
tramo continuo en una alternancia. Reduce el *tiempo* en posicion y aumenta el *numero de
cambios* -- y lo que cuesta dinero son los cambios, no el tiempo.

Esto no cambia la regla: congelada es congelada, y sustituirla ahora que se ve lo que hace seria
exactamente la seleccion que el pre-registro existe para impedir. Lo que cambia es el texto que
la acompana, y queda escrito aqui para que nadie repita el argumento.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.thesis_hybrid import combine  # noqa: E402


def test_agreement_keeps_the_ppo_exposure() -> None:
    """Cuando coinciden en signo, manda la magnitud del PPO -- el LLM solo vetea."""
    ppo = np.array([1.0, -0.5, 0.5, -1.0])
    llm = np.array([0.5, -1.0, 1.0, -0.5])
    np.testing.assert_array_equal(combine(ppo, llm), ppo)


def test_disagreement_flattens() -> None:
    ppo = np.array([1.0, -0.5, 0.5])
    llm = np.array([-1.0, 0.5, -0.5])
    np.testing.assert_array_equal(combine(ppo, llm), np.zeros(3))


def test_either_side_flat_flattens() -> None:
    """Cero no es una direccion: no puede haber acuerdo con la abstencion."""
    ppo = np.array([1.0, 0.0, 0.0])
    llm = np.array([0.0, 1.0, 0.0])
    np.testing.assert_array_equal(combine(ppo, llm), np.zeros(3))


def test_the_hybrid_never_takes_a_position_the_ppo_did_not_take() -> None:
    """El veto no puede inventar exposicion. Esa mitad de la justificacion SI se sostiene."""
    rng = np.random.default_rng(0)
    levels = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    for _ in range(200):
        ppo = rng.choice(levels, size=59)
        llm = rng.choice(levels, size=59)
        out = combine(ppo, llm)
        nonzero = out != 0.0
        np.testing.assert_array_equal(out[nonzero], ppo[nonzero])
        assert np.count_nonzero(out) <= np.count_nonzero(ppo)


def test_the_veto_can_INCREASE_the_number_of_changes() -> None:
    """La mitad falsa de la justificacion, fijada con un contraejemplo minimo.

    El PPO mantiene +1 las seis barras: **un** cambio (entrar). El LLM discrepa en las barras
    alternas, asi que el hibrido entra y sale tres veces: **seis** cambios. Multiplicar por seis
    el numero de transiciones es multiplicar por seis el peaje.
    """
    ppo = np.array([1.0] * 6)
    llm = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    hybrid = combine(ppo, llm)

    def changes(path: np.ndarray) -> int:
        return int(np.count_nonzero(np.diff(np.concatenate([[0.0], path]))))

    assert changes(ppo) == 1
    assert changes(hybrid) == 6
    assert changes(hybrid) > changes(ppo), (
        "si este test se pusiera verde al reves, habria que revisar la nota del pre-registro: "
        "significaria que el veto si reduce rotacion y el argumento original era correcto"
    )


@pytest.mark.parametrize("n", [1, 59])
def test_shape_is_preserved(n: int) -> None:
    assert combine(np.zeros(n), np.zeros(n)).shape == (n,)
