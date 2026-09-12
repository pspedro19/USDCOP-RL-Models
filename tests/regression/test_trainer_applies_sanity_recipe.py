"""La compuerta de sanidad tiene que GOBERNAR al entrenador, no solo informarle.

El 2026-09-11 se midio que no lo hacia. `thesis_train_ppo.py` leia
`outputs/thesis-repair/sanity_protocol_v2.json`, imprimia `sanity gate: receta=flat_init_no_turn`
y entrenaba la receta por defecto: el sesgo inicial de `action_net` -- que ES la sonda que abrio
S1-S4 -- vivia unicamente dentro del script de sanidad.

El efecto no es cosmetico. Sin ese sesgo, 300k pasos sobre el dataset v2 con la semilla 42 dan
909 operaciones en 226 sesiones y un coste de 0,4855 sobre un bruto de ~0,272: el cuadro exacto
de la tesis v1 rechazada. Con el, la misma receta se abstiene sobre ruido puro.

Una compuerta que valida un informe mientras corre otra receta no es una compuerta; es una nota
al pie. Estos tests fijan las tres propiedades que la convierten en compuerta:

  1. una sonda desconocida ABORTA en vez de degradarse al default,
  2. el sesgo se aplica de verdad sobre la politica (medido en la distribucion inicial),
  3. el entrenador declara en el artefacto que receta corrio.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.ppo_recipe import (  # noqa: E402
    CONTROL_ARM,
    apply_flat_bias,
    known_probes,
    recipe_for,
)

TRAINER = ROOT / "scripts" / "analysis" / "thesis_train_ppo.py"


def test_unknown_probe_aborts_instead_of_defaulting() -> None:
    with pytest.raises(ValueError, match="receta desconocida"):
        recipe_for("una_sonda_que_nadie_declaro")


def test_the_gate_winning_probe_is_declared() -> None:
    """`flat_init_no_turn` es la receta que abrio S1-S4; si no esta declarada, nada la aplica."""
    recipe = recipe_for("flat_init_no_turn")
    assert recipe.flat_bias_logit == 3.0
    assert recipe.kappa_turn == 0.0
    assert "flat_init_no_turn" in known_probes()


def test_control_arm_carries_no_bias() -> None:
    """El brazo de control existe para contrastar, asi que no puede llevar la receta dentro."""
    assert recipe_for(CONTROL_ARM).flat_bias_logit is None


def test_flat_bias_puts_the_mass_on_not_trading() -> None:
    """Medido sobre la politica real: el arranque tiene que ser practicamente plano."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    import gymnasium as gym

    from src.research.session_env import EXPOSURE_LEVELS

    class _Tiny(gym.Env):
        observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,))
        action_space = gym.spaces.Discrete(len(EXPOSURE_LEVELS))

        def reset(self, *, seed=None, options=None):
            return np.zeros(4, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(4, dtype=np.float32), 0.0, True, False, {}

    model = PPO("MlpPolicy", make_vec_env(_Tiny, n_envs=1), n_steps=8, batch_size=8,
                device="cpu", verbose=0, seed=0)

    obs = torch.zeros((1, 4))
    flat_index = EXPOSURE_LEVELS.index(0.0)

    assert apply_flat_bias(model, recipe_for("flat_init_no_turn")) is True
    with torch.no_grad():
        probs = model.policy.get_distribution(obs).distribution.probs[0]
    # 0,834 exactos con observacion nula: softmax([0,0,3,0,0]) = e^3/(e^3+4). El comentario
    # original de la sonda decia "~95 %", que es falso -- lo cazo este test. La cota se fija en
    # 0,80 para no atarse al decimal, pero el valor esperado esta escrito aqui a proposito.
    assert float(probs[flat_index]) > 0.80, (
        f"la receta debe arrancar casi plana y arranca con {float(probs[flat_index]):.3f} "
        "en 'no operar'"
    )
    assert abs(float(probs[flat_index]) - 0.8339) < 0.01

    assert apply_flat_bias(model, recipe_for(CONTROL_ARM)) is False, (
        "el brazo de control no debe tocar la politica: si lo hace, el contraste de una "
        "variable deja de serlo"
    )


def _trainer_tree() -> ast.Module:
    return ast.parse(TRAINER.read_text(encoding="utf-8"))


def _function(name: str) -> ast.FunctionDef:
    for node in ast.walk(_trainer_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} no existe en {TRAINER.name}")


def test_train_one_builds_the_environment_from_the_recipe() -> None:
    """`kappa_turn` es parametro del ENTORNO: si no viaja, la receta se pierde a mitad."""
    src = ast.unparse(_function("train_one"))
    assert "recipe_for(probe)" in src, "train_one debe resolver la receta de la sonda recibida"
    assert "kappa_turn=recipe.kappa_turn" in src, (
        "el entorno tiene que construirse con el kappa_turn de la receta, no con el default"
    )
    assert "apply_flat_bias(model, recipe)" in src, (
        "sin esta llamada la sonda seleccionada no llega a la politica: es justo el defecto "
        "que este test existe para impedir"
    )


def test_train_one_records_which_recipe_ran() -> None:
    src = ast.unparse(_function("train_one"))
    assert "'recipe_probe'" in src or '"recipe_probe"' in src, (
        "dos JSON de recetas distintas serian indistinguibles sin este campo"
    )


def test_main_feeds_the_gate_probe_into_training() -> None:
    src = ast.unparse(_function("main"))
    assert "sanity['selected_probe']" in src or 'sanity["selected_probe"]' in src, (
        "main debe TOMAR la receta del informe, no solo imprimirla"
    )
    assert "probe=probe" in src, "la receta seleccionada tiene que llegar a train_one"
    assert "known_probes()" in src, (
        "main debe abortar si la compuerta selecciona una receta que el entrenador no sabe "
        "aplicar, en vez de entrenar otra cosa en silencio"
    )
