"""La receta de PPO, en UN solo sitio.

## Por que existe este modulo

La compuerta de sanidad (`thesis_ppo_sanity.py`) prueba recetas sobre fixtures sinteticas y
declara cual pasa S1-S4. El entrenador de mercado (`thesis_train_ppo.py`) leia ese informe,
imprimia `sanity gate: receta=flat_init_no_turn` y entrenaba **otra receta**: el sesgo inicial
de la capa de accion -- que ES la sonda -- vivia solo en el script de sanidad. La compuerta
validaba un informe, no la receta que iba a correr.

El efecto medido no es cosmetico. Sin el sesgo, 300k pasos sobre el dataset v2, semilla 42:
909 operaciones en 226 sesiones y un coste de 0,4855 sobre un bruto de ~0,272 -- el mismo
cuadro de la tesis v1 rechazada. Con el sesgo, la misma receta se abstiene sobre ruido puro
(S1, 5/5 semillas).

Asi que la receta deja de ser codigo duplicado en dos scripts y pasa a ser un dato: una tabla
de sondas que ambos consumen. Una sonda desconocida **aborta**; nunca se degrada en silencio a
"entrena lo de siempre", que es exactamente como se colo el defecto.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[2]
FROZEN_SANITY_PROBE = "flat_init_no_turn"
PPO_KWARGS = {
    "learning_rate": 3e-4, "n_steps": 4096, "batch_size": 128, "n_epochs": 10,
    "gamma": 0.98, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01,
    "vf_coef": 0.5, "max_grad_norm": 0.5, "normalize_advantage": True,
}
NET_ARCH = {"pi": [256, 256], "vf": [256, 256]}

# Etiqueta del brazo que entrena SIN la receta de la compuerta. No es una sonda: es el control
# de una variable frente a la receta seleccionada, y se nombra para que ninguna tabla lo
# presente como "v2 validado".
CONTROL_ARM = "control_sin_sesgo_flat"


@dataclass(frozen=True)
class Recipe:
    """Lo que distingue a una sonda de otra. `None` = se conserva el valor congelado."""

    probe: str
    kappa_turn: float = 0.0
    ent_coef: float | None = None
    gamma: float | None = None
    norm_reward: bool = True
    flat_bias_logit: float | None = None
    """Logit que se planta en el nivel de exposicion 0.0 al inicializar `action_net`.

    3.0 deja **83,4 %** de la masa inicial en "no operar" con observacion nula
    (softmax([0,0,3,0,0]) = e^3/(e^3+4) = 0,8339). La sonda original documentaba "~95 %", que
    no es lo que hace el numero; lo midio `test_flat_bias_puts_the_mass_on_not_trading`. La
    politica arranca practicamente plana
    y tiene que APRENDER a salir, en vez de comprometerse con una direccion en la barra 0 y
    quedar encerrada ahi. No congela nada -- es solo el punto de partida.
    """


_RECIPES: dict[str, Recipe] = {
    "baseline": Recipe("baseline"),
    "ent_coef_zero": Recipe("ent_coef_zero", ent_coef=0.0),
    "norm_reward_off": Recipe("norm_reward_off", norm_reward=False),
    "gamma_one": Recipe("gamma_one", gamma=1.0),
    "kappa_turn_one": Recipe("kappa_turn_one", kappa_turn=1.0),
    "kappa_turn_one_ent_zero": Recipe("kappa_turn_one_ent_zero", kappa_turn=1.0, ent_coef=0.0),
    "kappa_turn_one_ent_high": Recipe("kappa_turn_one_ent_high", kappa_turn=1.0, ent_coef=0.05),
    "flat_init": Recipe("flat_init", kappa_turn=1.0, flat_bias_logit=3.0),
    "flat_init_no_turn": Recipe("flat_init_no_turn", kappa_turn=0.0, flat_bias_logit=3.0),
    CONTROL_ARM: Recipe(CONTROL_ARM),
}


def recipe_for(probe: str) -> Recipe:
    """Devuelve la receta de una sonda. Sonda desconocida = error, nunca un default."""
    try:
        return _RECIPES[probe]
    except KeyError:
        raise ValueError(
            f"receta desconocida: {probe!r}. Sondas declaradas: {sorted(_RECIPES)}. "
            "Un entrenador que no sabe aplicar la receta de la compuerta no debe entrenar: "
            "produciria numeros atribuidos a una receta que no corrio."
        ) from None


def known_probes() -> tuple[str, ...]:
    return tuple(sorted(_RECIPES))


def apply_flat_bias(model, recipe: Recipe) -> bool:
    """Sesga `action_net` hacia exposicion 0.0. Devuelve si toco algo.

    Falla ruidosamente si el espacio de acciones no contiene 0.0: sin el nivel plano la sonda
    no significa nada y seguir seria inventarse un punto de partida.
    """
    if recipe.flat_bias_logit is None:
        return False
    import torch

    from src.research.session_env import EXPOSURE_LEVELS

    if 0.0 not in EXPOSURE_LEVELS:
        raise ValueError(
            f"la receta {recipe.probe!r} sesga hacia exposicion 0.0 y el espacio congelado "
            f"{EXPOSURE_LEVELS} no la contiene"
        )
    flat_index = EXPOSURE_LEVELS.index(0.0)
    with torch.no_grad():
        bias = model.policy.action_net.bias
        bias.zero_()
        bias[flat_index] = float(recipe.flat_bias_logit)
    return True


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def effective_recipe(probe: str) -> dict:
    """Full effective configuration, shared by synthetic and market training."""
    recipe = recipe_for(probe)
    kwargs = deepcopy(PPO_KWARGS)
    if recipe.ent_coef is not None:
        kwargs["ent_coef"] = recipe.ent_coef
    if recipe.gamma is not None:
        kwargs["gamma"] = recipe.gamma
    return {
        "recipe": asdict(recipe), "algorithm": "PPO", "policy": "MlpPolicy",
        "device": "cpu", "ppo_kwargs": kwargs,
        "policy_kwargs": {"net_arch": deepcopy(NET_ARCH), "activation_fn": "torch.nn.Tanh"},
        "environment": {"reward_scale": 100.0, "shuffle": True,
                        "kappa_turn": recipe.kappa_turn, "lambda_dd": 0.0},
        "vecnormalize": {"norm_obs": False, "norm_reward": recipe.norm_reward,
                         "clip_reward": 10.0, "gamma": kwargs["gamma"]},
        "evaluation": {"deterministic": True, "normalizer_training": False},
    }


def build_ppo(sessions, *, seed: int, probe: str):
    """Construct (model, normalizer, recipe) once; never trains or reads data."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from src.research.session_gym import SessionTradingEnv

    recipe = recipe_for(probe)
    config = effective_recipe(probe)
    vec = DummyVecEnv([lambda: SessionTradingEnv(sessions, seed=seed,
                                                 **config["environment"])])
    normalizer = VecNormalize(vec, **config["vecnormalize"])
    model = PPO(config["policy"], normalizer, seed=seed, device=config["device"], verbose=0,
                policy_kwargs={"net_arch": deepcopy(NET_ARCH), "activation_fn": torch.nn.Tanh},
                **config["ppo_kwargs"])
    apply_flat_bias(model, recipe)
    return model, normalizer, recipe


def training_code_hashes() -> dict[str, str]:
    """Identity includes both callers, numerical environment and fee contract."""
    paths = (
        "src/research/ppo_recipe.py", "src/research/synthetic_sessions.py",
        "src/research/session_env.py", "src/research/session_gym.py",
        "src/research/cost_model.py", "src/research/cost_contract.py",
        "src/research/features.py", "config/research/cost_contract.yaml",
        "scripts/analysis/thesis_ppo_sanity.py", "scripts/analysis/thesis_train_ppo.py",
    )
    return {name: file_sha256(ROOT / name) for name in paths}


def library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "torch", "stable-baselines3", "gymnasium"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def sessions_sha256(sessions) -> str:
    """Hash actual arrays, fees and ordered dates, not merely a claimed dataset ID."""
    import numpy as np
    from src.research.cost_model import CostParameters

    digest = hashlib.sha256()
    for spec in sessions:
        parameters = getattr(spec, "cost_parameters", None) or CostParameters()
        header = {"date": str(spec.date), "spread": float(spec.spread_pips),
                  "cost_parameters": asdict(parameters)}
        digest.update(canonical_sha256(header).encode("ascii"))
        for array in (spec.close, spec.market, spec.context):
            values = np.asarray(array, dtype="<f8", order="C")
            digest.update(canonical_sha256(list(values.shape)).encode("ascii"))
            digest.update(values.tobytes())
    return digest.hexdigest()


def write_immutable_json(path: Path, value: dict) -> None:
    """Exclusive creation: a second invocation cannot silently overwrite evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
