"""Run pre-registered PPO sanity fixtures S1–S4.

The command is deliberately separate from market experiments.  Its output is
diagnostic evidence about the optimizer/contability recipe and is never added
to the market trial ledger.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# El script se invoca como fichero (`python scripts/analysis/...`), no como modulo,
# asi que la raiz del repo no esta en sys.path y `src` no resuelve. Los demas
# scripts de `scripts/analysis/` hacen exactamente esto; sin ello el runner
# committeado nunca pudo ejecutarse (ModuleNotFoundError en el primer import).
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.session_gym import SessionTradingEnv  # noqa: E402
from src.research.synthetic_sessions import Fixture, make_sessions, oracle_result


PPO_KWARGS = {
    "learning_rate": 3e-4,
    "n_steps": 4096,
    "batch_size": 128,
    "gamma": 0.98,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "policy_kwargs": {"net_arch": [256, 256]},
}
# `experiment-protocol.md` regla 2: estas cinco, sin excepciones. La lista tenia 2024 en
# lugar de 1337, lo que contradecia ademas la identidad congelada del pre-registro v3 y
# hacia que la evidencia de sanidad no fuera comparable por semilla con las corridas de
# mercado de `thesis_train_ppo.py`, que si usa las cinco correctas.
SEEDS = (42, 123, 456, 789, 1337)
PROBES = ("baseline", "ent_coef_zero", "norm_reward_off", "gamma_one", "kappa_turn_one",
          "kappa_turn_one_ent_zero", "kappa_turn_one_ent_high")
# `kappa_turn_one_ent_zero` se declara el 2026-09-11, despues de agotar las cuatro sondas
# originales y de MEDIR por que fallaron. No es una quinta prueba a ciegas: con kappa_turn=1 un
# cambio de posicion cuesta 1,0 en unidades de reward mientras el neto economico por barra es
# ~0,0005 -- una penalizacion dos mil veces mayor. Que aun asi dos de tres semillas operaran
# descarta que falte senal y deja como sospechoso al bono de entropia, que empuja la politica
# hacia la uniforme sobre las cinco acciones. Las dos piezas nunca se habian combinado: la
# sonda de entropia se probo SIN penalizacion de turnover, donde quitarla solo dejaba al agente
# apostar una direccion gratis.


def _train_one(fixture: Fixture, seed: int, timesteps: int, probe: str,
               checkpoint_dir: Path | None = None,
               resume: Path | None = None):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    sessions = make_sessions(fixture, n=500, seed=seed)
    vec = DummyVecEnv([lambda: SessionTradingEnv(sessions, seed=seed, shuffle=True)])
    norm = VecNormalize(vec, norm_obs=False, norm_reward=True, clip_reward=10.0,
                        gamma=PPO_KWARGS["gamma"])
    kwargs = dict(PPO_KWARGS)
    if probe == "ent_coef_zero":
        kwargs["ent_coef"] = 0.0
    elif probe == "gamma_one":
        kwargs["gamma"] = 1.0
        norm.gamma = 1.0
    elif probe == "kappa_turn_one":
        # A positive turn penalty is an environment parameter, not a PPO knob.
        vec = DummyVecEnv([lambda: SessionTradingEnv(
            sessions, seed=seed, shuffle=True, kappa_turn=1.0)])
        norm = VecNormalize(vec, norm_obs=False, norm_reward=True, clip_reward=10.0,
                            gamma=kwargs["gamma"])
    elif probe == "kappa_turn_one_ent_zero":
        kwargs["ent_coef"] = 0.0
        vec = DummyVecEnv([lambda: SessionTradingEnv(
            sessions, seed=seed, shuffle=True, kappa_turn=1.0)])
        norm = VecNormalize(vec, norm_obs=False, norm_reward=True, clip_reward=10.0,
                            gamma=kwargs["gamma"])
    elif probe == "kappa_turn_one_ent_high":
        # Una variable respecto a `kappa_turn_one`: mas exploracion, no menos. La direccion la
        # dicta la medicion, no una corazonada: quitar la entropia empeoro (0,052 -> 0,732) y
        # el conteo mostro por que -- sin exploracion la politica se compromete en la barra 0 y
        # la penalizacion de turnover la encierra, porque castiga el cambio y no la exposicion.
        # Si el fallo es comprometerse antes de aprender, retrasar el compromiso es el remedio.
        kwargs["ent_coef"] = 0.05
        vec = DummyVecEnv([lambda: SessionTradingEnv(
            sessions, seed=seed, shuffle=True, kappa_turn=1.0)])
        norm = VecNormalize(vec, norm_obs=False, norm_reward=True, clip_reward=10.0,
                            gamma=kwargs["gamma"])
    elif probe == "norm_reward_off":
        norm.norm_reward = False
    elif probe != "baseline":
        raise ValueError(f"unknown sanity probe: {probe}")
    # `PPO.load` restaura la politica, NO el wrapper de normalizacion: al reanudar, las
    # medias y varianzas corrientes del reward volverian a cero y el reward efectivo daria un
    # salto a mitad del entrenamiento. En un fixture cuyo proposito es diagnosticar si la
    # receta -- normalizacion incluida -- hace que el agente opere sobre ruido, eso contamina
    # justo lo que se mide. Se guarda y se restaura junto al checkpoint.
    norm_state = None if resume is None else Path(str(resume)).with_suffix(".vecnorm.pkl")
    if norm_state is not None and norm_state.is_file():
        from stable_baselines3.common.vec_env import VecNormalize as _VN
        restored = _VN.load(str(norm_state), vec)
        restored.training, restored.norm_reward = True, norm.norm_reward
        norm = restored
    model = (PPO.load(str(resume), env=norm, device="cpu") if resume is not None
             else PPO("MlpPolicy", norm, seed=seed, verbose=0, **kwargs))
    callback = None
    if checkpoint_dir is not None:
        from stable_baselines3.common.callbacks import CheckpointCallback
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callback = CheckpointCallback(save_freq=max(1, 10_000),
                                      save_path=str(checkpoint_dir),
                                      name_prefix=f"{fixture.value}_{probe}_seed{seed}",
                                      save_vecnormalize=True)
    model.learn(total_timesteps=timesteps, callback=callback,
                reset_num_timesteps=resume is None)
    if checkpoint_dir is not None:
        # Ultimo estado, con el mismo nombre que espera `--resume` en la llamada siguiente.
        final = checkpoint_dir / f"{fixture.value}_{probe}_seed{seed}_final.zip"
        model.save(str(final))
        norm.save(str(final.with_suffix(".vecnorm.pkl")))

    norm.training = False
    norm.norm_reward = False
    values, exposures = [], []
    for spec in sessions[:100]:
        eval_env = SessionTradingEnv([spec], seed=seed, shuffle=False)
        obs, _ = eval_env.reset()
        done = False
        weights = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _reward, done, _truncated, _info = eval_env.step(int(action))
            weights.append(eval_env._weights[-1])
        values.append(float(eval_env.last_result.daily_return))
        exposures.append(float(np.mean(np.abs(weights))))
    return {
        "seed": seed,
        "mean_net": float(np.mean(values)),
        "median_net": float(np.median(values)),
        "mean_abs_exposure": float(np.mean(exposures)),
        "n_eval_sessions": len(values),
    }


def run(fixture: Fixture | str, seeds: tuple[int, ...] = SEEDS,
        timesteps: int = 100_000, probe: str = "baseline",
        checkpoint_dir: Path | None = None,
        resume: Path | None = None) -> dict:
    fixture = Fixture(fixture)
    if resume is not None and len(seeds) != 1:
        raise ValueError("--resume requiere exactamente una semilla")
    rows = [_train_one(fixture, seed, timesteps, probe, checkpoint_dir=checkpoint_dir,
                       resume=resume) for seed in seeds]
    oracle = [oracle_result(fixture, s).daily_return
              for s in make_sessions(fixture, n=100, seed=991)]
    if fixture in (Fixture.NOISE_WITH_COST, Fixture.SIGNAL_BELOW_COST):
        passed = sum(row["mean_abs_exposure"] < 0.1 and row["mean_net"] > -0.005
                     for row in rows) >= 4
    else:
        passed = sum(row["mean_net"] > 0.0 for row in rows) >= 4
    return {
        "fixture": fixture.value,
        "synthetic_only": True,
        "market_trials_charged": 0,
        "timesteps": timesteps,
        "probe": probe,
        "rows": rows,
        "oracle_mean_net": float(np.mean(oracle)),
        "pass_rule": "4/5 seeds satisfy the pre-registered fixture criterion",
        "passed": bool(passed),
    }


def run_protocol(seeds: tuple[int, ...] = SEEDS, timesteps: int = 100_000) -> dict:
    """Ejecuta S1–S4 y sondas en el orden pre-registrado.

    Una receta solo se congela si pasa las cuatro fixtures. En cuanto una receta pasa, no se
    ejecutan sondas posteriores. El resultado es sintético y cobra cero trials de mercado.
    """
    attempts = []
    for probe in PROBES:
        fixture_reports = [run(f, seeds=seeds, timesteps=timesteps, probe=probe)
                           for f in Fixture]
        passed = all(r["passed"] for r in fixture_reports)
        attempts.append({"probe": probe, "fixtures": fixture_reports, "passed_all": passed})
        if passed:
            return {"synthetic_only": True, "market_trials_charged": 0,
                    "timesteps": timesteps, "selected_probe": probe,
                    "attempts": attempts,
                    "pass_rule": "primera receta que pasa S1-S4"}
    return {"synthetic_only": True, "market_trials_charged": 0,
            "timesteps": timesteps, "selected_probe": None,
            "attempts": attempts,
            "pass_rule": "primera receta que pasa S1-S4"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", choices=[f.value for f in Fixture], required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int,
                        help="ejecuta una sola semilla para jobs reanudables")
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="guarda checkpoints cada 10k pasos")
    parser.add_argument("--resume", type=Path,
                        help="reanuda un checkpoint de una sola semilla")
    parser.add_argument("--probe", choices=("baseline", "ent_coef_zero", "norm_reward_off",
                                              "gamma_one", "kappa_turn_one", "kappa_turn_one_ent_zero", "kappa_turn_one_ent_high"), default="baseline")
    parser.add_argument("--protocol", action="store_true",
                        help="ejecuta S1-S4 y detiene las sondas en la primera receta válida")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seeds = (args.seed,) if args.seed is not None else SEEDS
    report = (run_protocol(seeds=seeds, timesteps=args.timesteps) if args.protocol
              else run(args.fixture, seeds=seeds, timesteps=args.timesteps,
                       probe=args.probe, checkpoint_dir=args.checkpoint_dir,
                       resume=args.resume))
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report.get("passed", report.get("selected_probe") is not None) else 2


if __name__ == "__main__":
    raise SystemExit(main())
