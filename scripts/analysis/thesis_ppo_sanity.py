"""Run pre-registered PPO sanity fixtures S1–S4.

The command is deliberately separate from market experiments.  Its output is
diagnostic evidence about the optimizer/contability recipe and is never added
to the market trial ledger.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.research.session_gym import SessionTradingEnv
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
SEEDS = (42, 123, 456, 789, 2024)


def _train_one(fixture: Fixture, seed: int, timesteps: int, probe: str):
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
    elif probe == "norm_reward_off":
        norm.norm_reward = False
    elif probe != "baseline":
        raise ValueError(f"unknown sanity probe: {probe}")
    model = PPO("MlpPolicy", norm, seed=seed, verbose=0, **kwargs)
    model.learn(total_timesteps=timesteps)

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
        timesteps: int = 100_000, probe: str = "baseline") -> dict:
    fixture = Fixture(fixture)
    rows = [_train_one(fixture, seed, timesteps, probe) for seed in seeds]
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", choices=[f.value for f in Fixture], required=True)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--probe", choices=("baseline", "ent_coef_zero", "norm_reward_off",
                                              "gamma_one", "kappa_turn_one"), default="baseline")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.fixture, timesteps=args.timesteps, probe=args.probe)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
