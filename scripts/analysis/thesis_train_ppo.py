#!/usr/bin/env python
"""Entrena el brazo PPO de la tesis: 2 configuraciones x 5 semillas.

Contract: CTR-RESEARCH-PPO-001 · Date: 2026-08-25

## La ablación

`ppo_regime` y `ppo_backbone` comparten **todo**: hiperparámetros, semillas, datos, número de
pasos, arquitectura. La única diferencia son los 4 posteriores de régimen en la observación —
`ppo_backbone` los recibe puestos a cero.

Esa es la razón de que H2 sea decidible: cualquier diferencia entre los dos solo puede venir
de la información de régimen. Si además cambiaran los hiperparámetros, la comparación mediría
dos cosas a la vez y no se podría atribuir nada — la regla 1 de `experiment-protocol.md`.

## Sin HPO, y qué cuesta

Los hiperparámetros salen congelados de `config/experiments/v215b_baseline.yaml`. Son un prior
declarado ex-ante, no una elección hecha mirando estos datos, así que **no suman trials** al
contador del activo (constitución §2). El precio es que quedan dos candidatos en el universo,
y con un universo de 2 el White RC y el SPA pierden sentido: contrastan un ganador contra los
rivales que compitieron. Se declara que no se computan, en vez de publicarlos vacíos.

## Lo que NO se reutiliza

`src/training/engine.py:885-931` cablea `TradingEnvConfig` y `EnvironmentFactory` — el entorno
de PRODUCCIÓN, con costo plano de 2,5 bps y episodios de 2.400 barras. Contradice §9.1 y §9.3
punto por punto. Reutilizarlo habría sido más rápido y habría medido otra cosa.

## Selección del modelo

**El reward de evaluación NO elige el modelo** (`experiment-protocol.md` regla 4: seed 456 con
eval=131 perdió −20,6%; seed 1337 con eval=111 ganó +9,6%). Aquí se entrena un número fijo de
pasos y se guarda el modelo final. La selección entre configuraciones ocurre en el bloque de
SELECCIÓN, nunca en el hold-out.

Uso:
    python scripts/analysis/thesis_train_ppo.py --config ppo_regime --seed 42
    python scripts/analysis/thesis_train_ppo.py --all            # 10 corridas
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# La consola de Windows usa cp1252 y estos scripts imprimen `Δ`, `·`, `→`. Sin esto un
# UnicodeEncodeError aborta la corrida DESPUES de haber calculado todo, que es la peor
# forma de fallar: el trabajo esta hecho y no se escribe el JSON.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


from src.research.dataset import PORTABLE, load_or_build, load_portable  # noqa: E402
from src.research.features import GROUPS  # noqa: E402
from src.research.session_env import daily_series  # noqa: E402
from src.research.session_gym import SessionSpec, SessionTradingEnv  # noqa: E402
from scripts.diagnostics.audit_research_data_contract import require_contract  # noqa: E402
from src.research.sanity_gate import require_macro_identity, require_sanity_pass  # noqa: E402

SEEDS = (42, 123, 456, 789, 1337)          # `experiment-protocol.md` regla 2
CONFIGS = ("ppo_regime", "ppo_backbone")
OUT = Path(os.environ.get("THESIS_PPO_OUT", REPO / "outputs" / "thesis" / "ppo"))
N_REGIMES = len(GROUPS["regimen"])

# Congelados de config/experiments/v215b_baseline.yaml, seccion `training.ppo`.
PPO_KWARGS = dict(
    learning_rate=3e-4, n_steps=4096, batch_size=128, n_epochs=10,
    gamma=0.98, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    vf_coef=0.5, max_grad_norm=0.5, normalize_advantage=True,
)
NET_ARCH = dict(pi=[256, 256], vf=[256, 256])
TOTAL_TIMESTEPS = 300_000     # ver `_timesteps_note`


def _timesteps_note(n_dev: int, timesteps: int = TOTAL_TIMESTEPS) -> str:
    """Por que 300k y no los 2M del baseline de produccion.

    El baseline entrena sobre 70.000 barras continuas. Aqui desarrollo son 499 sesiones x 59
    decisiones = 29.441 pasos, asi que 2M serian **68 pasadas** sobre los mismos datos. El
    numero congelado no es transferible: lo que se congela es la RECETA (lr, arquitectura,
    clip...), no un conteo de pasos atado a otro tamano de muestra.

    300.000 son ~10 pasadas sobre desarrollo, que es el orden habitual para PPO sobre datos
    tabulares y mantiene el sobreajuste acotado. Es una decision declarada ex-ante y COMUN a
    las dos configuraciones, asi que no puede favorecer a ninguna: no altera la ablacion.
    """
    return f"{timesteps:,} pasos ~= {timesteps / (n_dev * 59):.1f} pasadas"


def strip_regimes(specs: list[SessionSpec]) -> list[SessionSpec]:
    """`ppo_backbone`: mismo vector, posteriores de regimen a cero.

    Se anulan en vez de eliminarse para que las dos configuraciones compartan dimension de
    observacion y arquitectura exacta. Si una red tuviera 4 entradas menos, tendria tambien
    menos parametros y la comparacion mediria capacidad ademas de informacion.
    """
    out = []
    for s in specs:
        ctx = s.context.copy()
        ctx[-N_REGIMES:] = 0.0
        out.append(SessionSpec(date=s.date, close=s.close, market=s.market,
                               context=ctx, spread_pips=s.spread_pips))
    return out


def evaluate(model, specs: list[SessionSpec]) -> dict:
    """Politica determinista sobre un bloque. Devuelve la serie diaria y sus metricas."""
    env = SessionTradingEnv(specs, seed=0, shuffle=False)
    results = []
    for _ in range(len(specs)):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(int(action))
        results.append(env.last_result)

    daily = daily_series(results)
    equity = float(np.prod(1.0 + daily))
    ann = 221.0                                   # sesiones/ano, derivado en Fase E
    mu, sd = float(np.mean(daily)), float(np.std(daily, ddof=1))
    sharpe = float(mu / sd * np.sqrt(ann)) if sd > 0 else 0.0
    return {
        "n_sessions": len(results),
        "total_return": equity - 1.0,
        "sharpe": sharpe,
        "mean_daily": mu,
        "std_daily": sd,
        "n_ops": int(sum(r.n_changes for r in results)),
        "mean_abs_exposure": float(np.mean([r.mean_abs_exposure for r in results])),
        "total_cost": float(sum(r.total_cost for r in results)),
        "daily_returns": [float(x) for x in daily],
        "daily_gross_returns": [float(r.gross_return) for r in results],
        "daily_costs": [float(r.total_cost) for r in results],
        "dates": [str(r.date) for r in results],
    }


def train_one(config: str, seed: int, data, timesteps: int = TOTAL_TIMESTEPS,
              verbose: bool = True, refit: bool = False,
              output_dir: Path | None = None) -> dict:
    """Entrena una configuracion.

    Con `refit=True` entrena sobre **desarrollo + seleccion**, que es lo que el pre-registro
    (§1, fila «Refit») compromete hacer UNA vez antes de abrir el hold-out. El modelo
    resultante nunca se evalua aqui sobre el hold-out: eso lo hace un paso aparte con el
    gate de la Regla B.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    dev = data.development if config == "ppo_regime" else strip_regimes(data.development)
    sel = data.selection if config == "ppo_regime" else strip_regimes(data.selection)
    train_specs = (dev + sel) if refit else dev

    def make():
        return SessionTradingEnv(train_specs, seed=seed, shuffle=True)

    # §6.6: norm_obs=False (las features ya vienen escaladas con el scaler de DESARROLLO;
    # renormalizarlas online reintroduciria estadisticos del bloque evaluado), norm_reward=True.
    venv = VecNormalize(DummyVecEnv([make]), norm_obs=False, norm_reward=True,
                        clip_reward=10.0, gamma=PPO_KWARGS["gamma"])

    model = PPO("MlpPolicy", venv, seed=seed, device="cpu", verbose=0,
                policy_kwargs=dict(net_arch=NET_ARCH), **PPO_KWARGS)

    t0 = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    elapsed = time.time() - t0

    # v2 must never overwrite the published v1 artifacts.  Callers can provide a
    # versioned directory; the environment variable remains backwards compatible
    # for the original v1 runner.
    out = output_dir or OUT
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{config}_refit_seed{seed}" if refit else f"{config}_seed{seed}"
    model.save(out / f"{tag}.zip")
    venv.save(str(out / f"{tag}_vecnorm.pkl"))

    # Evaluacion con VecNormalize CONGELADO: el modelo predice sobre las mismas
    # observaciones que vio entrenando, y el reward normalizado no interviene aqui.
    res = {
        "config": config, "seed": seed, "timesteps": timesteps, "refit": refit,
        "train_block": "development+selection" if refit else "development",
        "train_seconds": round(elapsed, 1),
        "development": evaluate(model, dev),
        "selection": evaluate(model, sel),
    }
    if refit:
        # En refit, «seleccion» ya es IN-SAMPLE. Se marca para que ninguna tabla la
        # presente como fuera de muestra: seria el error que §11 llama subperiodo in-sample.
        res["selection"]["in_sample"] = True
        res["development"]["in_sample"] = True
    (out / f"{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    if verbose:
        s = res["selection"]
        print(f"  {tag:<24} sel: ret {s['total_return']:+7.2%}  Sharpe {s['sharpe']:+6.2f}  "
              f"ops {s['n_ops']:>5}  |exp| {s['mean_abs_exposure']:.2f}  ({elapsed:.0f}s)")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=CONFIGS)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    ap.add_argument("--output-dir", type=Path,
                    help="directorio versionado; evita sobrescribir artefactos v1")
    ap.add_argument("--portable-path", type=Path,
                    help="dataset portable versionado; rechaza identidades obsoletas")
    ap.add_argument("--require-sanity", type=Path,
                    help="informe S1-S4 aprobado; obligatorio antes de un entrenamiento v2")
    ap.add_argument("--require-macro-identity", type=Path,
                    help="informe de reconciliación positiva de fuentes macro")
    ap.add_argument("--refit", action="store_true",
                    help="entrena sobre desarrollo+seleccion (paso F8 del pre-registro)")
    args = ap.parse_args()

    evidence = require_contract()
    print(f"data contract: structural={evidence['verdict']['structural_m5_clean']} "
          f"macro_complete={evidence['verdict']['macro_columns_complete']}")
    if args.require_sanity:
        sanity = require_sanity_pass(args.require_sanity)
        print(f"sanity gate: receta={sanity['selected_probe']}")
    if args.require_macro_identity:
        identity = require_macro_identity(args.require_macro_identity)
        print(f"macro identity gate: {len(identity.get('series', {}))} series verificadas")

    # El formato portable no lleva el objeto hmmlearn, asi que funciona dentro del
    # contenedor de Airflow, que no tiene esa dependencia instalada.
    portable_path = args.portable_path or PORTABLE
    if portable_path.is_file():
        data = load_portable(portable_path)
        print(f"dataset portable: {data.summary()}")
    else:
        data = load_or_build()
    print(_timesteps_note(len(data.development), args.timesteps))

    jobs = ([(c, s) for c in CONFIGS for s in SEEDS] if args.all
            else [(args.config, args.seed)])
    if any(c is None or s is None for c, s in jobs):
        ap.error("usa --all, o --config y --seed juntos")

    for config, seed in jobs:
        train_one(config, seed, data, timesteps=args.timesteps, refit=args.refit,
                  output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
