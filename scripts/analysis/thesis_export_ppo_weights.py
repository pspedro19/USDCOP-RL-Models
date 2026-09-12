#!/usr/bin/env python
"""Exporta la exposicion PPO **barra a barra**, que es lo que el hibrido necesita.

Los artefactos de entrenamiento guardan la serie DIARIA (retorno, coste, numero de cambios) pero
no la senda de 59 pesos. El hibrido pre-registrado combina PPO y LLM **por barra**, asi que sin
esta senda no se puede construir: el ledger del LLM si es por barra.

No entrena nada ni mira metricas: carga politicas ya congeladas y las rueda de forma
determinista sobre el bloque pedido, con el MISMO lazo que `thesis_train_ppo.evaluate` para que
la senda exportada sea exactamente la que produjo los numeros publicados.

La mediana entre semillas se calcula aqui, barra a barra, y se ajusta a la rejilla congelada:
el pre-registro prohibe elegir semilla, y una mediana de exposiciones puede caer fuera de los
cinco niveles permitidos (p. ej. 0,25), que no es una accion que el entorno acepte.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.dataset import load_portable  # noqa: E402
from src.research.session_env import EXPOSURE_LEVELS, OPERABLE_RETURNS  # noqa: E402
from src.research.session_gym import SessionTradingEnv  # noqa: E402

SEEDS = (42, 123, 456, 789, 1337)


def _snap(value: float) -> float:
    """Al nivel congelado mas cercano. Sin esto la mediana puede no ser una accion legal."""
    levels = np.asarray(EXPOSURE_LEVELS, dtype=float)
    return float(levels[int(np.argmin(np.abs(levels - value)))])


def export(model_dir: Path, config: str, specs, seeds=SEEDS) -> dict:
    from stable_baselines3 import PPO

    per_seed: dict[int, dict[str, list[float]]] = {}
    for seed in seeds:
        path = model_dir / f"{config}_seed{seed}.zip"
        if not path.is_file():
            raise FileNotFoundError(f"falta la politica congelada {path}")
        model = PPO.load(str(path), device="cpu")
        env = SessionTradingEnv(specs, seed=0, shuffle=False)
        weights: dict[str, list[float]] = {}
        for _ in range(len(specs)):
            obs, _ = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, _ = env.step(int(action))
            # `_weights` es la senda que el propio entorno acaba de ejecutar; se copia aqui en
            # vez de recomputarla para que no pueda divergir de la que produjo el resultado.
            weights[str(env.last_result.date)] = [float(w) for w in env._weights]
        per_seed[seed] = weights

    dates = list(per_seed[seeds[0]])
    median: dict[str, list[float]] = {}
    for date in dates:
        stack = np.asarray([per_seed[s][date] for s in seeds], dtype=float)
        median[date] = [_snap(v) for v in np.median(stack, axis=0)]
    return {
        "config": config,
        "seeds": list(seeds),
        "n_sessions": len(dates),
        "bars_per_session": OPERABLE_RETURNS,
        "exposure_levels": list(EXPOSURE_LEVELS),
        "aggregation": "median_across_seeds_snapped_to_frozen_grid",
        "per_seed": {str(s): per_seed[s] for s in seeds},
        "median": median,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--config", default="ppo_regime")
    ap.add_argument("--block", default="selection",
                    choices=("development", "selection", "holdout"))
    ap.add_argument("--portable", type=Path,
                    default=ROOT / "data" / "thesis" / "research_data_portable_v2.pkl")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    data = load_portable(args.portable)
    specs = getattr(data, args.block)
    payload = export(args.model_dir, args.config, specs)
    payload["block"] = args.block
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sessions": payload["n_sessions"],
                      "config": args.config, "block": args.block}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
