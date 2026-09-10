#!/usr/bin/env python
"""Descompone el resultado de la tesis en bruto, costo y senda de exposición.

Contract: CTR-RESEARCH-DECOMP-001 · Date: 2026-08-25

## El número que faltaba

La tesis se cerró con un rechazo: PPO no bate a `always_flat`, ΔSharpe −6,215 en hold-out,
p < 0,0001, 0/10 semillas positivas. **Eso no cambia.**

Lo que no estaba calculado es la descomposición. Como `retorno_neto = bruto − costo` es exacta
por sesión, el bruto se recupera de los JSON ya guardados:

    ppo_regime   bruto +27,95%   costos 107%   neto −54,9%
    ppo_backbone bruto +31,51%   costos 114%   neto −56,4%

**Las 10 corridas dan bruto positivo.** El agente aprende algo predictivo, y el costo de
ejecución de la frecuencia que elige se lo come cuatro veces. El rechazo sigue en pie, pero
ahora tiene mecanismo: **la restricción que ata es la ejecución, no la predicción**.

## Qué produce este script

Una sola pasada de replay sobre los 20 modelos guardados, y un artefacto que alimenta los
cuatro análisis sin volver a cargar un modelo:

- `gross_return`, `total_cost`, `terminal_cost` **por sesión** (los JSON solo tienen el costo
  sumado),
- `sum_abs_dw` y `sum_abs_dw_sigma`, que hacen el break-even resoluble en forma cerrada,
- la **senda de exposición de 59 barras**, que permite el re-scoring por frecuencia sin
  reentrenar nada.

## Lo que NO es

**No reabre la búsqueda.** El universo de `H-TESIS-RL-01` está cerrado y su hold-out abierto
una vez. Esto es descomposición descriptiva de un resultado ya obtenido —no selección de una
variante nueva— así que **cobra 0 trials**, con el mismo criterio con el que no se cobró la
selección de K del HMM.

**El bruto no es un claim de edge.** Es una cota superior contrafactual que exige costo cero,
que no existe. Se etiqueta así en todas las tablas.

## La verificación que hace fiable todo lo demás

Para cada corrida se comprueba que `Σ(gross) − Σ(cost)` reproduce el `Σ(daily_returns)` del
JSON original. Si el replay no fuese fiel —otra semilla, otro orden, otro modelo— ese cuadre
fallaría y nada de lo que sale de aquí valdría. Se aborta antes que publicar.

Uso:
    python scripts/analysis/thesis_decompose.py --block holdout
    python scripts/analysis/thesis_decompose.py --block selection --config ppo_regime
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.analysis.thesis_train_ppo import CONFIGS, SEEDS, strip_regimes  # noqa: E402
from src.research.cost_model import realized_vol_pips  # noqa: E402
from src.research.dataset import PORTABLE, load_or_build, load_portable  # noqa: E402
from src.research.session_env import OPERABLE_RETURNS, run_session  # noqa: E402
from src.research.session_gym import SessionTradingEnv  # noqa: E402

PPO_DIR = Path(os.environ.get("THESIS_PPO_OUT", REPO / "outputs" / "thesis" / "ppo"))
OUT = REPO / "outputs" / "thesis"
CROSSCHECK_TOL = 1e-9


def replay_one(model, specs) -> list[dict]:
    """Reproduce la política y descompone cada sesión.

    Se lee `gross_return` y `total_cost` directamente de `SessionResult` — `run_session` ya
    los devuelve por separado, no hay nada que recalcular ni ninguna fórmula que duplicar.
    Duplicarla sería exactamente el error que `test_engine_parity_independent.py` vigila.
    """
    env = SessionTradingEnv(specs, seed=0, shuffle=False)
    rows = []
    for _ in range(len(specs)):
        obs, _ = env.reset()
        weights, done = [], False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(int(action))
            weights.append(env._w_prev if not done else env.last_result.weights[-1])

        res = env.last_result
        w = np.asarray(res.weights, dtype=float)
        # `|Δw|` con `w_{-1}=0` (§9.1) mas el cierre terminal: es el turnover que paga costo.
        dw = np.abs(np.diff(np.concatenate([[0.0], w, [0.0]])))
        sigma = realized_vol_pips(np.asarray(res.__dict__.get("_close", []), dtype=float)) \
            if "_close" in res.__dict__ else None

        rows.append({
            "date": str(res.date),
            "gross_return": float(res.gross_return),
            "total_cost": float(res.total_cost),
            "terminal_cost": float(res.terminal_cost),
            "daily_return": float(res.daily_return),
            "n_changes": int(res.n_changes),
            "mean_abs_exposure": float(res.mean_abs_exposure),
            "spread_pips": float(res.spread_pips),
            "sum_abs_dw": float(dw.sum()),
            "weights": [float(x) for x in w],
        })
    return rows


def sigma_terms(specs, rows) -> None:
    """Anade `sum_abs_dw_sigma`, el termino de slippage, que el break-even necesita.

    `costo(s) = Σ|Δw|·(s/2 + 0.5) + 0.1·Σ|Δw|·σ12`. El segundo sumando no depende de `s`,
    asi que basta con guardarlo una vez para despejar `s` en forma cerrada despues.
    """
    by_date = {str(s.date): s for s in specs}
    for r in rows:
        spec = by_date[r["date"]]
        sigma = realized_vol_pips(spec.close)
        w = np.asarray(r["weights"], dtype=float)
        prev = np.concatenate([[0.0], w])
        acc = 0.0
        for b in range(len(w)):
            acc += abs(w[b] - prev[b]) * sigma[b]
        acc += abs(0.0 - w[-1]) * sigma[len(w) - 1]      # cierre terminal
        r["sum_abs_dw_sigma"] = float(acc)
        # Precio de referencia para pasar pips a retorno.
        r["mean_close"] = float(np.mean(spec.close))


def crosscheck(rows, original_daily: list[float]) -> tuple[bool, float]:
    """`Σ(bruto) − Σ(costo)` tiene que reproducir el `Σ(daily_returns)` guardado."""
    replayed = sum(r["gross_return"] - r["total_cost"] for r in rows)
    delta = abs(replayed - float(np.sum(original_daily)))
    return delta < CROSSCHECK_TOL, delta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="holdout",
                    choices=["development", "selection", "holdout"])
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--models", default="refit", choices=["refit", "development"])
    args = ap.parse_args()

    from stable_baselines3 import PPO

    data = load_portable() if PORTABLE.is_file() else load_or_build(verbose=False)
    specs = data.block(args.block)
    suffix = "_refit" if args.models == "refit" else ""
    print(f"bloque {args.block}: {len(specs)} sesiones "
          f"({specs[0].date} -> {specs[-1].date})\n")

    out_path = OUT / f"decomposition_{args.block}.json"
    blob = json.loads(out_path.read_text(encoding="utf-8")) if out_path.is_file() else {
        "contract": "CTR-RESEARCH-DECOMP-001", "block": args.block,
        "n_sessions": len(specs), "models": args.models, "runs": {}}

    for cfg in CONFIGS:
        if args.config and cfg != args.config:
            continue
        use = specs if cfg == "ppo_regime" else strip_regimes(specs)
        for seed in SEEDS:
            tag = f"{cfg}{suffix}_seed{seed}"
            zf = PPO_DIR / f"{tag}.zip"

            # De que JSON sale la evaluacion ORIGINAL contra la que se cuadra. No es una
            # sola ruta, y equivocarse la hace comparar dos modelos distintos:
            #
            #   holdout            -> `{cfg}_seed{n}.json["holdout"]`, porque
            #                         `thesis_open_holdout.py` escribio ahi la evaluacion
            #                         de los modelos REFIT.
            #   dev/seleccion      -> `{cfg}_refit_seed{n}.json` si se replayan los refit,
            #                         y `{cfg}_seed{n}.json` si se replayan los de desarrollo.
            #
            # El primer intento uso siempre `{cfg}_seed{n}.json` y el cuadre salto con
            # delta 0,234 — el guard funcionando, no un fallo numerico.
            if args.block == "holdout":
                jf = PPO_DIR / f"{cfg}_seed{seed}.json"
            else:
                jf = PPO_DIR / f"{tag}.json"
            if not zf.is_file() or not jf.is_file():
                print(f"  falta {tag}")
                continue

            original = json.loads(jf.read_text(encoding="utf-8")).get(args.block)
            if not original:
                print(f"  {tag}: sin bloque {args.block} en el JSON original")
                continue

            model = PPO.load(str(zf), device="cpu")
            rows = replay_one(model, use)
            sigma_terms(use, rows)

            ok, delta = crosscheck(rows, original["daily_returns"])
            if not ok:
                raise RuntimeError(
                    f"{tag}: el replay NO reproduce la evaluacion original "
                    f"(delta {delta:.3e}). Si el replay no es fiel, ninguna cifra que salga "
                    "de aqui vale; se aborta antes de publicar."
                )

            gross = sum(r["gross_return"] for r in rows)
            cost = sum(r["total_cost"] for r in rows)
            blob["runs"][f"{cfg}_seed{seed}"] = {
                "config": cfg, "seed": seed,
                "sum_gross": gross, "sum_cost": cost, "sum_net": gross - cost,
                "sum_abs_dw": sum(r["sum_abs_dw"] for r in rows),
                "sum_abs_dw_sigma": sum(r["sum_abs_dw_sigma"] for r in rows),
                "crosscheck_delta": delta,
                "sessions": rows,
            }
            print(f"  {tag:<32} bruto {gross:+7.2%}  costo {cost:7.2%}  "
                  f"neto {gross - cost:+7.2%}  (cuadre {delta:.1e})")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(f"\n{len(blob['runs'])} corridas -> {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
