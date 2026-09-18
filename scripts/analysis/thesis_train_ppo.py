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
eval=131 perdió -20,6%; seed 1337 con eval=111 ganó +9,6%). Aquí se entrena un número fijo de
pasos y se guarda el modelo final. La selección entre configuraciones ocurre en el bloque de
SELECCIÓN, nunca en el hold-out.

Uso:
    python scripts/analysis/thesis_train_ppo.py --config ppo_regime --seed 42
    python scripts/analysis/thesis_train_ppo.py --all            # 10 corridas
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# La consola de Windows usa cp1252 y estos scripts imprimen `Δ`, `·`, `→`. Sin esto un
# UnicodeEncodeError aborta la corrida DESPUES de haber calculado todo, que es la peor
# forma de fallar: el trabajo esta hecho y no se escribe el JSON.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


from scripts.diagnostics.audit_research_data_contract import require_contract  # noqa: E402
from src.research.dataset import PORTABLE, load_or_build, load_portable  # noqa: E402
from src.research.features import GROUPS  # noqa: E402
from src.research.ppo_recipe import (  # noqa: E402
    CONTROL_ARM,
    build_ppo,
    canonical_sha256,
    effective_recipe,
    file_sha256,
    known_probes,
    library_versions,
    recipe_for,
    sessions_sha256,
    training_code_hashes,
    write_immutable_json,
)
from src.research.ppo_recipe import (  # noqa: E402
    NET_ARCH as NET_ARCH,
)
from src.research.ppo_recipe import (  # noqa: E402
    PPO_KWARGS as PPO_KWARGS,
)
from src.research.sanity_gate import require_macro_identity, require_sanity_pass  # noqa: E402
from src.research.session_env import daily_series  # noqa: E402
from src.research.session_gym import SessionSpec, SessionTradingEnv  # noqa: E402

SEEDS = (42, 123, 456, 789, 1337)          # `experiment-protocol.md` regla 2
CONFIGS = ("ppo_regime", "ppo_backbone")
OUT = Path(os.environ.get("THESIS_PPO_OUT", REPO / "outputs" / "thesis" / "ppo"))
N_REGIMES = len(GROUPS["regimen"])

# PPO_KWARGS/NET_ARCH remain import-compatible, but are owned by ppo_recipe.
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
        from src.research.observation_contract import LEGACY_VERSION, observation_contract
        version = getattr(s, "observation_version", LEGACY_VERSION)
        ctx = s.context.copy()
        ctx[-observation_contract(version).regime_slots:] = 0.0
        out.append(SessionSpec(date=s.date, close=s.close, market=s.market,
                               context=ctx, spread_pips=s.spread_pips,
                               cost_parameters=getattr(s, "cost_parameters", None),
                               observation_version=version))
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
              output_dir: Path | None = None, probe: str = CONTROL_ARM,
              source_artifact: Path | None = None) -> dict:
    """Entrena una configuracion.

    Con `refit=True` entrena sobre **desarrollo + seleccion**, que es lo que el pre-registro
    (§1, fila «Refit») compromete hacer UNA vez antes de abrir el hold-out. El modelo
    resultante nunca se evalua aqui sobre el hold-out: eso lo hace un paso aparte con el
    gate de la Regla B.
    """
    if config not in CONFIGS or seed not in SEEDS or timesteps <= 0:
        raise ValueError("unknown config/seed or non-positive timesteps")
    from src.research.observation_contract import LEGACY_VERSION
    if any(getattr(s, "observation_version", LEGACY_VERSION) != LEGACY_VERSION
           for block in (data.development, data.selection) for s in block):
        raise ValueError("new observation version requires a separately frozen training admission; legacy runner refused")
    dev = data.development if config == "ppo_regime" else strip_regimes(data.development)
    sel = data.selection if config == "ppo_regime" else strip_regimes(data.selection)
    train_specs = (dev + sel) if refit else dev

    # La receta la dicta la compuerta de sanidad, no este fichero. Antes, `train_one`
    # construia SIEMPRE el entorno y la politica por defecto mientras `main` imprimia
    # `sanity gate: receta=flat_init_no_turn`: la compuerta validaba un informe y el
    # entrenador corria otra receta. Medido sin el sesgo inicial (300k pasos, v2, semilla 42):
    # 909 operaciones en 226 sesiones y coste 0,4855 sobre bruto ~0,272.
    recipe = recipe_for(probe)
    effective = effective_recipe(probe)
    kwargs = effective["ppo_kwargs"]
    out = output_dir or OUT
    tag = f"{config}_refit_seed{seed}" if refit else f"{config}_seed{seed}"
    paths = {"checkpoint": out / f"{tag}.zip", "vecnormalize": out / f"{tag}_vecnorm.pkl",
             "result": out / f"{tag}.json", "manifest": out / f"{tag}.manifest.json"}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"immutable PPO run already exists: {out / tag}")
    frozen_sources = training_code_hashes()
    manifest = {
        "schema_version": "research-grade-ppo-run-v1", "scope": "retrospective_diagnostic",
        "config": config, "seed": seed, "refit": refit,
        "timesteps_requested": timesteps, "effective_recipe": effective,
        "source_sha256": frozen_sources, "library_versions": library_versions(),
        "dataset_sha256": sessions_sha256(train_specs),
        "development_sha256": sessions_sha256(dev), "selection_sha256": sessions_sha256(sel),
        "input_artifact": ({"path": str(source_artifact.resolve()),
                            "sha256": file_sha256(source_artifact)} if source_artifact else None),
        "frozen_at_utc": datetime.now(UTC).isoformat(),
    }
    write_immutable_json(paths["manifest"], manifest)
    # The same constructor applies normalization, architecture AND the initial
    # action bias for market and sanity. There is no second implementation.
    model, venv, recipe = build_ppo(train_specs, seed=seed, probe=probe)
    flat_biased = recipe.flat_bias_logit is not None
    t0 = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    elapsed = time.time() - t0
    venv.training = False
    venv.norm_reward = False
    model.save(out / f"{tag}.zip")
    venv.save(str(out / f"{tag}_vecnorm.pkl"))

    # Evaluacion con VecNormalize CONGELADO: el modelo predice sobre las mismas
    # observaciones que vio entrenando, y el reward normalizado no interviene aqui.
    res = {
        "config": config, "seed": seed, "timesteps": timesteps, "refit": refit,
        "schema_version": "research-grade-ppo-run-v1",
        "scope": "retrospective_diagnostic", "manifest": manifest,
        "manifest_sha256": canonical_sha256(manifest),
        "timesteps_requested": timesteps, "timesteps_effective": int(model.num_timesteps),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "identity_unchanged": frozen_sources == training_code_hashes(),
        "artifacts": {name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
                      for name, path in paths.items() if name != "result"},
        # Sin este campo, dos JSON producidos por recetas distintas son indistinguibles y
        # cualquiera puede atribuirle a la receta validada los numeros de la que no lo esta.
        "recipe_probe": recipe.probe,
        "recipe": {"kappa_turn": recipe.kappa_turn, "ent_coef": kwargs["ent_coef"],
                   "gamma": kwargs["gamma"], "norm_reward": recipe.norm_reward,
                   "flat_bias_logit": recipe.flat_bias_logit,
                   "flat_bias_applied": flat_biased},
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
    write_immutable_json(paths["result"], res)
    venv.close()
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
    ap.add_argument("--dataset-version", choices=("v1", "v2"), default="v1",
                    help="v2 exige automáticamente sanidad sintética e identidad macro")
    ap.add_argument("--require-sanity", type=Path,
                    help="informe S1-S4 aprobado; obligatorio antes de un entrenamiento v2")
    ap.add_argument("--require-macro-identity", type=Path,
                    help="informe de reconciliación positiva de fuentes macro")
    ap.add_argument("--refit", action="store_true",
                    help="entrena sobre desarrollo+seleccion (paso F8 del pre-registro)")
    ap.add_argument("--diagnostic-retrospective", action="store_true",
                    help="permite una corrida diagnóstica v2 sin preregistro firmado; no es confirmatoria")
    ap.add_argument("--recipe-control", action="store_true",
                    help=("entrena el brazo de CONTROL (sin el sesgo inicial de la receta "
                          "seleccionada). Existe para poder medir el efecto de la receta con "
                          "una variable; no produce la serie validada."))
    args = ap.parse_args()

    evidence = require_contract()
    print(f"data contract: structural={evidence['verdict']['structural_m5_clean']} "
          f"macro_complete={evidence['verdict']['macro_columns_complete']}")
    sanity_path = args.require_sanity
    macro_path = args.require_macro_identity
    if args.dataset_version == "v2":
        sanity_path = sanity_path or (REPO / "outputs" / "thesis-repair" / "sanity_protocol_v2.json")
        macro_path = macro_path or (REPO / "outputs" / "thesis-repair" / "macro_identity_research_v2_latest.json")
        if not sanity_path.is_file() or not macro_path.is_file():
            ap.error("dataset v2 exige informes de sanidad e identidad macro existentes")
        prereg_path = REPO / ".claude" / "specs" / "planes" / "06-PRE-REGISTRATION-v3.md"
        prereg_text = prereg_path.read_text(encoding="utf-8") if prereg_path.is_file() else ""
        if "operator_signature: SIGNED" not in prereg_text and not args.diagnostic_retrospective:
            ap.error(
                "v2 confirmatorio bloqueado: 06-PRE-REGISTRATION-v3 no está SIGNED; "
                "use --diagnostic-retrospective solo para una medición retrospectiva explícita"
            )
        if args.diagnostic_retrospective:
            if args.output_dir is None:
                ap.error("diagnostic-retrospective exige --output-dir bajo outputs/thesis-repair")
            out_resolved = args.output_dir.resolve()
            if REPO / "outputs" / "thesis-repair" not in out_resolved.parents:
                ap.error("diagnostic-retrospective exige --output-dir bajo outputs/thesis-repair")
    probe = CONTROL_ARM
    if sanity_path:
        sanity = require_sanity_pass(sanity_path)
        probe = sanity["selected_probe"]
        print(f"sanity gate: receta={probe}")
        if probe not in known_probes():
            # Fail-closed a proposito. La alternativa -- seguir con la receta por defecto --
            # es como se colo el defecto que este bloque corrige: el informe decia una receta
            # y el entrenador corria otra, sin que nada en el artefacto lo delatara.
            ap.error(
                f"la compuerta selecciona la receta {probe!r} y el entrenador no sabe "
                f"aplicarla (declaradas: {list(known_probes())}). No se entrena."
            )
        if args.recipe_control:
            print(f"AVISO: --recipe-control ignora la receta de la compuerta y entrena "
                  f"{CONTROL_ARM!r}; sus JSON no son la serie validada.")
            probe = CONTROL_ARM
    if macro_path:
        clean_for_identity = None
        if args.dataset_version == "v2":
            clean_for_identity = REPO / "data" / "pipeline" / "04_cleaning" / "output" / "MACRO_RESEARCH_v2.parquet"
        identity = require_macro_identity(
            macro_path,
            availability=REPO / "config" / "research" / "macro_availability.yaml",
            clean=clean_for_identity,
        )
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
                  output_dir=args.output_dir, probe=probe,
                  source_artifact=portable_path if portable_path.is_file() else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
