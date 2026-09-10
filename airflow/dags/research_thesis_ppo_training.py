"""DAG del brazo PPO de la tesis: 2 configuraciones x 5 semillas.

Contract: CTR-RESEARCH-PPO-001 · Date: 2026-08-25

## Por qué un DAG y no un proceso en background

Diez corridas de ~21 minutos son ~3,5 horas de reloj. Un proceso hijo lanzado desde la sesión
ya se cortó dos veces durante este trabajo. Airflow sobrevive al cierre de la sesión, deja
logs por tarea, y reintenta la corrida que falle sin volver a lanzar las nueve que sí
terminaron.

## Paralelismo y por qué se limita

El contenedor ve 16 CPUs y torch arranca con 8 hilos. Diez tareas x 8 hilos = 80 hilos sobre
16 núcleos: el thrashing haría cada corrida más lenta que en secuencial. Se fija
`OMP_NUM_THREADS=2` y `max_active_tasks=6`, o sea ~12 hilos ocupados.

Esto **no afecta a los resultados**: PPO con semilla fija en CPU es determinista respecto al
número de hilos para MlpPolicy, y en todo caso el límite es idéntico para las dos
configuraciones, así que no puede sesgar la ablación.

## El dataset entra ya construido

`data/thesis/research_data_portable.pkl` se genera en el host (necesita `hmmlearn`, que este
contenedor no tiene) y llega por el montaje de `data/`. La primera tarea comprueba que exista
y falla rápido si no: es preferible a que las diez corridas mueran a los veinte minutos.

`manual` schedule: la tesis se lanza a mano, no en un cron.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from utils.run_status import fail_if_upstream_failed

REPO = "/opt/airflow"
PORTABLE = f"{REPO}/data/thesis/research_data_portable.pkl"
PPO_OUT = f"{REPO}/data/thesis/ppo"

SEEDS = (42, 123, 456, 789, 1337)          # `experiment-protocol.md` regla 2
CONFIGS = ("ppo_regime", "ppo_backbone")

DEFAULT_ARGS = {
    "owner": "research",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "depends_on_past": False,
}


def _prepare_env() -> None:
    """Limita los hilos ANTES de que torch se importe; despues ya no tiene efecto."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "2"
    os.environ["THESIS_PORTABLE"] = PORTABLE
    os.environ["THESIS_PPO_OUT"] = PPO_OUT
    if REPO not in sys.path:
        sys.path.insert(0, REPO)


def check_dataset(**_):
    """Falla rapido si falta el dataset, en vez de a los 20 minutos de la primera corrida."""
    from pathlib import Path

    p = Path(PORTABLE)
    if not p.is_file():
        raise FileNotFoundError(
            f"falta {PORTABLE}. Generalo en el HOST con:\n"
            "  python -c \"import sys;sys.path.insert(0,'.');"
            "from src.research.dataset import load_or_build,save_portable;"
            'save_portable(load_or_build())"'
        )

    _prepare_env()
    from src.research.dataset import load_portable

    data = load_portable(p)
    if not data.development or not data.selection:
        raise ValueError(f"dataset vacio: {data.summary()}")
    print(f"dataset OK: {data.summary()}")
    print(f"HMM: K={data.regime_model['k']} {data.regime_model['labels']} "
          f"ajustado en {data.regime_model['fit_range']}")


def train(config: str, seed: int, **context):
    """Entrena una corrida. Con `-c '{"refit": true}'` entrena sobre desarrollo+seleccion.

    El refit es el paso F8 que el pre-registro compromete antes de abrir el hold-out. Se
    activa por configuracion del DagRun y no por un DAG aparte, para que las dos corridas
    compartan exactamente el mismo codigo y los mismos hiperparametros: si divergieran, el
    refit dejaria de ser el mismo experimento con mas datos.
    """
    _prepare_env()
    import torch

    torch.set_num_threads(2)

    from scripts.analysis.thesis_train_ppo import TOTAL_TIMESTEPS, train_one
    from src.research.dataset import load_portable

    conf = (context.get("dag_run").conf or {}) if context.get("dag_run") else {}
    refit = bool(conf.get("refit", False))
    data = load_portable()
    res = train_one(config, seed, data, timesteps=TOTAL_TIMESTEPS, refit=refit)
    sel = res["selection"]
    print(f"{config} seed={seed}: seleccion ret={sel['total_return']:+.2%} "
          f"Sharpe={sel['sharpe']:+.2f} ops={sel['n_ops']}")
    # Sin veredicto aqui: la comparacion entre configuraciones y el contraste estadistico
    # se hacen en R5 sobre las 10 corridas juntas, no tarea a tarea.
    return {"config": config, "seed": seed,
            "selection_sharpe": sel["sharpe"],
            "selection_return": sel["total_return"]}


def collect(**context):
    """Junta las 10 corridas en un indice unico para que R5 lo consuma."""
    import json
    from pathlib import Path

    conf = (context.get("dag_run").conf or {}) if context.get("dag_run") else {}
    suffix = "_refit" if conf.get("refit") else ""
    rows = []
    for config in CONFIGS:
        for seed in SEEDS:
            f = Path(PPO_OUT) / f"{config}{suffix}_seed{seed}.json"
            if not f.is_file():
                print(f"AVISO: falta {f.name}")
                continue
            r = json.loads(f.read_text(encoding="utf-8"))
            rows.append({"config": config, "seed": seed,
                         "selection": {k: v for k, v in r["selection"].items()
                                       if k not in ("daily_returns", "dates")},
                         "development": {k: v for k, v in r["development"].items()
                                         if k not in ("daily_returns", "dates")}})

    index = Path(PPO_OUT) / f"index{suffix}.json"
    index.write_text(json.dumps({"runs": rows, "n": len(rows),
                                 "expected": len(CONFIGS) * len(SEEDS)}, indent=2),
                     encoding="utf-8")
    print(f"{len(rows)}/{len(CONFIGS) * len(SEEDS)} corridas -> {index}")
    for r in rows:
        s = r["selection"]
        print(f"  {r['config']:<14} seed {r['seed']:>4}: "
              f"ret {s['total_return']:+7.2%}  Sharpe {s['sharpe']:+6.2f}")

    # `collect_runs` es hoja y corre con `all_done` — a proposito: quiero el indice de las
    # corridas que SI terminaron aunque alguna muera. Pero una hoja `all_done` que sale
    # verde convierte un DagRun con entrenamientos caidos en un DagRun "exitoso", y esa es
    # exactamente la forma en que una tabla macro se quedo 13 dias stale sin que nadie lo
    # viera. El indice se escribe primero; el veredicto del run se emite despues.
    if len(rows) < len(CONFIGS) * len(SEEDS):
        raise RuntimeError(f"faltan corridas: {len(rows)} de {len(CONFIGS) * len(SEEDS)}")
    fail_if_upstream_failed(context, task_name="collect_runs")


with DAG(
    dag_id="research_thesis_ppo_training",
    description="Tesis · brazo PPO: ppo_regime vs ppo_backbone, 5 semillas (CTR-RESEARCH-PPO-001)",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 8, 1),
    schedule=None,                       # manual: la tesis no corre en cron
    catchup=False,
    max_active_tasks=6,                  # ~12 hilos sobre 16 nucleos
    tags=["research", "thesis", "rl", "ppo"],
) as dag:

    check = PythonOperator(task_id="check_dataset", python_callable=check_dataset)
    gather = PythonOperator(task_id="collect_runs", python_callable=collect,
                            trigger_rule="all_done")

    for cfg in CONFIGS:
        for sd in SEEDS:
            task = PythonOperator(
                task_id=f"train_{cfg}_seed{sd}",
                python_callable=train,
                op_kwargs={"config": cfg, "seed": sd},
                execution_timeout=timedelta(hours=2),
            )
            check >> task >> gather
