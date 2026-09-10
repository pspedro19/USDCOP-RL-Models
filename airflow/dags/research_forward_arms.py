"""DAG del carril forward: RL congelado frente a LLM, sobre las mismas sesiones.

Contract: CTR-RESEARCH-FORWARD-001 · Date: 2026-08-25

## Por qué dos ventanas y no una

El arnés exige que el job que **sella** y el que **liquida** no se toquen. No es purismo: si el
mismo proceso puede escribir la decisión y conocer el resultado, la garantía entera —«la
decisión existió antes del desenlace»— pasa a depender de que nadie se equivoque de orden.
Separarlos la hace estructural.

    12:15 UTC (07:15 COT)  sella el LLM      <- solo documentos anteriores al cutoff
    13:00 UTC (08:00 COT)  sella los RL      <- necesitan la barra 0
    18:30 UTC (13:30 COT)  liquida           <- despues del cierre de 12:55

## La asimetría de las 45 minutos

El LLM sella a las 07:15 con documentos estrictamente anteriores a las 08:00. El RL sella a las
08:00 porque su observación necesita la primera barra. **Los dos sellan antes de que exista
`r_1`**, que nace al cierre de la barra 1 — así que los dos son causalmente limpios, pero el RL
ve una barra que el LLM no ve.

Va en `information_edge` de cada registro RL y en el pre-registro. No es una nota al pie: es la
diferencia que un revisor buscaría primero.

## Un job tarde no se descarta

Si el sellado corre tarde, el arnés escribe igual con `sealed_before_open: false` y la
liquidación la excluye. Una fila excluida y visible es auditable; una fila ausente parece un día
que nunca existió.

`schedule` de lunes a viernes: la sesión de USD/COP no abre en fin de semana y un registro de
sábado sería una sesión inventada.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from utils.run_status import fail_if_upstream_failed

REPO = "/opt/airflow"
ARMS_LLM = ("llm_direct_fwd_v1",)
ARMS_RL = ("ppo_regime_fwd_k59", "ppo_regime_fwd_k1")

DEFAULT_ARGS = {
    "owner": "research",
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
    "depends_on_past": False,
}


def _prepare_env() -> None:
    """Rutas y limites de hilos ANTES de que torch se importe."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "2"
    os.environ.setdefault("FWD_DATA", f"{REPO}/data/forward")
    os.environ.setdefault("THESIS_PORTABLE",
                          f"{REPO}/data/thesis/research_data_portable.pkl")
    if REPO not in sys.path:
        sys.path.insert(0, REPO)


def _session_date(context) -> str:
    """La sesión COT que corresponde a esta corrida.

    Se toma de la fecha lógica del DagRun convertida a America/Bogota, no de
    `datetime.now()`: un reintento a medianoche UTC sellaria la sesion equivocada, y ese
    fallo no deja rastro — el registro parece correcto.
    """
    from datetime import timezone as _tz

    logical = context.get("logical_date") or context.get("execution_date")
    cot = logical.astimezone(_tz(timedelta(hours=-5)))
    return cot.date().isoformat()


def seal_llm(**context):
    _prepare_env()
    from src.research.llm_forward import decide

    session = _session_date(context)
    print(f"sellando brazo LLM para {session}")
    rc = 0
    for arm in ARMS_LLM:
        rc |= decide.run(session, arm_id=arm)
    if rc:
        raise RuntimeError("el sellado del brazo LLM devolvio error")


def seal_rl(**context):
    _prepare_env()
    import torch

    torch.set_num_threads(2)
    from src.research.llm_forward.arms import ppo_arm

    session = _session_date(context)
    print(f"sellando brazos RL para {session}")
    rc = 0
    for arm in ARMS_RL:
        rc |= ppo_arm.run(session, arm_id=arm)
    if rc:
        raise RuntimeError("el sellado de algun brazo RL devolvio error")


def settle(**context):
    """Liquida con el motor de la tesis, no con el del arnes.

    Los cierres salen de `usdcop_m5_ohlcv` —la misma serie que midio la tesis—, no de un CSV
    aparte: otra fuente introduciria una diferencia que ninguna tabla mostraria.
    """
    _prepare_env()
    import pandas as pd

    from src.research.llm_forward import settle_thesis

    session = _session_date(context)
    seed = f"{REPO}/seeds/latest/usdcop_m5_ohlcv.parquet"
    m5 = pd.read_parquet(seed)
    t = pd.to_datetime(m5["time"])
    day = m5[t.dt.date == pd.Timestamp(session).date()].sort_values("time")
    if "symbol" in day.columns:
        sym = day["symbol"].astype(str).str.upper().str.replace("/", "", regex=False)
        day = day[sym == "USDCOP"]

    closes = day["close"].to_numpy(dtype=float)
    print(f"{session}: {len(closes)} barras disponibles")
    settle_thesis.run({session: closes})
    fail_if_upstream_failed(context, task_name="settle_forward_arms")


def audit(**context):
    """Auditoria de la cadena. Barata, y es lo unico que prueba que el ledger vale."""
    _prepare_env()
    from src.research.llm_forward import verify

    verify.main()
    fail_if_upstream_failed(context, task_name="audit_forward_ledger")


with DAG(
    dag_id="research_forward_arms",
    description=("Carril forward: LLM vs PPO congelado sobre las mismas sesiones "
                 "(CTR-RESEARCH-FORWARD-001, exploratorio)"),
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 8, 26),
    # 07:15 COT. El sellado del RL y la liquidacion van dentro del mismo run como tareas
    # posteriores: Airflow no permite dos cron en un DAG, y separarlos en dos DAGs romperia
    # la garantia de que la liquidacion ve las decisiones de SU sesion.
    schedule="15 12 * * 1-5",
    catchup=False,
    max_active_runs=1,
    tags=["research", "forward", "llm", "rl", "exploratory"],
) as dag:

    t_llm = PythonOperator(task_id="seal_llm_arm", python_callable=seal_llm)

    # Espera hasta las 08:00 COT (13:00 UTC) para que exista la barra 0.
    from airflow.sensors.time_delta import TimeDeltaSensor

    wait_open = TimeDeltaSensor(
        task_id="wait_for_session_open",
        delta=timedelta(minutes=45),      # 07:15 -> 08:00 COT
        mode="reschedule",                # libera el worker mientras espera
    )

    t_rl = PythonOperator(task_id="seal_rl_arms", python_callable=seal_rl)

    wait_close = TimeDeltaSensor(
        task_id="wait_for_session_close",
        delta=timedelta(hours=6, minutes=15),   # 07:15 -> 13:30 COT
        mode="reschedule",
    )

    t_settle = PythonOperator(task_id="settle_forward_arms", python_callable=settle,
                              trigger_rule="all_done")
    t_audit = PythonOperator(task_id="audit_forward_ledger", python_callable=audit,
                             trigger_rule="all_done")

    t_llm >> wait_open >> t_rl >> wait_close >> t_settle >> t_audit
