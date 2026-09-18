"""Legacy research cohort orchestrator; NOT ready for prospective activation.

Its configured native RL arm needs 59 per-bar calls, but this DAG dispatches once.
C049 rejects that mismatch before provider calls and propagates audit failures.
The old08:00 wait also precedes the first M5 close at08:05; changing that wait alone
would not supply a live feed or the missing59-call controller. Calendar and actual
scheduler validation also remain blocking. The frozen cohort and schedule are not
silently modified here. Missing decisions are not zero returns. Separate jobs and
hash chains do not authenticate fills or timestamps.
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

    Se prefiere data_interval_end del DagRun convertido a America/Bogota, no
    `datetime.now()`: un reintento a medianoche UTC sellaria la sesion equivocada, y ese
    fallo no deja rastro — el registro parece correcto.
    """
    from src.research.llm_forward.session_date import session_date_from_context
    return session_date_from_context(context)


def seal_llm(**context):
    _prepare_env()
    _require_single_call_dispatch()
    from src.research.llm_forward import decide

    session = _session_date(context)
    print(f"sellando brazo LLM para {session}")
    rc = 0
    for arm in ARMS_LLM:
        rc |= decide.run(session, arm_id=arm)
    if rc:
        raise RuntimeError("el sellado del brazo LLM devolvio error")


def _require_single_call_dispatch():
    """Do not spend on a partial cohort while native RL has no per-bar dispatch.

    This legacy DAG calls each arm once. It cannot run a 59-decision treatment;
    the streaming controller must be wired and validated before that cohort can
    start. Removing an arm from the frozen cohort is not an operational repair.
    """
    from src.research.llm_forward.decide import arm_spec, load_preregistration
    from src.research.llm_forward.paths import PREREG_PATH

    spec, _ = load_preregistration(PREREG_PATH)
    for arm_id in ARMS_RL:
        arm = arm_spec(spec, arm_id)
        count = arm.get("decisions_per_session")
        if arm.get("kind") != "rl_frozen" or type(count) is not int or count != 1:
            raise RuntimeError(
                f"{arm_id}: single-call DAG cannot execute native 59-decision per-bar cohort"
            )


def seal_rl(**context):
    _prepare_env()
    _require_single_call_dispatch()
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
    rc = settle_thesis.run({session: closes})
    if type(rc) is not int or rc != 0:
        raise RuntimeError("forward settlement returned an error")
    fail_if_upstream_failed(context, task_name="settle_forward_arms")


def audit(**context):
    """Propagate ledger rejection; a valid chain is not scientific readiness."""
    _prepare_env()
    from src.research.llm_forward import verify

    rc = verify.main()
    if type(rc) is not int or rc != 0:
        raise RuntimeError("forward ledger verification failed")
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

    # Legacy wait08:00, NOT first-bar close08:05. Cohort remains blocked by preflight.
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
