# SPEC-10 — Orquestación con Airflow

## Propósito
Definir los DAGs que orquestan el sistema end-to-end. Usa **Airflow ≥2.8** con **Datasets** (scheduling data-aware), **TaskFlow API** y **dynamic task mapping** (para multi-seed). Todas las tasks **idempotentes**; datos pasan por paths/DVC, no por XCom (XCom solo metadata pequeña).

## Datasets (contratos de scheduling)
```python
from airflow.datasets import Dataset
RAW_GOLD      = Dataset("dvc://data/raw/gold/xauusd_m1")
PROCESSED     = Dataset("dvc://data/processed/gold_h1")
FEATURES      = Dataset("dvc://data/features/h1")
REGIME        = Dataset("dvc://data/features/regime_daily")
MODEL_CANDIDATE = Dataset("mlflow://models/gold_rl/candidate")
```
Un DAG que **produce** un Dataset dispara automáticamente al que lo **consume**. Elimina sensores frágiles y cron acoplados.

## Fábrica de pipelines y contrato de salida (ver SPEC-12)
Los DAGs `xau_*` de abajo son la **forma canónica de un pipeline**, pero **no se copian por activo/estrategia** (5 DAGs × N × M = explosión). El diseño escalable es una **fábrica config-driven**:
```
config/assets/<asset>.yaml + config/strategies/<strat>.yaml
        │  build_pipeline(asset, strategy)  →  emite train→signal→exec→monitor + validate
        ▼
airflow/dags/generated_pipelines.py  recorre el registro de pares habilitados y llama a la fábrica
```
Agregar Oro = una entrada de config, **0 archivos DAG nuevos**. Los DAGs `xau_*` de esta spec son el primer caso concreto que la fábrica debe poder generar. Además, **todo pipeline honra un contrato I/O**:
- **ENTRADA:** lee `AssetProfile` + `StrategyConfig`, sensores/Datasets upstream, **gate de frescura BLOQUEANTE**.
- **SALIDA:** su **tarea final es `register_bundle`** (valida → escribe backtest inmutable versionado → upsert atómico de `registry.json`). Un bundle inválido NO entra al registro. Ver SPEC-12 Contratos B/C.

## DAG 1 — `xau_data_ingestion` (SPEC-01)
Schedule: `@daily` (tras cierre NY). Tasks aisladas: una fuente caída no tumba las demás.

```python
@dag(schedule="0 22 * * 1-5", start_date=..., catchup=False,
     default_args={"retries": 5, "retry_delay": timedelta(minutes=10),
                   "retry_exponential_backoff": True}, tags=["xau","ingest"])
def xau_data_ingestion():
    @task(execution_timeout=timedelta(hours=2))
    def download_gold(): ...            # dukascopy incremental
    @task
    def download_fred(): ...            # DFII10, DGS10, T10YIE, DTWEXBGS
    @task
    def download_calendar(): ...        # eventos USD high-impact
    @task
    def validate_raw(): ...             # schema + sanity mínima
    @task(outlets=[RAW_GOLD])
    def dvc_add_push(): ...             # versiona y publica → dispara DAG 2
    [download_gold(), download_fred(), download_calendar()] >> validate_raw() >> dvc_add_push()
```

## DAG 2 — `xau_data_processing` (SPEC-02 + SPEC-03 + SPEC-04)
Schedule: `schedule=[RAW_GOLD]` (data-aware). Falla el DAG si la auditoría de calidad no pasa.

```python
@dag(schedule=[RAW_GOLD], catchup=False, tags=["xau","process"])
def xau_data_processing():
    @task def normalize_tz_dst(): ...          # SPEC-02 §1
    @task def handle_sunday(): ...             # SPEC-02 §2
    @task def quality_audit(): ...             # SPEC-02 §3 — raise si falla umbral
    @task def resample_h1_daily(): ...         # SPEC-02 §4
    @task(outlets=[PROCESSED]) def align_pit(): ...  # SPEC-02 §5 point-in-time
    @task def build_h1_features(): ...         # SPEC-03
    @task def build_daily_features(): ...      # SPEC-03
    @task(outlets=[REGIME, FEATURES]) def build_regime(): ...  # SPEC-04
    (normalize_tz_dst() >> handle_sunday() >> quality_audit()
     >> resample_h1_daily() >> align_pit()
     >> [build_h1_features(), build_daily_features()] >> build_regime())
```

## DAG 3 — `xau_training` (SPEC-08) — multi-seed con dynamic mapping
Schedule: manual o `@weekly`. Paraleliza seeds y folds. Usa `KubernetesPodOperator`/GPU si aplica.

```python
@dag(schedule=None, catchup=False, tags=["xau","train"])
def xau_training():
    @task def make_folds() -> list[dict]: ...   # SPEC-09 walk-forward folds
    @task def seed_grid(folds) -> list[dict]:   # producto folds × seeds
        return [{"fold": f, "seed": s} for f in folds for s in SEEDS]

    @task(execution_timeout=timedelta(hours=8), pool="gpu_pool")
    def train_one(spec: dict) -> str:           # entrena 1 (fold,seed) → mlflow run_id
        ...
    @task def aggregate(run_ids: list[str]): ...  # mediana/IQR por fold

    folds = make_folds()
    grid = seed_grid(folds)
    runs = train_one.expand(spec=grid)          # ← dynamic task mapping (multi-seed)
    aggregate(runs)
```

## DAG 4 — `xau_walkforward` (SPEC-09) — validación + gate
Schedule: `schedule=[MODEL_CANDIDATE]` o encadenado tras training.

```python
@dag(schedule=[MODEL_CANDIDATE], catchup=False, tags=["xau","validate"])
def xau_walkforward():
    @task def run_baselines(): ...              # B1, B2 (SPEC-07)
    @task def evaluate_agent(): ...             # métricas OOS mediana/IQR
    @task def attribution(): ...                # por régimen (SPEC-09)
    @task def stats_shield(): ...               # DSR + bootstrap CIs
    @task def decision_gate() -> bool:          # ¿mediana gana a B1 y B2?
        ...                                     # branch: promover o no
    @task def report(): ...                     # reports/walkforward_*.html
    @task def register_bundle(): ...            # SPEC-12: EXIT CONTRACT — solo si pasó el gate.
                                                # Valida → escribe backtests/<version>/<year> inmutable
                                                # → upsert atómico registry.json + manifest.json (aditivo).
    run_baselines() >> evaluate_agent() >> attribution() >> stats_shield() >> decision_gate() >> report() >> register_bundle()
```

## DAG 5 — `xau_paper_shadow` (SPEC-11) — fase de despliegue
Schedule: `@daily`. Genera señal, compara contra el shadow del backtest, alerta divergencias (train/serve skew).

```python
@dag(schedule="0 22 * * 1-5", catchup=False, tags=["xau","paper"])
def xau_paper_shadow():
    @task def pull_latest(): ...
    @task def generate_signal(): ...            # mismo env/risk que backtest
    @task def shadow_compare(): ...             # señal live vs backtest en mismos datos
    @task def drift_check(): ...                # PSI/KS sobre features (SPEC-11)
    @task def alert_on_divergence(): ...        # Slack/email si diverge
    pull_latest() >> generate_signal() >> shadow_compare() >> drift_check() >> alert_on_divergence()
```

## Convenciones transversales
- **Idempotencia:** reejecutar cualquier task no duplica ni corrompe (reescritura de partición-año OK).
- **Alerting:** `on_failure_callback` a Slack/email en todos los DAGs.
- **XCom:** solo IDs/paths/hashes; datos por DVC/filesystem.
- **Config:** montada desde `config/*.yaml`; `data_version` (DVC) y `git_sha` propagados como tags a MLflow.
- **Backfill:** `catchup=False` por defecto; backfills explícitos y controlados.
- **Secrets:** API keys (FRED, calendario, broker) vía Airflow Connections/Variables o un secrets backend, nunca en código.

## Criterios de aceptación
- [ ] Los 5 DAGs parsean sin error (`airflow dags list` + test de import).
- [ ] Scheduling data-aware: producir `RAW_GOLD` dispara DAG 2 (test de dataset).
- [ ] DAG 2 falla si la auditoría de calidad no pasa (test con datos corruptos).
- [ ] DAG 3 paraleliza `train_one` por dynamic mapping sobre folds×seeds.
- [ ] DAG 4 evalúa el gate y ramifica (promover/no) según mediana vs baselines.
- [ ] **DAG 4 termina en `register_bundle`** que publica un bundle inmutable versionado y hace upsert de `registry.json` de forma atómica y aditiva (SPEC-12: R3, R5, R9).
- [ ] Tasks idempotentes (test de doble ejecución).
- [ ] Fallas notifican por `on_failure_callback`.
- [ ] (Objetivo) los `xau_*` son **emitibles por la fábrica** desde `AssetProfile` + `StrategyConfig` — no hardcodeados por activo.

## Dependencias
Todas las SPEC-01…09 y SPEC-11 (orquesta sus entregables). **SPEC-12** (fábrica de pipelines + contrato de salida `register_bundle`).
