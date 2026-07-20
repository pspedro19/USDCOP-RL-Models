---
name: dag-change
description: Add, rename, reschedule, pause or deprecate an Airflow DAG without breaking the registry, the collision-free timeline or downstream sensors. Use when touching anything under airflow/dags, including the config-driven asset pipeline DAGs.
---

# DAG change

## Why this is dangerous

The registry drifted from disk and nobody noticed: **5 real DAGs had no constant at all**
(`core_watchdog`, `reconciliation_daily`, `rbac_entitlements_daily`, `forecast_weekly_generation`,
`forecast_asset_analysis_weekly`) while **7 constants advertised DAGs no module emits**. Every doc
quoting the registry inherited the error. `test_knowledge_inventory.py` now detects that; this
skill prevents it.

**DAG ids are persistent operational API keys** — Airflow history, REST triggers, sensors,
watchdog calls and the dashboard deploy path all reference them as literals.

## Checklist

1. **Registry constant** in `airflow/dags/contracts/dag_registry.py` (platform/ops constants live
   near `:206`).
2. **Add to `get_all_dag_ids()`** — unless it is config-driven, deprecated, or absent.
3. **Confirm `get_active_dag_ids()`**: active = `get_all_dag_ids()` − deprecated − absent +
   config-driven asset DAGs.
4. **`DAG_DEPENDENCIES`** if ordering changed (`dag_registry.py:301-347`).
5. **`DAG_TAGS`** — centralized in the registry (`:350-354`).
6. **Schedule collision** against the SSOT timeline:
   `.claude/specs/operations/elite-operations.md`. Pick a non-colliding slot; if behaviour
   changes, update that document in the same commit.
7. **`ExternalTaskSensor`** for cross-DAG gates. Live examples: Analysis-L8 waits for
   `news_daily_pipeline` (`analysis_l8_daily_generation.py:155-168`); H5-L5 waits for H5-L3
   (`forecast_h5_l5_weekly_signal.py:599-601`).
8. **Importability**, then **regenerate the inventory**.

## Asset DAGs are config-driven

`asset_{asset_id}_pipeline_weekly` is built by `airflow/dags/asset_pipeline_factory.py` from
`config/assets/pipelines.yaml`. **Never hardcode one in the registry** — change the YAML; the
registry derives them via `get_asset_dag_ids()`. Every stage `script:` path must exist.

## Never touch casually

| DAG(s) | Why |
|---|---|
| `rl_l4_01_experiment_runner`, `rl_l4_02_backtest_validation`, `rl_l4_03_scheduled_retraining` | **DEPRECATED on purpose.** `test_dag_registry_deprecated.py` requires they stay inactive |
| All `forecast_h1_*` | **Paused on purpose** (`is_paused_upon_creation=True`, audit A3-01). Do not unpause as part of unrelated cleanup |
| `forecast_h5_l4b_production_deploy` | **Event-driven and must survive cold boot UNPAUSED.** Do not "standardize" it to paused — you would silently break the post-Vote-2 deploy path |

## Verify

Local (Airflow optional — the importability test skips without it, and that skip is not a pass):

```powershell
python -m py_compile airflow/dags/contracts/dag_registry.py
python -m pytest tests/regression/test_dag_registry_deprecated.py -q
python -m pytest tests/regression/test_knowledge_inventory.py -q
python scripts/diagnostics/generate_inventory.py --check
```

In the scheduler container (the only place import errors are real):

```powershell
docker exec usdcop-airflow-scheduler airflow dags list-import-errors
docker exec usdcop-airflow-scheduler python -m pytest /opt/airflow/project/tests/regression/test_dag_importability.py -q
```

After an intentional change:

```powershell
python scripts/diagnostics/generate_inventory.py --write
python scripts/diagnostics/generate_inventory.py --check
```

## DO NOT

- Do NOT rename a DAG id without updating registry, sensors, triggers, dependencies, inventory and
  docs **in the same commit** — a rename orphans Airflow history and breaks REST triggers.
- Do NOT trust `airflow/dags/utils/dag_dependencies.py` as a source of truth: it still carries
  legacy ids (`v3.l2_preprocessing_pipeline`, `l4_feature_pipeline`, `l3_macro_ingest`).
- Do NOT hardcode an asset DAG id — it comes from `config/assets/pipelines.yaml`.
- Do NOT unpause deprecated RL-L4 or the intentionally paused H1 DAGs.
- Do NOT pause `forecast_h5_l4b_production_deploy`.
- Do NOT schedule without checking the collision-free timeline.
- Do NOT hand-edit `.claude/generated/inventory.json` — regenerate it.
- Do NOT treat a skipped importability test as a passing one.
