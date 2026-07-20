---
name: data-recovery
description: Diagnose and repair stale data in the USDCOP trading system. Use when training is blocked by a freshness gate, when a dashboard page shows "Sin datos", when OHLCV/macro/news/models look out of date, or after a container restart left the DB empty. Picks the right runbook from the symptom instead of guessing.
---

# Data recovery

Recovery here is **deterministic**: the symptom determines the runbook. The failure mode this
skill prevents is picking the wrong one under pressure — the procedures live in 5-8 documents
that do not fully agree with each other.

## Step 1 — Diagnose before acting

Never trigger a backfill blind. Establish which layer is stale:

```bash
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "
SELECT 'ohlcv' AS src, MAX(time)::text AS latest FROM usdcop_m5_ohlcv WHERE symbol='USD/COP'
UNION ALL SELECT 'macro', MAX(fecha)::text FROM macro_indicators_daily
UNION ALL SELECT 'news', MAX(published_at)::text FROM news_articles;"
```

Thresholds are SSOT in `.claude/rules/data-freshness.md`: **OHLCV 3 días · macro 7 días ·
modelos 10 días · news 24 h**. Anything within threshold is NOT the problem — keep looking.

Also check whether the upstream DAG even ran:

```bash
airflow dags list-runs -d core_l0_02_ohlcv_realtime --limit 5
```

## Step 2 — Apply the matching runbook

| Symptom | Action |
|---|---|
| OHLCV > 3d | `airflow dags trigger core_l0_01_ohlcv_backfill` |
| Macro > 7d | `airflow dags trigger core_l0_03_macro_backfill` |
| Modelos > 10d | `airflow dags trigger forecast_h5_l3_weekly_training` |
| News > 24h | verificar `feedparser`, luego `airflow dags trigger news_daily_pipeline` |
| `/analysis` sin charts macro | el CLEAN parquet está viejo → macro backfill, o `python scripts/ops/patch_analysis_macro_charts.py` |
| DB vacía tras reinicio | recuperación total → `.claude/specs/operations/freshness-recovery.md` §Recuperación total |

Full commands and the container-specific gotchas: `.claude/specs/operations/freshness-recovery.md`.

## Step 3 — Verify, don't assume

Re-run the Step 1 query. A DAG that finished is not the same as data that landed. State the
before/after timestamps explicitly in your report.

## Hard constraints

- **El track H1 está PAUSADO a propósito** (`is_paused_upon_creation=True`, audit A3-01).
  **NUNCA dispares un DAG `forecast_h1_*`** para "arreglar" frescura sin que el operador lo pida
  explícitamente — reactivarlo revierte una decisión deliberada. Verifica el estado de pausa antes.
- No hay backup de fin de semana (`0 20 * * 1-5`). Un lunes por la mañana, datos del viernes son
  normales, no un fallo.
- El restore de feature-data es **empty-table-only**: nunca sobrescribe tablas pobladas.
- Si el gate de training bloquea, **no lo desactives**. Está haciendo su trabajo: arregla el dato.
