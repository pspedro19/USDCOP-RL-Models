---
kind: as-built
status: IMPLEMENTED
contract: CTR-DQ-OPS-001
version: 2.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - airflow/dags/utils/data_quality.py
  - airflow/dags/core_watchdog.py
  - scripts/ops/patch_analysis_macro_charts.py
  - database/migrations
---
# SDD Spec: Recovery de frescura de datos (runbooks)

> **Responsibility**: procedimientos de recuperación por síntoma y tracking de migraciones.
> Los **umbrales** son SSOT de `../../rules/data-freshness.md` (auto-cargada) — no los repitas aquí.
> El **schedule de DAGs** es SSOT de `elite-operations.md` — no lo re-tabules aquí.

---

## Gate pre-training (BLOQUEANTE)

`airflow/dags/utils/data_quality.py::validate_training_data_freshness()` es la PRIMERA tarea de
los DAGs de training H1-L3 y H5-L3. Si falla, el entrenamiento NO procede.

```
validate_data_freshness >> load_and_build_features >> train_models >> ...
```

`check_model_freshness()` corre antes de inferencia y solo **avisa** (no bloquea): es preferible
operar con modelos algo viejos que detener el trading.

---

## Runbooks por síntoma

### OHLCV stale (>3 días)

```bash
airflow dags list-runs -d core_l0_02_ohlcv_realtime --limit 10   # ¿qué pasó?
airflow dags trigger core_l0_01_ohlcv_backfill                    # rellena el hueco
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading \
  -c "SELECT symbol, MAX(time) FROM usdcop_m5_ohlcv GROUP BY symbol;"
```

### Macro stale (>7 días)

```bash
airflow dags trigger core_l0_03_macro_backfill
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading \
  -c "SELECT MAX(fecha) FROM macro_indicators_daily;"
```

> **Tres bugs apilados impedían que esto funcionara en contenedor (corregidos 2026-07-07):**
> (1) faltaba `timescale_conn` → ahora durable vía `AIRFLOW_CONN_TIMESCALE_CONN` en compose;
> (2) `PROJECT_ROOT` ciego al contenedor + shadowing de `sys.path` en `l0_macro_backfill.py`;
> (3) la tarea de regeneración consultaba columnas inexistentes y escribía la forma equivocada.
> Si `/analysis` muestra "Sin datos" en los charts macro, el CLEAN parquet está viejo — corre
> esta recuperación o parchea con `scripts/ops/patch_analysis_macro_charts.py`.

### Modelos stale (>10 días)

```bash
airflow dags trigger forecast_h1_l3_weekly_training
airflow dags trigger forecast_h5_l3_weekly_training
ls -la outputs/forecasting/h5_weekly_models/latest/*.pkl
```

### News stale (>24 h)

```bash
docker exec usdcop-airflow-scheduler python -c "import feedparser" || \
  docker exec usdcop-airflow-scheduler python -m pip install feedparser
airflow dags trigger news_daily_pipeline
```

### Backup de seeds falló

```bash
airflow dags trigger core_l0_05_seed_backup
cat data/backups/seeds/backup_manifest.json
```

### Recuperación total (DB vacía tras reinicio)

```bash
docker-compose restart usdcop-postgres-timescale     # init-scripts auto-siembran
airflow dags trigger core_l0_01_ohlcv_backfill
airflow dags trigger core_l0_03_macro_backfill

# migraciones (en cold boot, init-scripts/26-restore-features.sh las aplica solo)
for m in 043 044 045 046 048 049 051 054; do
  docker exec -i usdcop-postgres-timescale psql -U admin -d usdcop_trading \
    < database/migrations/${m}_*.sql
done

# datos derivados (news/analysis/H5/asset_daily) — empty-table-only, nunca pisa datos
python -m scripts.ops.backup.feature_data_backup --mode restore --dir data/backups/features

airflow dags trigger forecast_h5_l3_weekly_training

# regenerar lo que sirve el dashboard
python scripts/pipeline/generate_weekly_forecasts.py --num-weeks 30
python scripts/pipeline/generate_asset_weekly_forecast.py --asset all --year all
python -m scripts.data.export_chart_ohlcv
python scripts/pipeline/train_and_export_smart_simple.py --phase both
```

> Estos outputs de `/forecasting` son regenerables y no están trackeados: un clone fresco muestra
> `/forecasting` vacío hasta correr el pipeline (o el stage `l5_weekly_forecast` del DAG por activo).

---

## Checklists de monitoreo

**Diario (Lun-Vie, tras cierre ~13:00 COT)**
- [ ] `SELECT MAX(time) FROM usdcop_m5_ohlcv WHERE symbol='USD/COP'` = hoy
- [ ] `SELECT MAX(fecha) FROM macro_indicators_daily` = hoy o ayer
- [ ] Pipeline de news completó sus 3 corridas

**Semanal (domingo noche / lunes)**
- [ ] H5-L3 training completó
- [ ] Señal H5-L5 generada el lunes
- [ ] `backup_manifest.json` fresco · MACRO_DAILY_CLEAN regenerado

---

## Tracking de migraciones

| Migración | Crea | Estado | Requerida para |
|-----------|------|--------|----------------|
| 043 | 5 tablas H5 + 2 vistas | Aplicada | Pipeline H5 |
| 044 | Columnas confidence + stops | Aplicada | Smart Simple v1.1 |
| 045 | 8 tablas News Engine | Aplicada | Pipeline de news |
| 046 | 4 tablas Analysis | Aplicada | Módulo de análisis |
| 049 | Columnas regime + DL | Aplicada | Smart Simple v2.0 |
| 051 | `asset_daily_ohlcv` + vista | Aplicada | Barras diarias Gold/BTC |
| 052 | 5 tablas crypto-native | Aditiva | Estrategia BTC |
| 054 | `UNIQUE(execution_id, subtrade_index)` | Aplicada | Upsert de subtrades en promoción |
| 055 | `entitlements` JSONB + `audit_log` | Aplicada | RBAC/monetización |
| 056 | `rbac_role_permissions` + overrides | Aplicada | Consola de roles dinámicos |

---

## Cross-References

| Concern | Doc |
|---------|-----|
| Umbrales de frescura (SSOT, auto-cargado) | `../../rules/data-freshness.md` |
| Schedule de DAGs (SSOT) | `elite-operations.md` |
| Backup y restore | `../data/backup-recovery.md` |
| Gobernanza L0 | `../../rules/data-governance.md` |
