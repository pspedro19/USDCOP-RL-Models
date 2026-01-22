# Plan de Backup/Restore v2.0 - Sistema Reproducible

## Estado Actual del Proyecto

### Tablas Críticas (DEBEN respaldarse)

| Tabla | Registros | Rango | Prioridad |
|-------|-----------|-------|-----------|
| `usdcop_m5_ohlcv` | 91,157 | 2020-01-02 → 2026-01-15 | **P0 - CRÍTICA** |
| `macro_indicators_daily` | 10,759 | 1954-07-31 → 2026-01-22 | **P0 - CRÍTICA** |
| `trades_history` | 636 | - | P1 - Historial |

### Problema Actual
- Backups dispersos en múltiples ubicaciones
- Rutas hardcodeadas que pueden no existir en nuevo servidor
- Sin versionado claro de snapshots
- gitignore puede excluir datos importantes

---

## Arquitectura Propuesta v2.0

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NUEVA ARQUITECTURA DE DATOS                       │
└─────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────┐
                    │   MinIO (S3-like)   │
                    │   bucket: seeds     │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │ ohlcv/       │   │ macro/       │   │ models/      │
    │ v2026.01.22/ │   │ v2026.01.22/ │   │ ppo_v20/     │
    │ data.parquet │   │ data.parquet │   │ model.onnx   │
    └──────────────┘   └──────────────┘   └──────────────┘
           │                   │                   │
           └───────────────────┼───────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   DAG: l0_backup    │
                    │   (Weekly + Manual) │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Init Script       │
                    │   04-seed-data.py   │
                    │   (Lee de MinIO)    │
                    └─────────────────────┘
```

---

## Plan de Implementación

### Fase 1: Snapshot Inmediato (AHORA)

#### 1.1 Crear snapshot de tablas críticas

```bash
# Ejecutar en Docker
docker exec usdcop-airflow-scheduler python /opt/airflow/scripts/create_snapshot.py
```

**Archivos generados:**
```
seeds/
├── v2026.01.22/
│   ├── usdcop_m5_ohlcv.parquet      # ~15 MB comprimido
│   ├── macro_indicators_daily.parquet # ~2 MB comprimido
│   ├── trades_history.parquet        # ~50 KB
│   ├── manifest.json                 # Metadata del snapshot
│   └── checksums.sha256              # Verificación integridad
```

#### 1.2 Subir a MinIO

```bash
# Bucket: seeds (versionado)
mc cp -r seeds/v2026.01.22/ minio/seeds/v2026.01.22/
```

### Fase 2: Actualizar Init Scripts

#### 2.1 Nuevo script de seeding: `init-scripts/04-seed-from-minio.py`

```python
"""
Seed data from MinIO (S3-compatible storage)

Priority:
1. MinIO bucket: seeds/latest/
2. Fallback: Local files in data/backups/
3. Fallback: Empty tables (DAG backfill later)
"""

SEED_SOURCES = [
    {
        'table': 'usdcop_m5_ohlcv',
        'minio_path': 'seeds/latest/usdcop_m5_ohlcv.parquet',
        'fallback_path': 'data/backups/usdcop_m5_ohlcv_*.csv.gz',
        'required': True,
    },
    {
        'table': 'macro_indicators_daily',
        'minio_path': 'seeds/latest/macro_indicators_daily.parquet',
        'fallback_path': 'data/pipeline/05_resampling/output/MACRO_DAILY_CONSOLIDATED.csv',
        'required': True,
    },
    {
        'table': 'trades_history',
        'minio_path': 'seeds/latest/trades_history.parquet',
        'fallback_path': None,
        'required': False,
    },
]
```

### Fase 3: DAG de Backup Automatizado

#### 3.1 Nuevo DAG: `l0_seed_backup.py`

```python
"""
DAG: l0_seed_backup
===================
Backup semanal de tablas críticas a MinIO.

Schedule: Domingos 02:00 UTC
Retention: 4 snapshots (1 mes)
"""

# Tasks:
# 1. export_ohlcv_to_parquet
# 2. export_macro_to_parquet
# 3. export_trades_to_parquet
# 4. create_manifest
# 5. upload_to_minio
# 6. update_latest_symlink
# 7. cleanup_old_snapshots
```

### Fase 4: Configuración de Rutas

#### 4.1 Nuevo archivo: `config/seed_config.yaml`

```yaml
# Seed Data Configuration
# =======================
# Single Source of Truth para rutas de datos seed

version: "2.0"

# MinIO/S3 Configuration
storage:
  type: minio  # minio | s3 | local
  endpoint: ${MINIO_ENDPOINT:-minio:9000}
  bucket: seeds
  access_key: ${MINIO_ACCESS_KEY}
  secret_key: ${MINIO_SECRET_KEY}
  use_ssl: false

# Seed Data Sources (in priority order)
sources:
  ohlcv:
    table: usdcop_m5_ohlcv
    primary: s3://seeds/latest/usdcop_m5_ohlcv.parquet
    fallback:
      - data/backups/usdcop_m5_ohlcv_*.csv.gz
      - data/backups/usdcop_backup.sql.gz
    required: true
    min_rows: 50000  # Validation threshold

  macro:
    table: macro_indicators_daily
    primary: s3://seeds/latest/macro_indicators_daily.parquet
    fallback:
      - data/pipeline/05_resampling/output/MACRO_DAILY_CONSOLIDATED.csv
    required: true
    min_rows: 5000

  trades:
    table: trades_history
    primary: s3://seeds/latest/trades_history.parquet
    fallback: null
    required: false

# Backup Configuration
backup:
  schedule: "0 2 * * 0"  # Sundays 02:00 UTC
  retention_count: 4     # Keep 4 weekly snapshots
  format: parquet        # parquet | csv.gz
  compression: snappy    # snappy | gzip | none

# Validation
validation:
  check_date_ranges: true
  check_row_counts: true
  check_null_ratios: true
  max_null_ratio: 0.1
```

### Fase 5: Actualizar .gitignore y DVC

#### 5.1 Modificar `.gitignore`

```gitignore
# === SEEDS (TRACKED via DVC) ===
# Los seeds se trackean con DVC, no con git
seeds/

# === BACKUPS (LOCAL ONLY) ===
# Backups locales no se suben a git
data/backups/*.csv.gz
data/backups/*.sql.gz

# === PERO INCLUIR ===
# Manifest de seeds (para saber qué versión usar)
!seeds/manifest.json
!seeds/LATEST_VERSION
```

#### 5.2 Configurar DVC para seeds

```bash
# Track seeds con DVC (almacena en MinIO)
dvc add seeds/
dvc push

# En otro servidor:
dvc pull  # Descarga seeds desde MinIO
```

---

## Flujo de Despliegue en Nuevo Servidor

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT FLOW (NUEVO SERVIDOR)                  │
└─────────────────────────────────────────────────────────────────────┘

1. git clone https://github.com/user/USDCOP-RL-Models.git
   │
2. dvc pull                    ← Descarga seeds desde MinIO remoto
   │
3. cp .env.example .env        ← Configurar credenciales
   │
4. docker-compose up -d
   │
   ├─→ PostgreSQL inicia
   │   └─→ init-scripts/00-*.sql ejecutan (schema)
   │
   ├─→ data-seeder inicia
   │   └─→ 04-seed-from-minio.py ejecuta:
   │       1. Intenta MinIO → seeds/latest/
   │       2. Fallback → data/backups/ (si DVC pull funcionó)
   │       3. Fallback → CSV locales
   │       4. Valida row counts
   │
   ├─→ Airflow inicia
   │   └─→ l0_ohlcv_backfill triggered (actualiza a fecha actual)
   │
5. Sistema listo con datos históricos + actualización automática
```

---

## Archivos a Crear/Modificar

### Nuevos Archivos

| Archivo | Propósito |
|---------|-----------|
| `config/seed_config.yaml` | Configuración centralizada de seeds |
| `scripts/create_snapshot.py` | Crear snapshot de tablas críticas |
| `scripts/restore_from_snapshot.py` | Restaurar desde snapshot |
| `init-scripts/04-seed-from-minio.py` | Seeder con fallback inteligente |
| `airflow/dags/l0_seed_backup.py` | DAG de backup semanal |
| `seeds/manifest.json` | Metadata del snapshot actual |

### Archivos a Modificar

| Archivo | Cambio |
|---------|--------|
| `.gitignore` | Excluir seeds/ pero incluir manifest |
| `.dvc/config` | Agregar remote para seeds |
| `docker-compose.yml` | Actualizar data-seeder |
| `init-scripts/04-data-seeding.py` | Deprecar → usar nuevo |

---

## Comandos de Operación

### Crear Snapshot Manual

```bash
# Desde el host
docker exec usdcop-airflow-scheduler \
  python /opt/airflow/scripts/create_snapshot.py \
  --version $(date +%Y.%m.%d) \
  --upload-to-minio

# Output:
# ✓ Exported usdcop_m5_ohlcv: 91,157 rows → 14.2 MB
# ✓ Exported macro_indicators_daily: 10,759 rows → 1.8 MB
# ✓ Exported trades_history: 636 rows → 48 KB
# ✓ Created manifest.json
# ✓ Uploaded to minio://seeds/v2026.01.22/
# ✓ Updated seeds/latest symlink
```

### Restaurar en Nuevo Servidor

```bash
# Opción 1: Desde MinIO (recomendado)
docker exec usdcop-data-seeder \
  python /app/restore_from_snapshot.py \
  --source minio \
  --version latest

# Opción 2: Desde DVC
dvc pull seeds/
docker exec usdcop-data-seeder \
  python /app/restore_from_snapshot.py \
  --source local \
  --path /app/seeds/latest/
```

### Verificar Estado

```bash
# Ver snapshots disponibles
mc ls minio/seeds/

# Ver manifest del último snapshot
mc cat minio/seeds/latest/manifest.json

# Verificar integridad
docker exec usdcop-airflow-scheduler \
  python /opt/airflow/scripts/verify_seed_integrity.py
```

---

## Manifest de Snapshot

```json
{
  "version": "2026.01.22",
  "created_at": "2026-01-22T16:30:00Z",
  "created_by": "l0_seed_backup DAG",
  "tables": {
    "usdcop_m5_ohlcv": {
      "file": "usdcop_m5_ohlcv.parquet",
      "rows": 91157,
      "date_range": ["2020-01-02", "2026-01-15"],
      "size_bytes": 14892544,
      "sha256": "abc123..."
    },
    "macro_indicators_daily": {
      "file": "macro_indicators_daily.parquet",
      "rows": 10759,
      "date_range": ["1954-07-31", "2026-01-22"],
      "size_bytes": 1887232,
      "sha256": "def456..."
    },
    "trades_history": {
      "file": "trades_history.parquet",
      "rows": 636,
      "date_range": null,
      "size_bytes": 49152,
      "sha256": "ghi789..."
    }
  },
  "config_snapshot": {
    "feature_config": "sha256:xxx",
    "norm_stats": "sha256:yyy"
  },
  "compatibility": {
    "min_schema_version": "2.0",
    "postgres_version": "15",
    "timescaledb_version": "2.x"
  }
}
```

---

## Checklist de Implementación

### Fase 1: Snapshot Inmediato
- [ ] Crear `scripts/create_snapshot.py`
- [ ] Ejecutar snapshot de tablas actuales
- [ ] Subir a MinIO bucket `seeds`
- [ ] Verificar integridad con checksums

### Fase 2: Init Scripts
- [ ] Crear `config/seed_config.yaml`
- [ ] Crear `init-scripts/04-seed-from-minio.py`
- [ ] Deprecar `init-scripts/04-data-seeding.py` (mover a fallback)
- [ ] Actualizar `docker-compose.yml`

### Fase 3: DAG de Backup
- [ ] Crear `airflow/dags/l0_seed_backup.py`
- [ ] Configurar schedule semanal
- [ ] Implementar retention policy

### Fase 4: DVC Integration
- [ ] Actualizar `.gitignore`
- [ ] Configurar DVC remote para seeds
- [ ] Documentar flujo de pull/push

### Fase 5: Testing
- [ ] Test deployment en servidor limpio
- [ ] Verificar fallback chain funciona
- [ ] Validar datos restaurados vs originales

---

## Beneficios de la Nueva Arquitectura

| Aspecto | Antes | Después |
|---------|-------|---------|
| **Ubicación de backups** | Dispersos en múltiples carpetas | Centralizados en MinIO |
| **Versionado** | Por fecha de archivo | Semántico con manifest |
| **Reproducibilidad** | Manual, propenso a errores | Automatizado con DAG |
| **Portabilidad** | Depende de rutas locales | S3-compatible, cualquier servidor |
| **Validación** | Ninguna | Row counts, checksums, date ranges |
| **Fallback** | Sin fallback claro | Chain: MinIO → DVC → Local |
| **gitignore** | Puede excluir datos críticos | Solo manifest en git, datos en DVC |
