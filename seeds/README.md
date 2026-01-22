# Seeds - Database Initialization Data

Los datos de seed se almacenan en **MinIO** (S3-compatible), no en git.

## Ubicación de Seeds

```
MinIO Bucket: seeds
├── LATEST_VERSION          # Versión actual (ej: "2026.01.22")
├── latest/                 # Siempre apunta a la versión más reciente
│   ├── usdcop_m5_ohlcv.parquet
│   ├── macro_indicators_daily.parquet
│   └── manifest.json
└── v2026.01.22/           # Snapshots versionados
    ├── usdcop_m5_ohlcv.parquet
    ├── macro_indicators_daily.parquet
    └── manifest.json
```

## Cómo Funcionan los Seeds

### Al hacer `docker-compose up`:

1. PostgreSQL ejecuta `init-scripts/00-03*.sql` (schema)
2. El servicio `data-seeder` ejecuta `04-seed-from-minio.py`:
   - **Prioridad 1**: MinIO `seeds/latest/`
   - **Prioridad 2**: Local `seeds/latest/` (si existe)
   - **Prioridad 3**: Legacy `data/backups/*.csv.gz`

### Backup Automático:

El DAG `l0_seed_backup` crea snapshots semanales:
- Schedule: Domingos 02:00 UTC
- Retención: 4 snapshots (1 mes)

## Comandos Útiles

```bash
# Ver contenido de MinIO
docker exec usdcop-airflow-scheduler python3 -c "
from minio import Minio
client = Minio('minio:9000', 'minioadmin', 'minioadmin123', secure=False)
for obj in client.list_objects('seeds', recursive=True):
    print(f'{obj.object_name}: {obj.size/1024:.1f} KB')
"

# Crear snapshot manual
docker exec usdcop-airflow-scheduler python3 /opt/airflow/scripts/create_snapshot.py \
  --version $(date +%Y.%m.%d) --upload-to-minio

# Trigger backup DAG
docker exec usdcop-airflow-scheduler airflow dags trigger l0_seed_backup
```

## Datos Actuales

| Tabla | Registros | Rango |
|-------|-----------|-------|
| `usdcop_m5_ohlcv` | 91,157 | 2020-01-02 → 2026-01-15 |
| `macro_indicators_daily` | 10,759 | 1954-07-31 → 2026-01-22 |
