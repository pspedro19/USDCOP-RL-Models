# Database Rollback Runbook
## USD/COP RL Trading System

**Contract**: PG-14
**Version**: 1.0.0
**Last Updated**: 2026-01-17

---

## Overview

This runbook provides procedures for rolling back PostgreSQL database changes including:

- Schema migrations (Alembic)
- Data corruption recovery
- TimescaleDB hypertable issues
- Point-in-time recovery

---

## Quick Reference

| Scenario | Recovery Time | Procedure |
|----------|---------------|-----------|
| Bad migration | 5-15 min | [Alembic Rollback](#alembic-migration-rollback) |
| Data corruption | 15-30 min | [Data Recovery](#data-corruption-recovery) |
| Full restore | 1-2 hours | [Full Restore](#full-database-restore) |

---

## Prerequisites

Before any rollback:

```bash
# 1. Verify database connectivity
pg_isready -h localhost -p 5432 -U ${POSTGRES_USER}

# 2. Check current migration version
cd database
alembic current

# 3. Create backup before rollback
pg_dump -h localhost -U ${POSTGRES_USER} ${POSTGRES_DB} > backup_$(date +%Y%m%d_%H%M%S).sql
```

---

## Alembic Migration Rollback

### Identify Current State

```bash
# List migration history
cd database
alembic history --verbose

# Check current head
alembic current

# List applied migrations
psql -h localhost -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "
SELECT version_num, applied_at
FROM alembic_version
ORDER BY applied_at DESC;
"
```

### Rollback One Migration

```bash
cd database

# Downgrade one step
alembic downgrade -1

# Verify
alembic current
```

### Rollback to Specific Version

```bash
cd database

# Downgrade to specific revision
alembic downgrade <revision_id>

# Example: rollback to initial state
alembic downgrade base

# Verify
alembic current
```

### Re-apply Migration After Fix

```bash
cd database

# After fixing the migration script
alembic upgrade head

# Verify
alembic current
```

---

## Data Corruption Recovery

### Symptoms
- Invalid data in tables
- Constraint violations
- Unexpected NULL values
- Duplicate records

### Recovery from Backup

```bash
# 1. Stop services that write to database
docker-compose stop airflow-scheduler

# 2. Identify last good backup
ls -la backups/*.sql

# 3. Create recovery database
psql -h localhost -U ${POSTGRES_USER} -c "CREATE DATABASE ${POSTGRES_DB}_recovery;"

# 4. Restore to recovery database
psql -h localhost -U ${POSTGRES_USER} -d ${POSTGRES_DB}_recovery < backups/backup_YYYYMMDD.sql

# 5. Verify data integrity
psql -h localhost -U ${POSTGRES_USER} -d ${POSTGRES_DB}_recovery -c "
SELECT COUNT(*) FROM usdcop_m5_ohlcv;
SELECT MAX(time) FROM usdcop_m5_ohlcv;
"

# 6. If valid, swap databases
psql -h localhost -U ${POSTGRES_USER} -c "
-- Disconnect all users
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '${POSTGRES_DB}';
-- Rename databases
ALTER DATABASE ${POSTGRES_DB} RENAME TO ${POSTGRES_DB}_corrupted;
ALTER DATABASE ${POSTGRES_DB}_recovery RENAME TO ${POSTGRES_DB};
"

# 7. Restart services
docker-compose start airflow-scheduler
```

### Selective Data Recovery

```bash
# Recover specific table from backup
pg_restore -h localhost -U ${POSTGRES_USER} -d ${POSTGRES_DB} \
    --table=usdcop_m5_ohlcv \
    --data-only \
    backup_YYYYMMDD.dump

# Or via SQL
psql -h localhost -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "
TRUNCATE usdcop_m5_ohlcv;
\copy usdcop_m5_ohlcv FROM 'backup_data.csv' WITH CSV HEADER;
"
```

---

## TimescaleDB Specific Recovery

### Hypertable Issues

```bash
# Check hypertable status
psql -h localhost -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "
SELECT hypertable_name, num_chunks, total_bytes
FROM timescaledb_information.hypertables;
"

# Check chunk health
psql -h localhost -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "
SELECT chunk_name, range_start, range_end, total_bytes
FROM timescaledb_information.chunks
WHERE hypertable_name = 'usdcop_m5_ohlcv'
ORDER BY range_start DESC
LIMIT 10;
"
```

### Recreate Hypertable

```bash
# If hypertable is corrupted
psql -h localhost -U ${POSTGRES_USER} -d ${POSTGRES_DB} <<EOF
-- Backup data to temp table
CREATE TABLE usdcop_m5_ohlcv_backup AS SELECT * FROM usdcop_m5_ohlcv;

-- Drop corrupted hypertable
DROP TABLE usdcop_m5_ohlcv;

-- Recreate table
CREATE TABLE usdcop_m5_ohlcv (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open NUMERIC(12,4),
    high NUMERIC(12,4),
    low NUMERIC(12,4),
    close NUMERIC(12,4),
    volume NUMERIC(20,2),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable
SELECT create_hypertable('usdcop_m5_ohlcv', 'time', chunk_time_interval => INTERVAL '7 days');

-- Restore data
INSERT INTO usdcop_m5_ohlcv SELECT * FROM usdcop_m5_ohlcv_backup;

-- Recreate indexes
CREATE INDEX idx_ohlcv_symbol_time ON usdcop_m5_ohlcv (symbol, time DESC);

-- Drop backup
DROP TABLE usdcop_m5_ohlcv_backup;

-- Re-enable compression
ALTER TABLE usdcop_m5_ohlcv SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);

-- Re-add retention policy
SELECT add_retention_policy('usdcop_m5_ohlcv', INTERVAL '365 days');
EOF
```

---

## Full Database Restore

### From pg_dump Backup

```bash
# 1. Stop all services
docker-compose down

# 2. Start only PostgreSQL
docker-compose up -d postgres

# 3. Wait for healthy
docker-compose ps postgres

# 4. Drop and recreate database
psql -h localhost -U ${POSTGRES_USER} -c "
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '${POSTGRES_DB}';
DROP DATABASE ${POSTGRES_DB};
CREATE DATABASE ${POSTGRES_DB};
"

# 5. Restore from backup
psql -h localhost -U ${POSTGRES_USER} -d ${POSTGRES_DB} < full_backup.sql

# 6. Verify
psql -h localhost -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "\dt"

# 7. Start all services
docker-compose up -d
```

### From MinIO Backup

```bash
# 1. List available backups
mc ls minio/99-common-trading-backups/

# 2. Download latest backup
mc cp minio/99-common-trading-backups/daily/backup_latest.sql ./

# 3. Follow restore procedure above
```

---

## Point-in-Time Recovery

### Using WAL Archives (if configured)

```bash
# 1. Stop PostgreSQL
docker-compose stop postgres

# 2. Create recovery.conf
cat > /var/lib/postgresql/data/recovery.conf <<EOF
restore_command = 'cp /wal_archive/%f %p'
recovery_target_time = '2026-01-17 10:00:00 UTC'
EOF

# 3. Start PostgreSQL
docker-compose start postgres

# 4. Monitor recovery
docker-compose logs -f postgres
```

---

## Verification Checklist

After any rollback:

- [ ] Database connectivity verified
- [ ] All tables exist (`\dt`)
- [ ] Row counts match expected values
- [ ] Indexes are present (`\di`)
- [ ] Constraints are valid
- [ ] Alembic version is correct
- [ ] Services can connect and query
- [ ] No error messages in logs

### Verification Script

```bash
#!/bin/bash
echo "=== Database Verification ==="

psql -h localhost -U ${POSTGRES_USER} -d ${POSTGRES_DB} <<EOF
-- Table counts
SELECT 'usdcop_m5_ohlcv' as table_name, COUNT(*) as row_count FROM usdcop_m5_ohlcv
UNION ALL
SELECT 'macro_indicators_daily', COUNT(*) FROM macro_indicators_daily
UNION ALL
SELECT 'model_registry', COUNT(*) FROM model_registry;

-- Latest data timestamps
SELECT 'OHLCV latest' as metric, MAX(time)::text as value FROM usdcop_m5_ohlcv
UNION ALL
SELECT 'Macro latest', MAX(fecha)::text FROM macro_indicators_daily;

-- Hypertable status
SELECT hypertable_name, num_chunks FROM timescaledb_information.hypertables;
EOF

echo "=== Verification Complete ==="
```

---

## Emergency Contacts

| Role | Responsibility |
|------|---------------|
| DBA On-Call | Database operations |
| DevOps | Infrastructure |
| Trading Team Lead | Business decisions |

---

*Document maintained by USDCOP Trading Team*
