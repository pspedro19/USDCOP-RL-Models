# USD/COP Market Data Backups

## 📊 Backup Information

This directory contains backups of USD/COP OHLCV (Open, High, Low, Close, Volume) market data from the TimescaleDB database.

### Current Backup

```
File: usdcop_m5_ohlcv_20251029_162859.csv.gz
Format: CSV (gzip compressed)
Records: 86,993 candles
Timeframe: 5 minutes
Date Range: 2020-01-02 to 2025-10-29
Size: 754 KB (compressed), ~13 MB (uncompressed)
```

---

## 🔧 Backup Process

Backups are created using PostgreSQL's COPY command to export all data from the TimescaleDB hypertable:

```bash
# Manual backup command
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading \
  -c "COPY (SELECT * FROM usdcop_m5_ohlcv ORDER BY time) TO STDOUT CSV HEADER" \
  > usdcop_m5_ohlcv_${BACKUP_DATE}.csv

# Compress the backup
gzip usdcop_m5_ohlcv_${BACKUP_DATE}.csv
```

---

## 🔄 Restoration

### Quick Restore

```bash
./restore_usdcop_data.sh usdcop_m5_ohlcv_20251029_162859.csv.gz
```

### Manual Restore (Advanced)

```bash
# 1. Extract backup
gunzip -c usdcop_m5_ohlcv_20251029_162859.csv.gz > temp_restore.csv

# 2. Copy to container
docker cp temp_restore.csv usdcop-postgres-timescale:/tmp/restore.csv

# 3. Restore data
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading << EOF
CREATE TEMP TABLE temp_restore (
    time TIMESTAMPTZ,
    symbol TEXT,
    open NUMERIC(12,6),
    high NUMERIC(12,6),
    low NUMERIC(12,6),
    close NUMERIC(12,6),
    volume BIGINT,
    source TEXT,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
);

\COPY temp_restore FROM '/tmp/restore.csv' CSV HEADER

INSERT INTO usdcop_m5_ohlcv
SELECT * FROM temp_restore
ON CONFLICT (time, symbol)
DO UPDATE SET
    open = EXCLUDED.open,
    high = EXCLUDED.high,
    low = EXCLUDED.low,
    close = EXCLUDED.close,
    volume = EXCLUDED.volume,
    updated_at = NOW();

DROP TABLE temp_restore;
EOF

# 4. Cleanup
rm temp_restore.csv
docker exec usdcop-postgres-timescale rm /tmp/restore.csv
```

---

## 📋 Data Schema

```sql
Table: usdcop_m5_ohlcv
Type: TimescaleDB Hypertable (partitioned by time)

Columns:
  time         TIMESTAMPTZ    PRIMARY KEY (with symbol)
  symbol       TEXT           PRIMARY KEY (with time)
  open         NUMERIC(12,6)  Opening price
  high         NUMERIC(12,6)  Highest price
  low          NUMERIC(12,6)  Lowest price
  close        NUMERIC(12,6)  Closing price
  volume       BIGINT         Trading volume
  source       TEXT           Data source (e.g., 'twelvedata')
  created_at   TIMESTAMPTZ    Record creation timestamp
  updated_at   TIMESTAMPTZ    Last update timestamp
```

---

## 🛡️ Safety Features

The restoration script includes:

- ✅ **Conflict Resolution:** Uses `ON CONFLICT DO UPDATE` to handle duplicate timestamps
- ✅ **Data Validation:** Verifies record counts before and after restoration
- ✅ **User Confirmation:** Requires explicit confirmation before proceeding
- ✅ **Error Handling:** Exits immediately on any error (`set -e`)
- ✅ **Cleanup:** Automatically removes temporary files

---

## 📅 Backup Schedule

**Recommended backup frequency:**
- **Daily:** Automated backup at 23:00 COT
- **Pre-deployment:** Before any system updates
- **On-demand:** When requested

**Retention policy:**
- Keep last 7 daily backups
- Keep last 4 weekly backups (Sunday)
- Keep last 12 monthly backups (1st of month)

---

## 🚨 Emergency Recovery

If the database is completely lost:

```bash
# 1. Ensure PostgreSQL container is running
docker-compose up -d postgres

# 2. Wait for database to be healthy
docker exec usdcop-postgres-timescale pg_isready -U admin

# 3. Restore from latest backup
./restore_usdcop_data.sh $(ls -t *.csv.gz | head -1)

# 4. Verify data
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading \
  -c "SELECT COUNT(*), MIN(time), MAX(time) FROM usdcop_m5_ohlcv;"
```

---

## 📊 Backup Verification

Check backup integrity:

```bash
# Count records in backup
gunzip -c usdcop_m5_ohlcv_20251029_162859.csv.gz | wc -l
# Expected: 86,994 lines (86,993 records + 1 header)

# Check date range in backup
gunzip -c usdcop_m5_ohlcv_20251029_162859.csv.gz | head -2 | tail -1
gunzip -c usdcop_m5_ohlcv_20251029_162859.csv.gz | tail -1

# Verify file integrity
gunzip -t usdcop_m5_ohlcv_20251029_162859.csv.gz
```

---

## 📝 Notes

- Backups use CSV format for portability and easy inspection
- gzip compression reduces file size by ~95% (13MB → 754KB)
- TimescaleDB hypertables require `SELECT *` in COPY command (not direct table COPY)
- Restoration preserves all original timestamps and metadata
- ON CONFLICT resolution ensures idempotent restoration (can run multiple times safely)

---

## 🔗 Related Documentation

- [TimescaleDB Documentation](https://docs.timescale.com/)
- [PostgreSQL COPY Command](https://www.postgresql.org/docs/current/sql-copy.html)
- [Project Architecture](/docs/ARCHITECTURE.md)
