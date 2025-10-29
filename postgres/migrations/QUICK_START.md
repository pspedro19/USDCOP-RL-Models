# Quick Start: OHLCV Migration

## TL;DR - Get Started in 3 Steps

```bash
# Step 1: Check current status
./scripts/check_migration_status.sh

# Step 2: Run migrations (includes automatic backup)
./scripts/run_migrations.sh

# Step 3: Verify completion
./scripts/check_migration_status.sh
```

## What This Does

Transforms your `market_data` table from:
```
timestamp | symbol | price | bid | ask | volume
```

To:
```
timestamp | symbol | timeframe | open | high | low | close | volume | bid | ask | spread | data_quality
```

## Expected Output

```
==========================================
USDCOP Database Migration Status Check
==========================================

Testing database connection...
✓ Connected to database

1. Table Structure
✓ All OHLCV columns exist - Migration 001 completed

2. Data Completeness
Total rows: 500000
Complete OHLC rows: 500000
Completeness: 100.00%

3. Index Status
Total indexes: 15
✓ Optimized indexes present - Migration 002 completed

4. Constraint Status
Total CHECK constraints: 11
✓ Validation constraints present - Migration 003 completed

Summary
Total Rows: 500000
Table Size: 185 MB
Indexes: 15
Constraints: 11

✓ MIGRATIONS APPLIED
```

## Time Estimates

| Your Data | Time Needed |
|-----------|-------------|
| < 100K rows | 2 minutes |
| 100K - 500K | 8 minutes |
| 500K - 1M | 13 minutes |
| > 1M rows | 30+ minutes |

## Rollback (If Needed)

```bash
./scripts/run_migrations.sh --rollback
```

This safely removes all OHLCV columns and restores the original structure.

## Need Help?

- Full documentation: `README.md` (in this directory)
- Complete summary: `/home/azureuser/USDCOP-RL-Models/MIGRATION_SUMMARY.md`
- Logs location: `/home/azureuser/USDCOP-RL-Models/logs/`

## Safety Features

✅ Automatic backup before migration
✅ Idempotent (safe to re-run)
✅ No data loss (original columns preserved)
✅ Complete logging
✅ Rollback available
✅ Validation at each step

## Migration Files

Located in `/home/azureuser/USDCOP-RL-Models/postgres/migrations/`:

1. `001_add_ohlcv_columns.sql` - Adds columns + backfill
2. `002_add_optimized_indexes.sql` - Creates 13 indexes
3. `003_add_constraints.sql` - Adds 11 validation constraints
4. `rollback_ohlcv.sql` - Reverts everything

## Common Issues

### "Cannot connect to database"
```bash
# Check if container is running
docker ps | grep postgres

# Check container logs
docker logs usdcop-postgres-timescale
```

### "Migration takes too long"
```bash
# Check table size first
./scripts/check_migration_status.sh

# For large tables (>1M rows), run during off-hours
```

### "Constraint violation"
Check logs in `/logs/migration_*.log` for specific errors and fix instructions.

---

**Ready to migrate?** Run: `./scripts/run_migrations.sh`
