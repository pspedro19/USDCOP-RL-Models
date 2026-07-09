-- =============================================================================
-- 054_h5_subtrades_unique.sql
-- =============================================================================
-- Adds the natural-key UNIQUE(execution_id, subtrade_index) constraint on
-- forecast_h5_subtrades. The H5 production seeding (train_and_export_smart_simple.py
-- seed_h5_db_tables) upserts subtrades with
--     INSERT ... ON CONFLICT (execution_id, subtrade_index) DO UPDATE ...
-- which REQUIRES a matching unique/exclusion constraint. Migrations 043/050 created
-- the table with only PK(id) + FK(execution_id), so the ON CONFLICT failed with
-- "there is no unique or exclusion constraint matching the ON CONFLICT specification",
-- rolling back the entire seed transaction (all 5 H5 tables stayed empty on deploy).
--
-- Idempotent: safe to re-run / apply on cold boot (picked up by 26-restore-features.sh
-- which globs 05[0-9]_*.sql). A duplicate subtrade (same execution + index) is a data
-- error, so any pre-existing dupes are surfaced rather than silently kept.
--
-- Applied: 2026-07-05
-- =============================================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'uq_h5_subtrade'
          AND conrelid = 'forecast_h5_subtrades'::regclass
    ) THEN
        ALTER TABLE forecast_h5_subtrades
            ADD CONSTRAINT uq_h5_subtrade UNIQUE (execution_id, subtrade_index);
        RAISE NOTICE '054: added uq_h5_subtrade UNIQUE(execution_id, subtrade_index)';
    ELSE
        RAISE NOTICE '054: uq_h5_subtrade already present, skipping';
    END IF;
EXCEPTION
    WHEN undefined_table THEN
        RAISE NOTICE '054: forecast_h5_subtrades absent (schema not yet created), skipping';
END $$;
