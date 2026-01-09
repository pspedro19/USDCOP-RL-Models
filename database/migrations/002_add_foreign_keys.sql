-- =====================================================================
-- Migration 002: Add Foreign Keys
-- =====================================================================
--
-- This migration adds missing foreign key constraints for data integrity.
--
-- Changes:
--   1. Add trading_date column to dw.fact_rl_inference (generated)
--   2. Add FK constraint to macro_indicators_daily
--
-- Author: Pedro @ Lean Tech Solutions
-- Date: 2025-12-17
-- Version: 1.0
--
-- Prerequisites:
--   - Migration 001 completed
--   - Tables dw.fact_rl_inference and macro_indicators_daily exist
--
-- Rollback:
--   ALTER TABLE dw.fact_rl_inference DROP CONSTRAINT IF EXISTS fk_inference_trading_date;
--   ALTER TABLE dw.fact_rl_inference DROP COLUMN IF EXISTS trading_date;
-- =====================================================================

\echo 'Adding foreign key constraints...'

-- =====================================================================
-- STEP 1: Add trading_date column (computed from timestamp_cot)
-- =====================================================================

-- Only add if column doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'dw'
        AND table_name = 'fact_rl_inference'
        AND column_name = 'trading_date'
    ) THEN
        ALTER TABLE dw.fact_rl_inference
        ADD COLUMN trading_date DATE GENERATED ALWAYS AS
            (DATE(timestamp_cot AT TIME ZONE 'America/Bogota')) STORED;

        RAISE NOTICE 'Added trading_date column to dw.fact_rl_inference';
    ELSE
        RAISE NOTICE 'trading_date column already exists';
    END IF;
END$$;

-- =====================================================================
-- STEP 2: Add Foreign Key to macro_indicators_daily
-- =====================================================================

-- Only add if constraint doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'fk_inference_trading_date'
        AND table_schema = 'dw'
        AND table_name = 'fact_rl_inference'
    ) THEN
        -- Note: This may fail if there are orphan records
        -- In that case, run a cleanup query first:
        -- DELETE FROM dw.fact_rl_inference
        -- WHERE trading_date NOT IN (SELECT date FROM macro_indicators_daily);

        ALTER TABLE dw.fact_rl_inference
        ADD CONSTRAINT fk_inference_trading_date
        FOREIGN KEY (trading_date)
        REFERENCES macro_indicators_daily(date)
        ON DELETE RESTRICT;

        RAISE NOTICE 'Added FK constraint fk_inference_trading_date';
    ELSE
        RAISE NOTICE 'FK constraint fk_inference_trading_date already exists';
    END IF;
EXCEPTION
    WHEN foreign_key_violation THEN
        RAISE WARNING 'FK constraint failed - orphan records exist. Run cleanup first.';
END$$;

-- =====================================================================
-- STEP 3: Create index for FK performance
-- =====================================================================

CREATE INDEX IF NOT EXISTS idx_fact_rl_inference_trading_date
    ON dw.fact_rl_inference(trading_date);

-- =====================================================================
-- STEP 4: Verification
-- =====================================================================

\echo 'Verifying foreign key constraints...'

SELECT
    tc.constraint_name,
    tc.table_schema,
    tc.table_name,
    kcu.column_name,
    ccu.table_schema AS foreign_table_schema,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
    AND tc.table_schema = kcu.table_schema
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'
    AND tc.table_schema = 'dw'
    AND tc.table_name = 'fact_rl_inference';

-- =====================================================================
-- MIGRATION COMPLETE
-- =====================================================================

SELECT
    'Migration 002 completed successfully' AS status,
    NOW() AS completed_at;
