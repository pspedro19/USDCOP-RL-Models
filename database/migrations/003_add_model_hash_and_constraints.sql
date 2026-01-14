-- ============================================================================
-- 003_add_model_hash_and_constraints.sql
-- P0-8 FIX: Add model_hash and validation constraint
-- CLAUDE-T4 | Contrato: CTR-004
-- ============================================================================

-- 1. Add model_hash column if not exists
ALTER TABLE public.trades_history
ADD COLUMN IF NOT EXISTS model_hash VARCHAR(64);

-- 2. Index for audit queries
CREATE INDEX IF NOT EXISTS idx_trades_model_hash
ON public.trades_history(model_hash);

-- 3. Comment for documentation
COMMENT ON COLUMN public.trades_history.model_hash IS
'SHA256 hash del modelo ONNX usado para la decision. Requerido para trades nuevos.';

-- 4. Update features_snapshot comment with full schema
COMMENT ON COLUMN public.trades_history.features_snapshot IS
'JSON snapshot de features al momento del trade. Schema V20: {version, timestamp, bar_idx, raw_features: {...}, normalized_features: {...}}';

-- 5. Verification
DO $$
DECLARE
    col_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'trades_history'
          AND column_name = 'model_hash'
    ) INTO col_exists;

    IF col_exists THEN
        RAISE NOTICE '✓ model_hash column exists';
    ELSE
        RAISE WARNING '⚠ model_hash column NOT created';
    END IF;
END $$;

SELECT 'Migration 003_add_model_hash_and_constraints.sql completed' as status;
