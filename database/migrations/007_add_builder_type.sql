-- ============================================================================
-- 007_add_builder_type.sql
-- Add explicit builder_type column to model_registry
-- CLAUDE-T17 | Plan Item: P2-CONSISTENCY | Contrato: CTR-011
-- ============================================================================
--
-- PURPOSE:
-- Replace fragile string matching with explicit builder type registration.
-- This ensures models use the correct observation builder.
--
-- BEFORE (FRAGILE):
--   if "v1" in model_id.lower():
--       return ObservationBuilderLegacy()
--
-- AFTER (EXPLICIT):
--   SELECT builder_type FROM model_registry WHERE model_id = 'ppo_primary'
--   -> 'current_15dim'
-- ============================================================================

-- 1. Add builder_type column
ALTER TABLE model_registry
ADD COLUMN IF NOT EXISTS builder_type VARCHAR(20);

-- 2. Set constraint for valid builder types
DO $$
BEGIN
    -- Drop existing constraint if it exists
    IF EXISTS (
        SELECT 1 FROM information_schema.constraint_column_usage
        WHERE table_name = 'model_registry' AND constraint_name = 'valid_builder_type'
    ) THEN
        ALTER TABLE model_registry DROP CONSTRAINT valid_builder_type;
    END IF;
END $$;

ALTER TABLE model_registry
ADD CONSTRAINT valid_builder_type
CHECK (builder_type IN ('legacy_32dim', 'current_15dim'));

-- 3. Update existing records with correct builder types
UPDATE model_registry
SET builder_type = CASE
    WHEN observation_dim = 32 THEN 'legacy_32dim'
    WHEN observation_dim = 15 THEN 'current_15dim'
    ELSE 'current_15dim'  -- Default for new models
END
WHERE builder_type IS NULL;

-- 4. Make builder_type NOT NULL after setting defaults
ALTER TABLE model_registry
ALTER COLUMN builder_type SET NOT NULL;

-- 5. Set default for new records
ALTER TABLE model_registry
ALTER COLUMN builder_type SET DEFAULT 'current_15dim';

-- 6. Add norm_stats_path column for explicit path storage
ALTER TABLE model_registry
ADD COLUMN IF NOT EXISTS norm_stats_path TEXT;

-- 7. Update existing records with default norm_stats paths
UPDATE model_registry
SET norm_stats_path = CASE
    WHEN builder_type = 'legacy_32dim' THEN 'config/norm_stats_legacy.json'
    WHEN builder_type = 'current_15dim' THEN 'config/norm_stats.json'
    ELSE 'config/norm_stats.json'
END
WHERE norm_stats_path IS NULL;

-- 8. Create index for builder_type queries
CREATE INDEX IF NOT EXISTS idx_model_registry_builder_type
    ON model_registry(builder_type);

-- 9. Add comments for documentation
COMMENT ON COLUMN model_registry.builder_type IS
    'Explicit builder type (legacy_32dim or current_15dim) - NO string matching on model_id';

COMMENT ON COLUMN model_registry.norm_stats_path IS
    'Path to normalization statistics JSON file - required for feature consistency';

-- 10. Create function to validate builder configuration
CREATE OR REPLACE FUNCTION validate_builder_config(p_model_id VARCHAR)
RETURNS TABLE (
    is_valid BOOLEAN,
    builder_type VARCHAR,
    observation_dim INTEGER,
    norm_stats_path TEXT,
    message TEXT
) AS $$
DECLARE
    v_builder_type VARCHAR;
    v_observation_dim INTEGER;
    v_norm_stats_path TEXT;
    v_expected_dim INTEGER;
BEGIN
    -- Get model configuration
    SELECT mr.builder_type, mr.observation_dim, mr.norm_stats_path
    INTO v_builder_type, v_observation_dim, v_norm_stats_path
    FROM model_registry mr
    WHERE mr.model_id = p_model_id;

    IF NOT FOUND THEN
        RETURN QUERY SELECT
            false,
            NULL::VARCHAR,
            NULL::INTEGER,
            NULL::TEXT,
            format('Model %s not found in registry', p_model_id);
        RETURN;
    END IF;

    -- Validate builder_type matches observation_dim
    v_expected_dim := CASE
        WHEN v_builder_type = 'legacy_32dim' THEN 32
        WHEN v_builder_type = 'current_15dim' THEN 15
        ELSE NULL
    END;

    IF v_expected_dim IS NULL THEN
        RETURN QUERY SELECT
            false,
            v_builder_type,
            v_observation_dim,
            v_norm_stats_path,
            format('Unknown builder_type: %s', v_builder_type);
        RETURN;
    END IF;

    IF v_observation_dim != v_expected_dim THEN
        RETURN QUERY SELECT
            false,
            v_builder_type,
            v_observation_dim,
            v_norm_stats_path,
            format('Dimension mismatch: builder_type %s expects %s dims, but got %s',
                   v_builder_type, v_expected_dim, v_observation_dim);
        RETURN;
    END IF;

    IF v_norm_stats_path IS NULL THEN
        RETURN QUERY SELECT
            false,
            v_builder_type,
            v_observation_dim,
            v_norm_stats_path,
            'norm_stats_path is NULL - this is required for inference';
        RETURN;
    END IF;

    -- All validations passed
    RETURN QUERY SELECT
        true,
        v_builder_type,
        v_observation_dim,
        v_norm_stats_path,
        'Builder configuration is valid';
END;
$$ LANGUAGE plpgsql;

-- 11. Update active_models view to include builder_type
CREATE OR REPLACE VIEW active_models AS
SELECT
    model_id,
    model_version,
    model_hash,
    builder_type,
    observation_dim,
    norm_stats_path,
    action_space,
    deployed_at,
    test_sharpe,
    test_win_rate
FROM model_registry
WHERE status = 'deployed'
ORDER BY deployed_at DESC;

-- 12. Verification
DO $$
DECLARE
    column_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'model_registry'
          AND column_name = 'builder_type'
    ) INTO column_exists;

    IF column_exists THEN
        RAISE NOTICE '✓ builder_type column added successfully';
    ELSE
        RAISE WARNING '⚠ builder_type column NOT added';
    END IF;
END $$;

SELECT 'Migration 007_add_builder_type.sql completed' as status;
