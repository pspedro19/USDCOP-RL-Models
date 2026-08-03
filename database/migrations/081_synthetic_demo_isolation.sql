-- Migration 081 / BL-43
-- Synthetic investor demonstrations live in demo.* and are rejected from every
-- production configuration, execution, inference and performance relation.

CREATE SCHEMA IF NOT EXISTS demo;
REVOKE ALL ON SCHEMA demo FROM PUBLIC;

CREATE TABLE IF NOT EXISTS demo.synthetic_model (
    model_id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    algorithm VARCHAR(50) NOT NULL DEFAULT 'SYNTHETIC',
    version VARCHAR(50) NOT NULL DEFAULT 'DEMO',
    environment TEXT NOT NULL DEFAULT 'demo',
    surface TEXT NOT NULL DEFAULT 'synthetic',
    execution_eligible BOOLEAN NOT NULL DEFAULT FALSE,
    color VARCHAR(7),
    description TEXT,
    display_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT synthetic_demo_only CHECK (
        algorithm = 'SYNTHETIC'
        AND environment = 'demo'
        AND surface = 'synthetic'
        AND execution_eligible = FALSE
    )
);

CREATE OR REPLACE FUNCTION demo.reject_synthetic_fact()
RETURNS TRIGGER AS $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM demo.synthetic_model
        WHERE model_id = NEW.model_id
    ) THEN
        RAISE EXCEPTION
            'synthetic demo model % cannot enter %.%',
            NEW.model_id,
            TG_TABLE_SCHEMA,
            TG_TABLE_NAME;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
DECLARE
    leaked_facts BIGINT;
BEGIN
    IF to_regclass('config.models') IS NULL THEN
        RAISE EXCEPTION
            'config.models is required before synthetic isolation can run';
    END IF;

    -- Fail before moving anything if a synthetic row has already contaminated a
    -- real fact table.  That incident requires evidence-led remediation.
    IF to_regclass('metrics.model_performance') IS NOT NULL THEN
        EXECUTE
            'SELECT count(*) '
            'FROM metrics.model_performance p '
            'JOIN config.models m ON m.model_id = p.model_id '
            'WHERE m.algorithm = ''SYNTHETIC'''
        INTO leaked_facts;
        IF leaked_facts > 0 THEN
            RAISE EXCEPTION
                'synthetic contamination: % rows in metrics.model_performance',
                leaked_facts;
        END IF;
    END IF;

    IF to_regclass('trading.model_inferences') IS NOT NULL THEN
        EXECUTE
            'SELECT count(*) '
            'FROM trading.model_inferences f '
            'JOIN config.models m ON m.model_id = f.model_id '
            'WHERE m.algorithm = ''SYNTHETIC'''
        INTO leaked_facts;
        IF leaked_facts > 0 THEN
            RAISE EXCEPTION
                'synthetic contamination: % rows in trading.model_inferences',
                leaked_facts;
        END IF;
    END IF;

    IF to_regclass('trading.model_trades') IS NOT NULL THEN
        EXECUTE
            'SELECT count(*) '
            'FROM trading.model_trades f '
            'JOIN config.models m ON m.model_id = f.model_id '
            'WHERE m.algorithm = ''SYNTHETIC'''
        INTO leaked_facts;
        IF leaked_facts > 0 THEN
            RAISE EXCEPTION
                'synthetic contamination: % rows in trading.model_trades',
                leaked_facts;
        END IF;
    END IF;

    INSERT INTO demo.synthetic_model (
        model_id,
        name,
        algorithm,
        version,
        environment,
        surface,
        execution_eligible,
        color,
        description,
        display_metadata,
        created_at,
        updated_at
    )
    SELECT
        model_id,
        name,
        'SYNTHETIC',
        version,
        'demo',
        'synthetic',
        FALSE,
        color,
        description,
        jsonb_build_object(
            'badge', 'DEMO - SYNTHETIC',
            'source_backtest_metrics', COALESCE(backtest_metrics, '{}'::jsonb)
        ),
        created_at,
        updated_at
    FROM config.models
    WHERE algorithm = 'SYNTHETIC'
    ON CONFLICT (model_id) DO UPDATE
    SET name = EXCLUDED.name,
        version = EXCLUDED.version,
        color = EXCLUDED.color,
        description = EXCLUDED.description,
        display_metadata = EXCLUDED.display_metadata,
        updated_at = EXCLUDED.updated_at;

    DELETE FROM config.models
    WHERE algorithm = 'SYNTHETIC';

    ALTER TABLE config.models
        DROP CONSTRAINT IF EXISTS chk_real_model_not_synthetic;
    ALTER TABLE config.models
        ADD CONSTRAINT chk_real_model_not_synthetic
        CHECK (algorithm <> 'SYNTHETIC') NOT VALID;
    ALTER TABLE config.models
        VALIDATE CONSTRAINT chk_real_model_not_synthetic;
END;
$$;

DO $$
BEGIN
    IF to_regclass('metrics.model_performance') IS NOT NULL THEN
        EXECUTE
            'DROP TRIGGER IF EXISTS trg_no_synthetic_performance '
            'ON metrics.model_performance';
        EXECUTE
            'CREATE TRIGGER trg_no_synthetic_performance '
            'BEFORE INSERT OR UPDATE OF model_id '
            'ON metrics.model_performance '
            'FOR EACH ROW EXECUTE FUNCTION demo.reject_synthetic_fact()';
    END IF;
    IF to_regclass('trading.model_inferences') IS NOT NULL THEN
        EXECUTE
            'DROP TRIGGER IF EXISTS trg_no_synthetic_inference '
            'ON trading.model_inferences';
        EXECUTE
            'CREATE TRIGGER trg_no_synthetic_inference '
            'BEFORE INSERT OR UPDATE OF model_id '
            'ON trading.model_inferences '
            'FOR EACH ROW EXECUTE FUNCTION demo.reject_synthetic_fact()';
    END IF;
    IF to_regclass('trading.model_trades') IS NOT NULL THEN
        EXECUTE
            'DROP TRIGGER IF EXISTS trg_no_synthetic_trade '
            'ON trading.model_trades';
        EXECUTE
            'CREATE TRIGGER trg_no_synthetic_trade '
            'BEFORE INSERT OR UPDATE OF model_id '
            'ON trading.model_trades '
            'FOR EACH ROW EXECUTE FUNCTION demo.reject_synthetic_fact()';
    END IF;
END;
$$;

CREATE OR REPLACE VIEW demo.synthetic_model_display AS
SELECT
    model_id,
    '[DEMO - SYNTHETIC] ' || name AS display_name,
    algorithm,
    environment,
    surface,
    execution_eligible,
    color,
    description,
    display_metadata,
    updated_at
FROM demo.synthetic_model;

COMMENT ON TABLE demo.synthetic_model IS
    'Isolated presentation-only models; never eligible for real metrics or execution.';
COMMENT ON VIEW demo.synthetic_model_display IS
    'Unmistakably labelled demo surface; excluded from production model views.';
