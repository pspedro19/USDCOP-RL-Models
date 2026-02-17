-- ============================================================================
-- DATABASE CLEANUP SCRIPT FOR FRESH DEMO
-- ============================================================================
-- Este script limpia todas las tablas relacionadas con modelos y propuestas
-- para empezar de cero con solo el "Investor Demo" visible.
--
-- USO: psql -U admin -d usdcop_trading -f scripts/db_cleanup_for_demo.sql
-- ============================================================================

BEGIN;

-- 1. Limpiar audit logs (depende de promotion_proposals)
DELETE FROM approval_audit_log;
RAISE NOTICE 'Deleted all approval_audit_log entries';

-- 2. Limpiar propuestas de promoción
DELETE FROM promotion_proposals;
RAISE NOTICE 'Deleted all promotion_proposals entries';

-- 3. Limpiar model_registry (mantener solo estructura)
DELETE FROM model_registry;
RAISE NOTICE 'Deleted all model_registry entries';

-- 4. Limpiar lineage records si existe
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'ml' AND table_name = 'lineage_records') THEN
        DELETE FROM ml.lineage_records;
        RAISE NOTICE 'Deleted all ml.lineage_records entries';
    END IF;
END $$;

-- 5. Limpiar model promotion audit si existe
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'ml' AND table_name = 'model_promotion_audit') THEN
        DELETE FROM ml.model_promotion_audit;
        RAISE NOTICE 'Deleted all ml.model_promotion_audit entries';
    END IF;
END $$;

-- 6. Verificar que config.models tiene el Investor Demo
-- Insertar si no existe
INSERT INTO config.models (model_id, name, algorithm, version, status, color, description, backtest_metrics)
VALUES (
    'investor_demo_v1',
    'Investor Demo',
    'SYNTHETIC',
    'V1',
    'active',
    '#F59E0B',
    'Modo demostración para visualizar el sistema sin modelo real',
    '{"sharpe_ratio": 1.5, "max_drawdown": 0.05, "win_rate": 0.65}'::jsonb
)
ON CONFLICT (model_id) DO UPDATE SET
    status = 'active',
    name = 'Investor Demo',
    algorithm = 'SYNTHETIC';

RAISE NOTICE 'Investor Demo model ensured in config.models';

COMMIT;

-- Verificación
SELECT 'config.models' as table_name, COUNT(*) as count FROM config.models
UNION ALL
SELECT 'model_registry', COUNT(*) FROM model_registry
UNION ALL
SELECT 'promotion_proposals', COUNT(*) FROM promotion_proposals
UNION ALL
SELECT 'approval_audit_log', COUNT(*) FROM approval_audit_log;
