-- ============================================================================
-- INSERT L4 PROPOSAL SCRIPT
-- ============================================================================
-- Este script inserta una propuesta de promoción simulando que L4 pasó
-- con recommendation = 'PROMOTE' y status = 'PENDING_APPROVAL'
--
-- USO: psql -U admin -d usdcop_trading -f scripts/insert_l4_proposal.sql
-- ============================================================================

BEGIN;

-- Variables para el modelo de prueba
-- Cambia estos valores según el modelo que quieras probar
DO $$
DECLARE
    v_model_id VARCHAR := 'ppo_ssot_' || TO_CHAR(NOW(), 'YYYYMMDD_HH24MISS');
    v_proposal_id VARCHAR := 'prop_' || TO_CHAR(NOW(), 'YYYYMMDD_HH24MISS');
    v_experiment_name VARCHAR := 'PPO SSOT ' || TO_CHAR(NOW(), 'YYYY-MM-DD');
BEGIN

    -- 1. Primero insertar el modelo en config.models si no existe
    INSERT INTO config.models (model_id, name, algorithm, version, status, color, description, backtest_metrics)
    VALUES (
        v_model_id,
        'PPO SSOT ' || TO_CHAR(NOW(), 'YYYY-MM-DD HH24:MI'),
        'PPO',
        'V1',
        'active',
        '#10B981',
        'PPO model trained with SSOT pipeline - awaiting approval',
        '{
            "sharpe_ratio": 1.42,
            "max_drawdown": 0.0766,
            "win_rate": 0.714,
            "total_return": 0.1364,
            "total_trades": 99,
            "profit_factor": 1.89
        }'::jsonb
    )
    ON CONFLICT (model_id) DO NOTHING;

    RAISE NOTICE 'Inserted model: %', v_model_id;

    -- 2. Insertar la propuesta de promoción con PENDING_APPROVAL
    INSERT INTO promotion_proposals (
        proposal_id,
        model_id,
        experiment_name,
        recommendation,
        confidence,
        reason,
        metrics,
        vs_baseline,
        criteria_results,
        lineage,
        status,
        created_at,
        expires_at
    ) VALUES (
        v_proposal_id,
        v_model_id,
        v_experiment_name,
        'PROMOTE',
        0.90,
        'Model passed all L4 validation gates with excellent metrics. Sharpe ratio above threshold, max drawdown within limits, and consistent win rate.',
        '{
            "totalReturn": 0.1364,
            "sharpeRatio": 1.42,
            "maxDrawdown": 0.0766,
            "winRate": 0.714,
            "profitFactor": 1.89,
            "totalTrades": 99,
            "avgTradeReturn": 0.00138,
            "sortinoRatio": 2.15,
            "calmarRatio": 1.78
        }'::jsonb,
        '{
            "returnDelta": 0.045,
            "sharpeDelta": 0.23,
            "drawdownDelta": -0.012,
            "winRateDelta": 0.05
        }'::jsonb,
        '[
            {"criterion": "Sharpe > 1.0", "passed": true, "value": 1.42, "threshold": 1.0, "weight": 0.25},
            {"criterion": "Max DD < 10%", "passed": true, "value": 0.0766, "threshold": 0.10, "weight": 0.25},
            {"criterion": "Win Rate > 50%", "passed": true, "value": 0.714, "threshold": 0.50, "weight": 0.20},
            {"criterion": "Min 50 Trades", "passed": true, "value": 99, "threshold": 50, "weight": 0.15},
            {"criterion": "Profit Factor > 1.5", "passed": true, "value": 1.89, "threshold": 1.5, "weight": 0.15}
        ]'::jsonb,
        '{
            "configHash": "abc123def456",
            "featureOrderHash": "789ghi012jkl",
            "modelHash": "mno345pqr678",
            "datasetHash": "stu901vwx234",
            "normStatsHash": "yza567bcd890",
            "rewardConfigHash": "efg123hij456",
            "modelPath": "/models/ppo_ssot/model.zip",
            "trainingStart": "2025-06-01",
            "trainingEnd": "2025-12-31"
        }'::jsonb,
        'PENDING_APPROVAL',
        NOW(),
        NOW() + INTERVAL '7 days'
    );

    RAISE NOTICE 'Inserted proposal: % for model: %', v_proposal_id, v_model_id;
    RAISE NOTICE '';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'L4 PROPOSAL CREATED SUCCESSFULLY';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Model ID: %', v_model_id;
    RAISE NOTICE 'Proposal ID: %', v_proposal_id;
    RAISE NOTICE 'Status: PENDING_APPROVAL';
    RAISE NOTICE 'Recommendation: PROMOTE';
    RAISE NOTICE '';
    RAISE NOTICE 'Next steps:';
    RAISE NOTICE '1. Go to Dashboard';
    RAISE NOTICE '2. Select the model from dropdown';
    RAISE NOTICE '3. See the floating approval panel';
    RAISE NOTICE '4. Run backtest to verify';
    RAISE NOTICE '5. Approve or Reject manually';
    RAISE NOTICE '========================================';

END $$;

COMMIT;

-- Verificación
SELECT
    pp.proposal_id,
    pp.model_id,
    pp.experiment_name,
    pp.recommendation,
    pp.confidence,
    pp.status,
    pp.created_at,
    pp.expires_at
FROM promotion_proposals pp
WHERE pp.status = 'PENDING_APPROVAL'
ORDER BY pp.created_at DESC
LIMIT 5;
