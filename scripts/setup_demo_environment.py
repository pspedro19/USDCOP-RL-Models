#!/usr/bin/env python3
"""
Setup Demo Environment Script
=============================
Este script:
1. Limpia la base de datos (opcional)
2. Inserta una propuesta L4 de prueba con PENDING_APPROVAL
3. Verifica que todo esté listo para la demo

USO:
    python scripts/setup_demo_environment.py --clean  # Limpia e inserta
    python scripts/setup_demo_environment.py          # Solo inserta propuesta
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)


def get_db_connection():
    """Get database connection from environment variables."""
    database_url = os.environ.get('DATABASE_URL') or os.environ.get('POSTGRES_URL')

    if not database_url:
        # Try to load from .env file
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith('DATABASE_URL='):
                        database_url = line.split('=', 1)[1].strip().strip('"\'')
                        break

    if not database_url:
        # Default for local development
        database_url = "postgresql://admin:admin123@localhost:5432/usdcop_trading"

    print(f"Connecting to database...")
    return psycopg2.connect(database_url)


def clean_database(conn):
    """Clean all model-related tables."""
    print("\n" + "="*60)
    print("CLEANING DATABASE")
    print("="*60)

    with conn.cursor() as cur:
        # Clean in order of dependencies
        tables = [
            ('approval_audit_log', 'DELETE FROM approval_audit_log'),
            ('promotion_proposals', 'DELETE FROM promotion_proposals'),
            ('model_registry', 'DELETE FROM model_registry'),
        ]

        for table_name, query in tables:
            try:
                cur.execute(query)
                print(f"  Cleaned {table_name}: {cur.rowcount} rows deleted")
            except Exception as e:
                print(f"  Warning: Could not clean {table_name}: {e}")

        # Ensure Investor Demo exists in config.models
        try:
            cur.execute("""
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
                    algorithm = 'SYNTHETIC'
            """)
            print("  Ensured Investor Demo exists in config.models")
        except Exception as e:
            print(f"  Warning: Could not ensure Investor Demo: {e}")

        conn.commit()

    print("Database cleaned successfully!")


def insert_l4_proposal(conn, model_name: str = None):
    """Insert a test L4 proposal with PENDING_APPROVAL status."""
    print("\n" + "="*60)
    print("INSERTING L4 PROPOSAL")
    print("="*60)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_id = f"ppo_ssot_{timestamp}"
    proposal_id = f"prop_{timestamp}"
    experiment_name = model_name or f"PPO SSOT {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    metrics = {
        "totalReturn": 0.1364,
        "sharpeRatio": 1.42,
        "maxDrawdown": 0.0766,
        "winRate": 0.714,
        "profitFactor": 1.89,
        "totalTrades": 99,
        "avgTradeReturn": 0.00138,
        "sortinoRatio": 2.15,
        "calmarRatio": 1.78
    }

    vs_baseline = {
        "returnDelta": 0.045,
        "sharpeDelta": 0.23,
        "drawdownDelta": -0.012,
        "winRateDelta": 0.05
    }

    criteria_results = [
        {"criterion": "Sharpe > 1.0", "passed": True, "value": 1.42, "threshold": 1.0, "weight": 0.25},
        {"criterion": "Max DD < 10%", "passed": True, "value": 0.0766, "threshold": 0.10, "weight": 0.25},
        {"criterion": "Win Rate > 50%", "passed": True, "value": 0.714, "threshold": 0.50, "weight": 0.20},
        {"criterion": "Min 50 Trades", "passed": True, "value": 99, "threshold": 50, "weight": 0.15},
        {"criterion": "Profit Factor > 1.5", "passed": True, "value": 1.89, "threshold": 1.5, "weight": 0.15}
    ]

    lineage = {
        "configHash": "abc123def456",
        "featureOrderHash": "789ghi012jkl",
        "modelHash": "mno345pqr678",
        "datasetHash": "stu901vwx234",
        "normStatsHash": "yza567bcd890",
        "rewardConfigHash": "efg123hij456",
        "modelPath": f"/models/{model_id}/model.zip",
        "trainingStart": "2025-06-01",
        "trainingEnd": "2025-12-31"
    }

    with conn.cursor() as cur:
        # Insert model in config.models
        try:
            cur.execute("""
                INSERT INTO config.models (model_id, name, algorithm, version, status, color, description, backtest_metrics)
                VALUES (%s, %s, 'PPO', 'V1', 'active', '#10B981', %s, %s)
                ON CONFLICT (model_id) DO NOTHING
            """, (
                model_id,
                experiment_name,
                f'PPO model trained with SSOT pipeline - awaiting approval',
                json.dumps(metrics)
            ))
            print(f"  Model inserted: {model_id}")
        except Exception as e:
            print(f"  Warning: Could not insert model: {e}")

        # Insert promotion proposal
        try:
            cur.execute("""
                INSERT INTO promotion_proposals (
                    proposal_id, model_id, experiment_name, recommendation, confidence,
                    reason, metrics, vs_baseline, criteria_results, lineage,
                    status, created_at, expires_at
                ) VALUES (%s, %s, %s, 'PROMOTE', 0.90, %s, %s, %s, %s, %s, 'PENDING_APPROVAL', NOW(), NOW() + INTERVAL '7 days')
            """, (
                proposal_id,
                model_id,
                experiment_name,
                'Model passed all L4 validation gates with excellent metrics.',
                json.dumps(metrics),
                json.dumps(vs_baseline),
                json.dumps(criteria_results),
                json.dumps(lineage)
            ))
            print(f"  Proposal inserted: {proposal_id}")
        except Exception as e:
            print(f"  ERROR inserting proposal: {e}")
            conn.rollback()
            return None, None

        conn.commit()

    print("\n" + "="*60)
    print("L4 PROPOSAL CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"  Model ID:      {model_id}")
    print(f"  Proposal ID:   {proposal_id}")
    print(f"  Status:        PENDING_APPROVAL")
    print(f"  Recommendation: PROMOTE")
    print(f"  Confidence:    90%")
    print()
    print("NEXT STEPS:")
    print("  1. Open Dashboard: http://localhost:3000/dashboard")
    print("  2. Select the model from dropdown")
    print("  3. See the floating approval panel at bottom")
    print("  4. Run backtest to verify metrics")
    print("  5. Click APROBAR or RECHAZAR")
    print("="*60)

    return model_id, proposal_id


def show_current_state(conn):
    """Show current state of the database."""
    print("\n" + "="*60)
    print("CURRENT DATABASE STATE")
    print("="*60)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Count records in each table
        tables_to_check = [
            ('config.models', 'SELECT COUNT(*) as count FROM config.models'),
            ('model_registry', 'SELECT COUNT(*) as count FROM model_registry'),
            ('promotion_proposals', 'SELECT COUNT(*) as count FROM promotion_proposals'),
            ('approval_audit_log', 'SELECT COUNT(*) as count FROM approval_audit_log'),
        ]

        print("\nRecord counts:")
        for table_name, query in tables_to_check:
            try:
                cur.execute(query)
                result = cur.fetchone()
                print(f"  {table_name}: {result['count']} records")
            except Exception as e:
                print(f"  {table_name}: ERROR - {e}")

        # Show pending proposals
        print("\nPending proposals:")
        try:
            cur.execute("""
                SELECT proposal_id, model_id, experiment_name, recommendation, confidence, status, created_at
                FROM promotion_proposals
                WHERE status = 'PENDING_APPROVAL'
                ORDER BY created_at DESC
                LIMIT 5
            """)
            proposals = cur.fetchall()
            if proposals:
                for p in proposals:
                    print(f"  - {p['model_id']}: {p['recommendation']} ({p['confidence']*100:.0f}% conf)")
            else:
                print("  (none)")
        except Exception as e:
            print(f"  ERROR: {e}")

        # Show models in config.models
        print("\nModels in config.models:")
        try:
            cur.execute("""
                SELECT model_id, name, algorithm, status
                FROM config.models
                WHERE status = 'active'
                ORDER BY model_id
            """)
            models = cur.fetchall()
            for m in models:
                algo_badge = "DEMO" if m['algorithm'] == 'SYNTHETIC' else m['algorithm']
                print(f"  - {m['model_id']}: {m['name']} [{algo_badge}]")
        except Exception as e:
            print(f"  ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description='Setup demo environment for L4 approval workflow')
    parser.add_argument('--clean', action='store_true', help='Clean database before inserting')
    parser.add_argument('--show', action='store_true', help='Only show current state, do not modify')
    parser.add_argument('--name', type=str, help='Custom experiment name')
    args = parser.parse_args()

    try:
        conn = get_db_connection()
        print("Connected to database!")

        if args.show:
            show_current_state(conn)
        else:
            if args.clean:
                clean_database(conn)

            insert_l4_proposal(conn, args.name)
            show_current_state(conn)

        conn.close()

    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
