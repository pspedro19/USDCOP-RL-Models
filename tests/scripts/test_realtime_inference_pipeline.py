#!/usr/bin/env python3
"""
Test Script: Real-time Inference Pipeline
==========================================

Script para probar el sistema de inferencia en tiempo real:
1. Ejecutar script SQL para crear tablas
2. Insertar datos de prueba
3. Simular inferencia del modelo
4. Verificar almacenamiento en DB
5. Probar API endpoints

Uso:
    python scripts/test_realtime_inference_pipeline.py --all
    python scripts/test_realtime_inference_pipeline.py --db-only
    python scripts/test_realtime_inference_pipeline.py --simulate-trading
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import random

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import psycopg2
from psycopg2.extras import RealDictCursor
import requests

# Configuration
DB_CONFIG = {
    'host': os.environ.get('POSTGRES_HOST', 'localhost'),
    'port': os.environ.get('POSTGRES_PORT', '5432'),
    'database': os.environ.get('POSTGRES_DB', 'usdcop_trading'),
    'user': os.environ.get('POSTGRES_USER', 'admin'),
    'password': os.environ.get('POSTGRES_PASSWORD', 'admin123'),
}

API_BASE_URL = os.environ.get('API_BASE_URL', 'http://localhost:3000')


def get_connection():
    """Get database connection"""
    return psycopg2.connect(**DB_CONFIG)


def test_database_connection():
    """Test database connectivity"""
    print("\n" + "=" * 60)
    print("TEST 1: Database Connection")
    print("=" * 60)

    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT version()")
        version = cur.fetchone()[0]
        print(f"[OK] Connected to PostgreSQL")
        print(f"    Version: {version[:50]}...")

        # Check TimescaleDB
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
        ts_version = cur.fetchone()
        if ts_version:
            print(f"[OK] TimescaleDB: v{ts_version[0]}")
        else:
            print("[WARN] TimescaleDB not installed")

        cur.close()
        conn.close()
        return True

    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        return False


def execute_sql_script():
    """Execute the SQL script to create tables"""
    print("\n" + "=" * 60)
    print("TEST 2: Execute SQL Script")
    print("=" * 60)

    sql_path = PROJECT_ROOT / 'init-scripts' / '11-realtime-inference-tables.sql'

    if not sql_path.exists():
        print(f"[ERROR] SQL script not found: {sql_path}")
        return False

    try:
        conn = get_connection()
        cur = conn.cursor()

        # Read and execute script
        with open(sql_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()

        # Split by statements and execute
        # Note: This is a simplified approach; for complex scripts use psql directly
        cur.execute(sql_content)
        conn.commit()

        print(f"[OK] SQL script executed successfully")

        # Verify tables created
        tables_to_check = [
            'dw.fact_rl_inference',
            'dw.fact_agent_actions',
            'dw.fact_session_performance',
            'dw.fact_equity_curve_realtime',
            'dw.fact_macro_realtime',
            'dw.fact_inference_alerts',
        ]

        for table in tables_to_check:
            schema, name = table.split('.')
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = %s
                )
            """, (schema, name))
            exists = cur.fetchone()[0]
            status = "[OK]" if exists else "[MISSING]"
            print(f"    {status} {table}")

        cur.close()
        conn.close()
        return True

    except Exception as e:
        print(f"[ERROR] SQL execution failed: {e}")
        return False


def insert_test_data():
    """Insert test data for simulation"""
    print("\n" + "=" * 60)
    print("TEST 3: Insert Test Data")
    print("=" * 60)

    try:
        conn = get_connection()
        cur = conn.cursor()

        now = datetime.utcnow()
        cot_now = now - timedelta(hours=5)
        session_date = cot_now.date()

        # Insert test macro data
        print("Inserting macro data...")
        cur.execute("""
            INSERT INTO dw.fact_macro_realtime (
                timestamp_utc, scrape_run_id,
                dxy, dxy_z, dxy_change_1d,
                vix, vix_z, vix_regime,
                embi, embi_z,
                brent, brent_change_1d,
                rate_spread,
                source, values_changed
            ) VALUES (
                %s, %s,
                103.5, 0.1, 0.002,
                18.5, -0.15, 1,
                285.0, -0.15,
                78.5, 0.005,
                -0.25,
                'test_script', 10
            ) RETURNING macro_id
        """, (now, 'test_run_001'))

        macro_id = cur.fetchone()[0]
        print(f"    [OK] Macro data inserted: ID={macro_id}")

        # Insert test inference
        print("Inserting inference data...")
        test_observation = [0.0] * 20  # 20 normalized features

        cur.execute("""
            INSERT INTO dw.fact_rl_inference (
                timestamp_utc, timestamp_cot,
                model_id, model_version, fold_id,
                observation,
                action_raw, action_discretized, confidence,
                close_price, raw_return_5m,
                position_before, position_after,
                portfolio_value_before, portfolio_value_after,
                latency_ms, inference_source
            ) VALUES (
                %s, %s,
                'ppo_usdcop_v14_fold0', 'v11.2', 0,
                %s,
                0.65, 'LONG', 0.65,
                4250.50, 0.0012,
                0.0, 0.65,
                10000.00, 10012.00,
                45, 'test_script'
            ) RETURNING inference_id
        """, (now, cot_now, test_observation))

        inference_id = cur.fetchone()[0]
        print(f"    [OK] Inference inserted: ID={inference_id}")

        # Insert test action
        print("Inserting agent action...")
        cur.execute("""
            INSERT INTO dw.fact_agent_actions (
                timestamp_utc, timestamp_cot,
                session_date, bar_number,
                action_type, side,
                price_at_action,
                position_before, position_after,
                equity_before, equity_after,
                pnl_action, pnl_daily,
                model_confidence, model_id,
                inference_id
            ) VALUES (
                %s, %s,
                %s, 15,
                'ENTRY_LONG', 'LONG',
                4250.50,
                0.0, 0.65,
                10000.00, 10012.00,
                12.00, 12.00,
                0.65, 'ppo_usdcop_v14_fold0',
                %s
            ) RETURNING action_id
        """, (now, cot_now, session_date, inference_id))

        action_id = cur.fetchone()[0]
        print(f"    [OK] Agent action inserted: ID={action_id}")

        # Insert equity curve point
        print("Inserting equity curve point...")
        cur.execute("""
            INSERT INTO dw.fact_equity_curve_realtime (
                timestamp_utc, timestamp_cot,
                session_date, bar_number,
                equity_value, log_equity,
                return_daily_pct, current_drawdown_pct,
                current_position, position_side,
                market_price, model_id
            ) VALUES (
                %s, %s,
                %s, 15,
                10012.00, 9.2115,
                0.12, 0.0,
                0.65, 'LONG',
                4250.50, 'ppo_usdcop_v14_fold0'
            )
        """, (now, cot_now, session_date))

        print(f"    [OK] Equity curve point inserted")

        # Update session performance
        print("Updating session performance...")
        cur.execute("SELECT dw.update_session_metrics(%s)", (session_date,))

        conn.commit()
        print(f"    [OK] Session metrics updated")

        cur.close()
        conn.close()
        return True

    except Exception as e:
        print(f"[ERROR] Test data insertion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def simulate_trading_session(num_bars: int = 10):
    """Simulate a trading session with multiple bars"""
    print("\n" + "=" * 60)
    print(f"TEST 4: Simulate Trading Session ({num_bars} bars)")
    print("=" * 60)

    try:
        conn = get_connection()
        cur = conn.cursor()

        now = datetime.utcnow()
        cot_now = now - timedelta(hours=5)
        session_date = cot_now.date()

        # Starting state
        position = 0.0
        equity = 10000.0
        daily_pnl = 0.0
        price = 4250.0

        action_types = ['HOLD', 'ENTRY_LONG', 'ENTRY_SHORT', 'EXIT_LONG', 'EXIT_SHORT',
                        'INCREASE_LONG', 'INCREASE_SHORT']

        print(f"Starting simulation: equity=${equity:.2f}, position={position:.2f}")
        print("-" * 60)

        for bar in range(1, num_bars + 1):
            # Simulate price movement
            price_change = random.uniform(-0.002, 0.002)
            price = price * (1 + price_change)

            # Simulate model action
            action_raw = random.uniform(-1, 1)
            if abs(action_raw - position) < 0.1:
                action_type = 'HOLD'
                new_position = position
            elif action_raw > 0.3 and position <= 0:
                action_type = 'ENTRY_LONG'
                new_position = action_raw
            elif action_raw < -0.3 and position >= 0:
                action_type = 'ENTRY_SHORT'
                new_position = action_raw
            elif action_raw > position + 0.1:
                action_type = 'INCREASE_LONG' if position > 0 else 'ENTRY_LONG'
                new_position = action_raw
            elif action_raw < position - 0.1:
                action_type = 'INCREASE_SHORT' if position < 0 else 'ENTRY_SHORT'
                new_position = action_raw
            else:
                action_type = 'HOLD'
                new_position = position

            # Calculate P&L
            pnl = position * price_change * equity
            equity += pnl
            daily_pnl += pnl

            bar_time = cot_now.replace(hour=8, minute=0) + timedelta(minutes=5 * bar)

            # Insert inference
            cur.execute("""
                INSERT INTO dw.fact_rl_inference (
                    timestamp_utc, timestamp_cot,
                    model_id, action_raw, action_discretized, confidence,
                    close_price, raw_return_5m,
                    position_before, position_after,
                    portfolio_value_before, portfolio_value_after,
                    latency_ms, inference_source, observation
                ) VALUES (
                    %s, %s,
                    'ppo_usdcop_v14_fold0', %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, 'simulation', %s
                ) RETURNING inference_id
            """, (
                bar_time + timedelta(hours=5), bar_time,
                action_raw,
                'LONG' if new_position > 0.3 else ('SHORT' if new_position < -0.3 else 'HOLD'),
                abs(action_raw),
                price, price_change,
                position, new_position,
                equity - pnl, equity,
                random.randint(20, 100),
                [0.0] * 20
            ))

            inference_id = cur.fetchone()[0]

            # Insert action
            cur.execute("""
                INSERT INTO dw.fact_agent_actions (
                    timestamp_utc, timestamp_cot,
                    session_date, bar_number,
                    action_type,
                    price_at_action,
                    position_before, position_after,
                    equity_before, equity_after,
                    pnl_action, pnl_daily,
                    model_confidence, model_id,
                    inference_id
                ) VALUES (
                    %s, %s,
                    %s, %s,
                    %s,
                    %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, 'ppo_usdcop_v14_fold0',
                    %s
                )
            """, (
                bar_time + timedelta(hours=5), bar_time,
                session_date, bar,
                action_type,
                price,
                position, new_position,
                equity - pnl, equity,
                pnl, daily_pnl,
                abs(action_raw),
                inference_id
            ))

            # Insert equity point
            cur.execute("""
                INSERT INTO dw.fact_equity_curve_realtime (
                    timestamp_utc, timestamp_cot,
                    session_date, bar_number,
                    equity_value,
                    return_daily_pct,
                    current_position,
                    market_price, model_id
                ) VALUES (
                    %s, %s,
                    %s, %s,
                    %s,
                    %s,
                    %s,
                    %s, 'ppo_usdcop_v14_fold0'
                )
            """, (
                bar_time + timedelta(hours=5), bar_time,
                session_date, bar,
                equity,
                daily_pnl / 10000 * 100,
                new_position,
                price
            ))

            position = new_position

            status = "[TRADE]" if action_type != 'HOLD' else "[HOLD] "
            print(f"  Bar {bar:2d}: {status} {action_type:15s} pos={position:+.2f} "
                  f"price={price:.2f} pnl=${pnl:+.2f} equity=${equity:.2f}")

        # Update session metrics
        cur.execute("SELECT dw.update_session_metrics(%s)", (session_date,))

        conn.commit()

        print("-" * 60)
        print(f"Simulation complete: equity=${equity:.2f}, daily_pnl=${daily_pnl:.2f}")

        cur.close()
        conn.close()
        return True

    except Exception as e:
        print(f"[ERROR] Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_endpoints():
    """Test API endpoints"""
    print("\n" + "=" * 60)
    print("TEST 5: API Endpoints")
    print("=" * 60)

    endpoints = [
        ('/api/agent/actions', 'GET', {'action': 'today'}),
        ('/api/agent/actions', 'GET', {'action': 'latest'}),
        ('/api/market/realtime', 'GET', None),
    ]

    all_passed = True

    for endpoint, method, params in endpoints:
        try:
            url = f"{API_BASE_URL}{endpoint}"
            if method == 'GET':
                response = requests.get(url, params=params, timeout=10)
            else:
                response = requests.post(url, json=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get('success', True):
                    print(f"[OK] {method} {endpoint}")
                    if 'data' in data:
                        print(f"     Response keys: {list(data['data'].keys()) if isinstance(data['data'], dict) else 'array'}")
                else:
                    print(f"[WARN] {method} {endpoint} - API returned error: {data.get('error')}")
            else:
                print(f"[FAIL] {method} {endpoint} - Status: {response.status_code}")
                all_passed = False

        except requests.exceptions.ConnectionError:
            print(f"[SKIP] {method} {endpoint} - Server not running")
        except Exception as e:
            print(f"[ERROR] {method} {endpoint} - {e}")
            all_passed = False

    return all_passed


def verify_data():
    """Verify inserted data"""
    print("\n" + "=" * 60)
    print("TEST 6: Verify Data")
    print("=" * 60)

    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Count records in each table
        tables = [
            'dw.fact_rl_inference',
            'dw.fact_agent_actions',
            'dw.fact_session_performance',
            'dw.fact_equity_curve_realtime',
            'dw.fact_macro_realtime',
        ]

        print("Record counts:")
        for table in tables:
            cur.execute(f"SELECT COUNT(*) as count FROM {table}")
            count = cur.fetchone()['count']
            print(f"    {table}: {count} records")

        # Check session performance
        print("\nLatest session performance:")
        cur.execute("""
            SELECT session_date, total_trades, win_rate, daily_pnl,
                   daily_return_pct, max_drawdown_intraday_pct, status
            FROM dw.fact_session_performance
            ORDER BY session_date DESC
            LIMIT 1
        """)
        perf = cur.fetchone()
        if perf:
            print(f"    Date: {perf['session_date']}")
            print(f"    Trades: {perf['total_trades']}")
            print(f"    Win Rate: {perf['win_rate']:.2%}" if perf['win_rate'] else "    Win Rate: N/A")
            print(f"    Daily P&L: ${perf['daily_pnl']:.2f}" if perf['daily_pnl'] else "    Daily P&L: N/A")
            print(f"    Status: {perf['status']}")

        # Check views
        print("\nViews data:")
        cur.execute("SELECT COUNT(*) as count FROM dw.v_latest_agent_actions")
        print(f"    v_latest_agent_actions: {cur.fetchone()['count']} records")

        cur.close()
        conn.close()
        return True

    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        return False


def cleanup_test_data():
    """Clean up test data"""
    print("\n" + "=" * 60)
    print("CLEANUP: Remove Test Data")
    print("=" * 60)

    try:
        conn = get_connection()
        cur = conn.cursor()

        # Delete test data (be careful with this!)
        today = datetime.now().date()

        tables = [
            'dw.fact_equity_curve_realtime',
            'dw.fact_agent_actions',
            'dw.fact_rl_inference',
            'dw.fact_session_performance',
            'dw.fact_macro_realtime',
        ]

        for table in tables:
            if 'session' in table or 'equity' in table or 'agent' in table:
                cur.execute(f"DELETE FROM {table} WHERE session_date = %s", (today,))
            elif 'inference' in table:
                cur.execute(f"DELETE FROM {table} WHERE DATE(timestamp_cot) = %s", (today,))
            elif 'macro' in table:
                cur.execute(f"DELETE FROM {table} WHERE scrape_run_id LIKE 'test%'")

            print(f"    Cleaned: {table}")

        conn.commit()
        cur.close()
        conn.close()
        print("[OK] Test data cleaned up")
        return True

    except Exception as e:
        print(f"[ERROR] Cleanup failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test Real-time Inference Pipeline')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--db-only', action='store_true', help='Test database only')
    parser.add_argument('--simulate-trading', action='store_true', help='Simulate trading session')
    parser.add_argument('--api', action='store_true', help='Test API endpoints')
    parser.add_argument('--cleanup', action='store_true', help='Clean up test data')
    parser.add_argument('--bars', type=int, default=20, help='Number of bars to simulate')

    args = parser.parse_args()

    print("=" * 60)
    print("REAL-TIME INFERENCE PIPELINE TEST")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Database: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")

    results = {}

    # Always test connection first
    results['connection'] = test_database_connection()

    if not results['connection']:
        print("\n[ABORT] Cannot proceed without database connection")
        sys.exit(1)

    if args.cleanup:
        results['cleanup'] = cleanup_test_data()
        sys.exit(0)

    if args.all or args.db_only:
        results['sql_script'] = execute_sql_script()
        results['test_data'] = insert_test_data()
        results['verify'] = verify_data()

    if args.all or args.simulate_trading:
        results['simulation'] = simulate_trading_session(args.bars)
        results['verify_sim'] = verify_data()

    if args.all or args.api:
        results['api'] = test_api_endpoints()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {test}")

    print("-" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        sys.exit(0)
    else:
        print("\n[WARNING] Some tests failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
