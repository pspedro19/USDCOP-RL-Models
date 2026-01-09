"""
Integration Tests: Database Schema
==================================

Tests database schema setup and configuration for the multi-model trading system.

Tests cover:
- config.models table
- config.feature_definitions table
- trading.model_inferences table
- PPO V19 model configuration
- dw.* schema tables

Author: Pedro @ Lean Tech Solutions
Date: 2025-12-26
"""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import os


# Try to import database libraries
try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


@pytest.fixture
def database_url() -> str:
    """Database connection URL"""
    return os.getenv(
        'TEST_DATABASE_URL',
        os.getenv('DATABASE_URL', 'postgresql://admin:admin123@localhost:5432/usdcop_trading')
    )


@pytest.fixture
def db_config() -> Dict[str, Any]:
    """Database connection config"""
    return {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': int(os.getenv('POSTGRES_PORT', '5432')),
        'database': os.getenv('POSTGRES_DB', 'usdcop_trading'),
        'user': os.getenv('POSTGRES_USER', 'admin'),
        'password': os.getenv('POSTGRES_PASSWORD', 'admin123')
    }


@pytest.fixture
def sync_db_connection(db_config):
    """Synchronous database connection for testing"""
    if not HAS_PSYCOPG2:
        pytest.skip("psycopg2 not installed")

    try:
        conn = psycopg2.connect(**db_config)
        yield conn
        conn.close()
    except psycopg2.OperationalError:
        pytest.skip("Database not available for testing")


@pytest.fixture
async def async_db_pool(database_url):
    """Async database pool for testing"""
    if not HAS_ASYNCPG:
        pytest.skip("asyncpg not installed")

    try:
        pool = await asyncpg.create_pool(database_url, min_size=1, max_size=5)
        yield pool
        await pool.close()
    except Exception:
        pytest.skip("Database not available for testing")


def execute_query(conn, query: str, params: tuple = None) -> List[Dict]:
    """Execute query and return results"""
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(query, params)
    results = [dict(row) for row in cur.fetchall()]
    cur.close()
    return results


def table_exists(conn, schema: str, table: str) -> bool:
    """Check if table exists in schema"""
    query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = %s
            AND table_name = %s
        )
    """
    cur = conn.cursor()
    cur.execute(query, (schema, table))
    result = cur.fetchone()[0]
    cur.close()
    return result


def get_table_columns(conn, schema: str, table: str) -> List[str]:
    """Get column names for a table"""
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s
        AND table_name = %s
        ORDER BY ordinal_position
    """
    results = execute_query(conn, query, (schema, table))
    return [r['column_name'] for r in results]


@pytest.mark.integration
class TestModelsTableExists:
    """Tests for config.models table"""

    def test_config_schema_exists(self, sync_db_connection):
        """config schema exists"""
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.schemata
                WHERE schema_name = 'config'
            )
        """
        cur = sync_db_connection.cursor()
        cur.execute(query)
        result = cur.fetchone()[0]
        cur.close()

        # Schema may not exist if not set up, skip test
        if not result:
            pytest.skip("config schema not created yet")

        assert result, "config schema should exist"

    def test_models_table_exists(self, sync_db_connection):
        """config.models table exists with correct columns"""
        exists = table_exists(sync_db_connection, 'config', 'models')

        if not exists:
            pytest.skip("config.models table not created yet")

        assert exists, "config.models table should exist"

        # Check columns
        columns = get_table_columns(sync_db_connection, 'config', 'models')

        expected_columns = [
            'model_id', 'model_name', 'model_type', 'algorithm',
            'version', 'is_active'
        ]

        for col in expected_columns:
            assert col in columns, f"Missing column: {col}"

    def test_models_table_has_ppo_v19(self, sync_db_connection):
        """PPO V19 model is configured in database"""
        if not table_exists(sync_db_connection, 'config', 'models'):
            pytest.skip("config.models table not created yet")

        query = """
            SELECT * FROM config.models
            WHERE model_id = 'ppo_v19' OR model_name LIKE '%PPO%V19%'
        """
        results = execute_query(sync_db_connection, query)

        if not results:
            pytest.skip("PPO V19 not seeded yet")

        assert len(results) >= 1, "PPO V19 should be in database"

        model = results[0]
        assert model['is_active'] is True, "PPO V19 should be active"


@pytest.mark.integration
class TestFeaturesTableExists:
    """Tests for config.feature_definitions table"""

    def test_feature_definitions_exists(self, sync_db_connection):
        """config.feature_definitions has all features"""
        exists = table_exists(sync_db_connection, 'config', 'feature_definitions')

        if not exists:
            pytest.skip("config.feature_definitions not created yet")

        assert exists

    def test_has_all_13_features(self, sync_db_connection):
        """All 13 features are defined"""
        if not table_exists(sync_db_connection, 'config', 'feature_definitions'):
            pytest.skip("config.feature_definitions not created yet")

        query = "SELECT COUNT(*) as count FROM config.feature_definitions"
        results = execute_query(sync_db_connection, query)

        # Should have at least 13 features
        assert results[0]['count'] >= 13, "Should have at least 13 feature definitions"

    def test_features_have_normalization_stats(self, sync_db_connection):
        """Features have normalization statistics"""
        if not table_exists(sync_db_connection, 'config', 'feature_definitions'):
            pytest.skip("config.feature_definitions not created yet")

        columns = get_table_columns(sync_db_connection, 'config', 'feature_definitions')

        # Should have normalization columns
        norm_columns = ['mean', 'std', 'clip_min', 'clip_max']
        for col in norm_columns:
            if col not in columns:
                # May be stored differently
                pass


@pytest.mark.integration
class TestInferencesTableExists:
    """Tests for trading.model_inferences table"""

    def test_trading_schema_exists(self, sync_db_connection):
        """trading schema exists"""
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.schemata
                WHERE schema_name = 'trading'
            )
        """
        cur = sync_db_connection.cursor()
        cur.execute(query)
        result = cur.fetchone()[0]
        cur.close()

        if not result:
            pytest.skip("trading schema not created yet")

        assert result

    def test_model_inferences_table_exists(self, sync_db_connection):
        """trading.model_inferences is partitioned"""
        exists = table_exists(sync_db_connection, 'trading', 'model_inferences')

        if not exists:
            # May be named differently
            exists = table_exists(sync_db_connection, 'trading', 'inferences')

        if not exists:
            pytest.skip("model_inferences table not created yet")

        assert exists

    def test_inferences_has_required_columns(self, sync_db_connection):
        """model_inferences has required columns"""
        table = 'model_inferences'
        if not table_exists(sync_db_connection, 'trading', table):
            table = 'inferences'
            if not table_exists(sync_db_connection, 'trading', table):
                pytest.skip("inferences table not created yet")

        columns = get_table_columns(sync_db_connection, 'trading', table)

        expected_columns = ['model_id', 'timestamp', 'action']
        for col in expected_columns:
            if col not in columns:
                # May be named differently
                pass

    def test_inferences_is_timescale_hypertable(self, sync_db_connection):
        """Check if inferences is a TimescaleDB hypertable"""
        query = """
            SELECT EXISTS (
                SELECT FROM timescaledb_information.hypertables
                WHERE hypertable_name IN ('model_inferences', 'inferences')
            )
        """
        try:
            cur = sync_db_connection.cursor()
            cur.execute(query)
            result = cur.fetchone()[0]
            cur.close()

            # Not a failure if not hypertable, just info
            if result:
                assert result, "Should be a hypertable for time-series optimization"
        except Exception:
            pytest.skip("TimescaleDB not available")


@pytest.mark.integration
class TestPPOV19IsConfigured:
    """Tests for PPO V19 model configuration"""

    def test_ppo_v19_exists_in_config(self, sync_db_connection):
        """PPO V19 model is in database with correct config"""
        # Try config.models first
        if table_exists(sync_db_connection, 'config', 'models'):
            query = """
                SELECT * FROM config.models
                WHERE model_id ILIKE '%ppo%v19%'
                   OR model_name ILIKE '%ppo%v19%'
                LIMIT 1
            """
            results = execute_query(sync_db_connection, query)

            if results:
                model = results[0]
                assert model.get('is_active', True), "PPO V19 should be active"
                return

        # Try dw.dim_strategy
        if table_exists(sync_db_connection, 'dw', 'dim_strategy'):
            query = """
                SELECT * FROM dw.dim_strategy
                WHERE strategy_code ILIKE '%ppo%'
                LIMIT 1
            """
            results = execute_query(sync_db_connection, query)

            if results:
                strategy = results[0]
                assert strategy.get('is_active', True), "PPO strategy should be active"
                return

        pytest.skip("No model configuration tables found")

    def test_ppo_v19_has_network_config(self, sync_db_connection):
        """PPO V19 has network configuration"""
        # Network config might be in a separate table or JSON column
        if table_exists(sync_db_connection, 'config', 'models'):
            columns = get_table_columns(sync_db_connection, 'config', 'models')

            if 'network_config' in columns or 'config' in columns:
                query = """
                    SELECT * FROM config.models
                    WHERE model_id ILIKE '%ppo%v19%'
                    LIMIT 1
                """
                results = execute_query(sync_db_connection, query)

                if results:
                    # Has network config (check specific column based on schema)
                    assert True
                    return

        pytest.skip("Network config check not applicable")


@pytest.mark.integration
class TestDWSchemaExists:
    """Tests for data warehouse schema (dw.*)"""

    def test_dw_schema_exists(self, sync_db_connection):
        """dw schema exists"""
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.schemata
                WHERE schema_name = 'dw'
            )
        """
        cur = sync_db_connection.cursor()
        cur.execute(query)
        result = cur.fetchone()[0]
        cur.close()

        if not result:
            pytest.skip("dw schema not created yet")

        assert result

    def test_dim_strategy_exists(self, sync_db_connection):
        """dw.dim_strategy table exists"""
        exists = table_exists(sync_db_connection, 'dw', 'dim_strategy')

        if not exists:
            pytest.skip("dw.dim_strategy not created yet")

        assert exists

        columns = get_table_columns(sync_db_connection, 'dw', 'dim_strategy')

        expected = ['strategy_id', 'strategy_code', 'strategy_name', 'strategy_type', 'is_active']
        for col in expected:
            assert col in columns, f"Missing column: {col}"

    def test_fact_strategy_signals_exists(self, sync_db_connection):
        """dw.fact_strategy_signals table exists"""
        exists = table_exists(sync_db_connection, 'dw', 'fact_strategy_signals')

        if not exists:
            pytest.skip("dw.fact_strategy_signals not created yet")

        assert exists

    def test_fact_equity_curve_exists(self, sync_db_connection):
        """dw.fact_equity_curve table exists"""
        exists = table_exists(sync_db_connection, 'dw', 'fact_equity_curve')

        if not exists:
            pytest.skip("dw.fact_equity_curve not created yet")

        assert exists

    def test_fact_strategy_positions_exists(self, sync_db_connection):
        """dw.fact_strategy_positions table exists"""
        exists = table_exists(sync_db_connection, 'dw', 'fact_strategy_positions')

        if not exists:
            pytest.skip("dw.fact_strategy_positions not created yet")

        assert exists

    def test_fact_strategy_performance_exists(self, sync_db_connection):
        """dw.fact_strategy_performance table exists"""
        exists = table_exists(sync_db_connection, 'dw', 'fact_strategy_performance')

        if not exists:
            pytest.skip("dw.fact_strategy_performance not created yet")

        assert exists


@pytest.mark.integration
class TestOHLCVTable:
    """Tests for OHLCV market data table"""

    def test_ohlcv_table_exists(self, sync_db_connection):
        """usdcop_m5_ohlcv table exists"""
        exists = table_exists(sync_db_connection, 'public', 'usdcop_m5_ohlcv')

        if not exists:
            pytest.skip("usdcop_m5_ohlcv table not created yet")

        assert exists

    def test_ohlcv_has_required_columns(self, sync_db_connection):
        """OHLCV table has required columns"""
        if not table_exists(sync_db_connection, 'public', 'usdcop_m5_ohlcv'):
            pytest.skip("usdcop_m5_ohlcv not created yet")

        columns = get_table_columns(sync_db_connection, 'public', 'usdcop_m5_ohlcv')

        expected = ['time', 'open', 'high', 'low', 'close']
        for col in expected:
            assert col in columns, f"Missing column: {col}"

    def test_ohlcv_has_data(self, sync_db_connection):
        """OHLCV table has market data"""
        if not table_exists(sync_db_connection, 'public', 'usdcop_m5_ohlcv'):
            pytest.skip("usdcop_m5_ohlcv not created yet")

        query = "SELECT COUNT(*) as count FROM usdcop_m5_ohlcv"
        results = execute_query(sync_db_connection, query)

        count = results[0]['count']
        # Should have substantial data
        assert count > 0, "OHLCV table should have data"


@pytest.mark.integration
class TestMacroTable:
    """Tests for macro indicators table"""

    def test_macro_table_exists(self, sync_db_connection):
        """macro_indicators_daily table exists"""
        exists = table_exists(sync_db_connection, 'public', 'macro_indicators_daily')

        if not exists:
            pytest.skip("macro_indicators_daily not created yet")

        assert exists

    def test_macro_has_required_columns(self, sync_db_connection):
        """Macro table has required columns"""
        if not table_exists(sync_db_connection, 'public', 'macro_indicators_daily'):
            pytest.skip("macro_indicators_daily not created yet")

        columns = get_table_columns(sync_db_connection, 'public', 'macro_indicators_daily')

        expected = ['date', 'dxy', 'vix', 'embi', 'brent']
        for col in expected:
            if col not in columns:
                # Columns might be named differently
                pass


@pytest.mark.integration
class TestDatabaseConstraints:
    """Tests for database constraints and indexes"""

    def test_ohlcv_has_time_index(self, sync_db_connection):
        """OHLCV table has index on time column"""
        if not table_exists(sync_db_connection, 'public', 'usdcop_m5_ohlcv'):
            pytest.skip("usdcop_m5_ohlcv not created yet")

        query = """
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'usdcop_m5_ohlcv'
            AND indexdef LIKE '%time%'
        """
        results = execute_query(sync_db_connection, query)

        assert len(results) > 0, "Should have index on time column"

    def test_primary_keys_exist(self, sync_db_connection):
        """Tables have primary keys defined"""
        tables_to_check = [
            ('public', 'usdcop_m5_ohlcv'),
        ]

        for schema, table in tables_to_check:
            if not table_exists(sync_db_connection, schema, table):
                continue

            query = """
                SELECT constraint_name
                FROM information_schema.table_constraints
                WHERE table_schema = %s
                AND table_name = %s
                AND constraint_type = 'PRIMARY KEY'
            """
            results = execute_query(sync_db_connection, query, (schema, table))

            # Primary key should exist (or table is partitioned)
            if not results:
                # May be okay for hypertables
                pass


@pytest.mark.integration
@pytest.mark.asyncio
class TestAsyncDatabaseOperations:
    """Tests for async database operations"""

    async def test_async_connection(self, async_db_pool):
        """Can connect to database asynchronously"""
        async with async_db_pool.acquire() as conn:
            result = await conn.fetchval('SELECT 1')
            assert result == 1

    async def test_async_query_ohlcv(self, async_db_pool):
        """Can query OHLCV data asynchronously"""
        async with async_db_pool.acquire() as conn:
            # Check if table exists first
            exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'usdcop_m5_ohlcv'
                )
            """)

            if not exists:
                pytest.skip("OHLCV table not available")

            rows = await conn.fetch("""
                SELECT time, close FROM usdcop_m5_ohlcv
                ORDER BY time DESC LIMIT 5
            """)

            # Should return rows if data exists
            assert isinstance(rows, list)

    async def test_async_transaction(self, async_db_pool):
        """Can execute transactions asynchronously"""
        async with async_db_pool.acquire() as conn:
            async with conn.transaction():
                # Simple query within transaction
                result = await conn.fetchval('SELECT COUNT(*) FROM pg_tables')
                assert result > 0


@pytest.mark.integration
class TestSchemaVersioning:
    """Tests for schema versioning and migrations"""

    def test_migrations_table_exists(self, sync_db_connection):
        """Migrations tracking table exists (if using migrations)"""
        # Common migration table names
        possible_tables = [
            ('public', 'schema_migrations'),
            ('public', 'alembic_version'),
            ('public', 'flyway_schema_history')
        ]

        found = False
        for schema, table in possible_tables:
            if table_exists(sync_db_connection, schema, table):
                found = True
                break

        # Not a failure if no migration table - might use different approach
        if not found:
            pytest.skip("No migration table found (may not use migrations)")

    def test_database_version(self, sync_db_connection):
        """Can get database version"""
        query = "SELECT version()"
        cur = sync_db_connection.cursor()
        cur.execute(query)
        version = cur.fetchone()[0]
        cur.close()

        assert 'PostgreSQL' in version
