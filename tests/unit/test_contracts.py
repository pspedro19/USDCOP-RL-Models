"""
Contract Validation Tests
=========================

Tests to ensure contract consistency between:
- L2 preprocessing pipeline contracts
- Shared schema system
- Feature store contracts
- Frontend/Backend alignment

Contract: CTR-TEST-001

NOTE: This file imports from airflow.dags.contracts which may conflict
with the installed Apache Airflow package. Module cache is cleared at
import time to prevent this conflict.
"""

import sys
from pathlib import Path

# CRITICAL: Clear Apache Airflow from module cache if it's from site-packages
# Our local airflow/ directory should take precedence
if 'airflow' in sys.modules:
    _airflow_mod = sys.modules.get('airflow')
    if _airflow_mod and hasattr(_airflow_mod, '__file__') and _airflow_mod.__file__:
        _airflow_path = str(Path(_airflow_mod.__file__).parent)
        if 'site-packages' in _airflow_path or 'lib' in _airflow_path:
            _keys_to_remove = [k for k in list(sys.modules.keys())
                              if k == 'airflow' or k.startswith('airflow.')]
            for _key in _keys_to_remove:
                del sys.modules[_key]

# Add project root to path FIRST for local airflow package
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Now safe to import other modules
import json
import datetime as dt
from typing import List

import pytest
from pydantic import ValidationError

# Check if airflow.dags module is available and correct
try:
    from airflow.dags.contracts.l2_preprocessing_contracts import OHLCVRecord
    AIRFLOW_CONTRACTS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    AIRFLOW_CONTRACTS_AVAILABLE = False


# =============================================================================
# L2 PREPROCESSING CONTRACTS
# =============================================================================


@pytest.mark.skipif(
    not AIRFLOW_CONTRACTS_AVAILABLE,
    reason="airflow.dags.contracts not available - run this file in isolation"
)
class TestL2PreprocessingContracts:
    """Tests for L2 preprocessing pipeline contracts."""

    def test_ohlcv_record_validation(self):
        """Test OHLCV record validates correctly."""
        from airflow.dags.contracts.l2_preprocessing_contracts import OHLCVRecord

        # Valid record
        record = OHLCVRecord(
            time=dt.datetime.now(),
            open=4200.50,
            high=4210.00,
            low=4195.00,
            close=4205.75,
            volume=100,
        )
        assert record.symbol == "USD/COP"

    def test_ohlcv_record_high_low_validation(self):
        """Test OHLCV validates high >= low."""
        from airflow.dags.contracts.l2_preprocessing_contracts import OHLCVRecord

        with pytest.raises(ValidationError) as exc_info:
            OHLCVRecord(
                time=dt.datetime.now(),
                open=4200.00,
                high=4190.00,  # Invalid: high < low
                low=4195.00,
                close=4205.00,
            )
        assert "high" in str(exc_info.value).lower()

    def test_macro_indicator_record(self):
        """Test macro indicator record creation."""
        from airflow.dags.contracts.l2_preprocessing_contracts import MacroIndicatorRecord

        record = MacroIndicatorRecord(
            date=dt.date(2025, 1, 1),
            dxy=103.5,
            vix=18.5,
            embi=300.0,
            treasury_10y=4.25,
            treasury_2y=4.10,
        )
        assert record.is_complete is False  # Set by validator

    def test_dataset_metadata(self):
        """Test dataset metadata creation."""
        from airflow.dags.contracts.l2_preprocessing_contracts import (
            DatasetMetadata,
            DatasetType,
            DatasetQualityChecks,
            TimeframeType,
        )

        quality = DatasetQualityChecks(
            no_nan_rows=True,
            temporal_ordered=True,
            no_duplicates=True,
            feature_ranges_valid=True,
            min_rows_satisfied=True,
            warmup_stripped=True,
        )

        metadata = DatasetMetadata(
            name=DatasetType.DS3_MACRO_CORE,
            row_count=100000,
            column_count=15,
            feature_count=15,
            start_date=dt.date(2020, 3, 1),
            end_date=dt.date(2025, 10, 31),
            feature_names=["log_ret_5m", "rsi_9"],
            quality_checks=quality,
        )

        assert metadata.version == "v3.0"
        assert quality.all_passed is True

    def test_feature_contract(self):
        """Test feature contract has correct dimensions."""
        from airflow.dags.contracts.l2_preprocessing_contracts import create_feature_contract

        contract = create_feature_contract()

        assert contract.observation_dim == 15
        assert len(contract.feature_order) == 15
        assert "log_ret_5m" in contract.feature_order
        assert "position" in contract.feature_order
        assert "time_normalized" in contract.feature_order

    def test_validation_result_states(self):
        """Test validation result status enum."""
        from airflow.dags.contracts.l2_preprocessing_contracts import (
            ValidationResult,
            ValidationStatus,
        )

        result = ValidationResult(
            status=ValidationStatus.PASSED,
            datasets_validated=10,
            passed_count=10,
            validation_duration_seconds=1.5,
        )

        assert result.is_valid is True


# =============================================================================
# SHARED SCHEMA CONTRACTS
# =============================================================================


class TestSharedSchemaContracts:
    """Tests for shared schema system."""

    @pytest.fixture(autouse=True)
    def setup_path(self):
        """Setup sys.path for schema imports."""
        import sys
        src_path = str(Path(__file__).parent.parent.parent / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

    def test_observation_dim(self):
        """Test observation dimension constant."""
        from shared.schemas import OBSERVATION_DIM, FEATURE_ORDER

        assert OBSERVATION_DIM == 15
        assert len(FEATURE_ORDER) == 15

    def test_named_features(self):
        """Test named features schema."""
        from shared.schemas import NamedFeatures

        features = NamedFeatures(
            log_ret_5m=0.001,
            log_ret_1h=0.005,
            log_ret_4h=0.01,
            rsi_9=55.0,
            atr_pct=0.5,
            adx_14=25.0,
            dxy_z=0.0,
            dxy_change_1d=0.001,
            vix_z=-0.5,
            embi_z=0.2,
            brent_change_1d=0.02,
            rate_spread=1.5,
            usdmxn_change_1d=0.005,
            position=0.0,
            time_normalized=0.5,
        )

        obs = features.to_observation()
        assert len(obs) == 15
        assert obs[0] == 0.001  # log_ret_5m
        assert obs[3] == 55.0  # rsi_9

    def test_observation_from_named_features_roundtrip(self):
        """Test roundtrip conversion observation <-> named features."""
        from shared.schemas import NamedFeatures, ObservationSchema

        original = [0.001, 0.005, 0.01, 50.0, 0.5, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]

        obs = ObservationSchema(values=original)
        named = obs.to_named_features()
        roundtrip = named.to_observation()

        for i, (orig, rt) in enumerate(zip(original, roundtrip)):
            assert abs(orig - rt) < 1e-6, f"Mismatch at index {i}: {orig} != {rt}"

    def test_signal_type_mapping(self):
        """Test signal type mapping between frontend and backend."""
        from shared.schemas.core import (
            SignalType,
            BackendAction,
            map_backend_to_signal,
            map_signal_to_backend,
        )

        # Backend -> Frontend
        assert map_backend_to_signal(BackendAction.LONG) == SignalType.BUY
        assert map_backend_to_signal(BackendAction.SHORT) == SignalType.SELL
        assert map_backend_to_signal(BackendAction.HOLD) == SignalType.HOLD

        # Frontend -> Backend
        assert map_signal_to_backend(SignalType.BUY) == BackendAction.LONG
        assert map_signal_to_backend(SignalType.SELL) == BackendAction.SHORT
        assert map_signal_to_backend(SignalType.HOLD) == BackendAction.HOLD

    def test_backtest_request_validation(self):
        """Test backtest request date validation."""
        from shared.schemas import BacktestRequestSchema

        # Valid request
        request = BacktestRequestSchema(
            start_date="2025-01-01",
            end_date="2025-06-30",
            model_id="ppo_primary",
        )
        assert request.model_id == "ppo_primary"

        # Invalid: end_date before start_date
        with pytest.raises(ValidationError):
            BacktestRequestSchema(
                start_date="2025-06-30",
                end_date="2025-01-01",
            )

        # Invalid: range > 365 days
        with pytest.raises(ValidationError):
            BacktestRequestSchema(
                start_date="2024-01-01",
                end_date="2025-06-30",  # 546 days
            )

    def test_trade_schema(self):
        """Test trade schema creation."""
        from shared.schemas import TradeSchema, TradeSide, TradeStatus

        trade = TradeSchema(
            trade_id="123",
            model_id="ppo_primary",
            timestamp="2025-01-15T10:30:00Z",
            entry_time="2025-01-15T10:30:00Z",
            side=TradeSide.LONG,
            entry_price=4200.50,
            exit_price=4210.00,
            pnl=9.5,
            pnl_percent=0.226,
            status=TradeStatus.CLOSED,
        )

        assert trade.side == "long"
        assert trade.status == "closed"


# =============================================================================
# CROSS-SYSTEM CONTRACT ALIGNMENT
# =============================================================================


@pytest.mark.skipif(
    not AIRFLOW_CONTRACTS_AVAILABLE,
    reason="airflow.dags.contracts not available - run this file in isolation"
)
class TestCrossSystemAlignment:
    """Tests to ensure contracts are aligned across systems."""

    @pytest.fixture(autouse=True)
    def setup_path(self):
        """Setup sys.path for schema imports."""
        import sys
        src_path = str(Path(__file__).parent.parent.parent / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

    def test_feature_order_alignment(self):
        """Test feature order is consistent across all contracts."""
        from shared.schemas import FEATURE_ORDER
        from airflow.dags.contracts.l2_preprocessing_contracts import create_feature_contract

        l2_contract = create_feature_contract()

        # Both should have same order
        assert len(FEATURE_ORDER) == len(l2_contract.feature_order)
        for i, (shared, l2) in enumerate(zip(FEATURE_ORDER, l2_contract.feature_order)):
            assert shared == l2, f"Feature order mismatch at index {i}: {shared} != {l2}"

    def test_observation_dim_alignment(self):
        """Test observation dimension is consistent."""
        from shared.schemas import OBSERVATION_DIM
        from airflow.dags.contracts.l2_preprocessing_contracts import create_feature_contract

        l2_contract = create_feature_contract()

        assert OBSERVATION_DIM == l2_contract.observation_dim == 15

    def test_trade_status_enums_alignment(self):
        """Test trade status enums are aligned."""
        from shared.schemas import TradeStatus as SharedTradeStatus
        from airflow.dags.contracts.l2_preprocessing_contracts import ValidationStatus

        # Both should support similar status concepts
        assert SharedTradeStatus.OPEN.value == "open"
        assert SharedTradeStatus.CLOSED.value == "closed"


# =============================================================================
# GENERATED TYPESCRIPT VALIDATION
# =============================================================================


class TestGeneratedTypeScript:
    """Tests to validate generated TypeScript files."""

    @pytest.fixture
    def generated_dir(self) -> Path:
        """Path to generated TypeScript directory."""
        return Path(__file__).parent.parent.parent / "usdcop-trading-dashboard" / "types" / "generated"

    def test_generated_files_exist(self, generated_dir: Path):
        """Test that generated files exist."""
        if not generated_dir.exists():
            pytest.skip("Generated TypeScript files not found. Run codegen first.")

        assert (generated_dir / "enums.ts").exists()
        assert (generated_dir / "types.ts").exists()
        assert (generated_dir / "schemas.ts").exists()
        assert (generated_dir / "index.ts").exists()

    def test_generated_enums_content(self, generated_dir: Path):
        """Test generated enums contain expected values."""
        if not generated_dir.exists():
            pytest.skip("Generated TypeScript files not found.")

        enums_content = (generated_dir / "enums.ts").read_text()

        # Check enums exist
        assert "SignalType" in enums_content
        assert "TradeSide" in enums_content
        assert "TradeStatus" in enums_content

        # Check constants
        assert "OBSERVATION_DIM" in enums_content
        assert "FEATURE_ORDER" in enums_content

    def test_generated_types_content(self, generated_dir: Path):
        """Test generated types contain expected interfaces."""
        if not generated_dir.exists():
            pytest.skip("Generated TypeScript files not found.")

        types_content = (generated_dir / "types.ts").read_text()

        # Check interfaces exist
        assert "interface NamedFeatures" in types_content
        assert "interface TradeSchema" in types_content
        assert "interface BacktestResponseSchema" in types_content


# =============================================================================
# SCHEMA SERIALIZATION
# =============================================================================


class TestSchemaSerialization:
    """Tests for schema JSON serialization."""

    @pytest.fixture(autouse=True)
    def setup_path(self):
        """Setup sys.path for schema imports."""
        import sys
        src_path = str(Path(__file__).parent.parent.parent / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

    def test_trade_schema_json_roundtrip(self):
        """Test trade schema JSON serialization roundtrip."""
        from shared.schemas import TradeSchema, TradeSide, TradeStatus

        trade = TradeSchema(
            trade_id="123",
            model_id="ppo_primary",
            timestamp="2025-01-15T10:30:00Z",
            entry_time="2025-01-15T10:30:00Z",
            side=TradeSide.LONG,
            entry_price=4200.50,
        )

        # Serialize to JSON
        json_str = trade.model_dump_json()
        data = json.loads(json_str)

        # Deserialize back
        trade2 = TradeSchema(**data)

        assert trade2.trade_id == trade.trade_id
        assert trade2.side == trade.side

    def test_backtest_request_json_schema(self):
        """Test backtest request generates valid JSON schema."""
        from shared.schemas import BacktestRequestSchema

        schema = BacktestRequestSchema.model_json_schema()

        assert "properties" in schema
        assert "start_date" in schema["properties"]
        assert "end_date" in schema["properties"]
        assert "model_id" in schema["properties"]


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestContractPerformance:
    """Performance tests for contract validation."""

    @pytest.fixture(autouse=True)
    def setup_path(self):
        """Setup sys.path for schema imports."""
        import sys
        src_path = str(Path(__file__).parent.parent.parent / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

    def test_observation_validation_performance(self):
        """Test observation validation is fast enough for inference."""
        import time
        from shared.schemas import ObservationSchema

        obs_values = [0.001, 0.005, 0.01, 50.0, 0.5, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]

        start = time.perf_counter()
        for _ in range(1000):
            ObservationSchema(values=obs_values)
        elapsed = time.perf_counter() - start

        # Should be < 100ms for 1000 validations
        assert elapsed < 0.1, f"Observation validation too slow: {elapsed:.3f}s for 1000 ops"

    def test_trade_schema_validation_performance(self):
        """Test trade schema validation performance."""
        import time
        from shared.schemas import TradeSchema, TradeSide

        start = time.perf_counter()
        for i in range(1000):
            TradeSchema(
                trade_id=str(i),
                model_id="ppo_primary",
                timestamp="2025-01-15T10:30:00Z",
                entry_time="2025-01-15T10:30:00Z",
                side=TradeSide.LONG,
                entry_price=4200.50 + i * 0.1,
            )
        elapsed = time.perf_counter() - start

        # Should be < 500ms for 1000 validations
        assert elapsed < 0.5, f"Trade validation too slow: {elapsed:.3f}s for 1000 ops"
