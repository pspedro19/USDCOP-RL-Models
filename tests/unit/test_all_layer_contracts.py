"""
Comprehensive Contract Tests for All Pipeline Layers
=====================================================

Tests to ensure contract consistency and validation across:
- L0: Data Acquisition
- L1: Feature Calculation
- L2: Preprocessing
- L3: Training
- L5: Inference

Contract Coverage Target: 100%

Contract: CTR-TEST-ALL-001

NOTE: This file must be run separately or at the start of the test suite
because it imports from airflow/dags/contracts which conflicts with
services/inference_api/contracts.
"""

import sys
from pathlib import Path

# CRITICAL: Clear any cached 'contracts' module BEFORE importing anything
# This prevents conflicts with services/inference_api/contracts
_contracts_keys = [k for k in list(sys.modules.keys()) if k == 'contracts' or k.startswith('contracts.')]
for _key in _contracts_keys:
    del sys.modules[_key]

# Add airflow/dags to path FIRST for contract imports
_airflow_dags_path = str(Path(__file__).parent.parent.parent / "airflow" / "dags")
if _airflow_dags_path not in sys.path:
    sys.path.insert(0, _airflow_dags_path)

# Now safe to import other modules
import datetime as dt
from decimal import Decimal

import pytest
from pydantic import ValidationError

# Check if contracts module is available and correct
try:
    from contracts.l0_data_contracts import L0XComKeys
    CONTRACTS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CONTRACTS_AVAILABLE = False

# Skip all tests in this file if contracts not available
pytestmark = pytest.mark.skipif(
    not CONTRACTS_AVAILABLE,
    reason="airflow/dags/contracts not available - run this file in isolation"
)


# =============================================================================
# L0 DATA ACQUISITION CONTRACTS
# =============================================================================


class TestL0DataContracts:
    """Tests for L0 data acquisition contracts."""

    def test_l0_xcom_keys_defined(self):
        """Test L0XComKeys enum has all required keys."""
        from contracts.l0_data_contracts import L0XComKeys

        assert L0XComKeys.OHLCV_ROWS_INSERTED.value == "ohlcv_rows_inserted"
        assert L0XComKeys.FRED_DATA.value == "fred_data"
        assert L0XComKeys.BACKUP_PATH.value == "backup_path"

    def test_twelvedata_bar_conversion(self):
        """Test TwelveData bar conversion to OHLCV record."""
        from contracts.l0_data_contracts import TwelveDataBar

        bar = TwelveDataBar(
            datetime="2025-01-15T10:30:00",
            open="4200.50",
            high="4210.00",
            low="4195.00",
            close="4205.75",
            volume="100"
        )

        record = bar.to_ohlcv_record()
        assert record.symbol == "USD/COP"
        assert record.close == Decimal("4205.75")

    def test_ohlcv_record_market_hours(self):
        """Test OHLCV record market hours validation."""
        from contracts.l0_data_contracts import OHLCVRecord

        # Within market hours (9:30 COT on Wednesday)
        record = OHLCVRecord(
            time=dt.datetime(2025, 1, 15, 9, 30),  # Wednesday 9:30
            open=Decimal("4200"),
            high=Decimal("4210"),
            low=Decimal("4195"),
            close=Decimal("4205"),
        )
        assert record.is_in_market_hours() is True

        # Outside market hours (15:00)
        record_late = OHLCVRecord(
            time=dt.datetime(2025, 1, 15, 15, 0),
            open=Decimal("4200"),
            high=Decimal("4210"),
            low=Decimal("4195"),
            close=Decimal("4205"),
        )
        assert record_late.is_in_market_hours() is False

    def test_ohlcv_record_high_low_validation(self):
        """Test OHLCV high >= low validation."""
        from contracts.l0_data_contracts import OHLCVRecord

        with pytest.raises(ValidationError):
            OHLCVRecord(
                time=dt.datetime.now(),
                open=Decimal("4200"),
                high=Decimal("4190"),  # Invalid: high < low
                low=Decimal("4195"),
                close=Decimal("4205"),
            )

    def test_acquisition_result_factory(self):
        """Test acquisition result factory."""
        from contracts.l0_data_contracts import (
            create_ohlcv_acquisition_result,
            AcquisitionStatus,
        )

        result = create_ohlcv_acquisition_result(rows_inserted=50, rows_filtered=5)
        assert result.status == AcquisitionStatus.SUCCESS
        assert result.rows_inserted == 50
        assert result.is_successful is True

        result_empty = create_ohlcv_acquisition_result(rows_inserted=0)
        assert result_empty.status == AcquisitionStatus.SKIPPED

    def test_l0_quality_report(self):
        """Test L0 quality report validation."""
        from contracts.l0_data_contracts import L0QualityReport

        report = L0QualityReport(
            ohlcv_bar_count=1000,
            ohlcv_latest_time=dt.datetime.now(),
            macro_coverage={"fxrt_index_dxy_usa_d_dxy": True, "volt_vix_usa_d_vix": True},
        )

        # Without critical columns, not production ready
        assert report.is_production_ready is False


# =============================================================================
# L1 FEATURE CALCULATION CONTRACTS
# =============================================================================


class TestL1FeatureContracts:
    """Tests for L1 feature calculation contracts."""

    def test_l1_xcom_keys_defined(self):
        """Test L1XComKeys enum has all required keys."""
        from contracts.l1_feature_contracts import L1XComKeys

        assert L1XComKeys.FEATURES_COUNT.value == "features_count"
        assert L1XComKeys.MACRO_ROWS_USED.value == "macro_rows_used"

    def test_feature_contract_dimensions(self):
        """Test feature contract has correct dimensions."""
        from contracts.l1_feature_contracts import FEATURE_CONTRACT

        assert FEATURE_CONTRACT.core_feature_count == 13
        assert FEATURE_CONTRACT.state_feature_count == 2
        assert FEATURE_CONTRACT.total_observation_dim == 15
        assert len(FEATURE_CONTRACT.feature_order) == 13

    def test_feature_definitions_count(self):
        """Test core feature definitions."""
        from contracts.l1_feature_contracts import CORE_FEATURE_DEFINITIONS

        assert len(CORE_FEATURE_DEFINITIONS) == 13
        names = [f.name for f in CORE_FEATURE_DEFINITIONS]
        assert "log_ret_5m" in names
        assert "rsi_9" in names
        assert "dxy_z" in names

    def test_calculated_features_to_array(self):
        """Test calculated features conversion to array."""
        from contracts.l1_feature_contracts import CalculatedFeatures

        features = CalculatedFeatures(
            time=dt.datetime.now(),
            log_ret_5m=0.001,
            log_ret_1h=0.005,
            log_ret_4h=0.01,
            rsi_9=55.0,
            atr_pct=0.5,
            adx_14=25.0,
            dxy_z=0.1,
            dxy_change_1d=0.002,
            vix_z=-0.5,
            embi_z=0.2,
            brent_change_1d=0.01,
            rate_spread=1.5,
            usdmxn_change_1d=0.003,
        )

        arr = features.to_observation_array()
        assert len(arr) == 13
        assert arr[0] == 0.001  # log_ret_5m
        assert arr[3] == 55.0  # rsi_9

    def test_feature_calculation_result_factory(self):
        """Test feature calculation result factory."""
        from contracts.l1_feature_contracts import (
            create_feature_calculation_result,
            FeatureCalculationStatus,
        )

        result = create_feature_calculation_result(
            rows_inserted=100,
            ohlcv_bars=150,
            macro_records=30,
        )
        assert result.status == FeatureCalculationStatus.SUCCESS
        assert result.is_successful is True


# =============================================================================
# L2 PREPROCESSING CONTRACTS
# =============================================================================


class TestL2PreprocessingContracts:
    """Tests for L2 preprocessing contracts."""

    def test_l2_xcom_keys_defined(self):
        """Test L2XComKeys has all required keys."""
        from contracts.l2_preprocessing_contracts import L2XComKeys

        # L2XComKeys uses class attributes (not Enum)
        assert L2XComKeys.EXPORT == "export_results"
        assert L2XComKeys.FUSION == "fusion_results"
        assert L2XComKeys.VALIDATION == "validation_results"

    def test_dataset_type_enum(self):
        """Test DatasetType enum values."""
        from contracts.l2_preprocessing_contracts import DatasetType

        assert DatasetType.DS3_MACRO_CORE.value == "RL_DS3_MACRO_CORE"
        assert len(DatasetType) == 10

    def test_feature_contract_factory(self):
        """Test feature contract factory."""
        from contracts.l2_preprocessing_contracts import create_feature_contract

        contract = create_feature_contract()
        assert contract.observation_dim == 15
        assert "log_ret_5m" in contract.feature_order
        assert "position" in contract.feature_order

    def test_dataset_quality_checks(self):
        """Test dataset quality checks all_passed property."""
        from contracts.l2_preprocessing_contracts import DatasetQualityChecks

        quality = DatasetQualityChecks(
            no_nan_rows=True,
            temporal_ordered=True,
            no_duplicates=True,
            feature_ranges_valid=True,
            min_rows_satisfied=True,
            warmup_stripped=True,
        )
        assert quality.all_passed is True

        quality_fail = DatasetQualityChecks(
            no_nan_rows=False,  # One failure
            temporal_ordered=True,
            no_duplicates=True,
            feature_ranges_valid=True,
            min_rows_satisfied=True,
            warmup_stripped=True,
        )
        assert quality_fail.all_passed is False


# =============================================================================
# L3 TRAINING CONTRACTS
# =============================================================================


class TestL3TrainingContracts:
    """Tests for L3 training contracts."""

    def test_l3_xcom_keys_defined(self):
        """Test L3XComKeys enum has all required keys."""
        from contracts.l3_training_contracts import L3XComKeys

        assert L3XComKeys.DATASET_PATH.value == "dataset_path"
        assert L3XComKeys.MODEL_PATH.value == "model_path"
        assert L3XComKeys.TRAINING_RESULT.value == "training_result"

    def test_training_config_defaults(self):
        """Test training config default values."""
        from contracts.l3_training_contracts import TrainingConfig

        config = TrainingConfig()
        assert config.version == "current"
        assert config.total_timesteps == 500_000
        assert config.observation_dim == 15
        assert config.market_feature_count == 13

    def test_training_config_ratio_validation(self):
        """Test training config ratio validation."""
        from contracts.l3_training_contracts import TrainingConfig

        # Valid ratios
        config = TrainingConfig(train_ratio=0.7, val_ratio=0.15)
        # Use approximate comparison for floating point
        assert abs(config.test_ratio - 0.15) < 1e-10

        # Invalid ratios (sum >= 1)
        with pytest.raises(ValidationError):
            TrainingConfig(train_ratio=0.8, val_ratio=0.3)

    def test_training_result_factory(self):
        """Test training result factory."""
        from contracts.l3_training_contracts import (
            create_training_result,
            TrainingStatus,
        )

        result = create_training_result(
            model_path="/path/to/model.zip",
            model_hash="abc123",
            best_reward=100.0,
            duration_seconds=3600.0,
        )
        assert result.status == TrainingStatus.SUCCESS
        assert result.is_successful is True

        result_failed = create_training_result(error="Training failed")
        assert result_failed.status == TrainingStatus.FAILED
        assert result_failed.is_successful is False

    def test_ppo_hyperparameters(self):
        """Test PPO hyperparameters defaults."""
        from contracts.l3_training_contracts import PPOHyperparameters

        params = PPOHyperparameters()
        assert params.learning_rate == 3e-4
        assert params.gamma == 0.90  # From config/trading_config.yaml SSOT
        assert params.clip_range == 0.2


# =============================================================================
# L5 INFERENCE CONTRACTS
# =============================================================================


class TestL5InferenceContracts:
    """Tests for L5 inference contracts."""

    def test_l5_xcom_keys_defined(self):
        """Test L5XComKeys enum has all required keys."""
        from contracts.l5_inference_contracts import L5XComKeys

        assert L5XComKeys.MARKET_FEATURES.value == "market_features"
        assert L5XComKeys.INFERENCE_RESULTS.value == "inference_results"
        assert L5XComKeys.EXECUTION_PRICE.value == "execution_price"

    def test_observation_contract(self):
        """Test observation contract dimensions."""
        from contracts.l5_inference_contracts import OBSERVATION_CONTRACT

        assert OBSERVATION_CONTRACT.total_dim == 15
        assert OBSERVATION_CONTRACT.market_feature_dim == 13
        assert OBSERVATION_CONTRACT.state_feature_dim == 2
        assert len(OBSERVATION_CONTRACT.feature_order) == 15

    def test_signal_action_enum(self):
        """Test SignalAction enum values."""
        from contracts.l5_inference_contracts import SignalAction

        assert SignalAction.LONG.value == "LONG"
        assert SignalAction.SHORT.value == "SHORT"
        assert SignalAction.HOLD.value == "HOLD"

    def test_full_observation_to_numpy(self):
        """Test full observation conversion to numpy."""
        from contracts.l5_inference_contracts import (
            MarketFeatures,
            StateFeatures,
            FullObservation,
        )

        market = MarketFeatures(
            log_ret_5m=0.001,
            rsi_9=55.0,
        )
        state = StateFeatures(position=1.0, time_normalized=0.5)

        obs = FullObservation(market_features=market, state_features=state)
        arr = obs.to_numpy()

        assert len(arr) == 15
        assert arr[13] == 1.0  # position
        assert arr[14] == 0.5  # time_normalized

    def test_inference_result_factory(self):
        """Test inference result factory with signal discretization."""
        from contracts.l5_inference_contracts import (
            create_inference_result,
            SignalAction,
        )

        # LONG signal (raw_action > threshold_long)
        result_long = create_inference_result(
            model_id="ppo_primary",
            model_name="PPO Primary",
            model_type="PPO",
            raw_action=0.5,  # > 0.10 threshold
        )
        assert result_long.signal == SignalAction.LONG
        assert result_long.is_trade_signal is True

        # SHORT signal (raw_action < threshold_short)
        result_short = create_inference_result(
            model_id="ppo_primary",
            model_name="PPO Primary",
            model_type="PPO",
            raw_action=-0.5,  # < -0.10 threshold
        )
        assert result_short.signal == SignalAction.SHORT

        # HOLD signal (within thresholds)
        result_hold = create_inference_result(
            model_id="ppo_primary",
            model_name="PPO Primary",
            model_type="PPO",
            raw_action=0.05,  # Between -0.10 and 0.10
        )
        assert result_hold.signal == SignalAction.HOLD
        assert result_hold.is_trade_signal is False

    def test_batch_inference_result_consensus(self):
        """Test batch inference consensus signal."""
        from contracts.l5_inference_contracts import (
            InferenceResult,
            BatchInferenceResult,
            SignalAction,
        )

        results = [
            InferenceResult(
                model_id="m1",
                model_name="M1",
                model_type="PPO",
                raw_action=0.5,
                signal=SignalAction.LONG,
                confidence=0.8,
                latency_ms=10.0,
                bar_number=1,
            ),
            InferenceResult(
                model_id="m2",
                model_name="M2",
                model_type="PPO",
                raw_action=0.6,
                signal=SignalAction.LONG,
                confidence=0.9,
                latency_ms=12.0,
                bar_number=1,
            ),
            InferenceResult(
                model_id="m3",
                model_name="M3",
                model_type="PPO",
                raw_action=-0.2,
                signal=SignalAction.SHORT,
                confidence=0.7,
                latency_ms=11.0,
                bar_number=1,
            ),
        ]

        batch = BatchInferenceResult(
            results=results,
            total_models=3,
            successful_count=3,
        )

        assert batch.signal_distribution["LONG"] == 2
        assert batch.signal_distribution["SHORT"] == 1
        assert batch.consensus_signal == SignalAction.LONG

    def test_risk_limits(self):
        """Test risk limits defaults."""
        from contracts.l5_inference_contracts import RiskLimits

        limits = RiskLimits()
        assert limits.max_drawdown_pct == 15.0
        assert limits.max_daily_loss_pct == 5.0
        assert limits.max_trades_per_day == 20


# =============================================================================
# CROSS-LAYER ALIGNMENT TESTS
# =============================================================================


class TestCrossLayerAlignment:
    """Tests to ensure contracts are aligned across layers."""

    def test_feature_order_alignment_l1_l2(self):
        """Test feature order alignment between L1 and L2."""
        from contracts.l1_feature_contracts import FEATURE_CONTRACT as L1_CONTRACT
        from contracts.l2_preprocessing_contracts import create_feature_contract

        l2_contract = create_feature_contract()

        # Core features should match (first 13)
        for i, feat in enumerate(L1_CONTRACT.feature_order):
            assert feat == l2_contract.feature_order[i], f"Mismatch at {i}: {feat}"

    def test_observation_dim_alignment_l1_l5(self):
        """Test observation dimension alignment between L1 and L5."""
        from contracts.l1_feature_contracts import FEATURE_CONTRACT as L1_CONTRACT
        from contracts.l5_inference_contracts import OBSERVATION_CONTRACT

        assert L1_CONTRACT.total_observation_dim == OBSERVATION_CONTRACT.total_dim
        assert L1_CONTRACT.core_feature_count == OBSERVATION_CONTRACT.market_feature_dim

    def test_xcom_keys_unique_across_layers(self):
        """Test XCom keys are unique within each layer."""
        from contracts.l0_data_contracts import L0XComKeys
        from contracts.l1_feature_contracts import L1XComKeys
        from contracts.l2_preprocessing_contracts import L2XComKeys
        from contracts.l3_training_contracts import L3XComKeys
        from contracts.l5_inference_contracts import L5XComKeys

        # Check Enum-based XCom keys have unique values
        for enum_cls in [L0XComKeys, L1XComKeys, L3XComKeys, L5XComKeys]:
            values = [e.value for e in enum_cls]
            assert len(values) == len(set(values)), f"Duplicate keys in {enum_cls.__name__}"

        # L2XComKeys uses class attributes (not Enum)
        l2_values = [getattr(L2XComKeys, k) for k in dir(L2XComKeys) if not k.startswith("_")]
        assert len(l2_values) == len(set(l2_values)), "Duplicate keys in L2XComKeys"

    def test_signal_action_mapping_consistency(self):
        """Test signal action mapping consistency."""
        from contracts.l5_inference_contracts import SignalAction

        # Ensure mapping to frontend values
        assert SignalAction.LONG.value == "LONG"
        assert SignalAction.SHORT.value == "SHORT"
        assert SignalAction.HOLD.value == "HOLD"


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================


class TestContractSerialization:
    """Tests for contract JSON serialization."""

    def test_l0_result_json_roundtrip(self):
        """Test L0 result JSON serialization roundtrip."""
        import json
        from contracts.l0_data_contracts import OHLCVAcquisitionResult, AcquisitionStatus

        result = OHLCVAcquisitionResult(
            status=AcquisitionStatus.SUCCESS,
            rows_inserted=100,
            rows_filtered=5,
        )

        json_str = result.model_dump_json()
        data = json.loads(json_str)
        result2 = OHLCVAcquisitionResult(**data)

        assert result2.rows_inserted == result.rows_inserted

    def test_l3_training_result_to_dict(self):
        """Test L3 training result to_dict method."""
        from contracts.l3_training_contracts import (
            TrainingResult,
            TrainingStatus,
            TrainingMetrics,
        )

        result = TrainingResult(
            status=TrainingStatus.SUCCESS,
            model_path="/path/model.zip",
            model_hash="abc123",
            metrics=TrainingMetrics(best_mean_reward=100.0, total_timesteps=500000),
        )

        d = result.to_dict()
        assert d["status"] == "success"
        assert d["best_mean_reward"] == 100.0

    def test_l5_inference_result_serialization(self):
        """Test L5 inference result serialization."""
        import json
        from contracts.l5_inference_contracts import InferenceResult, SignalAction

        result = InferenceResult(
            model_id="ppo_primary",
            model_name="PPO Primary",
            model_type="PPO",
            raw_action=0.5,
            signal=SignalAction.LONG,
            confidence=0.8,
            latency_ms=10.5,
            bar_number=30,
        )

        json_str = result.model_dump_json()
        data = json.loads(json_str)

        assert data["signal"] == "LONG"
        assert data["raw_action"] == 0.5


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================


class TestContractPerformance:
    """Performance tests for contract validation."""

    def test_l1_calculated_features_performance(self):
        """Test L1 calculated features validation is fast."""
        import time
        from contracts.l1_feature_contracts import CalculatedFeatures

        start = time.perf_counter()
        for i in range(1000):
            CalculatedFeatures(
                time=dt.datetime.now(),
                log_ret_5m=0.001 * i,
                rsi_9=50.0 + i % 50,
            )
        elapsed = time.perf_counter() - start

        # Should be < 200ms for 1000 validations
        assert elapsed < 0.2, f"Too slow: {elapsed:.3f}s"

    def test_l5_inference_result_performance(self):
        """Test L5 inference result validation is fast."""
        import time
        from contracts.l5_inference_contracts import InferenceResult, SignalAction

        start = time.perf_counter()
        for i in range(1000):
            InferenceResult(
                model_id=f"model_{i}",
                model_name=f"Model {i}",
                model_type="PPO",
                raw_action=0.5,
                signal=SignalAction.LONG,
                confidence=0.8,
                latency_ms=10.0,
                bar_number=i % 60 + 1,
            )
        elapsed = time.perf_counter() - start

        # Should be < 300ms for 1000 validations
        assert elapsed < 0.3, f"Too slow: {elapsed:.3f}s"


# =============================================================================
# IMPORTS TEST
# =============================================================================


class TestContractImports:
    """Test that all contracts can be imported from __init__."""

    def test_all_exports_importable(self):
        """Test all exports from contracts __init__ are importable."""
        from contracts import (
            # L0
            L0XComKeys,
            create_ohlcv_acquisition_result,
            # L1
            L1XComKeys,
            FEATURE_CONTRACT,
            # L2
            L2XComKeys,
            create_feature_contract,
            # L3
            L3XComKeys,
            create_training_result,
            # L5
            L5XComKeys,
            OBSERVATION_CONTRACT,
            create_inference_result,
        )

        # Just check they're callable/accessible
        assert L0XComKeys is not None
        assert L1XComKeys is not None
        assert L2XComKeys is not None
        assert L3XComKeys is not None
        assert L5XComKeys is not None
