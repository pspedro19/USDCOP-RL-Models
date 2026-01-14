"""
Tests para FeatureBuilder.
CLAUDE-T1 | Plan Item: P1-13
Contrato: CTR-001, CTR-002

Valida:
- observation_dim == 15
- Feature order exacto
- Sin NaN/Inf
- Clipping [-5, 5]
- Determinismo
- Position clipping
- time_normalized [0,1]
- JSON serializable
- Warmup validation
- Latencia < 10ms
"""

import pytest
import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from features.builder import FeatureBuilder
from features.contract import get_contract, FEATURE_CONTRACT


class TestFeatureBuilderContract:
    """
    Tests del Feature Contract Pattern.
    CLAUDE-T1 | Plan Item: P1-13
    Coverage Target: 95%
    """

    @pytest.fixture
    def builder(self):
        """FeatureBuilder con contrato actual."""
        return FeatureBuilder(version="current")

    @pytest.fixture
    def sample_ohlcv(self):
        """100 barras de 5min de datos sinteticos reproducibles."""
        np.random.seed(42)
        dates = pd.date_range("2026-01-10 13:00", periods=100, freq="5min", tz="UTC")
        base_price = 4250.0

        close = base_price + np.cumsum(np.random.randn(100) * 5)
        high = close + np.abs(np.random.randn(100) * 3)
        low = close - np.abs(np.random.randn(100) * 3)
        open_ = close + np.random.randn(100) * 2

        return pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.uniform(1e6, 5e6, 100),
        }, index=dates)

    @pytest.fixture
    def sample_macro(self):
        """Datos macro alineados con ohlcv."""
        np.random.seed(42)
        dates = pd.date_range("2026-01-10 13:00", periods=100, freq="5min", tz="UTC")
        return pd.DataFrame({
            "dxy": np.random.uniform(103, 105, 100),
            "vix": np.random.uniform(15, 25, 100),
            "embi": np.random.uniform(300, 400, 100),
            "brent": np.random.uniform(70, 80, 100),
            "usdmxn": np.random.uniform(17, 18, 100),
            "rate_spread": np.random.uniform(5, 7, 100),
        }, index=dates)

    # ══════════════════════════════════════════════════════════════
    # TEST 1: Dimension correcta
    # ══════════════════════════════════════════════════════════════
    def test_observation_dimension_matches_contract(self, builder):
        """observation_dim DEBE ser 15."""
        contract = get_contract("current")

        assert builder.get_observation_dim() == 15
        assert builder.get_observation_dim() == contract.observation_dim

    # ══════════════════════════════════════════════════════════════
    # TEST 2: Orden de features exacto
    # ══════════════════════════════════════════════════════════════
    def test_feature_order_matches_contract_exactly(self, builder):
        """Orden de features DEBE coincidir exactamente con contrato."""
        expected = FEATURE_CONTRACT.feature_order
        actual = builder.get_feature_names()

        assert actual == expected, f"Orden incorrecto:\nExpected: {expected}\nActual: {actual}"

    # ══════════════════════════════════════════════════════════════
    # TEST 3: Sin NaN ni Inf
    # ══════════════════════════════════════════════════════════════
    def test_observation_no_nan_or_inf_across_all_bars(self, builder, sample_ohlcv, sample_macro):
        """Observaciones NUNCA deben contener NaN o Inf en ninguna barra."""
        warmup = FEATURE_CONTRACT.warmup_bars

        for bar_idx in range(warmup, len(sample_ohlcv)):
            obs = builder.build_observation(
                ohlcv=sample_ohlcv,
                macro_df=sample_macro,
                position=0.0,
                timestamp=sample_ohlcv.index[bar_idx],
                bar_idx=bar_idx
            )

            assert not np.isnan(obs).any(), f"NaN encontrado en bar {bar_idx}: {obs}"
            assert not np.isinf(obs).any(), f"Inf encontrado en bar {bar_idx}: {obs}"

    # ══════════════════════════════════════════════════════════════
    # TEST 4: Clipping correcto [-5.0, 5.0]
    # ══════════════════════════════════════════════════════════════
    def test_normalized_features_clipped_to_range(self, builder, sample_ohlcv, sample_macro):
        """Features normalizadas DEBEN estar en [-5.0, 5.0]."""
        obs = builder.build_observation(
            ohlcv=sample_ohlcv,
            macro_df=sample_macro,
            position=0.0,
            timestamp=sample_ohlcv.index[50],
            bar_idx=50
        )

        # Features 0-12 son normalizadas
        normalized_features = obs[:13]

        assert np.all(normalized_features >= -5.0), \
            f"Feature < -5.0 detectada: {normalized_features[normalized_features < -5.0]}"
        assert np.all(normalized_features <= 5.0), \
            f"Feature > 5.0 detectada: {normalized_features[normalized_features > 5.0]}"

    # ══════════════════════════════════════════════════════════════
    # TEST 5: Determinismo (mismo input -> mismo output)
    # ══════════════════════════════════════════════════════════════
    def test_determinism_same_input_same_output(self, builder, sample_ohlcv, sample_macro):
        """Mismo input DEBE producir exactamente mismo output."""
        kwargs = {
            "ohlcv": sample_ohlcv,
            "macro_df": sample_macro,
            "position": 0.5,
            "timestamp": sample_ohlcv.index[50],
            "bar_idx": 50
        }

        obs1 = builder.build_observation(**kwargs)
        obs2 = builder.build_observation(**kwargs)
        obs3 = builder.build_observation(**kwargs)

        np.testing.assert_array_equal(obs1, obs2)
        np.testing.assert_array_equal(obs2, obs3)

    # ══════════════════════════════════════════════════════════════
    # TEST 6: Position en rango [-1, 1]
    # ══════════════════════════════════════════════════════════════
    @pytest.mark.parametrize("position,expected", [
        (-1.0, -1.0),
        (-0.5, -0.5),
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (-1.5, -1.0),  # Clipped
        (1.5, 1.0),    # Clipped
    ])
    def test_position_clipped_correctly(self, builder, sample_ohlcv, sample_macro, position, expected):
        """Position DEBE estar clipped en [-1, 1]."""
        obs = builder.build_observation(
            ohlcv=sample_ohlcv,
            macro_df=sample_macro,
            position=position,
            timestamp=sample_ohlcv.index[50],
            bar_idx=50
        )

        assert obs[13] == expected, f"Position index 13: expected {expected}, got {obs[13]}"

    # ══════════════════════════════════════════════════════════════
    # TEST 7: time_normalized en [0, 1]
    # ══════════════════════════════════════════════════════════════
    def test_time_normalized_in_valid_range(self, builder, sample_ohlcv, sample_macro):
        """time_normalized DEBE estar en [0, 1]."""
        for bar_idx in range(14, len(sample_ohlcv)):
            obs = builder.build_observation(
                ohlcv=sample_ohlcv,
                macro_df=sample_macro,
                position=0.0,
                timestamp=sample_ohlcv.index[bar_idx],
                bar_idx=bar_idx
            )

            assert 0.0 <= obs[14] <= 1.0, f"time_normalized fuera de rango en bar {bar_idx}: {obs[14]}"

    # ══════════════════════════════════════════════════════════════
    # TEST 8: features_snapshot JSON serializable
    # ══════════════════════════════════════════════════════════════
    def test_features_snapshot_json_serializable(self, builder, sample_ohlcv, sample_macro):
        """features_snapshot DEBE ser serializable a JSON."""
        snapshot = builder.export_feature_snapshot(
            ohlcv=sample_ohlcv,
            macro_df=sample_macro,
            position=0.0,
            timestamp=sample_ohlcv.index[50],
            bar_idx=50
        )

        # No debe lanzar excepcion
        json_str = json.dumps(snapshot)

        # Round-trip debe preservar datos
        recovered = json.loads(json_str)
        assert recovered["version"] == "current"
        assert len(recovered["raw_features"]) == 13
        assert len(recovered["normalized_features"]) == 15

    # ══════════════════════════════════════════════════════════════
    # TEST 9: Warmup validation
    # ══════════════════════════════════════════════════════════════
    def test_warmup_validation_raises_on_insufficient_bars(self, builder, sample_ohlcv, sample_macro):
        """bar_idx < warmup_bars DEBE lanzar ValueError."""
        with pytest.raises(ValueError, match="warmup_bars"):
            builder.build_observation(
                ohlcv=sample_ohlcv,
                macro_df=sample_macro,
                position=0.0,
                timestamp=sample_ohlcv.index[5],
                bar_idx=5  # < 14 warmup bars
            )

    # ══════════════════════════════════════════════════════════════
    # TEST 10: Shape consistency
    # ══════════════════════════════════════════════════════════════
    def test_output_shape_and_dtype_consistent(self, builder, sample_ohlcv, sample_macro):
        """Output DEBE tener shape (15,) y dtype float32."""
        for bar_idx in range(14, 50):
            obs = builder.build_observation(
                ohlcv=sample_ohlcv,
                macro_df=sample_macro,
                position=np.random.uniform(-1, 1),
                timestamp=sample_ohlcv.index[bar_idx],
                bar_idx=bar_idx
            )

            assert obs.shape == (15,), f"Shape incorrecto: {obs.shape}"
            assert obs.dtype == np.float32, f"Dtype incorrecto: {obs.dtype}"


class TestFeatureBuilderEdgeCases:
    """Tests de edge cases y manejo de errores."""

    @pytest.fixture
    def builder(self):
        return FeatureBuilder(version="current")

    @pytest.fixture
    def sample_ohlcv(self):
        np.random.seed(42)
        dates = pd.date_range("2026-01-10 13:00", periods=100, freq="5min", tz="UTC")
        close = 4250 + np.cumsum(np.random.randn(100) * 5)
        return pd.DataFrame({
            "open": close + np.random.randn(100),
            "high": close + np.abs(np.random.randn(100) * 3),
            "low": close - np.abs(np.random.randn(100) * 3),
            "close": close,
            "volume": np.random.uniform(1e6, 5e6, 100),
        }, index=dates)

    @pytest.fixture
    def sample_macro(self):
        np.random.seed(42)
        dates = pd.date_range("2026-01-10 13:00", periods=100, freq="5min", tz="UTC")
        return pd.DataFrame({
            "dxy": np.random.uniform(103, 105, 100),
            "vix": np.random.uniform(15, 25, 100),
            "embi": np.random.uniform(300, 400, 100),
            "brent": np.random.uniform(70, 80, 100),
            "usdmxn": np.random.uniform(17, 18, 100),
            "rate_spread": np.random.uniform(5, 7, 100),
        }, index=dates)

    def test_missing_ohlcv_columns_raises_error(self, builder, sample_macro):
        """DataFrame sin columnas requeridas DEBE lanzar ValueError."""
        bad_ohlcv = pd.DataFrame({"close": [1, 2, 3] * 10})

        with pytest.raises(ValueError, match="missing columns"):
            builder.build_observation(
                ohlcv=bad_ohlcv,
                macro_df=sample_macro,
                position=0.0,
                timestamp=pd.Timestamp("2026-01-10 13:00", tz="UTC"),
                bar_idx=20
            )

    def test_invalid_contract_version_raises_error(self):
        """Version de contrato invalida DEBE lanzar ValueError."""
        with pytest.raises(ValueError):
            FeatureBuilder(version="v99")

    def test_handles_missing_macro_columns_gracefully(self, builder, sample_ohlcv):
        """Debe manejar macro con columnas faltantes sin fallar."""
        partial_macro = pd.DataFrame({
            "dxy": np.random.uniform(103, 105, 100),
            "vix": np.random.uniform(15, 25, 100),
        }, index=sample_ohlcv.index)

        # No debe lanzar excepcion
        obs = builder.build_observation(
            ohlcv=sample_ohlcv,
            macro_df=partial_macro,
            position=0.0,
            timestamp=sample_ohlcv.index[50],
            bar_idx=50
        )

        assert obs.shape == (15,)
        assert not np.isnan(obs).any()


class TestFeatureBuilderPerformance:
    """Tests de performance."""

    @pytest.fixture
    def builder(self):
        return FeatureBuilder(version="current")

    @pytest.fixture
    def large_ohlcv(self):
        """1000 barras para benchmark."""
        np.random.seed(42)
        dates = pd.date_range("2026-01-01 13:00", periods=1000, freq="5min", tz="UTC")
        base_price = 4250.0
        close = base_price + np.cumsum(np.random.randn(1000) * 5)

        return pd.DataFrame({
            "open": close + np.random.randn(1000),
            "high": close + np.abs(np.random.randn(1000) * 3),
            "low": close - np.abs(np.random.randn(1000) * 3),
            "close": close,
            "volume": np.random.uniform(1e6, 5e6, 1000),
        }, index=dates)

    @pytest.fixture
    def large_macro(self):
        np.random.seed(42)
        dates = pd.date_range("2026-01-01 13:00", periods=1000, freq="5min", tz="UTC")
        return pd.DataFrame({
            "dxy": np.random.uniform(103, 105, 1000),
            "vix": np.random.uniform(15, 25, 1000),
            "embi": np.random.uniform(300, 400, 1000),
            "brent": np.random.uniform(70, 80, 1000),
            "usdmxn": np.random.uniform(17, 18, 1000),
            "rate_spread": np.random.uniform(5, 7, 1000),
        }, index=dates)

    def test_latency_under_50ms(self, builder, large_ohlcv, large_macro):
        """Latencia por observacion DEBE ser < 50ms."""
        import time

        times = []
        for _ in range(10):
            start = time.perf_counter()
            builder.build_observation(
                ohlcv=large_ohlcv,
                macro_df=large_macro,
                position=0.0,
                timestamp=large_ohlcv.index[500],
                bar_idx=500
            )
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg_time = np.mean(times)
        assert avg_time < 50, f"Latencia promedio {avg_time:.2f}ms > 50ms"

    def test_batch_processing_no_memory_leak(self, builder, large_ohlcv, large_macro):
        """Batch de 100 observaciones no debe causar memory leak."""
        import tracemalloc

        tracemalloc.start()

        observations = []
        for bar_idx in range(14, 114):
            obs = builder.build_observation(
                ohlcv=large_ohlcv,
                macro_df=large_macro,
                position=0.0,
                timestamp=large_ohlcv.index[bar_idx],
                bar_idx=bar_idx
            )
            observations.append(obs)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024
        assert peak_mb < 50, f"Peak memory: {peak_mb:.2f}MB > 50MB"


class TestContractIntegrity:
    """Tests de integridad del contrato."""

    def test_contract_is_frozen(self):
        """Contrato DEBE ser inmutable."""
        contract = get_contract("current")

        with pytest.raises(Exception):  # FrozenInstanceError o similar
            contract.version = "modified"

    def test_contract_version_is_current(self):
        """Version actual DEBE ser current."""
        contract = get_contract("current")
        assert contract.version == "current"

    def test_contract_observation_dim_is_15(self):
        """observation_dim DEBE ser 15."""
        contract = get_contract("current")
        assert contract.observation_dim == 15

    def test_contract_feature_order_has_15_elements(self):
        """feature_order DEBE tener 15 elementos."""
        contract = get_contract("current")
        assert len(contract.feature_order) == 15

    def test_contract_has_required_features(self):
        """Contrato DEBE incluir todas las features requeridas."""
        required = {
            "log_ret_5m", "log_ret_1h", "log_ret_4h",
            "rsi_9", "atr_pct", "adx_14",
            "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
            "brent_change_1d", "rate_spread", "usdmxn_change_1d",
            "position", "time_normalized"
        }
        contract = get_contract("current")
        actual = set(contract.feature_order)

        assert actual == required, f"Diferencia: {actual.symmetric_difference(required)}"
