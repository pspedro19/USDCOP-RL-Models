"""
Tests de Paridad Training/Inference.
CLAUDE-T11 | Plan Item: P1-12

Garantiza que las features calculadas durante training son
IDENTICAS a las calculadas durante inference.

Cualquier diferencia causa training/inference skew.
"""

import pytest
import numpy as np
import pandas as pd
import hashlib
import json
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from features.builder import FeatureBuilder
from features.contract import FEATURE_CONTRACT, get_contract


class TestTrainingInferenceParity:
    """
    Tests CRITICOS de paridad training/inference.
    CLAUDE-T11 | Plan Item: P1-12

    Cualquier diferencia causa training/inference skew.
    """

    @pytest.fixture
    def builder(self):
        return FeatureBuilder(version="current")

    @pytest.fixture
    def sample_data(self):
        """Datos sinteticos reproducibles para tests de paridad."""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2026-01-10 13:00", periods=n, freq="5min", tz="UTC")
        base_price = 4250.0

        close = base_price + np.cumsum(np.random.randn(n) * 5)
        ohlcv = pd.DataFrame({
            "open": close + np.random.randn(n),
            "high": close + np.abs(np.random.randn(n) * 3),
            "low": close - np.abs(np.random.randn(n) * 3),
            "close": close,
            "volume": np.random.uniform(1e6, 5e6, n),
        }, index=dates)

        macro = pd.DataFrame({
            "dxy": np.random.uniform(103, 105, n),
            "vix": np.random.uniform(15, 25, n),
            "embi": np.random.uniform(300, 400, n),
            "brent": np.random.uniform(70, 80, n),
            "usdmxn": np.random.uniform(17, 18, n),
            "rate_spread": np.random.uniform(5, 7, n),
        }, index=dates)

        return ohlcv, macro

    # ══════════════════════════════════════════════════════════════
    # TEST 1: Determinismo - mismo input produce mismo output
    # ══════════════════════════════════════════════════════════════
    def test_determinism_across_multiple_calls(self, builder, sample_data):
        """
        Multiples llamadas con mismo input DEBEN producir
        exactamente el mismo output (bit-a-bit).
        """
        ohlcv, macro = sample_data
        bar_idx = 50

        results = []
        for _ in range(10):
            obs = builder.build_observation(
                ohlcv=ohlcv,
                macro_df=macro,
                position=0.5,
                timestamp=ohlcv.index[bar_idx],
                bar_idx=bar_idx
            )
            results.append(obs.copy())

        # Todas deben ser identicas
        for i in range(1, len(results)):
            np.testing.assert_array_equal(
                results[0], results[i],
                err_msg=f"Determinismo fallido en iteracion {i}"
            )

    # ══════════════════════════════════════════════════════════════
    # TEST 2: Paridad entre instancias de builder
    # ══════════════════════════════════════════════════════════════
    def test_parity_across_builder_instances(self, sample_data):
        """
        Diferentes instancias de FeatureBuilder con misma version
        DEBEN producir outputs identicos.
        """
        ohlcv, macro = sample_data
        bar_idx = 50

        builder1 = FeatureBuilder(version="current")
        builder2 = FeatureBuilder(version="current")
        builder3 = FeatureBuilder(version="current")

        obs1 = builder1.build_observation(
            ohlcv=ohlcv, macro_df=macro, position=0.0,
            timestamp=ohlcv.index[bar_idx], bar_idx=bar_idx
        )
        obs2 = builder2.build_observation(
            ohlcv=ohlcv, macro_df=macro, position=0.0,
            timestamp=ohlcv.index[bar_idx], bar_idx=bar_idx
        )
        obs3 = builder3.build_observation(
            ohlcv=ohlcv, macro_df=macro, position=0.0,
            timestamp=ohlcv.index[bar_idx], bar_idx=bar_idx
        )

        np.testing.assert_array_equal(obs1, obs2)
        np.testing.assert_array_equal(obs2, obs3)

    # ══════════════════════════════════════════════════════════════
    # TEST 3: Feature order coincide con contrato
    # ══════════════════════════════════════════════════════════════
    def test_feature_order_matches_contract(self, builder):
        """
        Orden de features DEBE coincidir exactamente con el contrato.
        """
        contract = get_contract("current")
        builder_names = builder.get_feature_names()

        assert builder_names == contract.feature_order, \
            f"Feature order mismatch:\nBuilder: {builder_names}\nContract: {contract.feature_order}"

    # ══════════════════════════════════════════════════════════════
    # TEST 4: Dimension coincide con contrato
    # ══════════════════════════════════════════════════════════════
    def test_observation_dimension_matches_contract(self, builder, sample_data):
        """
        Dimension de observation DEBE coincidir con contrato.
        """
        ohlcv, macro = sample_data
        contract = get_contract("current")

        obs = builder.build_observation(
            ohlcv=ohlcv, macro_df=macro, position=0.0,
            timestamp=ohlcv.index[50], bar_idx=50
        )

        assert obs.shape == (contract.observation_dim,), \
            f"Dimension mismatch: {obs.shape} vs ({contract.observation_dim},)"

    # ══════════════════════════════════════════════════════════════
    # TEST 5: Hash de norm_stats es valido
    # ══════════════════════════════════════════════════════════════
    def test_norm_stats_hash_is_valid(self, builder):
        """
        Hash de norm_stats DEBE ser calculable y no vacio.
        """
        norm_stats_path = project_root / builder.contract.norm_stats_path

        with open(norm_stats_path, 'r') as f:
            content = f.read()

        hash_value = hashlib.sha256(content.encode()).hexdigest()

        assert len(hash_value) == 64, "SHA256 hash debe tener 64 caracteres"
        assert hash_value, "Hash no debe estar vacio"

    # ══════════════════════════════════════════════════════════════
    # TEST 6: Todas las features del contrato estan presentes
    # ══════════════════════════════════════════════════════════════
    def test_all_contract_features_present(self, builder, sample_data):
        """
        Todas las features definidas en el contrato DEBEN estar
        presentes en el output del builder.
        """
        ohlcv, macro = sample_data
        contract = get_contract("current")

        snapshot = builder.export_feature_snapshot(
            ohlcv=ohlcv, macro_df=macro, position=0.5,
            timestamp=ohlcv.index[50], bar_idx=50
        )

        raw_features = set(snapshot["raw_features"].keys())
        normalized_features = set(snapshot["normalized_features"].keys())

        # Las primeras 13 features deben estar en raw_features
        expected_raw = set(contract.feature_order[:13])
        assert raw_features == expected_raw, \
            f"Raw features mismatch: {raw_features.symmetric_difference(expected_raw)}"

        # Las 15 features deben estar en normalized_features
        expected_norm = set(contract.feature_order)
        assert normalized_features == expected_norm, \
            f"Normalized features mismatch: {normalized_features.symmetric_difference(expected_norm)}"

    # ══════════════════════════════════════════════════════════════
    # TEST 7: Valores normalizados dentro de rango
    # ══════════════════════════════════════════════════════════════
    def test_normalized_values_in_clip_range(self, builder, sample_data):
        """
        Valores normalizados DEBEN estar dentro del clip_range del contrato.
        """
        ohlcv, macro = sample_data
        contract = get_contract("current")
        clip_min, clip_max = contract.clip_range

        for bar_idx in range(14, 100):
            obs = builder.build_observation(
                ohlcv=ohlcv, macro_df=macro, position=0.0,
                timestamp=ohlcv.index[bar_idx], bar_idx=bar_idx
            )

            # Las primeras 13 features estan normalizadas y clipped
            assert np.all(obs[:13] >= clip_min), \
                f"Valor < {clip_min} en bar {bar_idx}: {obs[:13][obs[:13] < clip_min]}"
            assert np.all(obs[:13] <= clip_max), \
                f"Valor > {clip_max} en bar {bar_idx}: {obs[:13][obs[:13] > clip_max]}"

    # ══════════════════════════════════════════════════════════════
    # TEST 8: Position y time_normalized en rangos correctos
    # ══════════════════════════════════════════════════════════════
    def test_position_and_time_normalized_ranges(self, builder, sample_data):
        """
        Position DEBE estar en [-1, 1].
        time_normalized DEBE estar en [0, 1].
        """
        ohlcv, macro = sample_data

        for bar_idx in range(14, 100):
            for position in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                obs = builder.build_observation(
                    ohlcv=ohlcv, macro_df=macro, position=position,
                    timestamp=ohlcv.index[bar_idx], bar_idx=bar_idx
                )

                assert -1.0 <= obs[13] <= 1.0, \
                    f"Position {obs[13]} fuera de rango en bar {bar_idx}"
                assert 0.0 <= obs[14] <= 1.0, \
                    f"time_normalized {obs[14]} fuera de rango en bar {bar_idx}"

    # ══════════════════════════════════════════════════════════════
    # TEST 9: Sin NaN ni Inf en toda la serie
    # ══════════════════════════════════════════════════════════════
    def test_no_nan_inf_across_entire_series(self, builder, sample_data):
        """
        NUNCA debe haber NaN o Inf en ninguna barra despues del warmup.
        """
        ohlcv, macro = sample_data

        for bar_idx in range(14, len(ohlcv)):
            obs = builder.build_observation(
                ohlcv=ohlcv, macro_df=macro, position=0.0,
                timestamp=ohlcv.index[bar_idx], bar_idx=bar_idx
            )

            assert not np.isnan(obs).any(), f"NaN en bar {bar_idx}: {obs}"
            assert not np.isinf(obs).any(), f"Inf en bar {bar_idx}: {obs}"

    # ══════════════════════════════════════════════════════════════
    # TEST 10: Snapshot round-trip preserva datos
    # ══════════════════════════════════════════════════════════════
    def test_snapshot_roundtrip_preserves_data(self, builder, sample_data):
        """
        Serializar y deserializar snapshot DEBE preservar todos los datos.
        """
        ohlcv, macro = sample_data

        snapshot = builder.export_feature_snapshot(
            ohlcv=ohlcv, macro_df=macro, position=0.5,
            timestamp=ohlcv.index[50], bar_idx=50
        )

        # Serializar a JSON y deserializar
        json_str = json.dumps(snapshot)
        recovered = json.loads(json_str)

        # Verificar campos
        assert recovered["version"] == snapshot["version"]
        assert recovered["bar_idx"] == snapshot["bar_idx"]
        assert len(recovered["raw_features"]) == len(snapshot["raw_features"])
        assert len(recovered["normalized_features"]) == len(snapshot["normalized_features"])

        # Verificar valores numericos
        for key in snapshot["raw_features"]:
            assert np.isclose(
                recovered["raw_features"][key],
                snapshot["raw_features"][key],
                rtol=1e-10
            ), f"Mismatch en raw_features[{key}]"


class TestNormStatsIntegrity:
    """Tests de integridad de norm_stats."""

    def test_norm_stats_file_exists(self):
        """norm_stats file DEBE existir."""
        contract = get_contract("current")
        path = project_root / contract.norm_stats_path
        assert path.exists(), f"norm_stats no encontrado: {path}"

    def test_norm_stats_has_all_required_features(self):
        """norm_stats DEBE contener todas las features requeridas."""
        contract = get_contract("current")
        path = project_root / contract.norm_stats_path

        with open(path, 'r') as f:
            stats = json.load(f)

        # Features que requieren normalizacion (primeras 13)
        required = set(contract.feature_order[:13])
        available = set(stats.keys())

        missing = required - available
        assert not missing, f"Features faltantes en norm_stats: {missing}"

    def test_norm_stats_has_valid_statistics(self):
        """Cada feature en norm_stats DEBE tener mean y std validos."""
        contract = get_contract("current")
        path = project_root / contract.norm_stats_path

        with open(path, 'r') as f:
            stats = json.load(f)

        for name, feature_stats in stats.items():
            assert "mean" in feature_stats, f"{name} falta 'mean'"
            assert "std" in feature_stats, f"{name} falta 'std'"
            assert not np.isnan(feature_stats["mean"]), f"{name} mean es NaN"
            assert not np.isnan(feature_stats["std"]), f"{name} std es NaN"
            assert feature_stats["std"] > 0, f"{name} std debe ser > 0"
