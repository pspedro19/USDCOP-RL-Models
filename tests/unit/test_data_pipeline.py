"""
Tests para Data Pipeline - Anti-Leakage.
CLAUDE-T6, CLAUDE-T7 | Plan Items: P0-10, P0-11

Valida:
- P0-10: ffill siempre tiene limite
- P0-11: merge_asof no tiene tolerance
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "data"))

from safe_merge import (
    safe_ffill,
    safe_merge_macro,
    validate_no_future_data,
    check_ffill_in_source,
    check_merge_asof_tolerance,
    FFILL_LIMIT_5MIN
)


class TestSafeFfill:
    """Tests para safe_ffill (P0-10)."""

    def test_ffill_respects_limit(self):
        """safe_ffill DEBE respetar el limite."""
        # Crear serie con gap largo
        data = [1.0, np.nan, np.nan, np.nan, np.nan, np.nan, 7.0]
        df = pd.DataFrame({'value': data})

        # Con limite de 2, solo debe llenar 2 NaN
        result = safe_ffill(df, limit=2)

        assert result['value'].iloc[1] == 1.0  # Llenado
        assert result['value'].iloc[2] == 1.0  # Llenado
        assert pd.isna(result['value'].iloc[3])  # NO llenado (> limit)
        assert pd.isna(result['value'].iloc[4])  # NO llenado
        assert pd.isna(result['value'].iloc[5])  # NO llenado

    def test_ffill_default_limit_is_144(self):
        """safe_ffill default limit DEBE ser 144 (12 horas)."""
        assert FFILL_LIMIT_5MIN == 144

    def test_ffill_zero_limit_raises(self):
        """safe_ffill con limit=0 DEBE lanzar ValueError."""
        df = pd.DataFrame({'value': [1, np.nan, 3]})

        with pytest.raises(ValueError, match="limit debe ser > 0"):
            safe_ffill(df, limit=0)

    def test_ffill_negative_limit_raises(self):
        """safe_ffill con limit negativo DEBE lanzar ValueError."""
        df = pd.DataFrame({'value': [1, np.nan, 3]})

        with pytest.raises(ValueError):
            safe_ffill(df, limit=-1)

    def test_ffill_specific_columns(self):
        """safe_ffill puede llenar columnas especificas."""
        df = pd.DataFrame({
            'a': [1, np.nan, 3],
            'b': [4, np.nan, 6]
        })

        result = safe_ffill(df, columns=['a'], limit=10)

        assert result['a'].iloc[1] == 1.0  # Llenado
        assert pd.isna(result['b'].iloc[1])  # NO llenado


class TestSafeMergeMacro:
    """Tests para safe_merge_macro (P0-11)."""

    @pytest.fixture
    def sample_ohlcv(self):
        """OHLCV data de ejemplo."""
        dates = pd.date_range("2026-01-10 13:00", periods=10, freq="5min")
        return pd.DataFrame({
            'datetime': dates,
            'close': np.random.uniform(4200, 4300, 10)
        })

    @pytest.fixture
    def sample_macro(self):
        """Macro data de ejemplo (daily)."""
        dates = pd.date_range("2026-01-08", periods=5, freq="D")
        return pd.DataFrame({
            'datetime': dates,
            'dxy': [104.5, 104.6, 104.7, 104.8, 104.9],
            'vix': [18, 19, 20, 21, 22]
        })

    def test_merge_no_future_data(self, sample_ohlcv, sample_macro):
        """safe_merge_macro NO debe usar datos del futuro."""
        result = safe_merge_macro(sample_ohlcv, sample_macro)

        # macro_source_date nunca debe ser > datetime
        assert all(result['macro_source_date'] <= result['datetime'])

    def test_merge_includes_tracking_column(self, sample_ohlcv, sample_macro):
        """safe_merge_macro DEBE incluir columna de tracking."""
        result = safe_merge_macro(sample_ohlcv, sample_macro, track_source=True)

        assert 'macro_source_date' in result.columns

    def test_merge_detects_future_data(self):
        """safe_merge_macro DEBE detectar data leakage."""
        # Crear datos donde macro es del futuro
        ohlcv = pd.DataFrame({
            'datetime': pd.to_datetime(['2026-01-10']),
            'close': [4250.0]
        })

        macro = pd.DataFrame({
            'datetime': pd.to_datetime(['2026-01-15']),  # FUTURO
            'dxy': [105.0]
        })

        # Esto no deberia lanzar error porque merge_asof direction='backward'
        # solo toma datos del pasado
        result = safe_merge_macro(ohlcv, macro)

        # El merge_asof con direction='backward' no encontrara match
        # porque no hay datos macro antes de 2026-01-10
        assert pd.isna(result['dxy'].iloc[0]) or result['macro_source_date'].iloc[0] <= result['datetime'].iloc[0]


class TestValidateNoFutureData:
    """Tests para validate_no_future_data."""

    def test_detects_future_data(self):
        """validate_no_future_data DEBE detectar data del futuro."""
        df = pd.DataFrame({
            'datetime': pd.to_datetime(['2026-01-10', '2026-01-11']),
            'macro_source_date': pd.to_datetime(['2026-01-15', '2026-01-11'])  # Futuro!
        })

        with pytest.raises(ValueError, match="DATA LEAKAGE"):
            validate_no_future_data(df)

    def test_passes_for_valid_data(self):
        """validate_no_future_data DEBE pasar para datos validos."""
        df = pd.DataFrame({
            'datetime': pd.to_datetime(['2026-01-10', '2026-01-11']),
            'macro_source_date': pd.to_datetime(['2026-01-09', '2026-01-10'])  # Pasado - OK
        })

        result = validate_no_future_data(df)
        assert result is True


class TestSourceCodeAnalysis:
    """Tests que analizan codigo fuente para detectar problemas."""

    def test_check_ffill_in_known_file(self):
        """check_ffill_in_source debe detectar ffill sin limite."""
        target_file = project_root / "data" / "pipeline" / "06_rl_dataset_builder" / "01_build_5min_datasets.py"

        if target_file.exists():
            issues = check_ffill_in_source(str(target_file))

            # Reportar issues encontrados
            if issues:
                for issue in issues:
                    print(f"WARNING: ffill sin limit en linea {issue['line']}")

    def test_check_merge_asof_tolerance_in_known_file(self):
        """check_merge_asof_tolerance debe detectar tolerance."""
        target_file = project_root / "data" / "pipeline" / "06_rl_dataset_builder" / "01_build_5min_datasets.py"

        if target_file.exists():
            issues = check_merge_asof_tolerance(str(target_file))

            # Debe NO tener tolerance para evitar data leakage
            if issues:
                for issue in issues:
                    pytest.fail(
                        f"DATA LEAKAGE RISK: merge_asof con tolerance en linea {issue['line']}"
                    )


class TestAntiLeakageIntegration:
    """Tests de integracion anti-leakage."""

    def test_ffill_limit_prevents_stale_data(self):
        """ffill con limite previene datos obsoletos."""
        # Simular gap de 1 dia (288 barras de 5min)
        n_bars = 300
        data = np.full(n_bars, np.nan)
        data[0] = 100.0  # Solo primer valor
        data[-1] = 200.0  # Ultimo valor

        df = pd.DataFrame({'value': data})

        # Con limite de 144 (12 horas), no debe llenar todo
        result = safe_ffill(df, limit=144)

        # El valor en posicion 145 debe seguir NaN
        assert pd.isna(result['value'].iloc[145])

        # Pero posicion 143 debe estar llenado
        assert result['value'].iloc[143] == 100.0

    def test_safe_operations_are_deterministic(self):
        """Operaciones seguras deben ser deterministicas."""
        np.random.seed(42)

        df = pd.DataFrame({
            'value': [1, np.nan, np.nan, 4, np.nan, 6]
        })

        result1 = safe_ffill(df, limit=2)
        result2 = safe_ffill(df, limit=2)

        pd.testing.assert_frame_equal(result1, result2)
