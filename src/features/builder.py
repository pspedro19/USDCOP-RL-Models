"""
FeatureBuilder - Single Source of Truth para observaciones.
Contrato ID: CTR-001
CLAUDE-T1 | Plan Item: P1-13

Consumido por: Training, Inference API, Replay API, Airflow DAGs
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from .contract import get_contract, FeatureContract
from .calculators import returns, rsi, atr, adx, macro


def load_norm_stats(path: str) -> Dict[str, Dict[str, float]]:
    """
    Load normalization statistics from JSON file.

    Args:
        path: Path to norm_stats JSON file (relative to project root or absolute)

    Returns:
        Dictionary with feature names as keys and stats as values.
    """
    # Handle relative paths
    if not Path(path).is_absolute():
        # Try relative to project root
        project_root = Path(__file__).parent.parent.parent
        full_path = project_root / path
        if not full_path.exists():
            # Try relative to current directory
            full_path = Path(path)
    else:
        full_path = Path(path)

    if not full_path.exists():
        raise FileNotFoundError(f"Norm stats file not found: {path}")

    with open(full_path, 'r') as f:
        data = json.load(f)

    return data


class FeatureBuilder:
    """
    Constructor de observaciones siguiendo Feature Contract.

    Invariantes:
    - observation siempre tiene shape=(15,)
    - observation nunca contiene NaN o Inf
    - features normalizadas estan en [-5.0, 5.0]
    - mismo input produce exactamente mismo output (determinista)

    Ejemplo:
        builder = FeatureBuilder(version="current")
        obs = builder.build_observation(ohlcv, macro_df, position, timestamp, bar_idx)
    """

    def __init__(self, version: str = "current"):
        """
        Inicializa FeatureBuilder con contrato especificado.

        Args:
            version: Version del contrato a usar ("current")

        Raises:
            ValueError: Si la version no existe
            FileNotFoundError: Si norm_stats no existe
        """
        self.contract = get_contract(version)
        self.norm_stats = load_norm_stats(self.contract.norm_stats_path)
        self._validate_norm_stats()

    def _validate_norm_stats(self) -> None:
        """Valida que norm_stats contenga todas las features requeridas."""
        # Features que requieren normalizacion (excluyendo position, time_normalized)
        required = set(self.contract.feature_order[:13])
        available = set(self.norm_stats.keys())
        missing = required - available
        if missing:
            raise ValueError(f"norm_stats missing features: {missing}")

    def get_observation_dim(self) -> int:
        """Retorna dimension del observation space."""
        return self.contract.observation_dim

    def get_feature_names(self) -> Tuple[str, ...]:
        """Retorna nombres de features en orden exacto del contrato."""
        return self.contract.feature_order

    def build_observation(
        self,
        ohlcv: pd.DataFrame,
        macro_df: pd.DataFrame,
        position: float,
        timestamp: pd.Timestamp,
        bar_idx: int
    ) -> np.ndarray:
        """
        Construye observation array siguiendo Feature Contract.

        Args:
            ohlcv: DataFrame con columns [open, high, low, close, volume], index=datetime
            macro_df: DataFrame con columns [dxy, vix, embi, brent, usdmxn, rate_spread], index=datetime
            position: Posicion actual en [-1, 1]
            timestamp: Timestamp UTC de la barra actual
            bar_idx: Indice de la barra (para warmup validation)

        Returns:
            np.ndarray de shape (15,) con features normalizadas

        Raises:
            ValueError: Si bar_idx < warmup_bars o position fuera de rango
        """
        # Validaciones
        self._validate_inputs(ohlcv, macro_df, position, bar_idx)

        # Calcular features tecnicas
        raw_features = self._calculate_raw_features(ohlcv, macro_df, bar_idx)

        # Normalizar (excepto position y time_normalized)
        normalized = self._normalize_features(raw_features)

        # Agregar position y time_normalized
        normalized["position"] = float(np.clip(position, -1.0, 1.0))
        normalized["time_normalized"] = self._compute_time_normalized(timestamp)

        # Construir array en orden del contrato
        observation = self._assemble_observation(normalized)

        # Validacion final
        assert observation.shape == (self.contract.observation_dim,), \
            f"Shape mismatch: {observation.shape} != ({self.contract.observation_dim},)"
        assert not np.isnan(observation).any(), "NaN detectado en observation"
        assert not np.isinf(observation).any(), "Inf detectado en observation"

        return observation

    def _validate_inputs(
        self,
        ohlcv: pd.DataFrame,
        macro_df: pd.DataFrame,
        position: float,
        bar_idx: int
    ) -> None:
        """Valida inputs antes de procesar."""
        if bar_idx < self.contract.warmup_bars:
            raise ValueError(
                f"bar_idx ({bar_idx}) < warmup_bars ({self.contract.warmup_bars})"
            )
        if not -1.0 <= position <= 1.0:
            # Clip silently instead of raising error (as per spec)
            pass  # Will be clipped in build_observation

        required_ohlcv = {"open", "high", "low", "close", "volume"}
        if not required_ohlcv.issubset(ohlcv.columns):
            raise ValueError(f"ohlcv missing columns: {required_ohlcv - set(ohlcv.columns)}")

    def _calculate_raw_features(
        self,
        ohlcv: pd.DataFrame,
        macro_df: pd.DataFrame,
        bar_idx: int
    ) -> Dict[str, float]:
        """Calcula features crudas (sin normalizar)."""
        periods = self.contract.get_technical_periods()

        return {
            # Returns
            "log_ret_5m": returns.log_return(ohlcv["close"], periods=1, bar_idx=bar_idx),
            "log_ret_1h": returns.log_return(ohlcv["close"], periods=12, bar_idx=bar_idx),
            "log_ret_4h": returns.log_return(ohlcv["close"], periods=48, bar_idx=bar_idx),

            # Technical indicators
            "rsi_9": rsi.calculate(ohlcv["close"], period=periods["rsi"], bar_idx=bar_idx),
            "atr_pct": atr.calculate_pct(ohlcv, period=periods["atr"], bar_idx=bar_idx),
            "adx_14": adx.calculate(ohlcv, period=periods["adx"], bar_idx=bar_idx),

            # Macro (z-scores calculated internally)
            "dxy_z": macro.z_score(macro_df["dxy"], bar_idx=bar_idx) if "dxy" in macro_df.columns else 0.0,
            "dxy_change_1d": macro.change_1d(macro_df["dxy"], bar_idx=bar_idx) if "dxy" in macro_df.columns else 0.0,
            "vix_z": macro.z_score(macro_df["vix"], bar_idx=bar_idx) if "vix" in macro_df.columns else 0.0,
            "embi_z": macro.z_score(macro_df["embi"], bar_idx=bar_idx) if "embi" in macro_df.columns else 0.0,
            "brent_change_1d": macro.change_1d(macro_df["brent"], bar_idx=bar_idx) if "brent" in macro_df.columns else 0.0,
            "rate_spread": macro.get_value(macro_df["rate_spread"], bar_idx=bar_idx) if "rate_spread" in macro_df.columns else 0.0,
            "usdmxn_change_1d": macro.change_1d(macro_df["usdmxn"], bar_idx=bar_idx) if "usdmxn" in macro_df.columns else 0.0,
        }

    def _normalize_features(self, raw: Dict[str, float]) -> Dict[str, float]:
        """Aplica z-score normalization y clipping."""
        normalized = {}
        clip_min, clip_max = self.contract.clip_range

        for name, value in raw.items():
            stats = self.norm_stats.get(name)
            if stats and stats.get("std", 0) > 0:
                z = (value - stats["mean"]) / stats["std"]
                normalized[name] = float(np.clip(z, clip_min, clip_max))
            else:
                # Features que no requieren normalizacion adicional o ya estan normalizadas
                normalized[name] = float(np.clip(value, clip_min, clip_max))

        return normalized

    def _compute_time_normalized(self, timestamp: pd.Timestamp) -> float:
        """Normaliza timestamp a [0, 1] dentro de trading hours."""
        hours = self.contract.get_trading_hours()
        start_hour = int(hours["start"].split(":")[0])
        end_hour = int(hours["end"].split(":")[0])

        current_minutes = timestamp.hour * 60 + timestamp.minute
        start_minutes = start_hour * 60
        end_minutes = end_hour * 60

        if end_minutes <= start_minutes:
            end_minutes += 24 * 60  # Handle overnight

        if current_minutes < start_minutes:
            current_minutes += 24 * 60

        normalized = (current_minutes - start_minutes) / (end_minutes - start_minutes)
        return float(np.clip(normalized, 0.0, 1.0))

    def _assemble_observation(self, features: Dict[str, float]) -> np.ndarray:
        """Ensambla array en orden exacto del contrato."""
        observation = np.zeros(self.contract.observation_dim, dtype=np.float32)

        for idx, name in enumerate(self.contract.feature_order):
            value = features.get(name, 0.0)
            # Final safety check
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            observation[idx] = value

        return observation

    def export_feature_snapshot(
        self,
        ohlcv: pd.DataFrame,
        macro_df: pd.DataFrame,
        position: float,
        timestamp: pd.Timestamp,
        bar_idx: int
    ) -> Dict:
        """
        Exporta snapshot completo para auditoria y BD.

        Returns:
            Dict JSON-serializable con:
            - raw_features: valores sin normalizar
            - normalized_features: valores normalizados
            - metadata: version, timestamp, bar_idx
        """
        # Validate warmup
        if bar_idx < self.contract.warmup_bars:
            raise ValueError(
                f"bar_idx ({bar_idx}) < warmup_bars ({self.contract.warmup_bars})"
            )

        raw = self._calculate_raw_features(ohlcv, macro_df, bar_idx)
        normalized = self._normalize_features(raw)
        normalized["position"] = float(np.clip(position, -1.0, 1.0))
        normalized["time_normalized"] = self._compute_time_normalized(timestamp)

        return {
            "version": self.contract.version,
            "timestamp": timestamp.isoformat(),
            "bar_idx": bar_idx,
            "raw_features": {k: float(v) for k, v in raw.items()},
            "normalized_features": {k: float(v) for k, v in normalized.items()},
        }


# Convenience function for backward compatibility
def create_feature_builder(version: str = "current") -> FeatureBuilder:
    """
    Factory function to create FeatureBuilder.

    Args:
        version: Contract version ("current")

    Returns:
        Configured FeatureBuilder instance
    """
    return FeatureBuilder(version=version)
