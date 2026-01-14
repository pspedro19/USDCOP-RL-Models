"""
Multi-Horizon Features
======================

Añade features de múltiples horizontes temporales al dataset.

PROBLEMA QUE RESUELVE:
- Modelo opera en 5min pero no tiene contexto de tendencia macro
- Señales de corto plazo pueden ir contra la tendencia de largo plazo
- Similar a Didact AI que usa daily bars con decisión semanal

SOLUCIÓN:
- Calcular features en múltiples horizontes (1h, 4h, 1d, 5d)
- Añadir señal direccional de largo plazo
- El modelo ve tanto táctico (5min) como estratégico (1-5 días)

Author: Claude Code
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


class MultiHorizonFeatures:
    """
    Genera features de múltiples horizontes temporales.

    Para cada horizonte calcula:
    - Retorno acumulado
    - Volatilidad del período
    - Trend strength (retorno / volatilidad)
    - Momentum (retorno reciente vs anterior)

    Args:
        horizons: Lista de horizontes en barras [12, 48, 288] = [1h, 4h, 1d en 5min]
        return_column: Nombre de columna de retornos
    """

    def __init__(
        self,
        horizons: List[int] = None,
        return_column: str = 'log_ret_5m',
    ):
        # Default: 1h, 4h, 1d en barras de 5 minutos
        self.horizons = horizons or [12, 48, 288]
        self.return_column = return_column

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añadir features de múltiples horizontes al DataFrame.

        Args:
            df: DataFrame con columna de retornos

        Returns:
            DataFrame con columnas adicionales
        """
        df = df.copy()

        if self.return_column not in df.columns:
            raise ValueError(f"Column '{self.return_column}' not found in DataFrame")

        for h in self.horizons:
            # Retorno acumulado del horizonte
            df[f'ret_{h}bar'] = df[self.return_column].rolling(h).sum()

            # Volatilidad del horizonte
            df[f'vol_{h}bar'] = df[self.return_column].rolling(h).std()

            # Trend strength (retorno / volatilidad) - signal-to-noise ratio
            df[f'trend_{h}bar'] = (
                df[f'ret_{h}bar'] / (df[f'vol_{h}bar'] + 1e-10)
            ).clip(-3, 3)

            # Momentum (retorno reciente vs anterior del mismo horizonte)
            df[f'mom_{h}bar'] = (
                df[f'ret_{h}bar'] - df[f'ret_{h}bar'].shift(h)
            ).clip(-3, 3)

        # Fill NaNs with forward fill, then 0
        df = df.fillna(method='ffill').fillna(0)

        return df

    def get_feature_names(self) -> List[str]:
        """Obtener nombres de features generadas."""
        names = []
        for h in self.horizons:
            names.extend([
                f'ret_{h}bar',
                f'vol_{h}bar',
                f'trend_{h}bar',
                f'mom_{h}bar',
            ])
        return names


class DirectionalSignal:
    """
    Señal direccional de largo plazo.

    Combina múltiples horizontes para generar señal de 1-5 días.
    Similar al enfoque de Didact AI que mira tendencia semanal.

    Args:
        short_horizon: Horizonte corto en barras (default 48 = 4h en 5min)
        medium_horizon: Horizonte medio en barras (default 288 = 1d)
        long_horizon: Horizonte largo en barras (default 1440 = 5d)
        weights: Pesos para cada horizonte [short, medium, long]
    """

    def __init__(
        self,
        short_horizon: int = 48,
        medium_horizon: int = 288,
        long_horizon: int = 1440,
        weights: Tuple[float, float, float] = (0.2, 0.3, 0.5),
    ):
        self.short = short_horizon
        self.medium = medium_horizon
        self.long = long_horizon
        self.weights = weights

    def compute(
        self,
        df: pd.DataFrame,
        return_col: str = 'log_ret_5m',
    ) -> pd.Series:
        """
        Calcular señal direccional.

        Returns:
            Series con valores en [-1, 1] indicando dirección macro
        """
        # Retornos acumulados por horizonte
        ret_short = df[return_col].rolling(self.short).sum()
        ret_medium = df[return_col].rolling(self.medium).sum()
        ret_long = df[return_col].rolling(self.long).sum()

        # Combinar con pesos
        signal = (
            self.weights[0] * np.sign(ret_short) +
            self.weights[1] * np.sign(ret_medium) +
            self.weights[2] * np.sign(ret_long)
        )

        return signal.fillna(0).clip(-1, 1)

    def get_trend_alignment(
        self,
        df: pd.DataFrame,
        return_col: str = 'log_ret_5m',
    ) -> pd.Series:
        """
        Medir alineación de tendencias entre horizontes.

        Returns:
            Series con valores en [0, 1]
            1 = todos los horizontes en la misma dirección
            0 = horizontes en direcciones opuestas
        """
        ret_short = df[return_col].rolling(self.short).sum()
        ret_medium = df[return_col].rolling(self.medium).sum()
        ret_long = df[return_col].rolling(self.long).sum()

        # Signos
        sign_short = np.sign(ret_short)
        sign_medium = np.sign(ret_medium)
        sign_long = np.sign(ret_long)

        # Alignment: 1 si todos iguales, 0 si hay desacuerdo
        all_positive = (sign_short > 0) & (sign_medium > 0) & (sign_long > 0)
        all_negative = (sign_short < 0) & (sign_medium < 0) & (sign_long < 0)

        alignment = (all_positive | all_negative).astype(float)

        # También dar crédito parcial por alignment de 2/3
        two_positive = (
            ((sign_short > 0) & (sign_medium > 0)) |
            ((sign_short > 0) & (sign_long > 0)) |
            ((sign_medium > 0) & (sign_long > 0))
        )
        two_negative = (
            ((sign_short < 0) & (sign_medium < 0)) |
            ((sign_short < 0) & (sign_long < 0)) |
            ((sign_medium < 0) & (sign_long < 0))
        )

        partial_alignment = (two_positive | two_negative).astype(float) * 0.5

        # Combinar
        final_alignment = np.maximum(alignment, partial_alignment)

        return final_alignment.fillna(0.5)


class TrendRegimeClassifier:
    """
    Clasifica el régimen de tendencia actual.

    Regímenes:
    - STRONG_TREND: Señal direccional fuerte y alineada
    - WEAK_TREND: Señal direccional pero no totalmente alineada
    - RANGING: Sin tendencia clara
    """

    def __init__(
        self,
        signal_threshold: float = 0.5,
        alignment_threshold: float = 0.7,
    ):
        self.signal_threshold = signal_threshold
        self.alignment_threshold = alignment_threshold

    def classify(
        self,
        directional_signal: float,
        trend_alignment: float,
    ) -> str:
        """
        Clasificar régimen de tendencia.

        Returns:
            'STRONG_TREND', 'WEAK_TREND', o 'RANGING'
        """
        abs_signal = abs(directional_signal)

        if abs_signal >= self.signal_threshold and trend_alignment >= self.alignment_threshold:
            return 'STRONG_TREND'
        elif abs_signal >= self.signal_threshold * 0.5:
            return 'WEAK_TREND'
        else:
            return 'RANGING'

    def get_features(
        self,
        directional_signal: float,
        trend_alignment: float,
    ) -> List[float]:
        """
        Obtener features de régimen de tendencia.

        Returns:
            [is_strong_trend, is_weak_trend, is_ranging, regime_confidence]
        """
        regime = self.classify(directional_signal, trend_alignment)

        is_strong = 1.0 if regime == 'STRONG_TREND' else 0.0
        is_weak = 1.0 if regime == 'WEAK_TREND' else 0.0
        is_ranging = 1.0 if regime == 'RANGING' else 0.0

        # Confidence basada en fuerza de señal y alignment
        confidence = abs(directional_signal) * trend_alignment

        return [is_strong, is_weak, is_ranging, confidence]


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def add_horizon_features(
    df: pd.DataFrame,
    return_column: str = 'log_ret_5m',
    horizons: Optional[List[int]] = None,
    add_directional_signal: bool = True,
    add_trend_alignment: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Añadir features de horizonte mixto al dataset.

    Llamar ANTES de entrenar.

    Args:
        df: DataFrame con datos
        return_column: Columna de retornos
        horizons: Lista de horizontes en barras
        add_directional_signal: Si añadir señal direccional
        add_trend_alignment: Si añadir alignment de tendencias

    Returns:
        (DataFrame con features, lista de nombres de features añadidas)
    """
    # Multi-horizon features
    mh = MultiHorizonFeatures(
        horizons=horizons or [12, 48, 288],  # 1h, 4h, 1d
        return_column=return_column,
    )
    df = mh.compute_features(df)
    feature_names = mh.get_feature_names()

    # Directional signal
    if add_directional_signal:
        ds = DirectionalSignal()
        df['directional_signal'] = ds.compute(df, return_column)
        feature_names.append('directional_signal')

    # Trend alignment
    if add_trend_alignment:
        ds = DirectionalSignal()
        df['trend_alignment'] = ds.get_trend_alignment(df, return_column)
        feature_names.append('trend_alignment')

    print(f"Added {len(feature_names)} horizon features: {feature_names}")

    return df, feature_names


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("MULTI-HORIZON FEATURES - USD/COP RL Trading System")
    print("=" * 70)

    # Crear datos de prueba
    np.random.seed(42)
    n_samples = 500

    # Simular retornos con tendencia
    trend = np.linspace(0, 0.1, n_samples)
    noise = np.random.normal(0, 0.01, n_samples)
    returns = np.diff(trend) + noise[:-1]
    returns = np.insert(returns, 0, 0)

    df = pd.DataFrame({
        'log_ret_5m': returns,
        'close': 4500 * np.exp(np.cumsum(returns)),
    })

    # Test MultiHorizonFeatures
    print("\n1. MultiHorizonFeatures Test:")
    print("-" * 50)

    mh = MultiHorizonFeatures(horizons=[12, 48, 288])
    df = mh.compute_features(df)

    print(f"  Features added: {mh.get_feature_names()}")
    print(f"  Sample values at end:")
    for feat in mh.get_feature_names()[:4]:  # Solo primeros 4
        print(f"    {feat}: {df[feat].iloc[-1]:.4f}")

    # Test DirectionalSignal
    print("\n2. DirectionalSignal Test:")
    print("-" * 50)

    ds = DirectionalSignal()
    df['directional_signal'] = ds.compute(df, 'log_ret_5m')
    df['trend_alignment'] = ds.get_trend_alignment(df, 'log_ret_5m')

    print(f"  Directional signal range: [{df['directional_signal'].min():.2f}, {df['directional_signal'].max():.2f}]")
    print(f"  Trend alignment range: [{df['trend_alignment'].min():.2f}, {df['trend_alignment'].max():.2f}]")
    print(f"  Final directional signal: {df['directional_signal'].iloc[-1]:.3f}")
    print(f"  Final trend alignment: {df['trend_alignment'].iloc[-1]:.3f}")

    # Test TrendRegimeClassifier
    print("\n3. TrendRegimeClassifier Test:")
    print("-" * 50)

    classifier = TrendRegimeClassifier()

    test_cases = [
        (0.8, 0.9),   # Strong trend
        (0.4, 0.6),   # Weak trend
        (0.1, 0.3),   # Ranging
    ]

    for signal, alignment in test_cases:
        regime = classifier.classify(signal, alignment)
        features = classifier.get_features(signal, alignment)
        print(f"  Signal={signal:.1f}, Alignment={alignment:.1f}")
        print(f"    -> Regime: {regime}, Features: {features}")

    # Test convenience function
    print("\n4. add_horizon_features() Test:")
    print("-" * 50)

    df_test = pd.DataFrame({
        'log_ret_5m': np.random.normal(0, 0.01, 1000),
    })

    df_test, feature_names = add_horizon_features(df_test)
    print(f"  Total features added: {len(feature_names)}")

    print("\n" + "=" * 70)
    print("MultiHorizonFeatures ready for integration")
    print("=" * 70)
