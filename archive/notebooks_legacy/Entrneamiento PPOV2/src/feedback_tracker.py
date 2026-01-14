"""
Feedback Tracker for USD/COP RL Trading System
===============================================

Rastrea la precisión de predicciones recientes para que el modelo
pueda aprender a ser más conservador cuando sus predicciones fallan.

PROBLEMA QUE RESUELVE:
- El modelo no sabe si sus predicciones recientes fueron correctas
- No puede adaptar su comportamiento a rachas perdedoras

SOLUCIÓN:
- Track direcciones predichas vs realizadas
- Proporcionar 3 features al observation:
  1. accuracy: Precisión reciente (0-1)
  2. accuracy_trend: Tendencia de precisión (-1 a +1)
  3. consecutive_wrong: Errores consecutivos (0-1)

Author: Claude Code
Version: 1.0.0
"""

import numpy as np
from typing import List, Tuple
from collections import deque


class FeedbackTracker:
    """
    Rastrea precisión de predicciones recientes.

    El modelo recibe feedback sobre qué tan bien están funcionando
    sus predicciones. Si fallan, puede aprender a ser más conservador.

    Features generadas (3):
    1. accuracy: Precisión rolling (0-1)
    2. accuracy_trend: Cambio en precisión (últimos 5 vs anteriores 5)
    3. consecutive_wrong: Errores consecutivos normalizados

    Args:
        window_size: Tamaño de ventana para cálculos rolling
        action_threshold: Umbral para considerar acción como dirección
    """

    def __init__(
        self,
        window_size: int = 20,
        action_threshold: float = 0.1,
    ):
        self.window_size = window_size
        self.action_threshold = action_threshold

        # Buffers para tracking
        self.predictions: deque = deque(maxlen=window_size)
        self.actuals: deque = deque(maxlen=window_size)

        # Métricas
        self.consecutive_wrong = 0
        self.total_predictions = 0
        self.total_correct = 0

    def reset(self):
        """Reset del tracker para nuevo episodio."""
        self.predictions.clear()
        self.actuals.clear()
        self.consecutive_wrong = 0
        self.total_predictions = 0
        self.total_correct = 0

    def update(self, predicted_direction: float, actual_return: float) -> None:
        """
        Actualizar con nueva observación.

        Args:
            predicted_direction: Acción del modelo (>threshold = LONG, <-threshold = SHORT)
            actual_return: Retorno real del mercado
        """
        # Discretizar dirección predicha
        if predicted_direction > self.action_threshold:
            pred_dir = 1  # LONG
        elif predicted_direction < -self.action_threshold:
            pred_dir = -1  # SHORT
        else:
            pred_dir = 0  # HOLD

        # Discretizar dirección real
        if actual_return > 1e-6:
            actual_dir = 1  # UP
        elif actual_return < -1e-6:
            actual_dir = -1  # DOWN
        else:
            actual_dir = 0  # FLAT

        # Añadir a buffers
        self.predictions.append(pred_dir)
        self.actuals.append(actual_dir)

        # Actualizar métricas
        self.total_predictions += 1

        # HOLD siempre se considera "correcto" (no predijo dirección)
        if pred_dir == 0:
            is_correct = True
        else:
            is_correct = (pred_dir == actual_dir)

        if is_correct:
            self.total_correct += 1
            self.consecutive_wrong = 0
        else:
            self.consecutive_wrong += 1

    def get_accuracy(self) -> float:
        """
        Calcular precisión rolling.

        Returns:
            Precisión en ventana (0-1)
        """
        if len(self.predictions) < 5:
            return 0.5  # Default neutral

        correct = 0
        total = 0

        for pred, actual in zip(self.predictions, self.actuals):
            if pred == 0:  # HOLD siempre correcto
                correct += 1
            elif pred == actual:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.5

    def get_accuracy_trend(self) -> float:
        """
        Calcular tendencia de precisión.

        Returns:
            Diferencia entre precisión reciente y anterior (-1 a +1)
        """
        if len(self.predictions) < 10:
            return 0.0

        # Últimos 5
        recent_preds = list(self.predictions)[-5:]
        recent_actuals = list(self.actuals)[-5:]

        # Anteriores 5
        older_preds = list(self.predictions)[-10:-5]
        older_actuals = list(self.actuals)[-10:-5]

        def calc_acc(preds, actuals):
            correct = sum(1 for p, a in zip(preds, actuals) if p == 0 or p == a)
            return correct / len(preds) if preds else 0.5

        acc_recent = calc_acc(recent_preds, recent_actuals)
        acc_older = calc_acc(older_preds, older_actuals)

        return acc_recent - acc_older

    def get_consecutive_wrong_normalized(self) -> float:
        """
        Obtener errores consecutivos normalizados.

        Returns:
            Errores consecutivos / 5 (capped a 1.0)
        """
        return min(self.consecutive_wrong / 5.0, 1.0)

    def get_features(self) -> List[float]:
        """
        Obtener las 3 features para el observation.

        Returns:
            [accuracy, accuracy_trend, consecutive_wrong_normalized]
        """
        return [
            self.get_accuracy(),
            self.get_accuracy_trend(),
            self.get_consecutive_wrong_normalized(),
        ]

    def get_stats(self) -> dict:
        """Obtener estadísticas completas."""
        return {
            'total_predictions': self.total_predictions,
            'total_correct': self.total_correct,
            'lifetime_accuracy': self.total_correct / max(1, self.total_predictions),
            'rolling_accuracy': self.get_accuracy(),
            'accuracy_trend': self.get_accuracy_trend(),
            'consecutive_wrong': self.consecutive_wrong,
            'buffer_size': len(self.predictions),
        }


class RegimeFeatureGenerator:
    """
    Genera features de régimen para el observation.

    PROBLEMA:
    - El modelo NO VE el régimen, solo su acción es modificada
    - No puede APRENDER cuándo reducir exposición

    SOLUCIÓN:
    - Añadir 6 features de régimen al observation:
      1. is_crisis (0/1)
      2. is_volatile (0/1)
      3. is_normal (0/1)
      4. regime_confidence (0-1)
      5. vix_trend (cambio reciente)
      6. days_in_regime (normalizado)

    Args:
        vix_lookback: Ventana para calcular tendencia de VIX
    """

    def __init__(
        self,
        vix_lookback: int = 20,
        crisis_threshold_vix: float = 2.0,
        crisis_threshold_embi: float = 2.0,
        crisis_threshold_vol: float = 95.0,
        volatile_threshold_vix: float = 1.0,
        volatile_threshold_embi: float = 1.0,
        volatile_threshold_vol: float = 75.0,
    ):
        self.vix_lookback = vix_lookback

        # Thresholds
        self.crisis_vix = crisis_threshold_vix
        self.crisis_embi = crisis_threshold_embi
        self.crisis_vol = crisis_threshold_vol
        self.volatile_vix = volatile_threshold_vix
        self.volatile_embi = volatile_threshold_embi
        self.volatile_vol = volatile_threshold_vol

        # Tracking
        self.vix_history: deque = deque(maxlen=vix_lookback)
        self.current_regime = 'NORMAL'
        self.days_in_regime = 0

    def reset(self):
        """Reset para nuevo episodio."""
        self.vix_history.clear()
        self.current_regime = 'NORMAL'
        self.days_in_regime = 0

    def detect_regime(
        self,
        vix_z: float,
        embi_z: float,
        vol_pct: float,
    ) -> str:
        """
        Detectar régimen de mercado.

        Args:
            vix_z: VIX z-score
            embi_z: EMBI z-score
            vol_pct: Percentil de volatilidad (0-100)

        Returns:
            'CRISIS', 'VOLATILE', o 'NORMAL'
        """
        # CRISIS check
        if (vix_z > self.crisis_vix or
            embi_z > self.crisis_embi or
            vol_pct > self.crisis_vol):
            return 'CRISIS'

        # VOLATILE check
        if (vix_z > self.volatile_vix or
            embi_z > self.volatile_embi or
            vol_pct > self.volatile_vol):
            return 'VOLATILE'

        return 'NORMAL'

    def get_regime_probs(
        self,
        vix_z: float,
        embi_z: float,
        vol_pct: float,
    ) -> dict:
        """
        Obtener probabilidades soft de cada régimen.

        Usa transiciones suaves en lugar de thresholds duros.
        """
        def sigmoid(x, threshold, steepness=2.0):
            return 1.0 / (1.0 + np.exp(-steepness * (x - threshold)))

        # Crisis probability
        crisis_vix = sigmoid(vix_z, self.crisis_vix)
        crisis_embi = sigmoid(embi_z, self.crisis_embi)
        crisis_vol = sigmoid(vol_pct, self.crisis_vol)
        crisis_prob = max(crisis_vix, crisis_embi, crisis_vol)

        # Volatile probability (si no es crisis)
        volatile_vix = sigmoid(vix_z, self.volatile_vix)
        volatile_embi = sigmoid(embi_z, self.volatile_embi)
        volatile_vol = sigmoid(vol_pct, self.volatile_vol)
        volatile_prob = max(volatile_vix, volatile_embi, volatile_vol) * (1 - crisis_prob)

        # Normal probability
        normal_prob = 1.0 - crisis_prob - volatile_prob

        return {
            'CRISIS': crisis_prob,
            'VOLATILE': volatile_prob,
            'NORMAL': max(0, normal_prob),
        }

    def get_features(
        self,
        vix_z: float,
        embi_z: float,
        vol_pct: float,
    ) -> List[float]:
        """
        Obtener las 6 features de régimen.

        Args:
            vix_z: VIX z-score
            embi_z: EMBI z-score
            vol_pct: Percentil de volatilidad (0-100)

        Returns:
            Lista de 6 features:
            [is_crisis, is_volatile, is_normal,
             regime_confidence, vix_trend, days_in_regime]
        """
        # Detectar régimen
        regime = self.detect_regime(vix_z, embi_z, vol_pct)
        probs = self.get_regime_probs(vix_z, embi_z, vol_pct)

        # Actualizar tracking de VIX
        self.vix_history.append(vix_z)

        # One-hot encoding
        is_crisis = 1.0 if regime == 'CRISIS' else 0.0
        is_volatile = 1.0 if regime == 'VOLATILE' else 0.0
        is_normal = 1.0 if regime == 'NORMAL' else 0.0

        # Confianza (max probabilidad)
        regime_confidence = max(probs.values())

        # Tendencia de VIX
        if len(self.vix_history) >= 10:
            recent = np.mean(list(self.vix_history)[-5:])
            older = np.mean(list(self.vix_history)[-10:-5])
            vix_trend = np.clip(recent - older, -2.0, 2.0)
        else:
            vix_trend = 0.0

        # Días en régimen
        if regime != self.current_regime:
            self.current_regime = regime
            self.days_in_regime = 0
        else:
            self.days_in_regime += 1

        days_normalized = min(self.days_in_regime / 50.0, 1.0)

        return [
            is_crisis,
            is_volatile,
            is_normal,
            regime_confidence,
            vix_trend / 2.0,  # Normalizar a [-1, 1]
            days_normalized,
        ]


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("FEEDBACK TRACKER - USD/COP RL Trading System")
    print("=" * 70)

    # Test FeedbackTracker
    print("\n1. FeedbackTracker Test:")
    print("-" * 50)

    tracker = FeedbackTracker(window_size=20)

    # Simular algunas predicciones
    predictions = [
        (0.5, 0.001),   # LONG correcto
        (0.5, -0.001),  # LONG incorrecto
        (-0.5, -0.001), # SHORT correcto
        (0.0, 0.001),   # HOLD (siempre ok)
        (0.3, 0.001),   # LONG correcto
        (-0.3, 0.001),  # SHORT incorrecto
    ]

    for pred, actual in predictions:
        tracker.update(pred, actual)
        features = tracker.get_features()
        print(f"  Pred={pred:+.1f}, Actual={actual:+.4f}")
        print(f"    -> Accuracy={features[0]:.2f}, Trend={features[1]:+.2f}, ConsecWrong={features[2]:.2f}")

    print(f"\n  Stats: {tracker.get_stats()}")

    # Test RegimeFeatureGenerator
    print("\n2. RegimeFeatureGenerator Test:")
    print("-" * 50)

    regime_gen = RegimeFeatureGenerator()

    test_cases = [
        ("Normal", 0.5, 0.3, 45.0),
        ("Volatile", 1.5, 0.8, 78.0),
        ("Crisis", 2.5, 2.2, 97.0),
    ]

    for name, vix, embi, vol in test_cases:
        features = regime_gen.get_features(vix, embi, vol)
        print(f"\n  {name} Market (VIX_z={vix}, EMBI_z={embi}, Vol%={vol}):")
        print(f"    is_crisis={features[0]:.0f}, is_volatile={features[1]:.0f}, is_normal={features[2]:.0f}")
        print(f"    confidence={features[3]:.2f}, vix_trend={features[4]:+.2f}, days={features[5]:.2f}")

    print("\n" + "=" * 70)
    print("FeedbackTracker ready for integration")
    print("=" * 70)
