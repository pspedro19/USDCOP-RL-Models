# pipeline_limpio_regresion/validation/purged_kfold.py
"""
Cross-Validation OPTIMIZADO para datos financieros con overlapping targets.

AUDITORIA 2025-01: Problemas corregidos:
1. PurgedKFold usaba datos FUTUROS (look-ahead bias)
2. Embargo formula incorrecta (min() lo limitaba siempre a 5%)
3. Buffer +5 constante en lugar de proporcional
4. Demasiados datos eliminados por fold

SOLUCION: Walk-Forward Expanding Window con purge adaptativo
- Solo entrena con datos PASADOS (simula produccion real)
- Embargo proporcional al horizonte
- Folds adaptativos segun horizonte

Benchmark esperado:
- Mejor generalizacion en out-of-sample
- DA mas conservador pero realista en CV
"""

import numpy as np
from typing import Iterator, Tuple, Optional
from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit
import logging

logger = logging.getLogger(__name__)


class PurgedKFold(BaseCrossValidator):
    """
    K-Fold con purge y embargo para eliminar data leakage temporal.

    En datos financieros con targets overlapping (como returns a 30 dias),
    las observaciones consecutivas comparten informacion. PurgedKFold
    elimina esta contaminacion creando gaps entre train y test.

    Ejemplo para H=30:
    ```
    Datos: [0, 1, 2, ..., 100, 101, ..., 200]

    Sin Purge (MALO):
      Train: [0-100], Test: [101-150]
      Problema: Obs 71-100 en train usan datos de 101-130 (overlap con test)

    Con Purge (BUENO):
      Train: [0-70], Test: [101-150]  (purge elimina 71-100)
      Embargo: [0-70], Test: [101-150], Skip: [151-160]
    ```

    Attributes:
        n_splits: Numero de folds
        horizon: Horizonte de prediccion (dias de overlap)
        embargo_pct: Porcentaje del dataset como embargo post-test
    """

    def __init__(
        self,
        n_splits: int = 5,
        horizon: int = 30,
        embargo_pct: float = 0.01
    ):
        """
        Args:
            n_splits: Numero de folds (default: 5)
            horizon: Horizonte de prediccion en dias/barras (default: 30)
            embargo_pct: Porcentaje del dataset como embargo (default: 1%)
        """
        self.n_splits = n_splits
        self.horizon = horizon
        self.embargo_pct = embargo_pct

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Retorna el numero de folds."""
        return self.n_splits

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Genera indices de train/test con purge y embargo.

        Args:
            X: Features array o DataFrame
            y: Target array (no usado, por compatibilidad)
            groups: Groups array (no usado, por compatibilidad)

        Yields:
            Tuple de (train_indices, test_indices)
        """
        # Obtener numero de samples
        if hasattr(X, 'shape'):
            n_samples = X.shape[0]
        elif hasattr(X, '__len__'):
            n_samples = len(X)
        else:
            raise ValueError("X debe tener atributo shape o __len__")

        # Calcular embargo en numero de samples
        embargo = int(n_samples * self.embargo_pct)

        # Crear array de indices
        indices = np.arange(n_samples)

        # Tamano de cada fold
        fold_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            # Definir test set
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples

            # Purge: eliminar 'horizon' observaciones ANTES del test
            # Estas observaciones usan datos que se solapan con el test
            purge_start = max(0, test_start - self.horizon)

            # Embargo: eliminar 'embargo' observaciones DESPUES del test
            # Evita leakage residual por autocorrelacion
            embargo_end = min(n_samples, test_end + embargo)

            # Train: todo ANTES del purge + todo DESPUES del embargo
            train_before = indices[:purge_start] if purge_start > 0 else np.array([], dtype=int)
            train_after = indices[embargo_end:] if embargo_end < n_samples else np.array([], dtype=int)

            train_indices = np.concatenate([train_before, train_after])
            test_indices = indices[test_start:test_end]

            # Solo yield si hay suficientes datos
            min_train_samples = 100
            min_test_samples = 10

            if len(train_indices) >= min_train_samples and len(test_indices) >= min_test_samples:
                logger.debug(
                    f"Fold {i+1}: train={len(train_indices)}, test={len(test_indices)}, "
                    f"purge=[{purge_start}:{test_start}], embargo=[{test_end}:{embargo_end}]"
                )
                yield train_indices, test_indices
            else:
                logger.warning(
                    f"Fold {i+1} skipped: insufficient samples "
                    f"(train={len(train_indices)}, test={len(test_indices)})"
                )


class WalkForwardPurged(BaseCrossValidator):
    """
    Walk-Forward Validation con purge OPTIMIZADO (expanding window).

    CORRECCION: Esta es la implementacion RECOMENDADA para series temporales.
    Solo usa datos PASADOS para entrenar (sin look-ahead bias).

    Parametros optimizados:
    - purge_pct: Proporcional al horizonte (no constante)
    - min_train_samples: Absoluto en lugar de porcentaje
    - n_splits adaptativo segun horizonte
    """

    def __init__(
        self,
        n_splits: int = 5,
        horizon: int = 30,
        min_train_size: float = 0.3,
        min_train_samples: int = 500,
        purge_multiplier: float = 1.2
    ):
        """
        Args:
            n_splits: Numero de folds
            horizon: Horizonte de prediccion
            min_train_size: Fraccion minima de datos para primer train set
            min_train_samples: Numero minimo absoluto de muestras de train
            purge_multiplier: Multiplicador del horizonte para purge (1.2 = 20% extra)
        """
        self.n_splits = n_splits
        self.horizon = horizon
        self.min_train_size = min_train_size
        self.min_train_samples = min_train_samples
        self.purge_multiplier = purge_multiplier

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        indices = np.arange(n_samples)

        # Purge zone: horizon * multiplier (redondeado)
        purge_zone = int(self.horizon * self.purge_multiplier)

        # Tamano minimo de train (el mayor entre porcentaje y absoluto)
        min_train_pct = int(n_samples * self.min_train_size)
        min_train = max(min_train_pct, self.min_train_samples)

        # Espacio disponible para test despues de min_train
        available_for_test = n_samples - min_train - purge_zone

        if available_for_test < self.n_splits * 20:
            logger.warning(
                f"Insufficient data for {self.n_splits} folds. "
                f"Available: {available_for_test}, needed: {self.n_splits * 20}"
            )
            # Reducir folds automaticamente
            effective_splits = max(2, available_for_test // 50)
        else:
            effective_splits = self.n_splits

        # Reduce folds for very long horizons to preserve data quality
        # AUDITORIA 2025-01: Minimo 3 folds, no 2 (mejor estabilidad estadistica)
        if self.horizon >= 20:
            effective_splits = min(effective_splits, 3)

        # Tamano de cada test fold
        test_size = available_for_test // effective_splits

        folds_yielded = 0
        for i in range(effective_splits):
            # Test set
            test_start = min_train + purge_zone + i * test_size
            test_end = test_start + test_size if i < effective_splits - 1 else n_samples

            # Train set: todo ANTES del purge zone (expanding window)
            # FIXED: Direct calculation to avoid double-gap bug
            train_end = min_train + i * test_size

            train_indices = indices[:train_end]
            test_indices = indices[test_start:test_end]

            if len(train_indices) >= self.min_train_samples and len(test_indices) >= 20:
                logger.debug(
                    f"WalkForward Fold {i+1}: train={len(train_indices)} samples "
                    f"[0:{train_end}], purge=[{train_end}:{test_start}], "
                    f"test={len(test_indices)} samples [{test_start}:{test_end}]"
                )
                folds_yielded += 1
                yield train_indices, test_indices
            else:
                logger.warning(
                    f"WalkForward Fold {i+1} skipped: train={len(train_indices)}, "
                    f"test={len(test_indices)} (min_train={self.min_train_samples})"
                )

        if folds_yielded == 0:
            raise ValueError(
                f"No valid folds generated. n_samples={n_samples}, horizon={self.horizon}, "
                f"min_train={self.min_train_samples}. Consider reducing horizon or min_train_samples."
            )


def get_cv_for_horizon(
    horizon: int,
    n_splits: int = 5,
    method: str = 'auto',
    n_samples: int = None
) -> BaseCrossValidator:
    """
    Factory function OPTIMIZADA para obtener el CV apropiado segun horizonte.

    AUDITORIA 2025-01: Cambios criticos:
    1. Usar WalkForwardPurged por defecto (sin look-ahead bias)
    2. Folds adaptativos segun horizonte y tamano de datos
    3. Purge proporcional (1.2x horizonte) en lugar de constante +5
    4. Min train samples absoluto (500) para evitar underfitting

    Args:
        horizon: Horizonte de prediccion
        n_splits: Numero de folds (se adapta automaticamente si es necesario)
        method: 'auto', 'walkforward', 'purged', o 'timeseries'
        n_samples: Numero de muestras (opcional, para adaptar folds)

    Returns:
        Cross-validator apropiado

    Recomendaciones por horizonte (AUDITORIA 2025-01):
    - H=1-5:   5 folds, purge=1.5x horizonte
    - H=10-15: 4 folds, purge=1.8x horizonte (era 1.3x)
    - H=20-30: 3 folds, purge=2.0x horizonte (era 1.2x, ahora H=30 -> 60 obs)
    """
    # Adaptar numero de folds segun horizonte
    # AUDITORIA 2025-01: Aumentar purge_multiplier para mejor embargo temporal
    if horizon >= 20:
        effective_splits = min(n_splits, 3)  # Minimo 3 folds para H>=20
        purge_mult = 2.0  # AUMENTADO de 1.2 a 2.0 (H=30 -> 60 obs embargo)
        min_train = 400
    elif horizon >= 10:
        effective_splits = min(n_splits, 4)  # Maximo 4 folds para H>=10
        purge_mult = 1.8  # AUMENTADO de 1.3 a 1.8
        min_train = 450
    else:
        effective_splits = n_splits  # 5 folds para H<10
        purge_mult = 1.5  # Mantener
        min_train = 500

    # =========================================================================
    # METODO RECOMENDADO: WalkForwardPurged (default para 'auto')
    # =========================================================================
    if method == 'auto' or method == 'walkforward':
        logger.info(
            f"Using WalkForwardPurged for H={horizon}: "
            f"{effective_splits} folds, purge={purge_mult}x, min_train={min_train}"
        )
        return WalkForwardPurged(
            n_splits=effective_splits,
            horizon=horizon,
            min_train_size=0.35,  # 35% minimo como porcentaje
            min_train_samples=min_train,
            purge_multiplier=purge_mult
        )

    # =========================================================================
    # METODO LEGACY: PurgedKFold (mantener por compatibilidad)
    # ADVERTENCIA: Usa datos futuros - NO RECOMENDADO para produccion
    # =========================================================================
    elif method == 'purged':
        # Embargo proporcional al horizonte (no constante)
        embargo_pct = min(0.08, 0.01 + horizon / 500)  # H=30: 7%, H=10: 3%
        purge_horizon = int(horizon * purge_mult)
        logger.warning(
            f"Using PurgedKFold for H={horizon} (LEGACY - uses future data!). "
            f"Embargo={embargo_pct:.1%}, purge={purge_horizon}"
        )
        return PurgedKFold(
            n_splits=effective_splits,
            horizon=purge_horizon,
            embargo_pct=embargo_pct
        )

    # =========================================================================
    # METODO SIMPLE: TimeSeriesSplit (para horizontes muy cortos)
    # =========================================================================
    elif method == 'timeseries':
        # Gap proporcional al horizonte
        gap = int(horizon * purge_mult)
        logger.info(f"Using TimeSeriesSplit for H={horizon} (gap={gap})")
        return TimeSeriesSplit(
            n_splits=effective_splits,
            gap=gap
        )

    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Valid options: 'auto', 'walkforward', 'purged', 'timeseries'"
        )


def compare_cv_methods(
    X: np.ndarray,
    y: np.ndarray,
    horizon: int,
    model_factory,
    n_splits: int = 5
) -> dict:
    """
    Compara diferentes metodos de CV para un horizonte dado.

    AUDITORIA 2025-01: Ahora incluye WalkForwardPurged como metodo recomendado.

    Args:
        X: Features
        y: Target
        horizon: Horizonte de prediccion
        model_factory: Callable que retorna un modelo nuevo
        n_splits: Numero de folds

    Returns:
        Dict con resultados de cada metodo, incluyendo:
        - mse_mean/std: Error cuadratico medio
        - da_mean/std: Direction Accuracy
        - n_folds: Numero de folds efectivos
        - train_sizes: Lista de tamanos de train por fold
    """
    from sklearn.metrics import mean_squared_error

    results = {}

    # Determinar parametros adaptativos
    if horizon >= 20:
        eff_splits = min(n_splits, 3)
        purge_mult = 1.2
    elif horizon >= 10:
        eff_splits = min(n_splits, 4)
        purge_mult = 1.3
    else:
        eff_splits = n_splits
        purge_mult = 1.5

    methods = {
        'TimeSeriesSplit (gap=0)': TimeSeriesSplit(n_splits=n_splits, gap=0),
        f'TimeSeriesSplit (gap={horizon})': TimeSeriesSplit(n_splits=n_splits, gap=horizon),
        'PurgedKFold (legacy)': PurgedKFold(n_splits=eff_splits, horizon=horizon, embargo_pct=0.02),
        'WalkForwardPurged (RECOMMENDED)': WalkForwardPurged(
            n_splits=eff_splits,
            horizon=horizon,
            min_train_size=0.35,
            min_train_samples=400,
            purge_multiplier=purge_mult
        ),
    }

    for name, cv in methods.items():
        scores = []
        da_scores = []
        train_sizes = []

        try:
            for train_idx, test_idx in cv.split(X):
                model = model_factory()

                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]

                train_sizes.append(len(train_idx))

                model.fit(X_train, y_train)
                pred = model.predict(X_test)

                mse = mean_squared_error(y_test, pred)
                da = (np.sign(pred) == np.sign(y_test)).mean()

                scores.append(mse)
                da_scores.append(da)

            results[name] = {
                'mse_mean': np.mean(scores),
                'mse_std': np.std(scores),
                'da_mean': np.mean(da_scores),
                'da_std': np.std(da_scores),
                'n_folds': len(scores),
                'train_sizes': train_sizes,
                'avg_train_size': np.mean(train_sizes) if train_sizes else 0
            }
        except Exception as e:
            results[name] = {
                'error': str(e),
                'mse_mean': np.nan,
                'da_mean': np.nan,
                'n_folds': 0
            }

    return results
