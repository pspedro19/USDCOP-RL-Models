"""
USD/COP RL Trading System - Purged Cross-Validation
=====================================================

Implementación de Purged K-Fold CV para series temporales financieras.

PROBLEMA QUE RESUELVE:
- CV estándar tiene data leakage en series temporales
- Autocorrelación causa sobreestimación del rendimiento
- No hay gap entre train/test

SOLUCIÓN:
- Purged K-Fold con embargo entre folds
- Walk-Forward validation
- Gaps configurables según autocorrelación

Referencias:
- "Advances in Financial Machine Learning" - Marcos López de Prado
- "The Perils of K-Fold Cross-Validation with Time Series"

Author: Claude Code
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Generator, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class CVFold:
    """Representa un fold de cross-validation."""
    fold_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    embargo_size: int

    @property
    def train_indices(self) -> np.ndarray:
        return np.arange(self.train_start, self.train_end)

    @property
    def test_indices(self) -> np.ndarray:
        return np.arange(self.test_start, self.test_end)

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


class PurgedKFoldCV:
    """
    Purged K-Fold Cross-Validation para series temporales.

    CARACTERÍSTICAS:
    1. Embargo: Gap entre train y test para evitar leakage
    2. Purging: Elimina samples del train que están cerca del test
    3. Respeta orden temporal

    CONFIGURACIÓN RECOMENDADA:
    - n_splits: 5
    - bars_per_day: 60 (5min), 20 (15min), 5 (1h)
    - embargo_days: 3 (calculado automáticamente)
    - purge_bars: 1 día de buffer

    Args:
        n_splits: Número de folds
        embargo_bars: Barras de embargo (si None, usa embargo_days * bars_per_day)
        purge_bars: Barras adicionales a purgar (si None, usa bars_per_day)
        bars_per_day: Barras por día para calcular embargo proporcional
        embargo_days: Días de embargo (usado si embargo_bars es None)
        min_train_size: Tamaño mínimo de train set
        min_test_size: Tamaño mínimo de test set
        verbose: Nivel de verbosidad
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_bars: int = None,
        purge_bars: int = None,
        bars_per_day: int = 60,
        embargo_days: int = 3,
        min_train_size: int = 10000,
        min_test_size: int = 2000,
        verbose: int = 1,
    ):
        self.n_splits = n_splits
        self.bars_per_day = bars_per_day

        # Calcular embargo proporcional al timeframe si no se especifica
        if embargo_bars is None:
            self.embargo_bars = bars_per_day * embargo_days
        else:
            self.embargo_bars = embargo_bars

        if purge_bars is None:
            self.purge_bars = bars_per_day  # 1 día de purge
        else:
            self.purge_bars = purge_bars

        self.min_train_size = min_train_size
        self.min_test_size = min_test_size
        self.verbose = verbose

        if self.verbose > 0:
            print(f"[PurgedKFoldCV] embargo={self.embargo_bars} bars ({self.embargo_bars/bars_per_day:.1f} days), "
                  f"purge={self.purge_bars} bars")

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generar índices de train/test para cada fold.

        Args:
            X: Datos de entrada
            y: Labels (ignorados)
            groups: Grupos (ignorados)

        Yields:
            Tuple de (train_indices, test_indices)
        """
        n_samples = len(X)
        total_gap = self.embargo_bars + self.purge_bars

        # Calcular tamaño de cada fold
        test_size = (n_samples - total_gap * (self.n_splits - 1)) // self.n_splits

        if test_size < self.min_test_size:
            raise ValueError(
                f"Test size {test_size} < min_test_size {self.min_test_size}. "
                f"Reduce n_splits or embargo_bars."
            )

        for fold in range(self.n_splits):
            # Test set: cada fold es un bloque consecutivo
            test_start = fold * (test_size + total_gap)
            test_end = test_start + test_size

            # Ajustar último fold para cubrir todo el dataset
            if fold == self.n_splits - 1:
                test_end = n_samples

            # Train set: todo lo demás con embargo
            train_indices = []

            # Bloque antes del test
            if test_start > 0:
                train_end = max(0, test_start - self.embargo_bars)
                if train_end > 0:
                    train_indices.extend(range(0, train_end))

            # Bloque después del test (con purge)
            if test_end < n_samples:
                train_start = min(n_samples, test_end + self.purge_bars)
                if train_start < n_samples:
                    train_indices.extend(range(train_start, n_samples))

            train_indices = np.array(train_indices)
            test_indices = np.arange(test_start, test_end)

            # Validar tamaños mínimos
            if len(train_indices) < self.min_train_size:
                if self.verbose > 0:
                    print(f"Warning: Fold {fold} train size {len(train_indices)} < min {self.min_train_size}")
                continue

            if len(test_indices) < self.min_test_size:
                if self.verbose > 0:
                    print(f"Warning: Fold {fold} test size {len(test_indices)} < min {self.min_test_size}")
                continue

            if self.verbose > 0:
                print(f"Fold {fold}: Train={len(train_indices):,}, Test={len(test_indices):,}, "
                      f"Gap={self.embargo_bars + self.purge_bars}")

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Retornar número de folds."""
        return self.n_splits

    def get_folds(self, n_samples: int) -> List[CVFold]:
        """
        Obtener lista de folds con información detallada.

        Args:
            n_samples: Número total de samples

        Returns:
            Lista de CVFold
        """
        folds = []
        dummy_X = np.zeros(n_samples)

        for fold_idx, (train_idx, test_idx) in enumerate(self.split(dummy_X)):
            fold = CVFold(
                fold_idx=fold_idx,
                train_start=int(train_idx[0]) if len(train_idx) > 0 else 0,
                train_end=int(train_idx[-1]) + 1 if len(train_idx) > 0 else 0,
                test_start=int(test_idx[0]),
                test_end=int(test_idx[-1]) + 1,
                embargo_size=self.embargo_bars,
            )
            folds.append(fold)

        return folds


class WalkForwardValidator:
    """
    Walk-Forward Validation para trading.

    A diferencia de K-Fold, WFV siempre entrena en el pasado
    y testea en el futuro inmediato, simulando condiciones reales.

    ESQUEMA:
    |----TRAIN----|--GAP--|--TEST--|
                  |----TRAIN----|--GAP--|--TEST--|
                                |----TRAIN----|--GAP--|--TEST--|

    Args:
        n_splits: Número de períodos de test
        train_size: Tamaño del train en barras (o fracción)
        test_size: Tamaño del test en barras (o fracción)
        embargo_bars: Gap entre train y test (si None, usa embargo_days * bars_per_day)
        bars_per_day: Barras por día para calcular embargo proporcional
        embargo_days: Días de embargo (usado si embargo_bars es None)
        expanding: Si usar expanding window (True) o sliding (False)
        min_train_size: Tamaño mínimo de train
        verbose: Nivel de verbosidad
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        embargo_bars: int = None,
        bars_per_day: int = 60,
        embargo_days: int = 3,
        expanding: bool = False,
        min_train_size: int = 10000,
        verbose: int = 1,
    ):
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.bars_per_day = bars_per_day

        # Calcular embargo proporcional al timeframe
        if embargo_bars is None:
            self.embargo_bars = bars_per_day * embargo_days
        else:
            self.embargo_bars = embargo_bars

        self.expanding = expanding
        self.min_train_size = min_train_size
        self.verbose = verbose

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generar splits walk-forward.
        """
        n_samples = len(X)

        # Determinar tamaños
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 2)
        else:
            test_size = self.test_size

        if self.train_size is None:
            train_size = test_size * 3  # 3:1 ratio por defecto
        else:
            train_size = self.train_size

        # Punto de inicio (dejando espacio para train inicial)
        start_offset = train_size + self.embargo_bars

        for fold in range(self.n_splits):
            # Calcular posiciones
            test_start = start_offset + fold * test_size
            test_end = test_start + test_size

            if test_end > n_samples:
                break

            # Train: antes del embargo
            if self.expanding:
                # Expanding window: desde el inicio
                train_start = 0
            else:
                # Sliding window: tamaño fijo
                train_start = max(0, test_start - self.embargo_bars - train_size)

            train_end = test_start - self.embargo_bars

            if train_end - train_start < self.min_train_size:
                if self.verbose > 0:
                    print(f"Warning: Fold {fold} train too small, skipping")
                continue

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            if self.verbose > 0:
                train_pct = train_end / n_samples * 100
                test_pct = test_end / n_samples * 100
                print(f"Fold {fold}: Train=[0-{train_pct:.0f}%], Test=[{test_start/n_samples*100:.0f}%-{test_pct:.0f}%]")

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class TimeSeriesSplit:
    """
    Split simple de series temporales con embargo.

    Para cuando solo necesitas train/validation/test.

    Args:
        train_pct: Porcentaje para training
        val_pct: Porcentaje para validación
        test_pct: Porcentaje para test (resto)
        embargo_bars: Barras de embargo (si None, usa embargo_days * bars_per_day)
        bars_per_day: Barras por día para calcular embargo proporcional
        embargo_days: Días de embargo (usado si embargo_bars es None)
    """

    def __init__(
        self,
        train_pct: float = 0.70,
        val_pct: float = 0.15,
        test_pct: float = 0.15,
        embargo_bars: int = None,
        bars_per_day: int = 60,
        embargo_days: int = 3,
    ):
        assert abs(train_pct + val_pct + test_pct - 1.0) < 0.001
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = test_pct
        self.bars_per_day = bars_per_day

        # Calcular embargo proporcional al timeframe
        if embargo_bars is None:
            self.embargo_bars = bars_per_day * embargo_days
        else:
            self.embargo_bars = embargo_bars

    def split(self, n_samples: int) -> Dict[str, Tuple[int, int]]:
        """
        Obtener índices de cada split.

        Returns:
            Dict con 'train', 'val', 'test' -> (start, end)
        """
        total_embargo = self.embargo_bars * 2  # Entre train-val y val-test

        usable = n_samples - total_embargo

        train_size = int(usable * self.train_pct)
        val_size = int(usable * self.val_pct)
        test_size = usable - train_size - val_size

        train_end = train_size
        val_start = train_end + self.embargo_bars
        val_end = val_start + val_size
        test_start = val_end + self.embargo_bars
        test_end = n_samples

        return {
            'train': (0, train_end),
            'val': (val_start, val_end),
            'test': (test_start, test_end),
        }

    def get_dataframes(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Dividir DataFrame en train/val/test.

        Args:
            df: DataFrame completo

        Returns:
            Dict con DataFrames para cada split
        """
        splits = self.split(len(df))

        result = {}
        for name, (start, end) in splits.items():
            result[name] = df.iloc[start:end].copy()

        return result
