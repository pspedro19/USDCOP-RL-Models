"""Validación temporal purgada (SDD-004 §4-§5, ADR-005).

`sklearn.model_selection.KFold` está prohibido en este repositorio: las
etiquetas financieras se solapan y barajar filtra el futuro al train.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

__all__ = ["CPCV", "PurgedKFold"]

DEFAULT_EMBARGO_DAYS = 252  # ADR-005: el 1-2% habitual no cubre features de memoria anual


def _embargo_end(test_end: pd.Timestamp, embargo_days: int) -> pd.Timestamp:
    if embargo_days <= 0:
        return test_end
    return test_end + pd.tseries.offsets.BDay(embargo_days)


def _purge_and_embargo(
    index: pd.DatetimeIndex,
    t1: pd.Series,
    test_pos: np.ndarray,
    embargo_days: int,
) -> np.ndarray:
    """Índices posicionales de train tras purgar solapamientos y aplicar embargo."""
    starts = index.asi8                       # int64 ns, tz-safe
    ends = pd.DatetimeIndex(t1).asi8

    test_start = starts[test_pos].min()
    test_end_ts = index[test_pos].max()
    test_end = test_end_ts.value
    emb_end = _embargo_end(test_end_ts, embargo_days).value

    overlaps = (starts <= test_end) & (ends >= test_start)
    embargoed = (starts > test_end) & (starts <= emb_end)

    keep = ~(overlaps | embargoed)
    keep[test_pos] = False
    return np.flatnonzero(keep)


@dataclass
class PurgedKFold:
    """K-Fold con folds de test CONTIGUOS, purging por `t1` y embargo posterior."""

    n_splits: int = 5
    embargo_days: int = DEFAULT_EMBARGO_DAYS

    def split(
        self, index: pd.DatetimeIndex, t1: pd.Series
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if len(index) != len(t1):
            raise ValueError("index y t1 deben tener el mismo largo")

        bounds = np.array_split(np.arange(len(index)), self.n_splits)
        for test_pos in bounds:
            train_pos = _purge_and_embargo(index, t1, test_pos, self.embargo_days)
            yield train_pos, test_pos


@dataclass
class CPCV:
    """Combinatorially Purged Cross-Validation (López de Prado 2018, cap. 12).

    N grupos contiguos, k de test por combinación:
        n_splits = C(N, k)          n_paths = C(N, k)·k / N = C(N-1, k-1)
    Con N=12, k=2  ->  66 splits, 11 backtest paths.
    """

    n_groups: int = 12
    n_test_groups: int = 2
    embargo_days: int = DEFAULT_EMBARGO_DAYS

    @property
    def n_splits(self) -> int:
        return math.comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        return self.n_splits * self.n_test_groups // self.n_groups

    def _groups(self, n: int) -> list[np.ndarray]:
        return np.array_split(np.arange(n), self.n_groups)

    def _combos(self) -> list[tuple[int, ...]]:
        return list(combinations(range(self.n_groups), self.n_test_groups))

    def split(
        self, index: pd.DatetimeIndex, t1: pd.Series
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        groups = self._groups(len(index))
        for combo in self._combos():
            test_pos = np.concatenate([groups[g] for g in combo])
            train_pos = _purge_and_embargo(index, t1, test_pos, self.embargo_days)
            yield train_pos, test_pos

    def paths(
        self, index: pd.DatetimeIndex, t1: pd.Series
    ) -> Iterator[list[tuple[np.ndarray, np.ndarray]]]:
        """Cada path recorre los N grupos exactamente una vez como test.

        El grupo `g` es testeado en C(N-1, k-1) = n_paths combinaciones distintas;
        el path `j` toma la j-ésima de ellas.
        """
        groups = self._groups(len(index))
        combos = self._combos()

        combos_testing: dict[int, list[tuple[int, ...]]] = {
            g: [c for c in combos if g in c] for g in range(self.n_groups)
        }

        for j in range(self.n_paths):
            path: list[tuple[np.ndarray, np.ndarray]] = []
            for g in range(self.n_groups):
                combo = combos_testing[g][j]
                test_all = np.concatenate([groups[x] for x in combo])
                train_pos = _purge_and_embargo(index, t1, test_all, self.embargo_days)
                path.append((train_pos, groups[g]))
            yield path
