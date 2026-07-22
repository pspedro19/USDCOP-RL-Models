"""E1 — Harness reutilizable de validación cross-asset: purged K-fold + wrappers SSOT (0 trials).

Contract: CTR-QUANT-CONSTITUTION-001 · Skill: .claude/skills/xasset-alpha-engine

**0 trials — ningún OOS abierto; cualquier estudio futuro requiere pre-registro en
HYPOTHESIS-REGISTRY + aprobación del operador.**

Este módulo NO se ejecuta contra ningún target real. Su `smoke()` usa EXCLUSIVAMENTE
datos sintéticos (numpy default_rng(42)) para probar la mecánica: particiones purgadas
sin fuga, embargo aplicado, wrappers al SSOT constitucional devolviendo dicts sanos y
JSON-safe. Correrlo no abre ningún test — no lee DB, ni seeds, ni parquets de señales.

Qué contiene y de dónde viene (DRY — no re-implementa el gate):
  purged_kfold_indices     generalización posicional (purge + embargo) del
                           `purged_kfold_idx` ya usado en
                           scripts/analysis/cop_risk_family_screen.py (H-RISK-FAM-01)
  purged_kfold_splits_t1   variante por tiempos-de-etiqueta t1 (AFML cap.7), adaptada
                           de .claude/skills/xasset-alpha-engine/scripts/xasset/
                           validation.py::purged_kfold_splits
  dsr_report               RE-EXPORT del SSOT services/common/metrics.py — único punto
  circular_block_bootstrap de entrada DSR/bootstrap (quant-constitution §2). Este
  deflated_sharpe_ratio    módulo NO trae implementación propia: dos implementaciones
  probabilistic_sharpe_ratio  del gate es peor que ninguna.

Uso previsto (FUTURO, tras pre-registro):
    from scripts.analysis.xasset_cv_harness import purged_kfold_indices, dsr_report
    for tr, te in purged_kfold_indices(n, n_splits=5, purge=1, embargo=4):
        ...fit en tr, predecir en te...
    # y el claim final SIEMPRE via dsr_report con el conteo de trials del registry.

Smoke (sin targets):
    python -m scripts.analysis.xasset_cv_harness   # exit 0 = PASS
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Wrappers/re-exports al SSOT constitucional (services/common/metrics.py).
from services.common.metrics import (  # noqa: E402
    circular_block_bootstrap,
    deflated_sharpe_ratio,
    dsr_report,
    probabilistic_sharpe_ratio,
    trial_aware_moments,
)
from src.contracts.strategy_schema import safe_json_dumps  # noqa: E402

__all__ = [
    "purged_kfold_indices",
    "purged_kfold_splits_t1",
    "dsr_report",
    "circular_block_bootstrap",
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "trial_aware_moments",
    "smoke",
]


# ---------------------------------------------------------------------------
# Purged K-fold (posicional): folds contiguos + purga simétrica + embargo post-test
# ---------------------------------------------------------------------------

def purged_kfold_indices(
    n_samples: int, n_splits: int = 5, purge: int = 1, embargo: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """K-fold purgado sobre índices posicionales de una serie temporal ordenada.

    Generaliza `cop_risk_family_screen.purged_kfold_idx` añadiendo embargo explícito:
      - purge:   se eliminan del train los `purge` índices adyacentes a CADA lado del
                 bloque test (etiquetas solapadas — p.ej. horizonte semanal ⇒ purge=1
                 en datos semanales, purge=5 en diarios).
      - embargo: se eliminan ADEMÁS `embargo` índices inmediatamente POSTERIORES al
                 test (la autocorrelación serial sigue llevando información del test
                 hacia adelante; AFML cap.7).

    Yields (train_idx, test_idx) posicionales. Lanza ValueError si algún fold se
    queda sin train (horizonte demasiado largo para la muestra) — mejor ruido que
    silencio.
    """
    if n_splits < 2:
        raise ValueError("n_splits debe ser >= 2")
    if purge < 0 or embargo < 0:
        raise ValueError("purge y embargo deben ser >= 0")
    if n_samples < n_splits * 2:
        raise ValueError(f"n_samples={n_samples} demasiado corto para {n_splits} folds")

    idx = np.arange(n_samples)
    for test in np.array_split(idx, n_splits):
        lo, hi = int(test[0]), int(test[-1])
        keep = (idx < lo - purge) | (idx > hi + purge + embargo)
        train = idx[keep]
        if train.size == 0:
            raise ValueError(
                f"Purga+embargo vaciaron el train (n={n_samples}, splits={n_splits}, "
                f"purge={purge}, embargo={embargo})")
        yield train, test


# ---------------------------------------------------------------------------
# Purged K-fold por tiempos de etiqueta (t1) — para holding periods irregulares
# ---------------------------------------------------------------------------

def purged_kfold_splits_t1(
    t1: pd.Series, n_splits: int = 5, embargo_pct: float = 0.01,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Splits purgados usando los tiempos de FIN de etiqueta (AFML cap.7).

    `t1`: Serie de tiempos de fin de etiqueta, indexada por tiempo de inicio,
    ordenada ascendente (para un holding de 5 días: starts + 5 días hábiles).
    Purga toda etiqueta de train cuyo intervalo [start, t1] solape la ventana de
    test; embarga `embargo_pct` de la muestra tras el final del test.

    Adaptado de la skill xasset-alpha-engine (validation.py::purged_kfold_splits);
    preferir esta variante cuando los holding periods no son constantes.
    """
    if not isinstance(t1, pd.Series):
        raise TypeError("t1 debe ser pd.Series de tiempos de fin de etiqueta")
    if not t1.index.is_monotonic_increasing:
        raise ValueError("el índice de t1 debe estar ordenado ascendente")
    if n_splits < 2:
        raise ValueError("n_splits debe ser >= 2")
    if not 0.0 <= embargo_pct < 1.0:
        raise ValueError("embargo_pct debe estar en [0, 1)")

    n = len(t1)
    indices = np.arange(n)
    embargo = int(n * embargo_pct)

    for test_idx in np.array_split(indices, n_splits):
        test_start_time = t1.index[test_idx[0]]
        test_end_time = t1.iloc[test_idx].max()
        before = indices[(t1.values < test_start_time)]
        after_pos = int(t1.index.searchsorted(test_end_time, side="right"))
        after = indices[min(after_pos + embargo, n):]
        train_idx = np.concatenate([before, after])
        if train_idx.size == 0:
            raise ValueError(
                f"La purga vació el train de un fold (n={n}, n_splits={n_splits}): "
                "horizonte de etiqueta demasiado largo para la muestra.")
        yield train_idx, test_idx


# ---------------------------------------------------------------------------
# Smoke — SOLO datos sintéticos con seed. Ningún target real, ningún OOS abierto.
# ---------------------------------------------------------------------------

def smoke() -> dict:
    """Prueba la MECÁNICA del harness con ruido sintético (default_rng(42)).

    Ninguna de estas cifras es un resultado de trading: el "retorno" es ruido
    gaussiano generado localmente. Verifica particiones, purga, embargo, wrappers
    SSOT y JSON-safety. 0 trials.
    """
    rng = np.random.default_rng(42)
    checks: dict[str, bool] = {}

    # --- 1) posicional: partición completa, sin solape, purga+embargo respetados
    n, purge, embargo = 520, 2, 4
    seen = np.zeros(n, dtype=int)
    ok_no_leak, ok_gap = True, True
    for tr, te in purged_kfold_indices(n, n_splits=5, purge=purge, embargo=embargo):
        seen[te] += 1
        if np.intersect1d(tr, te).size:
            ok_no_leak = False
        lo, hi = te[0], te[-1]
        banned = np.arange(max(lo - purge, 0), min(hi + purge + embargo, n - 1) + 1)
        if np.intersect1d(tr, banned).size:
            ok_gap = False
    checks["pos_partition_covers_all_once"] = bool((seen == 1).all())
    checks["pos_no_train_test_overlap"] = ok_no_leak
    checks["pos_purge_embargo_respected"] = ok_gap

    # --- 2) t1: etiquetas solapadas de 5 días — la purga debe excluirlas
    starts = pd.bdate_range("2019-01-01", periods=400)
    t1 = pd.Series(starts + pd.offsets.BDay(5), index=starts)
    ok_t1 = True
    for tr, te in purged_kfold_splits_t1(t1, n_splits=5, embargo_pct=0.01):
        test_start = t1.index[te[0]]
        test_end = t1.iloc[te].max()
        # ninguna etiqueta de train puede solapar la ventana de test
        overlaps = (t1.index[tr] <= test_end) & (t1.iloc[tr].values >= test_start)
        if overlaps.any():
            ok_t1 = False
    checks["t1_no_label_overlap"] = ok_t1

    # --- 3) wrappers SSOT sobre ruido sintético
    fake_returns = rng.normal(0.0005, 0.01, size=260)     # ruido, no un backtest
    m = trial_aware_moments(fake_returns)
    rep = dsr_report(m["sharpe_per_period"], m["n_obs"], n_trials=25,
                     skew=m["skew"], kurtosis=m["kurtosis"], periods_per_year=52)
    checks["dsr_report_shape"] = all(k in rep for k in
                                     ("headline_dsr", "cells", "bar", "passes"))
    checks["dsr_in_unit_interval"] = (rep["headline_dsr"] is None or
                                      0.0 <= rep["headline_dsr"] <= 1.0)

    bb = circular_block_bootstrap(fake_returns, np.mean, n_boot=500, block=4, seed=42)
    checks["bootstrap_ci_finite"] = (bb["ci95"][0] is not None and
                                     np.isfinite(bb["ci95"][0]) and
                                     np.isfinite(bb["ci95"][1]))

    # --- 4) JSON-safety del reporte completo (sin Infinity/NaN)
    payload = {"checks": checks, "dsr_report_synthetic": rep, "bootstrap_synthetic": bb,
               "nan_probe": float("nan"), "inf_probe": float("inf")}
    js = safe_json_dumps(payload)
    checks["json_safe"] = ("Infinity" not in js) and ("NaN" not in js)

    passed = all(checks.values())
    return {"passed": passed, "checks": checks,
            "note": ("smoke sintetico default_rng(42) — 0 trials, ningun OOS abierto; "
                     "cualquier estudio futuro requiere pre-registro en "
                     "HYPOTHESIS-REGISTRY + aprobacion del operador")}


def main() -> int:
    res = smoke()
    for k, v in res["checks"].items():
        print(f"  [{'OK' if v else 'FAIL'}] {k}")
    print(("[PASS] " if res["passed"] else "[FAIL] ") + res["note"])
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
