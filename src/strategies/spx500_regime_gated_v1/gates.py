"""Gates de aceptación y kill criteria (README §3, SDD-000 §4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["RLReport", "gate_g4", "gate_g6", "validate_report_benchmarks", "validate_rl_report"]

REQUIRED_BENCHMARKS = ("SPY_TR", "SIXTY_FORTY", "VOL_TARGET_10", "MA200")
MIN_RL_SEEDS = 10


def gate_g4(dsr: float, pbo: float) -> bool:
    """DSR > 0.95 (K1) y PBO < 0.50 (K2)."""
    return dsr > 0.95 and pbo < 0.50


def gate_g6(break_even_bps: float, real_cost_bps: float, margin: float = 3.0) -> bool:
    """break_even >= 3 x costo real, si no -> kill criterion K3."""
    return break_even_bps >= margin * real_cost_bps


def validate_report_benchmarks(benchmarks) -> None:
    missing = [b for b in REQUIRED_BENCHMARKS if b not in set(benchmarks)]
    if missing:
        raise ValueError(
            "SDD-006 §3: toda tabla debe reportar los cuatro benchmarks. "
            f"Faltan: {', '.join(missing)}"
        )


@dataclass(frozen=True, slots=True)
class RLReport:
    headline: float   # mediana, NUNCA el mejor seed
    iqr: float
    n_seeds: int


def validate_rl_report(sharpes) -> RLReport:
    s = np.asarray(sharpes, dtype=float)
    if s.size < MIN_RL_SEEDS:
        raise ValueError(
            f"SDD-005 §5.4 exige al menos {MIN_RL_SEEDS} seeds; se recibieron {s.size}. "
            "Reportar el mejor de pocos seeds equivale a trials ocultos."
        )
    q75, q25 = np.percentile(s, [75, 25])
    return RLReport(headline=float(np.median(s)), iqr=float(q75 - q25), n_seeds=int(s.size))
