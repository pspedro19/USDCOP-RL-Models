"""Contraste estadístico de la tesis: bootstrap estacionario **pareado** (§11.4).

Contract: CTR-RESEARCH-INFERENCE-001 · Date: 2026-08-25

## Por qué pareado

Dos estrategias evaluadas sobre las MISMAS sesiones producen series de retorno fuertemente
correlacionadas: comparten el mercado. Remuestrear cada una por su lado destruye esa
correlación y ensancha el intervalo de la diferencia — se acaba declarando «indecidible» un
contraste que sí lo era.

Aquí se remuestrean **los mismos índices** para ambas series. Lo que se bootstrapea es la
diferencia, no cada Sharpe por separado.

## Por qué estacionario y no de bloque fijo

El bootstrap de bloque fijo (Künsch) trocea la serie en bloques de longitud L constante, y el
resultado depende de dónde caen los cortes. El estacionario de Politis-Romano sortea la
longitud de cada bloque de una geométrica de media L, lo que hace la serie remuestreada
**estacionaria** y elimina esa dependencia. §11.4 pide L en 5-20; se promedia sobre ese rango
en vez de elegir un L, porque elegirlo mirando el resultado sería otro trial.

## Lo que NO se computa, y por qué

**White RC y Hansen SPA no se calculan.** Contrastan un ganador contra el universo de
candidatos que compitió por serlo. Tras renunciar al HPO, ese universo tiene **dos** miembros.
Un SPA sobre dos candidatos produce un número con aspecto de rigor y contenido nulo. Se
declara la omisión y su razón, que es más honesto que publicar el número vacío.

PBO y DSR sí se computan: ninguno depende del tamaño del universo — el DSR depende del conteo
de trials, que es otra cosa y se hereda del registro del activo (n=111).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ANN_SESSIONS = 221.0          # sesiones por año, derivado en la Fase E
DEFAULT_BLOCKS = (5, 10, 15, 20)
N_BOOT = 10_000


def sharpe(returns: np.ndarray, ann: float = ANN_SESSIONS) -> float:
    r = np.asarray(returns, dtype=float)
    sd = np.std(r, ddof=1)
    return float(np.mean(r) / sd * np.sqrt(ann)) if sd > 0 else 0.0


def stationary_bootstrap_indices(n: int, mean_block: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """Índices de Politis-Romano: bloques de longitud geométrica, envolviendo el final.

    `p = 1/mean_block` es la probabilidad de arrancar un bloque nuevo en cada paso. El
    envolvimiento circular es lo que mantiene la muestra estacionaria: sin él, las
    observaciones del final de la serie aparecerían menos que las del principio.

    Vectorizado a propósito. La versión en bucle Python tardaba minutos en las 10.000
    réplicas que pide §11.4, y un contraste que cuesta minutos acaba corriéndose con menos
    réplicas de las declaradas.
    """
    p = 1.0 / max(mean_block, 1.0)
    fresh = rng.random(n) < p
    fresh[0] = True                                   # el primero siempre abre bloque
    starts = rng.integers(0, n, n)

    t = np.arange(n)
    # Para cada t, el instante en que empezó su bloque, y dónde arrancó ese bloque.
    block_start_t = np.maximum.accumulate(np.where(fresh, t, 0))
    block_origin = starts[block_start_t]
    return (block_origin + (t - block_start_t)) % n


def _bootstrap_matrix(n: int, n_boot: int, blocks, rng) -> np.ndarray:
    """Los mismos índices para todas las series: es lo que hace el contraste pareado."""
    per_block = max(1, n_boot // len(blocks))
    return np.stack([stationary_bootstrap_indices(n, L, rng)
                     for L in blocks for _ in range(per_block)])


def _sharpe_rows(x: np.ndarray, idx: np.ndarray, ann: float) -> np.ndarray:
    """Sharpe de cada fila remuestreada, de una sola pasada."""
    sample = x[idx]
    sd = sample.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sd > 0, sample.mean(axis=1) / sd * np.sqrt(ann), 0.0)
    return out


@dataclass
class PairedTest:
    """Resultado de un contraste pareado. `p_value` es a dos colas."""

    name_a: str
    name_b: str
    n: int
    sharpe_a: float
    sharpe_b: float
    diff: float
    ci_low: float
    ci_high: float
    p_value: float
    n_boot: int
    blocks: tuple = field(default_factory=tuple)
    correlation: float = 0.0

    @property
    def decisive(self) -> bool:
        """Decidible = el IC del 95% excluye el cero."""
        return not (self.ci_low <= 0.0 <= self.ci_high)

    def verdict(self) -> str:
        if not self.decisive:
            return (f"INDECIDIBLE: la diferencia de Sharpe ({self.diff:+.3f}) tiene un IC 95% "
                    f"[{self.ci_low:+.3f}, {self.ci_high:+.3f}] que incluye el cero con "
                    f"n={self.n}. No es 'sin diferencia': es que estos datos no bastan.")
        signo = "supera a" if self.diff > 0 else "queda por debajo de"
        return (f"DECIDIBLE: {self.name_a} {signo} {self.name_b} "
                f"(ΔSharpe {self.diff:+.3f}, IC 95% [{self.ci_low:+.3f}, {self.ci_high:+.3f}], "
                f"p={self.p_value:.4f}, n={self.n}).")

    def to_dict(self) -> dict:
        return {**self.__dict__, "decisive": self.decisive, "verdict": self.verdict()}


def paired_sharpe_test(a, b, name_a: str = "A", name_b: str = "B", n_boot: int = N_BOOT,
                       blocks=DEFAULT_BLOCKS, seed: int = 42,
                       ann: float = ANN_SESSIONS) -> PairedTest:
    """Bootstrap estacionario pareado de `Sharpe(a) − Sharpe(b)`.

    Ambas series tienen que estar alineadas sesión a sesión: es lo que hace válido el
    remuestreo con índices comunes. Si no lo estuvieran, el contraste compararía días
    distintos y su correlación no significaría nada.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(x) != len(y):
        raise ValueError(f"series no alineadas: {len(x)} vs {len(y)}; el contraste pareado "
                         "exige las MISMAS sesiones en el mismo orden")
    n = len(x)
    observed = sharpe(x, ann) - sharpe(y, ann)

    rng = np.random.default_rng(seed)
    idx = _bootstrap_matrix(n, n_boot, blocks, rng)
    diffs = _sharpe_rows(x, idx, ann) - _sharpe_rows(y, idx, ann)
    k = len(diffs)

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # p a dos colas por el metodo del percentil: cuanto de la distribucion remuestreada
    # cae al otro lado del cero respecto al efecto observado.
    centred = diffs - diffs.mean()
    p = 2.0 * min(np.mean(centred >= abs(observed)), np.mean(centred <= -abs(observed)))

    return PairedTest(
        name_a=name_a, name_b=name_b, n=n,
        sharpe_a=sharpe(x, ann), sharpe_b=sharpe(y, ann),
        diff=float(observed), ci_low=float(lo), ci_high=float(hi),
        p_value=float(min(p, 1.0)), n_boot=k, blocks=tuple(blocks),
        correlation=float(np.corrcoef(x, y)[0, 1]) if n > 2 else 0.0,
    )


def bootstrap_sharpe_ci(returns, n_boot: int = N_BOOT, blocks=DEFAULT_BLOCKS,
                        seed: int = 42, ann: float = ANN_SESSIONS) -> dict:
    """IC del Sharpe de UNA serie, con el mismo remuestreo (para las tablas de la §4.3)."""
    r = np.asarray(returns, dtype=float)
    rng = np.random.default_rng(seed)
    vals = _sharpe_rows(r, _bootstrap_matrix(len(r), n_boot, blocks, rng), ann)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"sharpe": sharpe(r, ann), "ci_low": float(lo), "ci_high": float(hi),
            "n": len(r), "n_boot": len(vals)}


def dsr_with_inherited_trials(returns, n_trials: int, ann: float = ANN_SESSIONS) -> dict:
    """DSR deflactado con el N del ACTIVO, no con el de esta tesis.

    `partition.yaml` declara `inherited_n: 111` y el ADR-0023 fija que la partición jamás
    resetea N. Un DSR calculado con los trials de este trabajo solamente sería un número
    distinto del que exige la constitución §2, y más favorable.
    """
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from services.common.metrics import dsr_report

    from scipy import stats as _st

    r = np.asarray(returns, dtype=float)
    sd = np.std(r, ddof=1)
    sr_per_period = float(np.mean(r) / sd) if sd > 0 else 0.0

    # `dsr_report` no toma la serie: quiere el Sharpe POR PERIODO mas sus momentos, y
    # `periods_per_year` para convertir el sigma anualizado de la rejilla. Pasarle 252
    # (su default) con sesiones de 221 desplazaria el umbral del nulo.
    rep = dsr_report(
        sharpe_per_period=sr_per_period, n_obs=len(r), n_trials=n_trials,
        periods_per_year=int(round(ann)),
        skew=float(_st.skew(r)), kurtosis=float(_st.kurtosis(r, fisher=False)),
    )
    rep["n_trials"] = n_trials
    rep["sharpe_annualized"] = sharpe(r, ann)
    rep["sharpe_per_period"] = sr_per_period
    return rep


WHITE_SPA_OMISSION = (
    "White Reality Check y Hansen SPA NO se computan. Ambos contrastan un ganador contra el "
    "universo de candidatos que compitio por serlo; tras renunciar al HPO ese universo tiene "
    "DOS miembros (ppo_regime y ppo_backbone). Un SPA sobre dos candidatos da un numero con "
    "apariencia de rigor y contenido nulo. Se declara la omision en vez de publicarlo vacio. "
    "PBO y DSR si se reportan: no dependen del tamano del universo."
)
