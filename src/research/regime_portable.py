"""El HMM de régimen, sin `hmmlearn`: parámetros exportados + recursión hacia adelante.

Contract: CTR-RESEARCH-REGIME-PORTABLE-001 · Date: 2026-08-25

## Por qué existe

El brazo RL forward tiene que construir la observación de una sesión viva, y esa observación
incluye los 4 posteriores de régimen. Calcularlos exige el HMM ajustado, y el contenedor de
Airflow **no tiene `hmmlearn`** — el mismo obstáculo que ya obligó a inventar el formato
portable del dataset (`dataset.py::save_portable`), que resuelve el problema *descartando* el
modelo. Aquí no se puede descartar: se necesita evaluarlo cada día.

Instalarlo en el contenedor fue el primer intento y falló (el `pip` de `root` no escribe en el
site-packages de `airflow`). Y aunque hubiera funcionado, dejaría el carril forward dependiendo
de un `pip install` manual que no sobrevive a recrear el contenedor.

## Qué se hace en su lugar

Un HMM gaussiano evaluado es aritmética: `startprob`, `transmat`, `means_`, `covars_`. Se
exportan a un JSON y el posterior filtrado se calcula con la **recursión alfa**, que son treinta
líneas de numpy.

## Y de paso es más correcto

`FrozenRegimeModel.filtered_posterior` llamaba a `predict_proba(X)[-1]`. `predict_proba` es
forward-**backward**: calcula el posterior SUAVIZADO de toda la ventana y luego se tira todo
menos la última fila. En la última fila suavizado y filtrado coinciden —no hay futuro más allá
de `t` dentro de la ventana— así que el resultado era correcto, pero pagando una pasada hacia
atrás completa por cada sesión y cada día.

La recursión alfa calcula **directamente** lo que se quiere: `P(estado_t | datos ≤ t)`.

`test_regime_portable_matches_hmmlearn` compara las dos implementaciones sobre las sesiones del
hold-out. Si divergieran, el brazo forward correría con posteriores distintos de los que
produjeron el resultado de la tesis, y la comparación no mediría lo que dice medir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO / "config" / "research" / "regime_hmm_frozen.json"


@dataclass(frozen=True)
class PortableRegimeModel:
    """Todo lo necesario para evaluar el HMM, sin la librería que lo ajustó."""

    k: int
    startprob: np.ndarray          # (k,)
    transmat: np.ndarray           # (k, k)
    means: np.ndarray              # (k, d) en el espacio ESTANDARIZADO
    covars: np.ndarray             # (k, d, d)
    std_means: np.ndarray          # (d,) estandarizacion de las observaciones
    std_scales: np.ndarray         # (d,)
    vol_order: tuple[int, ...]
    labels: tuple[str, ...]
    feature_names: tuple[str, ...]
    fit_range: tuple[str, str]

    # -- emisiones ---------------------------------------------------------
    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """`log N(x_t | mu_k, Sigma_k)` para cada t y k, vía Cholesky.

        Cholesky y no `inv()`: invertir una covarianza mal condicionada da un número sin
        avisar, mientras que `cholesky` lanza. Con `covariance_type='full'` y 9 features eso
        no es teórico.
        """
        n, d = X.shape
        out = np.empty((n, self.k), dtype=float)
        for j in range(self.k):
            cov = self.covars[j]
            chol = np.linalg.cholesky(cov)
            diff = X - self.means[j]
            sol = np.linalg.solve_triangular(chol, diff.T, lower=True) \
                if hasattr(np.linalg, "solve_triangular") else \
                _solve_lower(chol, diff.T)
            maha = np.sum(sol ** 2, axis=0)
            log_det = 2.0 * np.sum(np.log(np.diag(chol)))
            out[:, j] = -0.5 * (maha + log_det + d * np.log(2.0 * np.pi))
        return out

    # -- posterior ---------------------------------------------------------
    def filtered_posterior(self, obs_window: np.ndarray) -> np.ndarray:
        """`P(estado_t | datos ≤ t)` con `t` = última fila, reordenado por volatilidad.

        Recursión alfa con normalización en cada paso. La normalización no es cosmética: sin
        ella `alpha` cae por debajo del mínimo de `float64` en unas pocas decenas de pasos y
        el posterior sale `nan` — y la ventana típica aquí son cientos de sesiones.
        """
        X = (np.asarray(obs_window, dtype=float) - self.std_means) / self.std_scales
        log_b = self._log_emission(X)

        # Se trabaja en escala lineal con renormalizacion por paso, que es exactamente lo
        # que hace hmmlearn internamente y evita divergencias numericas entre las dos.
        alpha = _renorm(self.startprob * np.exp(log_b[0] - log_b[0].max()), self.startprob)

        for t in range(1, len(X)):
            bt = np.exp(log_b[t] - log_b[t].max())
            alpha = _renorm((alpha @ self.transmat) * bt, self.startprob)

        return alpha[list(self.vol_order)]

    # -- serializacion -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "contract": "CTR-RESEARCH-REGIME-PORTABLE-001",
            "k": self.k,
            "startprob": self.startprob.tolist(),
            "transmat": self.transmat.tolist(),
            "means": self.means.tolist(),
            "covars": self.covars.tolist(),
            "std_means": self.std_means.tolist(),
            "std_scales": self.std_scales.tolist(),
            "vol_order": list(self.vol_order),
            "labels": list(self.labels),
            "feature_names": list(self.feature_names),
            "fit_range": list(self.fit_range),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PortableRegimeModel":
        return cls(
            k=int(d["k"]),
            startprob=np.asarray(d["startprob"], dtype=float),
            transmat=np.asarray(d["transmat"], dtype=float),
            means=np.asarray(d["means"], dtype=float),
            covars=np.asarray(d["covars"], dtype=float),
            std_means=np.asarray(d["std_means"], dtype=float),
            std_scales=np.asarray(d["std_scales"], dtype=float),
            vol_order=tuple(d["vol_order"]),
            labels=tuple(d["labels"]),
            feature_names=tuple(d["feature_names"]),
            fit_range=tuple(d["fit_range"]),
        )

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "PortableRegimeModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _renorm(alpha: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Normaliza un paso de la recursion, con red para la masa nula.

    Se aplica en TODOS los pasos, incluido el primero. La primera version lo omitia en el
    paso inicial y una observacion suficientemente extrema producia una division por cero
    ahi — el test del outlier pasaba igual, pero dejando un `RuntimeWarning: invalid value
    encountered in divide`. Un aviso que se ignora hoy es un `nan` que aparece en produccion
    el dia que la observacion llega peor.

    Masa nula significa "esta observacion es imposible bajo todos los estados": se cae al
    prior en vez de propagar `nan`.
    """
    total = alpha.sum()
    if total <= 0.0 or not np.isfinite(total):
        alpha = fallback.copy()
        total = alpha.sum()
    return alpha / total


def _solve_lower(chol: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sustitución hacia adelante para `L x = b`, sin depender de `scipy`."""
    from scipy.linalg import solve_triangular

    return solve_triangular(chol, b, lower=True)


def export_from_frozen(model, path: Path = DEFAULT_PATH) -> Path:
    """Vuelca un `FrozenRegimeModel` (con hmmlearn) al formato portable.

    Se corre UNA vez, en el host, y el JSON resultante viaja al contenedor por el montaje de
    `config/`. El modelo ajustado sigue siendo la fuente; esto es una proyección de solo
    lectura de sus parámetros.
    """
    hmm = model.model
    covars = hmm.covars_
    if covars.ndim == 2:                       # covariance_type='diag'
        covars = np.stack([np.diag(row) for row in covars])

    portable = PortableRegimeModel(
        k=model.k,
        startprob=np.asarray(hmm.startprob_, dtype=float),
        transmat=np.asarray(hmm.transmat_, dtype=float),
        means=np.asarray(hmm.means_, dtype=float),
        covars=np.asarray(covars, dtype=float),
        std_means=np.asarray(model.means, dtype=float),
        std_scales=np.asarray(model.scales, dtype=float),
        vol_order=tuple(model.vol_order),
        labels=tuple(model.state_labels()),
        feature_names=tuple(model.feature_names),
        fit_range=tuple(model.fit_range),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(portable.to_dict(), indent=2), encoding="utf-8")
    return path
