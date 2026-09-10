"""Construye el `SessionSpec` de UNA sesión viva, sin `hmmlearn` y sin reconstruir el dataset.

Contract: CTR-RESEARCH-FORWARD-001 · Date: 2026-08-25

## El problema

`build_research_data` construye los tres bloques de golpe: ajusta el HMM, calcula features
sobre toda la serie, ajusta el escalador en desarrollo y emite 1.317 `SessionSpec`. Tarda
minutos y necesita `hmmlearn`.

El brazo RL forward necesita **una sesión**, hoy, dentro de un contenedor de Airflow que no
tiene `hmmlearn`. Reconstruir el dataset entero cada mañana para sacar un spec sería absurdo, y
además imposible en ese contenedor.

## Qué se reutiliza y qué se congela

Lo que el brazo forward **calcula** cada día:

- las features de mercado de la sesión (`build_market_features`, causales por construcción),
- el posterior de régimen del cierre de `d−1` (`PortableRegimeModel`, numpy puro),
- las features macro con `merge_asof(backward)`.

Lo que viene **congelado** del entrenamiento y no se recalcula nunca:

- el escalador (media y escala ajustadas SOLO en desarrollo),
- los parámetros del HMM.

Esa separación es la que impide que el brazo forward derive: si el escalador se reajustara con
datos nuevos, la política congelada estaría viendo un espacio de observación distinto del que
aprendió, y el resultado no sería comparable con nada.

## La comprobación que lo hace fiable

`test_live_spec_reproduces_the_batch_spec` construye con esta función el spec de una fecha del
hold-out y lo compara **elemento a elemento** con el que produjo `build_research_data`. Si
divergen, el brazo forward no está corriendo la misma política que evaluó la tesis, y la
comparación RL-vs-LLM mediría otra cosa sin decirlo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.dataset import (CLIP, MACRO_FEATURES, MARKET_FEATURES,  # noqa: F401
                                  N_REGIMES, SEED_M5)
from src.research.evaluation_mask import build_mask
from src.research.features import attach_macro_features, build_market_features
from src.research.regime_hmm import build_regime_observations
from src.research.regime_portable import PortableRegimeModel
from src.research.session_env import BARS_PER_SESSION
from src.research.session_gym import SessionSpec

REPO = Path(__file__).resolve().parents[2]
SCALER_PATH = REPO / "config" / "research" / "feature_scaler_frozen.json"

# Los mismos niveles de §8.4. Se importan de `regime_hmm` para que exista UNA definicion.
from src.research.regime_hmm import SPREAD_PIPS_BY_LEVEL  # noqa: E402

# Contexto minimo de sesiones para que el posterior filtrado sea estable. Igual que el
# `min_context` de `spread_series`: por debajo, `build_research_data` DESCARTA la sesion, y
# el brazo forward tiene que descartarla por la misma razon, no inventar un posterior.
MIN_CONTEXT_SESSIONS = 60


@dataclass(frozen=True)
class FrozenScaler:
    """Media y escala del escalador ajustado en desarrollo. Nunca se reajusta."""

    mean: np.ndarray
    scale: np.ndarray
    features: tuple[str, ...]

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale

    @classmethod
    def load(cls, path: Path = SCALER_PATH) -> "FrozenScaler":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(mean=np.asarray(d["mean"], dtype=float),
                   scale=np.asarray(d["scale"], dtype=float),
                   features=tuple(d["features"]))

    def to_dict(self) -> dict:
        return {"contract": "CTR-RESEARCH-FORWARD-001",
                "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "features": list(self.features)}


def export_scaler(data, path: Path = SCALER_PATH) -> Path:
    """Vuelca el escalador de un `ResearchData` al formato congelado. Se corre UNA vez."""
    scaler = FrozenScaler(mean=np.asarray(data.scaler_mean, dtype=float),
                          scale=np.asarray(data.scaler_scale, dtype=float),
                          features=tuple(MARKET_FEATURES))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scaler.to_dict(), indent=2), encoding="utf-8")
    return path


def _levels_for_k(k: int) -> np.ndarray:
    lo, mid, hi = SPREAD_PIPS_BY_LEVEL
    if k == 3:
        return np.array([lo, mid, hi], dtype=float)
    return np.interp(np.linspace(0.0, 1.0, k), [0.0, 0.5, 1.0], [lo, mid, hi])


def build_live_spec(session_date: str | date, m5: pd.DataFrame | None = None,
                    scaler: FrozenScaler | None = None,
                    regime: PortableRegimeModel | None = None) -> SessionSpec:
    """`SessionSpec` de una sola sesión, con artefactos congelados.

    Args:
        session_date: la sesión a construir, `YYYY-MM-DD` o `date`.
        m5: serie de 5 minutos; por defecto el seed. Debe incluir la sesión pedida **y**
            al menos `MIN_CONTEXT_SESSIONS` anteriores.

    Raises:
        ValueError: si la sesión no está completa (≠ 60 barras) o no hay contexto suficiente
            para el posterior. Se lanza en vez de rellenar: una sesión incompleta produce un
            spec silenciosamente distinto del que vio el entrenamiento.
    """
    target = pd.Timestamp(session_date).date() if not isinstance(session_date, date) \
        else session_date
    m5 = pd.read_parquet(SEED_M5) if m5 is None else m5
    scaler = scaler or FrozenScaler.load()
    regime = regime or PortableRegimeModel.load()

    # La serie se acota DOS veces, y las dos importan:
    #
    # 1. Por fecha: nada posterior a la sesion entra. Las ventanas largas (`rv_78`, `EMA_72`,
    #    `zscore_60`) necesitan pasado, pero solo pasado.
    # 2. Por la MASCARA DE EVALUACION. `build_research_data` calcula las features sobre la
    #    serie enmascarada —solo sesiones validas, concatenadas— asi que sus ventanas saltan
    #    festivos e incompletas. Omitir este filtro parecia inofensivo y no lo era: la primera
    #    version daba deltas de ~1e-7 en fechas normales y **4,27 en 2024-01-02**, sobre una
    #    feature acotada a +-5. Es decir, una observacion completamente distinta justo en la
    #    primera sesion del hold-out, y solo se vio al comparar VARIAS fechas en vez de una.
    valid = {d for d in build_mask().valid if d <= target}
    t = pd.to_datetime(m5["time"])
    hist = m5[t.dt.date <= target]
    if hist.empty:
        raise ValueError(f"{target}: sin datos hasta esa fecha")
    if target not in valid:
        raise ValueError(
            f"{target}: la mascara de evaluacion la excluye (festivo o sesion incompleta). "
            "`build_research_data` no genera spec para estas fechas."
        )

    feats = build_market_features(hist, valid_sessions=valid)
    day = feats[feats["_session"] == target]
    if len(day) != BARS_PER_SESSION:
        raise ValueError(
            f"{target}: {len(day)} barras, se esperan {BARS_PER_SESSION}. Una sesion "
            "incompleta produce un spec distinto del que vio el entrenamiento."
        )

    # --- posterior de regimen del cierre de d-1 (§8.1) --------------------
    obs = build_regime_observations(hist, valid_sessions=valid).dropna()
    obs.index = [x.date() if hasattr(x, "date") else x for x in obs.index]
    prior = obs[[d < target for d in obs.index]]
    if len(prior) < MIN_CONTEXT_SESSIONS:
        raise ValueError(
            f"{target}: solo {len(prior)} sesiones de contexto, hacen falta "
            f"{MIN_CONTEXT_SESSIONS}. `build_research_data` descarta estas sesiones; el "
            "brazo forward las descarta por la misma razon."
        )
    probs = regime.filtered_posterior(prior.to_numpy(dtype=float))
    spread = float(np.dot(probs, _levels_for_k(regime.k)))

    if len(probs) < N_REGIMES:
        probs = np.pad(probs, (0, N_REGIMES - len(probs)))

    # --- macro as-of ------------------------------------------------------
    macro = attach_macro_features([target])
    ctx = np.concatenate([
        macro.loc[pd.Timestamp(target), MACRO_FEATURES].to_numpy(dtype=float),
        probs[:N_REGIMES],
    ])

    X = scaler.transform(day[MARKET_FEATURES].to_numpy(dtype=float))

    dt = pd.to_datetime(hist["time"])
    same_day = hist[dt.dt.date == target]
    if "symbol" in same_day.columns:
        sym = same_day["symbol"].astype(str).str.upper().str.replace("/", "", regex=False)
        same_day = same_day[sym == "USDCOP"]
    closes = same_day.sort_values("time")["close"].to_numpy(dtype=float)

    return SessionSpec(
        date=target,
        close=closes,
        market=np.clip(X, -CLIP, CLIP).astype(np.float32),
        context=np.clip(ctx, -CLIP, CLIP).astype(np.float32),
        spread_pips=spread,
    )


def session_spread(session_date: str | date, m5: pd.DataFrame | None = None,
                   regime: PortableRegimeModel | None = None) -> float:
    """Spread esperado de una sesión, **sin** necesitar sus barras.

    El spread sale del posterior de régimen del cierre de `d−1` (§8.1/§8.4), así que se
    conoce **antes de la apertura**. Eso permite que el brazo LLM —que sella a las 07:15,
    cuando la sesión aún no tiene ni una barra— selle también su contrato de costos.

    Sin esto, el costo del LLM se calcularía al liquidar, es decir DESPUES de conocer el
    resultado. Nadie lo manipularía a propósito, pero el diseño entero de este carril consiste
    en no tener que confiar en eso.
    """
    target = pd.Timestamp(session_date).date() if not isinstance(session_date, date) \
        else session_date
    m5 = pd.read_parquet(SEED_M5) if m5 is None else m5
    regime = regime or PortableRegimeModel.load()

    valid = {d for d in build_mask().valid if d < target}
    t = pd.to_datetime(m5["time"])
    hist = m5[t.dt.date < target]

    obs = build_regime_observations(hist, valid_sessions=valid).dropna()
    if len(obs) < MIN_CONTEXT_SESSIONS:
        raise ValueError(
            f"{target}: solo {len(obs)} sesiones de contexto, hacen falta "
            f"{MIN_CONTEXT_SESSIONS} para un posterior estable."
        )
    probs = regime.filtered_posterior(obs.to_numpy(dtype=float))
    return float(np.dot(probs, _levels_for_k(regime.k)))
