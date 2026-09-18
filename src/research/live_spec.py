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
- el posterior de régimen del cierre de `d-1` (`PortableRegimeModel`, numpy puro),
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
from datetime import date
from decimal import Decimal
from numbers import Real
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.dataset import (
    CLIP,
    MACRO_FEATURES,
    MARKET_FEATURES,
    N_REGIMES,
    SEED_M5,
)
from src.research.evaluation_mask import _mask_from_frame, colombia_holidays, usa_holidays
from src.research.features import attach_macro_features, build_market_features
from src.research.observation_contract import LEGACY_VERSION
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
    macro_mean: np.ndarray | None = None
    macro_scale: np.ndarray | None = None

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.scale

    @classmethod
    def load(cls, path: Path = SCALER_PATH) -> FrozenScaler:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        expected_identity = None
        if d.get("dataset_identity") is not None:
            from src.research.dataset import dataset_identity
            expected_identity = dataset_identity()
            if d["dataset_identity"] != expected_identity:
                raise ValueError(
                    "frozen scaler identity mismatch; export scaler after rebuilding v2"
                )
        elif path.resolve() == SCALER_PATH.resolve():
            raise ValueError("frozen scaler has no dataset identity; legacy artifact refused")
        return cls(mean=np.asarray(d["mean"], dtype=float),
                   scale=np.asarray(d["scale"], dtype=float),
                   features=tuple(d["features"]),
                   macro_mean=(np.asarray(d["macro_mean"], dtype=float)
                               if d.get("macro_mean") is not None else None),
                   macro_scale=(np.asarray(d["macro_scale"], dtype=float)
                                if d.get("macro_scale") is not None else None))

    def to_dict(self) -> dict:
        from src.research.dataset import dataset_identity
        return {"contract": "CTR-RESEARCH-FORWARD-001",
                "dataset_identity": dataset_identity(),
                "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "features": list(self.features),
                "macro_mean": (self.macro_mean.tolist() if self.macro_mean is not None else None),
                "macro_scale": (self.macro_scale.tolist() if self.macro_scale is not None else None)}


@dataclass(frozen=True)
class PartialLiveSpec:
    """Observaciones disponibles hasta una barra cerrada, sin inventar el resto.

    No es un ``SessionSpec`` entrenable: deliberadamente no satisface el contrato de 60
    cierres. El carril forward debe acumular estos prefijos y solo liquidar cuando la sesión
    esté completa; exponer este tipo evita que una sesión parcial se cuele como confirmatoria.
    """

    date: date
    bars_received: int
    market: np.ndarray
    context: np.ndarray
    closes: np.ndarray
    spread_pips: float
    observation_version: str = LEGACY_VERSION

    @property
    def sealed_before_next_bar(self) -> bool:
        return self.bars_received > 0


def export_scaler(data, path: Path = SCALER_PATH) -> Path:
    """Vuelca el escalador de un `ResearchData` al formato congelado. Se corre UNA vez."""
    scaler = FrozenScaler(mean=np.asarray(data.scaler_mean, dtype=float),
                          scale=np.asarray(data.scaler_scale, dtype=float),
                          features=tuple(MARKET_FEATURES),
                          macro_mean=(None if data.macro_scaler_mean is None
                                      else np.asarray(data.macro_scaler_mean, dtype=float)),
                          macro_scale=(None if data.macro_scaler_scale is None
                                       else np.asarray(data.macro_scaler_scale, dtype=float)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scaler.to_dict(), indent=2), encoding="utf-8")
    return path


def _levels_for_k(k: int) -> np.ndarray:
    lo, mid, hi = SPREAD_PIPS_BY_LEVEL
    if k == 3:
        return np.array([lo, mid, hi], dtype=float)
    return np.interp(np.linspace(0.0, 1.0, k), [0.0, 0.5, 1.0], [lo, mid, hi])


def _validate_regime_width(regime: PortableRegimeModel) -> None:
    """A live observation obeys the same slot limit as the dataset producer."""
    k = regime.k
    if (isinstance(k, bool | np.bool_) or not isinstance(k, int | np.integer)
            or not 1 <= k <= N_REGIMES):
        raise ValueError(
            f"HMM k={k!r} must be an integer in [1, {N_REGIMES}]; "
            "posterior truncation is forbidden. A new preregistration/schema decision is required."
        )


def _checked_regime_posterior(regime: PortableRegimeModel, prior: np.ndarray) -> np.ndarray:
    """Reject malformed probabilities; never repair or renormalize model output."""
    _validate_regime_width(regime)
    raw = regime.filtered_posterior(prior)
    if np.ma.isMaskedArray(raw) and np.ma.getmaskarray(raw).any():
        raise ValueError("HMM regime posterior contains masked/unknown probabilities")
    if isinstance(raw, list | tuple) and any(isinstance(p, bool | np.bool_) for p in raw):
        raise ValueError("HMM regime posterior contains boolean probabilities")
    try:
        probs = np.asarray(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("HMM regime posterior must be a numeric vector") from exc
    if probs.shape != (regime.k,) or probs.dtype.kind not in "fiu":
        raise ValueError("HMM regime posterior must be a real numeric vector of shape (k,)")
    probs = probs.astype(float, copy=False)
    if (not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any()
            or not np.isclose(probs.sum(), 1.0, atol=1e-9, rtol=0.0)):
        raise ValueError("HMM regime posterior must be finite in [0, 1] with unit mass (atol=1e-9)")
    return probs


def _cot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize known instants, never guess the timezone of market timestamps."""
    if "time" not in frame.columns:
        raise ValueError("market data must contain time")
    times = frame["time"]
    if isinstance(times.dtype, pd.DatetimeTZDtype):
        if times.isna().any():
            raise ValueError("market time must be non-missing and timezone-aware")
        normalized = times.dt.tz_convert("America/Bogota")
    else:
        try:
            parsed = [pd.Timestamp(value) for value in times]
        except (TypeError, ValueError) as exc:
            raise ValueError("market time must contain known timezone-aware instants") from exc
        if any(pd.isna(value) or value.tzinfo is None for value in parsed):
            raise ValueError("market time must be non-missing and timezone-aware")
        normalized = pd.to_datetime(parsed, utc=True).tz_convert("America/Bogota")
    result = frame.copy()
    result["time"] = normalized
    return result


def _validated_prefix(target: date, bars: pd.DataFrame) -> pd.DataFrame:
    """Admit only the received, ordered USD/COP prefix; do not consult today's mask."""
    if bars is None or bars.empty:
        raise ValueError(f"{target}: se necesita al menos una barra cerrada")
    required = {"time", "open", "high", "low", "close"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"faltan columnas de barras: {sorted(missing)}")
    if len(bars) > BARS_PER_SESSION:
        raise ValueError(f"{target}: {len(bars)} barras, maximo {BARS_PER_SESSION}")
    if (target.weekday() >= 5 or target.isoformat() in colombia_holidays()
            or target.isoformat() in usa_holidays()):
        raise ValueError(f"{target}: calendario excluye festivo o weekend")
    frame = _cot_frame(bars)
    expected = pd.date_range(f"{target} 08:00", periods=len(frame), freq="5min",
                             tz="America/Bogota")
    if not (pd.DatetimeIndex(frame["time"]) == expected).all():
        raise ValueError(f"{target}: prefijo debe ser ordenado y contiguo desde 08:00, cada 5 min")
    if "symbol" in frame.columns:
        symbols = frame["symbol"].astype(str).str.upper().str.replace("/", "", regex=False)
        if not (symbols == "USDCOP").all():
            raise ValueError("prefix contains a non-USDCOP or missing symbol")
    else:
        # This builder is explicitly USD/COP-scoped. Without this assignment,
        # concatenating with a symbol-bearing history would drop the entire prefix.
        frame["symbol"] = "USDCOP"
    raw = frame[["open", "high", "low", "close"]]
    if any(isinstance(value, bool | np.bool_) or not isinstance(value, str | Real | Decimal)
           for value in raw.to_numpy(dtype=object).flat):
        raise ValueError("prefix OHLC contains a non-price scalar type")
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    if (not np.isfinite(numeric.to_numpy()).all() or (numeric <= 0).any().any()
            or (numeric["high"] < numeric[["open", "close"]].max(axis=1)).any()
            or (numeric["low"] > numeric[["open", "close"]].min(axis=1)).any()
            or (numeric["high"] < numeric["low"]).any()):
        raise ValueError("prefix OHLC must be finite, positive and internally coherent")
    frame[["open", "high", "low", "close"]] = numeric
    return frame


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

    _validate_regime_width(regime)

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
    m5 = _cot_frame(m5)
    t = m5["time"]
    hist = m5[t.dt.date <= target]
    valid = set(_mask_from_frame(hist, source="live_input").valid)
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
    probs = _checked_regime_posterior(regime, prior.to_numpy(dtype=float))
    spread = float(np.dot(probs, _levels_for_k(regime.k)))

    if len(probs) < N_REGIMES:
        probs = np.pad(probs, (0, N_REGIMES - len(probs)))

    # --- macro as-of ------------------------------------------------------
    macro = attach_macro_features([target])
    macro_values = macro.loc[pd.Timestamp(target), MACRO_FEATURES].to_numpy(dtype=float)
    if scaler.macro_mean is not None and scaler.macro_scale is not None:
        macro_values = (macro_values - scaler.macro_mean) / scaler.macro_scale
    ctx = np.concatenate([
        macro_values,
        probs,
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

    El spread sale del posterior de régimen del cierre de `d-1` (§8.1/§8.4), así que se
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

    m5 = _cot_frame(m5)
    t = m5["time"]
    hist = m5[t.dt.date < target]
    valid = set(_mask_from_frame(hist, source="live_input").valid)

    obs = build_regime_observations(hist, valid_sessions=valid).dropna()
    if len(obs) < MIN_CONTEXT_SESSIONS:
        raise ValueError(
            f"{target}: solo {len(obs)} sesiones de contexto, hacen falta "
            f"{MIN_CONTEXT_SESSIONS} para un posterior estable."
        )
    probs = regime.filtered_posterior(obs.to_numpy(dtype=float))
    return float(np.dot(probs, _levels_for_k(regime.k)))


def build_live_spec_partial(session_date: str | date, bars: pd.DataFrame,
                            scaler: FrozenScaler | None = None,
                            regime: PortableRegimeModel | None = None) -> PartialLiveSpec:
    """Construye únicamente el prefijo causal de una sesión.

    ``bars`` debe contener las barras cerradas recibidas hasta el instante de sellado. Se
    rechazan filas futuras, duplicadas y sesiones distintas. Las ventanas se calculan sobre
    la serie truncada, por lo que agregar barras posteriores no puede reescribir el prefijo.
    """
    target = pd.Timestamp(session_date).date() if not isinstance(session_date, date) else session_date
    frame = _validated_prefix(target, bars)

    # The seed is read once, but only days strictly before target may influence
    # admission or features. Today's received prefix is admitted independently.
    full = _cot_frame(pd.read_parquet(SEED_M5))
    history = full[full["time"].dt.date < target]
    if "symbol" not in history.columns:
        history = history.assign(symbol="USDCOP")
    valid = set(_mask_from_frame(history, source="live_history").valid)
    valid.add(target)
    hist = pd.concat([history, frame], ignore_index=True)
    scaler = scaler or FrozenScaler.load()
    regime = regime or PortableRegimeModel.load()
    _validate_regime_width(regime)
    feats = build_market_features(hist, valid_sessions=valid)
    day = feats[feats["_session"] == target]
    if len(day) != len(frame):
        raise ValueError(f"{target}: prefijo de features inconsistente ({len(day)} != {len(frame)})")

    obs = build_regime_observations(hist, valid_sessions=valid).dropna()
    obs.index = [x.date() if hasattr(x, "date") else x for x in obs.index]
    prior = obs[[d < target for d in obs.index]]
    if len(prior) < MIN_CONTEXT_SESSIONS:
        raise ValueError(f"{target}: contexto insuficiente para posterior filtrado")
    probs = _checked_regime_posterior(regime, prior.to_numpy(dtype=float))
    spread = float(np.dot(probs, _levels_for_k(regime.k)))
    if len(probs) < N_REGIMES:
        probs = np.pad(probs, (0, N_REGIMES - len(probs)))
    macro = attach_macro_features([target])
    macro_row = macro.loc[pd.Timestamp(target), MACRO_FEATURES].to_numpy(dtype=float)
    if not np.isfinite(macro_row).all():
        raise ValueError(f"{target}: macro ausente o stale; prefijo no sellable")
    if scaler.macro_mean is not None and scaler.macro_scale is not None:
        macro_row = (macro_row - scaler.macro_mean) / scaler.macro_scale
    ctx = np.concatenate([macro_row, probs])
    X = scaler.transform(day[MARKET_FEATURES].to_numpy(dtype=float))
    return PartialLiveSpec(date=target, bars_received=len(frame),
                           market=np.clip(X, -CLIP, CLIP).astype(np.float32),
                           context=np.clip(ctx, -CLIP, CLIP).astype(np.float32),
                           closes=frame["close"].to_numpy(dtype=float),
                           spread_pips=spread)
