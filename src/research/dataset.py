"""Ensambla los `SessionSpec` de cada bloque, con el escalado de §6.6.

Contract: CTR-RESEARCH-DATASET-001 · Date: 2026-08-25

## La regla que gobierna este módulo

**El escalador se ajusta SOLO sobre desarrollo.** Selección y hold-out lo reciben congelado.

No es una preferencia metodológica: es la capa 1 del anti-look-ahead de la constitución
(«normalización global»). Un `StandardScaler` ajustado sobre todo el período filtra la media
y la varianza del futuro dentro de cada observación del pasado. El efecto es pequeño por
observación, invisible en cualquier tabla, y sistemáticamente favorable — la clase de fuga
que sobrevive a una revisión porque no produce ningún número raro.

`test_scaler_is_blind_to_evaluation_data` lo mide en vez de confiar en la lectura del código:
altera el hold-out entero y comprueba que ni un solo valor del escalador se mueve.

## Cómo se arma una sesión

    features de mercado (27)  ->  build_market_features, escaladas con el scaler de desarrollo
    estado de posición  (5)   ->  lo produce el Env dentro del episodio (endógeno, §10.6)
    macro as-of         (3)   ->  merge_asof(backward), constante en el día
    régimen             (4)   ->  posterior FILTRADO del cierre de d-1 (§8.1)

El `spread_pips` de la sesión sale del mismo posterior: es el spread esperado, no el del
`argmax` (§8.4). Un régimen incierto entre calmo y shock cobra un spread intermedio, que es
lo honesto.

## Sesiones sin contexto de régimen

Las primeras `min_context` sesiones de la serie no tienen historial suficiente para un
posterior filtrado. No se inventan: se **descartan**, y el conteo descartado se reporta.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.research.evaluation_mask import build_mask
from src.research.features import GROUPS, attach_macro_features, build_market_features
from src.research.regime_hmm import build_regime_observations, fit_frozen, spread_series
from src.research.session_env import BARS_PER_SESSION
from src.research.session_gym import SessionSpec

REPO = Path(__file__).resolve().parents[2]
PARTITION = REPO / "config" / "research" / "partition.yaml"
SEED_M5 = REPO / "seeds" / "latest" / "usdcop_m5_ohlcv.parquet"

MARKET_GROUPS = ("precio", "volatilidad", "tendencia", "temporal")
MARKET_FEATURES = [f for g in MARKET_GROUPS for f in GROUPS[g]]
MACRO_FEATURES = list(GROUPS["macro"])
N_REGIMES = len(GROUPS["regimen"])
CLIP = 5.0


@dataclass
class ResearchData:
    """Los tres bloques, más lo que hizo falta descartar para armarlos."""

    development: list[SessionSpec]
    selection: list[SessionSpec]
    holdout: list[SessionSpec]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    regime_model: object
    dropped: dict[str, int]

    def block(self, name: str) -> list[SessionSpec]:
        return {"development": self.development, "selection": self.selection,
                "holdout": self.holdout}[name]

    def summary(self) -> str:
        return (f"desarrollo {len(self.development)} · selección {len(self.selection)} · "
                f"hold-out {len(self.holdout)} · descartadas {self.dropped}")


def load_partition() -> dict:
    return yaml.safe_load(PARTITION.read_text(encoding="utf-8"))


def _block_dates(part: dict, name: str, valid) -> list:
    blk = part["blocks"][name]
    lo = pd.Timestamp(blk["start"]).date()
    hi = pd.Timestamp(blk["end"]).date()
    return [d for d in valid if lo <= d <= hi]


def fit_dev_scaler(feats: pd.DataFrame, dev_dates: set) -> tuple[np.ndarray, np.ndarray]:
    """Media y escala de §6.6, ajustadas **solo** sobre desarrollo.

    Funcion aparte para que el test pueda comprobarlo sin reconstruir el dataset entero:
    la propiedad que importa es que alterar seleccion o hold-out no mueva ni un valor.
    """
    dev_rows = feats[[d in dev_dates for d in feats["_session"]]][MARKET_FEATURES]
    if dev_rows.empty:
        raise ValueError("el escalador no vio ninguna sesion de desarrollo")
    mean = dev_rows.mean().to_numpy(dtype=float)
    scale = dev_rows.std(ddof=0).to_numpy(dtype=float)
    scale[scale == 0] = 1.0          # feature constante: escalar por 0 daria inf
    return mean, scale


def build_research_data(m5: pd.DataFrame | None = None, min_context: int = 60,
                        verbose: bool = True) -> ResearchData:
    """Arma los tres bloques. El escalador y el HMM se ajustan solo sobre desarrollo."""
    part = load_partition()
    mask = build_mask()
    valid = list(mask.valid)
    if m5 is None:
        m5 = pd.read_parquet(SEED_M5)

    dev_dates = set(_block_dates(part, "development", valid))

    # --- HMM congelado sobre desarrollo (§8.2) ---------------------------
    # `build_regime_observations` indexa por Timestamp y `fit_frozen` cuenta con ello
    # (llama a `.date()` sobre el indice). El resto del carril —mascara, features,
    # particion— usa `date`. Se convierte la SALIDA, no la entrada: mutar `obs_all`
    # rompia `fit_frozen`, y dejarlo sin convertir hacia que ningun filtro de bloque
    # casara y el HMM recibiera 0 filas. Los dos fallos ya ocurrieron.
    obs_all = build_regime_observations(m5, valid_sessions=set(valid))
    dev_obs = obs_all[[t.date() in dev_dates for t in obs_all.index]]
    model = fit_frozen(dev_obs)
    regimes = spread_series(model, obs_all, min_context=min_context,
                            shift_to_next_session=True)
    regimes.index = [t.date() for t in regimes.index]
    if verbose:
        print(f"HMM: K={model.k} ajustado en {model.fit_range} "
              f"({', '.join(model.state_labels())})")

    # --- features de mercado --------------------------------------------
    feats = build_market_features(m5, valid_sessions=set(valid))
    macro = attach_macro_features(valid)

    mean, scale = fit_dev_scaler(feats, dev_dates)

    prob_cols = [c for c in regimes.columns if c.startswith("p_")]
    blocks: dict[str, list[SessionSpec]] = {}
    dropped = {"sin_regimen": 0, "barras_incompletas": 0}

    by_session = {d: g for d, g in feats.groupby("_session")}
    for name in ("development", "selection", "holdout"):
        specs: list[SessionSpec] = []
        for d in _block_dates(part, name, valid):
            g = by_session.get(d)
            if g is None or len(g) != BARS_PER_SESSION:
                dropped["barras_incompletas"] += 1
                continue
            if d not in regimes.index or not np.isfinite(regimes.at[d, "spread_pips"]):
                dropped["sin_regimen"] += 1
                continue

            X = (g[MARKET_FEATURES].to_numpy(dtype=float) - mean) / scale
            probs = regimes.loc[d, prob_cols].to_numpy(dtype=float)
            if len(probs) < N_REGIMES:                       # K < 4: se rellena con ceros
                probs = np.pad(probs, (0, N_REGIMES - len(probs)))
            ctx = np.concatenate([
                macro.loc[pd.Timestamp(d), MACRO_FEATURES].to_numpy(dtype=float),
                probs[:N_REGIMES]])

            specs.append(SessionSpec(
                date=d,
                close=_closes(m5, d),
                market=np.clip(X, -CLIP, CLIP).astype(np.float32),
                context=np.clip(ctx, -CLIP, CLIP).astype(np.float32),
                spread_pips=float(regimes.at[d, "spread_pips"]),
            ))
        blocks[name] = specs
        if verbose:
            print(f"  {name:<12} {len(specs):>4} sesiones")

    if verbose and any(dropped.values()):
        print(f"  descartadas: {dropped}")

    return ResearchData(development=blocks["development"], selection=blocks["selection"],
                        holdout=blocks["holdout"], scaler_mean=mean, scaler_scale=scale,
                        regime_model=model, dropped=dropped)


_CLOSE_CACHE_KEY: tuple | None = None
_CLOSE_CACHE: dict = {}


def _close_cache_key(m5: pd.DataFrame) -> tuple:
    """Identity of the price input; never reuse closes from another dataframe."""
    t = pd.to_datetime(m5["time"])
    values = pd.util.hash_pandas_object(
        pd.DataFrame({"time": t, "close": m5["close"]}), index=False
    ).to_numpy(dtype="uint64", copy=False)
    return len(m5), int(values.sum(dtype="uint64")), int(values[-1]) if len(values) else 0


def _closes(m5: pd.DataFrame, d) -> np.ndarray:
    """Cierres de la sesión `d`, cacheados: se piden una vez por sesión y bloque."""
    global _CLOSE_CACHE_KEY, _CLOSE_CACHE
    cache_key = _close_cache_key(m5)
    if cache_key != _CLOSE_CACHE_KEY:
        _CLOSE_CACHE_KEY = cache_key
        _CLOSE_CACHE = {}
        t = pd.to_datetime(m5["time"])
        df = m5.assign(_t=t, _d=t.dt.date).sort_values("_t")
        if "symbol" in df.columns:
            sym = df["symbol"].astype(str).str.upper().str.replace("/", "", regex=False)
            df = df[sym == "USDCOP"]
        for day, g in df.groupby("_d"):
            _CLOSE_CACHE[day] = g["close"].to_numpy(dtype=float)
    return _CLOSE_CACHE[d]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
# Configurable porque en Airflow `outputs/` NO esta montado: el contenedor solo ve
# `data/`, `src/`, `scripts/`, `config/` y `seeds/`. Sin esto los artefactos de una
# corrida de 3 horas se quedarian dentro del contenedor.
CACHE = Path(os.environ.get("THESIS_CACHE",
                            REPO / "outputs" / "thesis" / "research_data.pkl"))


def load_or_build(rebuild: bool = False, verbose: bool = True) -> ResearchData:
    """Arma el dataset una vez y lo reutiliza.

    El ajuste del HMM mas los 1.383 posteriores filtrados cuestan varios minutos, y el
    entrenamiento multi-semilla pide el dataset una vez por corrida. La cache es una
    optimizacion, no una fuente: se invalida sola si cambia el hash de la mascara o el
    del esquema de features, que son los dos artefactos que definen el contenido.
    """
    import pickle

    from src.research.evaluation_mask import build_mask as _bm
    from src.research.features import SCHEMA as _S

    def digest(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    # Names alone are insufficient: a corrected macro parquet or partition with the
    # same schema must invalidate the cache as well.
    inputs = {
        "seed_m5": digest(SEED_M5),
        "macro": digest(attach_macro_features.__globals__["MACRO_CLEAN"]),
        "partition": digest(PARTITION),
    }
    key = {
        "mask": _bm().sha256,
        "schema": _S.sha256,
        "formula_versions": _S.formula_versions,
        "inputs": inputs,
    }
    if CACHE.is_file() and not rebuild:
        with CACHE.open("rb") as fh:
            blob = pickle.load(fh)
        if blob.get("key") == key:
            if verbose:
                print(f"dataset desde cache ({CACHE.name}): {blob['data'].summary()}")
            return blob["data"]
        if verbose:
            print("cache invalida (cambio la mascara o el esquema): se reconstruye")

    data = build_research_data(verbose=verbose)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("wb") as fh:
        pickle.dump({"key": key, "data": data}, fh)
    return data


# ---------------------------------------------------------------------------
# Formato portable: los specs SIN el objeto del HMM
# ---------------------------------------------------------------------------
PORTABLE = Path(os.environ.get(
    "THESIS_PORTABLE", REPO / "data" / "thesis" / "research_data_portable.pkl"))


def save_portable(data: ResearchData, path: Path = PORTABLE) -> Path:
    """Serializa los specs sin `regime_model`, para que el consumidor no necesite hmmlearn.

    El HMM hace falta para CONSTRUIR los specs (posteriores, spread esperado), no para
    consumirlos: una vez construidos, un `SessionSpec` es numpy puro. Guardar el objeto
    ajustado obligaba a tener `hmmlearn` instalado en cualquier sitio que abriera el
    fichero — y el contenedor de Airflow no lo tiene.

    De lo que se descarta se conserva la FICHA (K, etiquetas, rango de ajuste, BIC), que
    es lo que la tesis tiene que reportar en §8. El modelo ajustado sigue en la cache del
    host para reproducir.
    """
    import pickle

    m = data.regime_model
    meta = {
        "k": m.k, "labels": m.state_labels(), "fit_range": m.fit_range,
        "bic_by_k": {int(k): float(v) for k, v in m.bic_by_k.items()},
        "covariance_type": m.covariance_type, "feature_names": list(m.feature_names),
    }
    blob = {
        "development": data.development, "selection": data.selection,
        "holdout": data.holdout, "scaler_mean": data.scaler_mean,
        "scaler_scale": data.scaler_scale, "dropped": data.dropped,
        "regime_meta": meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(blob, fh)
    return path


def load_portable(path: Path = PORTABLE) -> ResearchData:
    """Lee el formato portable. `regime_model` queda como la ficha, no como el objeto."""
    import pickle

    with path.open("rb") as fh:
        blob = pickle.load(fh)
    return ResearchData(
        development=blob["development"], selection=blob["selection"],
        holdout=blob["holdout"], scaler_mean=blob["scaler_mean"],
        scaler_scale=blob["scaler_scale"], regime_model=blob["regime_meta"],
        dropped=blob["dropped"])
