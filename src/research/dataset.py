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
import json
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
    dropped: dict[str, list]
    macro_scaler_mean: np.ndarray | None = None
    macro_scaler_scale: np.ndarray | None = None

    def block(self, name: str) -> list[SessionSpec]:
        return {
            "development": self.development,
            "selection": self.selection,
            "holdout": self.holdout,
        }[name]

    def summary(self) -> str:
        return (
            f"desarrollo {len(self.development)} · selección {len(self.selection)} · "
            f"hold-out {len(self.holdout)} · descartadas "
            f"{ {k: len(v) for k, v in self.dropped.items()} }"
        )


def load_partition() -> dict:
    path = Path(os.environ.get("THESIS_PARTITION_CONFIG", str(PARTITION)))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def partition_path() -> Path:
    """Return the active partition path, allowing a versioned preregistered run."""
    return Path(os.environ.get("THESIS_PARTITION_CONFIG", str(PARTITION)))


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
    scale[scale == 0] = 1.0  # feature constante: escalar por 0 daria inf
    return mean, scale


def fit_dev_macro_scaler(macro: pd.DataFrame, dev_dates: set) -> tuple[np.ndarray, np.ndarray]:
    """Fit the macro scaler on development dates only.

    Macro levels are part of the policy observation, so leaving them in raw units while
    market features are standardized gives the policy an unintended scale imbalance.
    This function is separate to make the no-look-ahead boundary testable.
    """
    rows = macro.loc[macro.index.isin(pd.to_datetime(list(dev_dates)))][MACRO_FEATURES]
    if rows.empty:
        raise ValueError("el escalador macro no vio ninguna sesion de desarrollo")
    mean = rows.mean().to_numpy(dtype=float)
    scale = rows.std(ddof=0).to_numpy(dtype=float)
    scale[scale == 0] = 1.0
    return mean, scale


def build_research_data(
    m5: pd.DataFrame | None = None, min_context: int = 60, verbose: bool = True
) -> ResearchData:
    """Arma los tres bloques. El escalador y el HMM se ajustan solo sobre desarrollo."""
    part = load_partition()
    mask = build_mask()
    valid = list(mask.valid)
    if m5 is None:
        m5 = pd.read_parquet(SEED_M5)

    train_valid = list(mask.train_valid) or valid
    dev_dates = set(_block_dates(part, "development", train_valid))

    # --- HMM congelado sobre desarrollo (§8.2) ---------------------------
    # `build_regime_observations` indexa por Timestamp y `fit_frozen` cuenta con ello
    # (llama a `.date()` sobre el indice). El resto del carril —mascara, features,
    # particion— usa `date`. Se convierte la SALIDA, no la entrada: mutar `obs_all`
    # rompia `fit_frozen`, y dejarlo sin convertir hacia que ningun filtro de bloque
    # casara y el HMM recibiera 0 filas. Los dos fallos ya ocurrieron.
    obs_all = build_regime_observations(m5, valid_sessions=set(valid))
    dev_obs = obs_all[[t.date() in dev_dates for t in obs_all.index]]
    model = fit_frozen(dev_obs)
    if model.k > N_REGIMES:
        raise ValueError(
            f"HMM k={model.k} exceeds {N_REGIMES} observation slots; "
            "posterior truncation is forbidden. A new preregistration/schema decision is required."
        )
    regimes = spread_series(model, obs_all, min_context=min_context, shift_to_next_session=True)
    regimes.index = [t.date() for t in regimes.index]
    if verbose:
        print(f"HMM: K={model.k} ajustado en {model.fit_range} ({', '.join(model.state_labels())})")

    # --- features de mercado --------------------------------------------
    feats = build_market_features(m5, valid_sessions=set(valid))
    macro = attach_macro_features(valid)

    mean, scale = fit_dev_scaler(feats, dev_dates)
    macro_mean, macro_scale = fit_dev_macro_scaler(macro, dev_dates)

    prob_cols = [c for c in regimes.columns if c.startswith("p_")]
    blocks: dict[str, list[SessionSpec]] = {}
    # Preserve dates, not just counts: a published dataset must explain every
    # session that disappeared before model fitting/evaluation.
    dropped: dict[str, list] = {reason: list(dates) for reason, dates in mask.excluded.items()}
    dropped.setdefault("sin_regimen", [])
    dropped.setdefault("barras_incompletas", [])
    dropped.setdefault("sin_macro", [])

    by_session = {key: group for key, group in feats.groupby("_session")}  # noqa: C416
    for name in ("development", "selection", "holdout"):
        specs: list[SessionSpec] = []
        for d in _block_dates(part, name, valid):
            g = by_session.get(d)
            if g is None or len(g) != BARS_PER_SESSION:
                dropped["barras_incompletas"].append(d)
                continue
            if d not in regimes.index or not np.isfinite(regimes.at[d, "spread_pips"]):
                dropped["sin_regimen"].append(d)
                continue
            macro_row = macro.reindex([pd.Timestamp(d)])[MACRO_FEATURES].iloc[0]
            if not np.isfinite(macro_row.to_numpy(dtype=float)).all():
                dropped["sin_macro"].append(d)
                continue

            X = (g[MARKET_FEATURES].to_numpy(dtype=float) - mean) / scale
            probs = regimes.loc[d, prob_cols].to_numpy(dtype=float)
            if len(probs) < N_REGIMES:  # K < 4: se rellena con ceros
                probs = np.pad(probs, (0, N_REGIMES - len(probs)))
            macro_values = macro.loc[pd.Timestamp(d), MACRO_FEATURES].to_numpy(dtype=float)
            macro_values = (macro_values - macro_mean) / macro_scale
            ctx = np.concatenate([macro_values, probs[:N_REGIMES]])

            specs.append(
                SessionSpec(
                    date=d,
                    close=_closes(m5, d),
                    market=np.clip(X, -CLIP, CLIP).astype(np.float32),
                    context=np.clip(ctx, -CLIP, CLIP).astype(np.float32),
                    spread_pips=float(regimes.at[d, "spread_pips"]),
                )
            )
        blocks[name] = specs
        if verbose:
            print(f"  {name:<12} {len(specs):>4} sesiones")

    if verbose and any(dropped.values()):
        print(f"  descartadas: { {k: len(v) for k, v in dropped.items()} }")

    return ResearchData(
        development=blocks["development"],
        selection=blocks["selection"],
        holdout=blocks["holdout"],
        scaler_mean=mean,
        scaler_scale=scale,
        regime_model=model,
        dropped=dropped,
        macro_scaler_mean=macro_mean,
        macro_scaler_scale=macro_scale,
    )


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
CACHE = Path(os.environ.get("THESIS_CACHE", REPO / "outputs" / "thesis" / "research_data.pkl"))


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
        "partition": digest(partition_path()),
        # Cache identity must include implementation changes that alter the numeric
        # observation, not only the source files' data hashes.  In particular, the v2
        # macro scaler changed contexts while preserving the feature schema names.
        "dataset_code": digest(Path(__file__)),
        "live_spec_code": digest(REPO / "src" / "research" / "live_spec.py"),
    }
    key = {
        "identity_v3": dataset_identity(),
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
PORTABLE = Path(
    os.environ.get("THESIS_PORTABLE", REPO / "data" / "thesis" / "research_data_portable.pkl")
)


def frozen_artifact_paths() -> tuple[Path, Path]:
    """Return the scaler/regime artifacts for the active dataset lane.

    Confirmatory rebuilds must not overwrite the historical v1/v2 artifacts.  The
    lane therefore opts into explicit files through environment variables; the
    defaults preserve the legacy behaviour for existing callers.
    """
    scaler = Path(
        os.environ.get(
            "THESIS_SCALER_CONFIG", REPO / "config" / "research" / "feature_scaler_frozen.json"
        )
    )
    regime = Path(
        os.environ.get(
            "THESIS_REGIME_CONFIG", REPO / "config" / "research" / "regime_hmm_frozen.json"
        )
    )
    # During a first build the lane-specific files do not exist yet.  The builder
    # may use canonical inputs to fit, then materializes identical numeric content
    # into the lane paths before serializing.  Never silently fall back when a
    # configured file exists but is unreadable or malformed.
    canonical_scaler = REPO / "config" / "research" / "feature_scaler_frozen.json"
    canonical_regime = REPO / "config" / "research" / "regime_hmm_frozen.json"
    if not scaler.is_file() and scaler != canonical_scaler:
        scaler = canonical_scaler
    if not regime.is_file() and regime != canonical_regime:
        regime = canonical_regime
    return scaler, regime


def _file_digest(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing dataset identity input: {path}")
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_artifact_key(path: Path) -> str:
    """Return a platform-independent key for a frozen artifact in the identity."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO.resolve()).as_posix()
    except ValueError:
        # External artifacts remain identifiable by an absolute, normalized POSIX path.
        return resolved.as_posix()


def dataset_identity_manifest() -> dict:
    """Versioned input/transform contract; never silently legitimizes older portables.

    Frozen artifacts embed their dataset identity. Hash their content excluding that
    back-reference only, preventing a circular SHA fixed-point requirement. A build
    must export frozen numeric artifacts before capturing its final manifest.
    """
    from src.research.evaluation_mask import build_mask
    from src.research.features import SCHEMA, attach_macro_features

    source_files = {
        name: _file_digest(REPO / name)
        for name in (
            "src/research/features.py",
            "src/research/dataset.py",
            "src/research/regime_hmm.py",
            "src/research/cost_model.py",
            "src/research/evaluation_mask.py",
            "src/research/macro_asof.py",
            "src/research/live_spec.py",
            "src/research/regime_portable.py",
            "src/research/session_env.py",
            "src/research/session_gym.py",
        )
    }
    configuration = {
        name: _file_digest(REPO / name)
        for name in (
            "config/research/macro_availability.yaml",
            "config/trading_calendar.json",
            "config/research/cost_contract.yaml",
            "config/research/source_lineage.yaml",
            "config/research/feature_schema.json",
        )
    }
    frozen_content = {}
    scaler_path, regime_path = frozen_artifact_paths()
    for name, artifact in (
        (_stable_artifact_key(scaler_path), scaler_path),
        (_stable_artifact_key(regime_path), regime_path),
    ):
        content = json.loads(artifact.read_text(encoding="utf-8"))
        content.pop("dataset_identity", None)
        frozen_content[name] = hashlib.sha256(
            json.dumps(content, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
    payload = {
        "identity_version": 4,
        "mask": build_mask().sha256,
        "schema": SCHEMA.sha256,
        "formula_versions": SCHEMA.formula_versions,
        "seed_m5": _file_digest(SEED_M5),
        "macro": _file_digest(attach_macro_features.__globals__["MACRO_CLEAN"]),
        "partition": _file_digest(partition_path()),
        "source_files": source_files,
        "configuration": configuration,
        "frozen_content_excluding_dataset_identity": frozen_content,
    }
    return payload


def dataset_identity() -> str:
    """Current v3 identity; legacy files require explicit historical verification."""
    return hashlib.sha256(
        json.dumps(dataset_identity_manifest(), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def load_historical_portable(path: Path, *, expected_sha256: str) -> ResearchData:
    """Read one trusted archived portable by exact file SHA, without current rebuilding.

    Diagnostic-only: this does not accredit its causal design or current identity.
    Pickle is executable serialization; use only locally trusted project snapshots.
    Hash the same bytes that are deserialized, so a path replacement cannot bypass
    the caller's expected archive hash.
    """
    import pickle

    artifact = Path(path).resolve()
    for part in artifact.parts:
        name = part.lower()
        if (
            name == "secrets"
            or name == ".env"
            or name.startswith(".env.")
            or name.endswith((".pem", ".key"))
            or (name.startswith(("credentials", "service-account")) and name.endswith(".json"))
        ):
            raise ValueError("sensitive path is not a historical research artifact")
    raw = artifact.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("historical portable file hash mismatch")
    blob = pickle.loads(raw)
    return ResearchData(
        development=blob["development"],
        selection=blob["selection"],
        holdout=blob["holdout"],
        scaler_mean=blob["scaler_mean"],
        scaler_scale=blob["scaler_scale"],
        regime_model=blob["regime_meta"],
        dropped=blob["dropped"],
        macro_scaler_mean=blob.get("macro_scaler_mean"),
        macro_scaler_scale=blob.get("macro_scaler_scale"),
    )


def save_portable(
    data: ResearchData,
    path: Path = PORTABLE,
    *,
    scaler_path: Path | None = None,
    regime_path: Path | None = None,
) -> Path:
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
    regime_path = regime_path or (REPO / "config" / "research" / "regime_hmm_frozen.json")
    scaler_path = scaler_path or (REPO / "config" / "research" / "feature_scaler_frozen.json")
    frozen_scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    for field, actual in (
        ("mean", data.scaler_mean),
        ("scale", data.scaler_scale),
        ("macro_mean", data.macro_scaler_mean),
        ("macro_scale", data.macro_scaler_scale),
    ):
        expected = frozen_scaler.get(field)
        if (actual is None) != (expected is None) or (
            actual is not None and not np.array_equal(np.asarray(expected), actual)
        ):
            raise ValueError(
                f"frozen scaler {field} does not match ResearchData; export numeric artifacts first"
            )
    if frozen_scaler.get("features") != MARKET_FEATURES:
        raise ValueError("frozen scaler feature order differs from ResearchData")
    frozen_regime = json.loads(regime_path.read_text(encoding="utf-8"))
    covars = m.model.covars_
    if covars.ndim == 2:
        covars = np.stack([np.diag(row) for row in covars])
    for field, actual in (
        ("startprob", m.model.startprob_),
        ("transmat", m.model.transmat_),
        ("means", m.model.means_),
        ("covars", covars),
        ("std_means", m.means),
        ("std_scales", m.scales),
    ):
        if not np.array_equal(np.asarray(frozen_regime.get(field)), actual):
            raise ValueError(
                f"frozen regime {field} does not match ResearchData; export numeric artifacts first"
            )
    if (
        frozen_regime.get("k") != m.k
        or frozen_regime.get("labels") != list(m.state_labels())
        or frozen_regime.get("vol_order") != list(m.vol_order)
        or frozen_regime.get("feature_names") != list(m.feature_names)
        or frozen_regime.get("fit_range") != list(m.fit_range)
    ):
        raise ValueError("frozen regime metadata differs from ResearchData")
    manifest = dataset_identity_manifest()
    identity = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if (
        frozen_scaler.get("dataset_identity") != identity
        or frozen_regime.get("dataset_identity") != identity
    ):
        raise ValueError(
            "frozen artifact identity is stale; bind identity after exporting both numeric artifacts"
        )
    meta = {
        "k": m.k,
        "labels": m.state_labels(),
        "fit_range": m.fit_range,
        "bic_by_k": {int(k): float(v) for k, v in m.bic_by_k.items()},
        "covariance_type": m.covariance_type,
        "feature_names": list(m.feature_names),
    }
    blob = {
        "identity": identity,
        "identity_manifest": manifest,
        "development": data.development,
        "selection": data.selection,
        "holdout": data.holdout,
        "scaler_mean": data.scaler_mean,
        "scaler_scale": data.scaler_scale,
        "macro_scaler_mean": data.macro_scaler_mean,
        "macro_scaler_scale": data.macro_scaler_scale,
        "dropped": data.dropped,
        "regime_artifact_sha256": _file_digest(regime_path),
        "regime_meta": meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(blob, fh)
    return path


def load_portable(path: Path = PORTABLE, *, allow_stale: bool = False) -> ResearchData:
    """Lee el portable y rechaza artefactos de otra identidad por defecto.

    ``allow_stale`` existe únicamente para auditorías retrospectivas explícitas;
    los carriles de entrenamiento, evaluación y forward deben dejarlo en False.
    """
    import pickle

    with path.open("rb") as fh:
        blob = pickle.load(fh)
    expected = dataset_identity()
    actual = blob.get("identity")
    if actual != expected and not allow_stale:
        raise ValueError(
            "portable dataset identity mismatch; rebuild v2 before training/evaluation "
            f"(expected {expected[:16]}, found {str(actual)[:16]})"
        )
    _scaler_path, regime_path = frozen_artifact_paths()
    current_regime_digest = _file_digest(regime_path)
    if (blob.get("regime_artifact_sha256") != current_regime_digest) and not allow_stale:
        raise ValueError("portable dataset regime artifact mismatch; rebuild v2")
    return ResearchData(
        development=blob["development"],
        selection=blob["selection"],
        holdout=blob["holdout"],
        scaler_mean=blob["scaler_mean"],
        scaler_scale=blob["scaler_scale"],
        regime_model=blob["regime_meta"],
        dropped=blob["dropped"],
        macro_scaler_mean=blob.get("macro_scaler_mean"),
        macro_scaler_scale=blob.get("macro_scaler_scale"),
    )
