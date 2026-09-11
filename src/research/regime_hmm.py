"""Régimen HMM causal con fit congelado, y el spread esperado que sale de él.

Contract: CTR-RESEARCH-REGIME-001 · Date: 2026-08-24

Implementa §8.1-§8.4 del plan de tesis
(`.claude/specs/planes/06-tesis-rl-llm-hibrido.md`) y su **test 3**.

## Las dos propiedades que lo hacen usable

**1. Posterior FILTRADO, nunca suavizado (§8.2).** `predict_proba(serie_completa)[t]` corre
forward-backward: el posterior en `t` incorpora observaciones POSTERIORES a `t`. Aquí se
evalúa siempre sobre la serie truncada en `t` y se toma el último elemento — que dentro de esa
ventana es el forward, porque no hay futuro que filtrar hacia atrás. El plan lo exige
literalmente: *"recursión forward filtrada; nunca Viterbi sobre la secuencia completa ni
posterior suavizada"*. Es lo que verifica el **test 3**.

**2. Fit CONGELADO.** El modelo se ajusta UNA vez sobre desarrollo y no se re-ajusta. Un
refit rodante —como el de `src/forecasting/regime_features.py::hmm_regime_features`—
**re-etiqueta el pasado** cada vez que reajusta, que es la capa 2 del anti-look-ahead de
`quant-constitution.md` §4. Aquel módulo es correcto para features de producción; no lo es
para un hold-out que se abre una sola vez.

## Por qué el vector de observación es multivariante (§8.1)

Una primera versión de este módulo observaba **solo log-retornos**, y el resultado fue
degenerado: BIC elegía K=2 con un estado "alto" de persistencia 0,23 —un salto puntual, no un
régimen— y solo 2 de 584 sesiones del hold-out superaban 0,5 de posterior en él. Costear con
eso deja el spread pegado al suelo del rango declarado, o sea costos optimistas.

La causa era el input: una serie de retornos **no tiene estructura de volatilidad que
separar**. §8.1 pide features al cierre de `d−1`: volatilidad realizada y log-volatilidad, ATR
normalizado, retorno y su absoluto, rango/ATR, autocorrelación intradía, y retornos as-of de
DXY y Brent.

**Omitido y declarado**: la *dummy de evento macro* de §8.1 necesita un calendario de eventos
que el repo no tiene como SSOT causal. Se deja fuera en vez de fabricarla; su ausencia se
declara aquí y debe repetirse en el capítulo, no descubrirse leyendo el código.

## K se elige una vez, con la regla del plan (§8.2)

BIC sobre `K ∈ {2,3,4,5}` dentro de **desarrollo**, y —textual— *"`K=3` se mantiene salvo que
otro mejore el BIC en más de 10 puntos"*. Esa histéresis está implementada: sin ella, una
diferencia de BIC de 2 puntos cambiaría el régimen del sistema entero. Los `K` no elegidos
quedan como sensibilidad descriptiva.

## Del posterior al spread (§8.4)

    spread_d = Σ_k p_{d,k} · spread_k      con spread = {2, 3, 6} pips por nivel de vol

Continuo a propósito: `argmax` produciría saltos de turnover arbitrarios en la frontera entre
regímenes, y el plan lo reserva *"solo para el desglose descriptivo por régimen en las tablas,
nunca para costear"*.

Consistencia verificable: el round-trip mínimo sin slippage de §9.3 es `2·(spread/2 + 0.5)` =
**3 / 4 / 7 pips** para spread ∈ {2, 3, 6}. Coincide con lo que declara §9.3.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MACRO_CLEAN = REPO / "data" / "pipeline" / "04_cleaning" / "output" / "MACRO_DAILY_CLEAN.parquet"

# Spread por nivel de volatilidad, en pips (§8.4). 1 pip = 1,00 COP.
SPREAD_PIPS_BY_LEVEL: tuple[float, float, float] = (2.0, 3.0, 6.0)
COMMISSION_PIPS_PER_SIDE = 0.5           # §9.3

# §8.2, textual: K ∈ {2,3,4,5}; K=3 se mantiene salvo que otro mejore el BIC en >10 puntos.
K_CANDIDATES: tuple[int, ...] = (2, 3, 4, 5)
K_DEFAULT = 3
BIC_HYSTERESIS = 10.0

N_ITER = 500                             # §8.2
N_INITS = 20                             # §8.2: 20 inicializaciones, gana la de mayor loglik
RANDOM_STATE = 42

FEATURE_NAMES = (
    "rv", "log_rv", "atr_norm", "ret", "abs_ret", "range_over_atr",
    "intraday_autocorr", "dxy_ret", "brent_ret",
)


# ---------------------------------------------------------------------------
# Features (§8.1) — todas al cierre de d-1
# ---------------------------------------------------------------------------

def build_regime_observations(m5: pd.DataFrame, valid_sessions=None) -> pd.DataFrame:
    """Vector de observación diario del HMM, causal por construcción.

    Cada fila `d` resume lo ocurrido HASTA el cierre de `d`; quien decide en `d+1` usa esta
    fila. El desplazamiento a `d+1` lo hace `spread_series`, no esta función, para que el
    vector siga siendo legible como "resumen del día d".
    """
    df = m5.copy()
    t = pd.to_datetime(df["time"])
    if "symbol" in df.columns:
        keep = df["symbol"].astype(str).str.upper().str.replace("/", "", regex=False) == "USDCOP"
        df, t = df[keep], t[keep]
    df = df.assign(_t=t, _d=t.dt.date).sort_values("_t")
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    if valid_sessions is not None:
        df = df[df["_d"].isin(set(valid_sessions))]

    df["_bar_ret"] = np.log(df["close"] / df.groupby("_d")["close"].shift(1))

    g = df.groupby("_d")
    daily = pd.DataFrame({
        "close": g["close"].last(),
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        # Volatilidad REALIZADA: la suma de cuadrados intradía, no una desviación de cierres.
        "rv": np.sqrt(g["_bar_ret"].apply(lambda s: np.nansum(s ** 2))),
        # Autocorrelación intradía de orden 1: mide si el día fue de arrastre o de reversión.
        "intraday_autocorr": g["_bar_ret"].apply(
            lambda s: s.autocorr(lag=1) if s.notna().sum() > 5 else np.nan),
    })
    daily.index = pd.to_datetime(list(daily.index))
    daily = daily.sort_index()

    daily["ret"] = np.log(daily["close"] / daily["close"].shift(1))
    daily["abs_ret"] = daily["ret"].abs()
    daily["log_rv"] = np.log(daily["rv"].replace(0.0, np.nan))

    prev_close = daily["close"].shift(1)
    true_range = pd.concat([
        daily["high"] - daily["low"],
        (daily["high"] - prev_close).abs(),
        (daily["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = true_range.rolling(14, min_periods=5).mean()
    daily["atr_norm"] = atr14 / daily["close"]
    daily["range_over_atr"] = (daily["high"] - daily["low"]) / atr14.replace(0.0, np.nan)

    daily = _attach_macro(daily)
    return daily[list(FEATURE_NAMES)]


def _attach_macro(daily: pd.DataFrame) -> pd.DataFrame:
    """Retornos as-of de DXY y Brent (§8.1).

    `merge_asof(direction='backward')`: para la sesión `d` se toma el ÚLTIMO valor macro
    publicado en o antes de `d`. Es la defensa de la capa 1 de `quant-constitution.md` §4;
    un `reindex().ffill()` ingenuo daría lo mismo hoy pero no declara la intención.
    """
    for name in ("dxy_ret", "brent_ret"):
        daily[name] = 0.0
    if not MACRO_CLEAN.is_file():
        raise FileNotFoundError(f"macro artifact missing: {MACRO_CLEAN}")
    macro = pd.read_parquet(MACRO_CLEAN)
    cols = {"dxy_ret": "FXRT_INDEX_DXY_USA_D_DXY", "brent_ret": "COMM_OIL_BRENT_GLB_D_BRENT"}
    for out, src in cols.items():
        if src not in macro.columns:
            continue
        s = macro[src].dropna().sort_index()
        r = np.log(s / s.shift(1)).dropna()
        left = pd.DataFrame({"d": pd.to_datetime(daily.index)})
        right = pd.DataFrame({"d": pd.to_datetime(r.index), out: r.to_numpy()})
        merged = pd.merge_asof(
            left.sort_values("d"), right.sort_values("d"), on="d",
            direction="backward", allow_exact_matches=False,
        )
        daily[out] = merged[out].fillna(0.0).to_numpy()
    return daily


# ---------------------------------------------------------------------------
# Fit congelado
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrozenRegimeModel:
    model: object
    k: int
    vol_order: tuple[int, ...]
    fit_range: tuple[str, str]
    bic_by_k: dict[int, float]
    feature_names: tuple[str, ...]
    means: np.ndarray = field(default_factory=lambda: np.array([]))
    scales: np.ndarray = field(default_factory=lambda: np.array([]))
    covariance_type: str = "full"

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.means) / self.scales

    def filtered_posterior(self, obs_window: np.ndarray) -> np.ndarray:
        """P(estado_t | datos hasta t), con t = última fila de `obs_window`."""
        X = self._standardize(np.asarray(obs_window, dtype=float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probs = self.model.predict_proba(X)[-1]
        return probs[list(self.vol_order)]

    def state_labels(self) -> list[str]:
        """Etiquetas de §8.3: bajo -> calmo, alto -> shock, medio segun persistencia."""
        if self.k == 2:
            return ["calmo", "shock"]
        persist = np.diag(self.model.transmat_)[list(self.vol_order)]
        labels = ["calmo"] + ["intermedio"] * (self.k - 2) + ["shock"]
        for i in range(1, self.k - 1):
            # "tendencial" solo si su persistencia respalda que es un regimen y no un salto.
            if persist[i] >= 0.8:
                labels[i] = "tendencial"
        return labels


def _fit_one(X: np.ndarray, k: int, cov_type: str):
    from hmmlearn.hmm import GaussianHMM
    best, best_ll = None, -np.inf
    for seed in range(N_INITS):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = GaussianHMM(n_components=k, covariance_type=cov_type, n_iter=N_ITER,
                            random_state=RANDOM_STATE + seed)
            try:
                m.fit(X)
                ll = m.score(X)
            except Exception:  # noqa: BLE001 - una init puede degenerar; se prueba la siguiente
                continue
        if np.isfinite(ll) and ll > best_ll:
            best, best_ll = m, ll
    return best, best_ll


def fit_frozen(dev_obs: pd.DataFrame, k_candidates: tuple[int, ...] = K_CANDIDATES
               ) -> FrozenRegimeModel:
    """Ajusta UNA vez sobre desarrollo, elige K por BIC con la histéresis de §8.2."""
    obs = dev_obs.dropna()
    if len(obs) < 200:
        raise ValueError(f"muy pocas observaciones para el HMM: {len(obs)}")
    raw = obs.to_numpy(dtype=float)
    means, scales = raw.mean(axis=0), raw.std(axis=0)
    scales[scales == 0] = 1.0
    X = (raw - means) / scales
    n, d = X.shape

    cov_type = "full"
    bic_by_k: dict[int, float] = {}
    fitted: dict[int, object] = {}
    for k in k_candidates:
        m, ll = _fit_one(X, k, cov_type)
        if m is None:                       # fallback declarado en §8.2
            cov_type = "diag"
            m, ll = _fit_one(X, k, cov_type)
        if m is None:
            continue
        n_cov = k * d * (d + 1) / 2 if cov_type == "full" else k * d
        n_params = k * (k - 1) + (k - 1) + k * d + n_cov
        bic_by_k[k] = float(-2.0 * ll + n_params * np.log(n))
        fitted[k] = m
    if not fitted:
        raise RuntimeError("ningun K convergio")

    # §8.2: K=3 se mantiene salvo que otro mejore el BIC en MAS de 10 puntos.
    best_k = min(bic_by_k, key=bic_by_k.get)
    if K_DEFAULT in bic_by_k and best_k != K_DEFAULT:
        if bic_by_k[K_DEFAULT] - bic_by_k[best_k] <= BIC_HYSTERESIS:
            best_k = K_DEFAULT

    model = fitted[best_k]
    var = (np.array([np.diag(c).mean() for c in model.covars_])
           if getattr(model, "covariance_type", cov_type) == "full"
           else model.covars_.mean(axis=1))
    return FrozenRegimeModel(
        model=model, k=best_k, vol_order=tuple(int(i) for i in np.argsort(var)),
        fit_range=(str(obs.index[0].date()), str(obs.index[-1].date())),
        bic_by_k=bic_by_k, feature_names=tuple(dev_obs.columns),
        means=means, scales=scales, covariance_type=cov_type,
    )


# ---------------------------------------------------------------------------
# Spread esperado (§8.4)
# ---------------------------------------------------------------------------

def _levels_for_k(k: int) -> np.ndarray:
    lo, mid, hi = SPREAD_PIPS_BY_LEVEL
    if k == 3:
        return np.array([lo, mid, hi], dtype=float)
    return np.interp(np.linspace(0.0, 1.0, k), [0.0, 0.5, 1.0], [lo, mid, hi])


def spread_series(model: FrozenRegimeModel, obs: pd.DataFrame, min_context: int = 60,
                  shift_to_next_session: bool = True) -> pd.DataFrame:
    """`spread_d` y posteriores filtrados, sesión a sesión, sin mirar el futuro.

    Con `shift_to_next_session` (el default y lo que pide §8.1) el posterior calculado con
    datos hasta el cierre de `d−1` es el que rige DURANTE `d`. Sin el desplazamiento, `d`
    usaría su propio cierre para costear sus propias operaciones — look-ahead de un día.
    """
    clean = obs.dropna()
    arr = clean.to_numpy(dtype=float)
    levels = _levels_for_k(model.k)
    rows = []
    for i in range(len(clean)):
        if i + 1 < min_context:
            rows.append({"spread_pips": np.nan})
            continue
        p = model.filtered_posterior(arr[: i + 1])
        rows.append({"spread_pips": float(np.dot(p, levels)),
                     **{f"p_{lab}_{j}": float(p[j])
                        for j, lab in enumerate(model.state_labels())}})
    out = pd.DataFrame(rows, index=clean.index)
    if shift_to_next_session:
        out = out.shift(1)
    out["round_trip_pips_min"] = 2.0 * (out["spread_pips"] / 2.0 + COMMISSION_PIPS_PER_SIDE)
    return out
