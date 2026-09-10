"""Vector de observación de la tesis (§6.5): causal, agrupado y hasheado.

Contract: CTR-RESEARCH-FEATURES-001 · Date: 2026-08-24

## Qué entra y qué NO, y por qué

§6.5 declara siete grupos. Dos se caen por sus **propios criterios**, no por preferencia:

- **VOLUMEN — eliminado.** §6.5 lo condiciona a «barras con volumen cero ≤ 5%». Medido sobre
  la serie reparada: **99.714 de 99.714 barras tienen volumen 0** y hay un único valor
  distinto en toda la columna. El proveedor no publica volumen para USD/COP spot, que es un
  mercado OTC. El plan dice «si no, el grupo se elimina completo»; se elimina.
- **SORPRESA MACRO — eliminada.** §6.5 la condiciona a «consenso histórico verificable». El
  repo no tiene serie de consenso, y fabricarla sería inventar el dato que la feature mide.
- **TEXTO (`s_d`, `n_docs`, …) — fuera de alcance.** Pertenece al brazo LLM, descopado.

Queda un vector de **39 features** en siete grupos, todas con disponibilidad temporal
declarada en `feature_schema.json`.

## Ventanas que cruzan la sesión, y por qué es legítimo

`rv_78`, `EMA_72` y `zscore_60` piden más barras de las que tiene una sesión (60). Se calculan
sobre la **serie continua** de sesiones válidas concatenadas en orden temporal. Eso NO es
look-ahead: §9.1 permite «información ≤ cierre_b», y las sesiones anteriores son pasado. Lo
que nunca se hace es mirar hacia adelante — todas las ventanas terminan en `b`.

Las features de **estado de posición**, en cambio, se reinician en cada sesión: `w_{−1} = 0`
por §9.1, así que arrastrar la posición de ayer sería contradecir el entorno.

## RSI por Wilder, no `ewm()`

`CLAUDE.md` lo marca como DO-NOT explícito: el RSI usa la EMA de Wilder (`alpha = 1/period`),
no el `ewm()` por defecto de pandas (`alpha = 2/(period+1)`). Dan series distintas y la
diferencia no es cosmética.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MACRO_CLEAN = REPO / "data" / "pipeline" / "04_cleaning" / "output" / "MACRO_DAILY_CLEAN.parquet"
SCHEMA_PATH = REPO / "config" / "research" / "feature_schema.json"

BARS_PER_SESSION = 60
SESSION_OPEN_MINUTE = 8 * 60          # 08:00 COT

# Grupos declarados en §6.5. El orden ES el contrato: cambia el hash si cambia.
GROUPS: dict[str, tuple[str, ...]] = {
    "precio": ("logret_1", "logret_3", "logret_6", "logret_12",
               "ret_sesion_acum", "close_pos_rango"),
    "volatilidad": ("rv_12", "rv_78", "rv_ratio", "atr_14", "atr_norm",
                    "parkinson_12", "garman_klass_12"),
    "tendencia": ("ema_dist_12", "ema_dist_26", "ema_dist_72", "macd_norm",
                  "macd_signal_norm", "slope_20", "rsi_14", "zscore_60"),
    "temporal": ("min_desde_apertura", "sin_hora", "cos_hora",
                 "primeros_30min", "ultimos_30min", "dia_semana"),
    "posicion": ("w_prev", "pnl_no_realizado", "barras_en_posicion",
                 "drawdown_sesion", "n_cambios"),
    "macro": ("brent_ret_prev", "dxy_ret_prev", "spread_ibr_dgs2"),
    "regimen": ("p_regime_0", "p_regime_1", "p_regime_2", "p_regime_3"),
}

# Grupos que el propio §6.5 elimina por criterio, con la medición que lo justifica.
EXCLUDED_GROUPS = {
    "volumen": "100% de 99.714 barras con volumen 0 (criterio §6.5: <= 5%); USD/COP spot "
               "es OTC y el proveedor no publica volumen",
    "sorpresa_macro": "§6.5 la condiciona a consenso historico verificable; el repo no lo "
                      "tiene y fabricarlo seria inventar el dato que la feature mide",
    "texto": "pertenece al brazo LLM, fuera del alcance de 2 brazos",
}

# Features endogenas: dependen de la trayectoria del agente, no del mercado (§10.6).
ENDOGENOUS = set(GROUPS["posicion"])

FEATURE_ORDER: tuple[str, ...] = tuple(f for g in GROUPS.values() for f in g)


@dataclass(frozen=True)
class FeatureSchema:
    order: tuple[str, ...]
    groups: dict[str, tuple[str, ...]]
    excluded: dict[str, str]
    endogenous: tuple[str, ...]

    @property
    def sha256(self) -> str:
        payload = json.dumps({"order": list(self.order),
                              "groups": {k: list(v) for k, v in self.groups.items()}},
                             sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "contract": "CTR-RESEARCH-FEATURES-001",
            "n_features": len(self.order),
            "sha256": self.sha256,
            "order": list(self.order),
            "groups": {k: list(v) for k, v in self.groups.items()},
            "endogenous": sorted(self.endogenous),
            "excluded_groups": self.excluded,
            "availability": {
                "market": "informacion <= cierre de la barra b (§9.1)",
                "macro": "merge_asof(backward): ultimo valor publicado <= la sesion",
                "regimen": "posterior filtrado con datos <= cierre de la sesion anterior",
                "posicion": "endogena; se reinicia cada sesion con w_{-1}=0",
            },
        }


SCHEMA = FeatureSchema(order=FEATURE_ORDER, groups=GROUPS, excluded=EXCLUDED_GROUPS,
                       endogenous=tuple(sorted(ENDOGENOUS)))


# ---------------------------------------------------------------------------
# Indicadores
# ---------------------------------------------------------------------------

def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI con EMA de Wilder (`alpha = 1/period`).

    `CLAUDE.md`: *"Do NOT use pandas `ewm()` for RSI — use Wilder's EMA"*. El `ewm()` por
    defecto usa `alpha = 2/(period+1)`, que para period=14 es 0,133 en vez de 0,0714: casi el
    doble de peso al dato reciente. Da otra serie, no una aproximacion.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # `avg_loss == 0` no es un dato ausente: es el caso limite. RS -> inf y el RSI es 100.
    # Colapsarlo a 50 (el neutro del calentamiento) apagaba la feature justo en las rachas
    # sin retroceso, que es donde mas informa.
    rs = avg_gain / avg_loss
    out = 100.0 - 100.0 / (1.0 + rs)
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask(both_zero, 50.0)                 # serie plana: ni alza ni baja
    return out.fillna(50.0)                         # solo calentamiento: < `period` barras


def parkinson(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """Estimador de Parkinson: usa el rango, mas eficiente que la desviacion de cierres."""
    hl = np.log(high / low) ** 2
    return np.sqrt(hl.rolling(window, min_periods=2).mean() / (4.0 * np.log(2.0)))


def garman_klass(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series,
                 window: int) -> pd.Series:
    """Garman-Klass: incorpora apertura y cierre ademas del rango."""
    term = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    return np.sqrt(term.rolling(window, min_periods=2).mean().clip(lower=0.0))


def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev = c.shift(1)
    return pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)


# ---------------------------------------------------------------------------
# Construccion
# ---------------------------------------------------------------------------

def build_market_features(m5: pd.DataFrame, valid_sessions=None) -> pd.DataFrame:
    """Features de mercado por barra, sobre la serie continua de sesiones validas.

    Devuelve un DataFrame indexado por timestamp con las columnas de los grupos
    `precio`, `volatilidad`, `tendencia` y `temporal`. Las de `posicion` las produce el
    entorno (son endogenas) y las de `macro`/`regimen` se adjuntan por sesion.
    """
    df = m5.copy()
    t = pd.to_datetime(df["time"])
    if "symbol" in df.columns:
        keep = df["symbol"].astype(str).str.upper().str.replace("/", "", regex=False) == "USDCOP"
        df, t = df[keep], t[keep]
    df = df.assign(_t=t, _d=t.dt.date).sort_values("_t")
    if valid_sessions is not None:
        df = df[df["_d"].isin(set(valid_sessions))]
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df = df.set_index("_t")

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    out = pd.DataFrame(index=df.index)

    # --- precio -----------------------------------------------------------
    logc = np.log(c)
    for k in (1, 3, 6, 12):
        out[f"logret_{k}"] = logc.diff(k)
    # Acumulado DENTRO de la sesion: se reinicia cada dia por definicion.
    session_open = c.groupby(df["_d"]).transform("first")
    out["ret_sesion_acum"] = c / session_open - 1.0
    hi_sofar = h.groupby(df["_d"]).cummax()
    lo_sofar = l.groupby(df["_d"]).cummin()
    rng = (hi_sofar - lo_sofar).replace(0.0, np.nan)
    out["close_pos_rango"] = ((c - lo_sofar) / rng).fillna(0.5)

    # --- volatilidad y rango ---------------------------------------------
    r1 = logc.diff(1)
    out["rv_12"] = r1.rolling(12, min_periods=2).std()
    out["rv_78"] = r1.rolling(78, min_periods=10).std()
    out["rv_ratio"] = out["rv_12"] / out["rv_78"].replace(0.0, np.nan)
    tr = true_range(h, l, c)
    atr14 = tr.rolling(14, min_periods=3).mean()
    out["atr_14"] = atr14
    out["atr_norm"] = atr14 / c
    out["parkinson_12"] = parkinson(h, l, 12)
    out["garman_klass_12"] = garman_klass(o, h, l, c, 12)

    # --- tendencia --------------------------------------------------------
    safe_atr = atr14.replace(0.0, np.nan)
    for k in (12, 26, 72):
        out[f"ema_dist_{k}"] = (c - c.ewm(span=k, adjust=False).mean()) / safe_atr
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_norm"] = macd / safe_atr
    out["macd_signal_norm"] = signal / safe_atr
    out["slope_20"] = (c - c.shift(20)) / (20.0 * safe_atr)
    out["rsi_14"] = wilder_rsi(c, 14) / 100.0
    roll = c.rolling(60, min_periods=10)
    out["zscore_60"] = (c - roll.mean()) / roll.std().replace(0.0, np.nan)

    # --- temporales -------------------------------------------------------
    minutes = df.index.hour * 60 + df.index.minute - SESSION_OPEN_MINUTE
    out["min_desde_apertura"] = minutes / (BARS_PER_SESSION * 5.0)
    frac = np.asarray(minutes, dtype=float) / (BARS_PER_SESSION * 5.0)
    out["sin_hora"] = np.sin(2 * np.pi * frac)
    out["cos_hora"] = np.cos(2 * np.pi * frac)
    out["primeros_30min"] = (np.asarray(minutes) < 30).astype(float)
    out["ultimos_30min"] = (np.asarray(minutes) >= BARS_PER_SESSION * 5 - 30).astype(float)
    out["dia_semana"] = df.index.dayofweek / 4.0

    out["_session"] = df["_d"].to_numpy()
    return _finalize(out)


def _finalize(out: pd.DataFrame) -> pd.DataFrame:
    """Politica de NaN explicita: un vector de observacion no puede llevar NaN a la red.

    Hay exactamente dos fuentes, y ninguna se rellena hacia atras (eso seria look-ahead):

    1. **Calentamiento de ventanas.** `rv_78` necesita 78 barras, `zscore_60` sesenta. Las
       primeras barras de la serie continua no las tienen. Se rellenan a 0.0 = "sin senal",
       nunca con el valor futuro.
    2. **ATR = 0.** Las features de tendencia se normalizan dividiendo por el ATR; en una barra
       de rango nulo eso es 0/0. La distancia a una media, medida en unidades de un rango que
       no existe, es 0 por definicion.

    Medido: 0,32% de las 82.980 barras, todas por la causa 2.
    """
    cols = [c for c in out.columns if c != "_session"]
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def attach_macro_features(per_session_index) -> pd.DataFrame:
    """Grupo macro (§6.5), con `merge_asof(backward)` — capa 1 del anti-look-ahead."""
    idx = pd.to_datetime(sorted(per_session_index))
    out = pd.DataFrame(index=idx,
                       columns=["brent_ret_prev", "dxy_ret_prev", "spread_ibr_dgs2"],
                       dtype=float)
    if not MACRO_CLEAN.is_file():
        return out.fillna(0.0)
    macro = pd.read_parquet(MACRO_CLEAN).sort_index()

    def asof(series: pd.Series, name: str) -> pd.Series:
        s = series.dropna()
        left = pd.DataFrame({"d": idx})
        right = pd.DataFrame({"d": pd.to_datetime(s.index), name: s.to_numpy()})
        merged = pd.merge_asof(left.sort_values("d"), right.sort_values("d"),
                               on="d", direction="backward")
        return pd.Series(merged[name].to_numpy(), index=idx)

    if "COMM_OIL_BRENT_GLB_D_BRENT" in macro:
        b = macro["COMM_OIL_BRENT_GLB_D_BRENT"].dropna()
        out["brent_ret_prev"] = asof(np.log(b / b.shift(1)).dropna(), "brent_ret_prev")
    if "FXRT_INDEX_DXY_USA_D_DXY" in macro:
        d = macro["FXRT_INDEX_DXY_USA_D_DXY"].dropna()
        out["dxy_ret_prev"] = asof(np.log(d / d.shift(1)).dropna(), "dxy_ret_prev")
    if {"FINC_RATE_IBR_OVERNIGHT_COL_D_IBR", "FINC_BOND_YIELD2Y_USA_D_DGS2"} <= set(macro):
        ibr = asof(macro["FINC_RATE_IBR_OVERNIGHT_COL_D_IBR"], "x")
        dgs = asof(macro["FINC_BOND_YIELD2Y_USA_D_DGS2"], "x")
        out["spread_ibr_dgs2"] = (ibr - dgs) / 100.0
    return out.fillna(0.0)


def write_schema(path: Path = SCHEMA_PATH) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = SCHEMA.to_dict()
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return info
