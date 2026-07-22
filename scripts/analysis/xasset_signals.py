"""E1 — Libro cross-asset: señales trend/value en escala comparable (INFRAESTRUCTURA, 0 trials).

Contract: CTR-QUANT-CONSTITUTION-001 · Skill: .claude/skills/xasset-alpha-engine

**0 trials — ningún OOS abierto; cualquier estudio futuro requiere pre-registro en
HYPOTHESIS-REGISTRY + aprobación del operador.**

Este script construye SEÑALES y un reporte de disponibilidad — NO evalúa performance,
NO calcula Sharpe/retornos de estrategia, NO toca ningún dato posterior a 2024-12-31.
La línea roja se aplica en DOS capas: (1) el SQL corta en <= 2024-12-31; (2) un assert
aborta si cualquier fecha cargada excede el cutoff. 2025 = backtest sellado, 2026 =
forward: ninguno se abre aquí, por eso este trabajo cuenta como 0 trials.

Universo (asset_daily_ohlcv, verificado 2026-07-21):
  USD/COP 1989→ · USD/MXN 1990→ · USD/BRL 1994→ · XAU/USD 1979→ · BTC/USDT 2017→ ·
  SPX/500 1995→ · SPY 1993→ (SPY se excluye del z-score cross-sectional: mismo
  subyacente que SPX/500; incluir ambos duplicaría el peso equity en el ranking).
  NO hay USD/CLP en la tabla (solo existe como columna macro USDCLP en MACRO_DAILY_CLEAN).

Señales (todas causales — shift(1) semanal antes del z-score cross-sectional):
  TREND  TSMOM votos 3/6/12 meses (13/26/52 semanas W-FRI), media de signos en [-1,1],
         luego z-score cross-sectional por fecha (Moskowitz-Ooi-Pedersen 2012).
  VALUE  Reversión a media 3-5 años: -z del log-precio vs su propia ventana de 260
         semanas (min 156 = 3 años), luego z-score cross-sectional
         (Asness-Moskowitz-Pedersen 2013, definición "commodity-style" — la única
         aplicable sin cash flows ni PPP ingestado).
  CARRY  PLACEHOLDER: columna NULL. Requiere datos no ingestados (ver CARRY_MISSING
         por asset abajo y en el JSON). Para BTC el funding de Binance YA está en DB
         (crypto_derivatives_daily.funding_rate, 2506 días desde 2019-09-10) pero su
         uso está gateado por el plan pre-registrado (memoria
         btc-binance-derivatives-plan + .claude/specs/assets/btcusdt/design/
         HYPOTHESIS-REGISTRY.md, S4/H-POS-01): funding→z_funding solo tras
         pre-registro DSR/OOS — no se construye aquí.

Escala comparable = z-score cross-sectional por señal y por fecha (media 0, sigma 1
sobre el universo XS de 6 activos). Alineación en semanas W-FRI para evitar el
problema de calendarios mixtos (BTC 24/7 vs FX 24/5 vs equity horario bolsa): con
alineación diaria solo ~68.5% de fechas son comunes y el ffill sesga la vol (skill
xasset-alpha-engine, sección "Build a cross-asset book").

Salidas (data/pipeline/xasset_signals/ — añadido a .gitignore: intermedio regenerable,
política de tracking 2026-07-09):
  xasset_signals_weekly.parquet   panel largo [date, symbol, close_wfri, trend_raw,
                                  trend_z, value_raw, value_z, carry_z(NULL)]
  availability_report.json        años por símbolo, NaN%, carry faltante por asset,
                                  atestación del cutoff (JSON sin Infinity/NaN vía
                                  safe_json_dump)

Uso (repo root, DB arriba, .env cargado):
  set -a; . ./.env; set +a
  python -m scripts.analysis.xasset_signals
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.contracts.strategy_schema import safe_json_dump  # noqa: E402

# ---------------------------------------------------------------------------
# Diseño congelado ANTES de correr (ex-ante, sin grid): cambiar cualquiera de
# estas constantes = otra versión del archivo, no un "ajuste".
# ---------------------------------------------------------------------------
DESIGN_CUTOFF = pd.Timestamp("2024-12-31")          # línea roja absoluta
TREND_LOOKBACKS_W = (13, 26, 52)                    # 3/6/12 meses en semanas
VALUE_WINDOW_W = 260                                # 5 años
VALUE_MIN_W = 156                                   # mínimo 3 años
MIN_XS_ASSETS = 3                                   # mínimo de activos para z-score XS

UNIVERSE = ["USD/COP", "USD/MXN", "USD/BRL", "XAU/USD", "BTC/USDT", "SPX/500", "SPY"]
# SPY = mismo subyacente que SPX/500; se computan sus señales pero NO entra al
# ranking cross-sectional (duplicaría el peso equity). Declarado ex-ante.
XS_EXCLUDED = {"SPY": "redundante: mismo subyacente que SPX/500"}

# Qué falta para CARRY, por asset (la columna queda NULL hasta que se ingiera y
# se pre-registre el estudio):
CARRY_MISSING = {
    "USD/COP": ("Falta la pata corta USD diaria (Fed funds/SOFR) y los forward points "
                "COP (NDF). La pata larga SI existe: IBR overnight diario "
                "(MACRO_DAILY_CLEAN::FINC_RATE_IBR_OVERNIGHT_COL_D_IBR) y TPM mensual; "
                "UST2Y/prime no son la pata de financiacion."),
    "USD/MXN": ("Faltan TIIE/Cetes 28d (o fondeo Banxico) y/o forward points MXN. "
                "Nada de esto esta ingestado en macro_indicators_daily."),
    "USD/BRL": ("Faltan Selic/DI y/o puntos NDF BRL. Por controles de capital el carry "
                "forward-implied (NDF) es el honesto (skill xasset-alpha-engine, "
                "references/em-fx.md); no ingestado."),
    "XAU/USD": ("Falta la curva de futuros GC (COMEX) para roll yield, o lease rates. "
                "Solo hay spot en DB."),
    "BTC/USDT": ("Funding perp Binance YA en DB (crypto_derivatives_daily.funding_rate, "
                 "2506 dias desde 2019-09-10) — NO se usa aqui: gateado por el plan "
                 "pre-registrado (.claude/specs/assets/btcusdt/design/"
                 "HYPOTHESIS-REGISTRY.md S4/H-POS-01 y memoria "
                 "btc-binance-derivatives-plan); requiere pre-registro DSR/OOS antes "
                 "de entrar al libro."),
    "SPX/500": ("Falta dividend yield del indice y financing rate (carry equity = "
                "div yield - financiacion). No ingestado."),
    "SPY": ("Igual que SPX/500 (dividend yield - financing rate); ademas excluido del "
            "XS por redundancia."),
}

OUT_DIR = REPO / "data" / "pipeline" / "xasset_signals"


# ---------------------------------------------------------------------------
# Carga (SQL corta en el cutoff — capa 1 de la línea roja)
# ---------------------------------------------------------------------------

def load_weekly_closes() -> dict[str, pd.Series]:
    """Cierres semanales W-FRI por símbolo, SOLO <= 2024-12-31 (corte en SQL)."""
    import psycopg2
    conn = psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
    )
    out: dict[str, pd.Series] = {}
    try:
        for sym in UNIVERSE:
            d = pd.read_sql(
                "SELECT time, close FROM asset_daily_ohlcv "
                "WHERE symbol=%s AND time <= %s ORDER BY time",
                conn, params=(sym, DESIGN_CUTOFF.strftime("%Y-%m-%d 23:59:59+00")),
            )
            if d.empty:
                continue
            d["time"] = pd.to_datetime(d["time"], utc=True).dt.tz_localize(None)
            s = d.set_index("time")["close"].astype(float)
            s = s[s > 0]
            out[sym] = s.resample("W-FRI").last().dropna()
    finally:
        conn.close()
    if not out:
        raise RuntimeError("asset_daily_ohlcv vacio o DB inaccesible — nada que construir")
    # Capa 2 de la línea roja: nada posterior al cutoff puede estar en memoria.
    for sym, s in out.items():
        # el label W-FRI puede caer hasta 4 días después del último dato real
        assert s.index.max() <= DESIGN_CUTOFF + pd.Timedelta(days=4), (
            f"LINEA ROJA: {sym} tiene datos post-cutoff ({s.index.max()})")
    return out


# ---------------------------------------------------------------------------
# Señales por símbolo (series de tiempo, causales)
# ---------------------------------------------------------------------------

def tsmom_votes(close: pd.Series, lookbacks=TREND_LOOKBACKS_W) -> pd.Series:
    """Media de sign(ret_w) para w en lookbacks — continua en [-1, 1].

    Votos multi-horizonte en vez de un solo lookback: un único horizonte es un
    parámetro libre pidiendo overfit (skill xasset-alpha-engine::blended_tsmom).
    NaN hasta tener el lookback más largo completo — sin relleno parcial.
    """
    votes = [np.sign(close.pct_change(w)) for w in lookbacks]
    v = pd.concat(votes, axis=1)
    out = v.mean(axis=1)
    out[v.isna().any(axis=1)] = np.nan
    return out


def value_reversal(close: pd.Series, window=VALUE_WINDOW_W, min_p=VALUE_MIN_W) -> pd.Series:
    """-z del log-precio vs su propia media rolling 5y (min 3y) — positivo = barato.

    Definición de value estilo commodity (Asness et al. 2013): sin cash flows ni
    PPP en DB, la única definición honesta disponible para FX/oro/BTC/índice por
    igual. El signo negativo hace que barato-vs-su-historia sea señal LARGA.
    """
    lp = np.log(close)
    mu = lp.rolling(window, min_periods=min_p).mean()
    sd = lp.rolling(window, min_periods=min_p).std(ddof=1)
    return -(lp - mu) / sd.replace(0.0, np.nan)


def xs_zscore(wide: pd.DataFrame, min_assets=MIN_XS_ASSETS) -> pd.DataFrame:
    """Z-score cross-sectional por fecha (fila). Filas con < min_assets → NaN.

    Esta es la "escala comparable": media 0 / sigma 1 sobre el universo XS en
    cada fecha, para que +1 en BTC y +1 en COP signifiquen lo mismo en el libro.
    """
    mu = wide.mean(axis=1)
    sd = wide.std(axis=1, ddof=1)
    z = wide.sub(mu, axis=0).div(sd.replace(0.0, np.nan), axis=0)
    z[wide.notna().sum(axis=1) < min_assets] = np.nan
    return z


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build() -> tuple[pd.DataFrame, dict]:
    closes = load_weekly_closes()
    syms = [s for s in UNIVERSE if s in closes]

    close_w = pd.DataFrame({s: closes[s] for s in syms})
    # La última semana parcial (cierres 30-31 dic) llevaría label W-FRI 2025-01-03.
    # Se DESCARTA en vez de re-etiquetarse: el panel termina visiblemente <= cutoff
    # y ninguna fila lleva fecha 2025 aunque su dato subyacente fuera de 2024.
    close_w = close_w[close_w.index <= DESIGN_CUTOFF]

    # señales por símbolo — CAUSALES: shift(1) ANTES del z-score XS, de modo que
    # la fila t solo contiene información conocible al cierre del viernes t-1.
    trend_raw = pd.DataFrame({s: tsmom_votes(close_w[s].dropna()) for s in syms})
    value_raw = pd.DataFrame({s: value_reversal(close_w[s].dropna()) for s in syms})
    trend_raw = trend_raw.reindex(close_w.index).shift(1)
    value_raw = value_raw.reindex(close_w.index).shift(1)

    xs_cols = [s for s in syms if s not in XS_EXCLUDED]
    trend_z = xs_zscore(trend_raw[xs_cols])
    value_z = xs_zscore(value_raw[xs_cols])

    rows = []
    for s in syms:
        df = pd.DataFrame({
            "date": close_w.index,
            "symbol": s,
            "close_wfri": close_w[s].values,
            "trend_raw": trend_raw[s].values,
            "trend_z": trend_z[s].values if s in xs_cols else np.nan,
            "value_raw": value_raw[s].values,
            "value_z": value_z[s].values if s in xs_cols else np.nan,
            # CARRY placeholder — ver CARRY_MISSING en el docstring y el JSON.
            "carry_z": np.nan,
        })
        rows.append(df[df["close_wfri"].notna()])
    panel = pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"])

    # reporte de disponibilidad
    per_symbol = {}
    for s in syms:
        sub = panel[panel["symbol"] == s]
        n = len(sub)
        first, last = sub["date"].min(), sub["date"].max()
        per_symbol[s] = {
            "first_week": str(first.date()), "last_week": str(last.date()),
            "years_of_history": round((last - first).days / 365.25, 1),
            "n_weeks": n,
            "nan_pct": {
                "trend_raw": round(float(sub["trend_raw"].isna().mean()) * 100, 1),
                "trend_z": round(float(sub["trend_z"].isna().mean()) * 100, 1),
                "value_raw": round(float(sub["value_raw"].isna().mean()) * 100, 1),
                "value_z": round(float(sub["value_z"].isna().mean()) * 100, 1),
                "carry_z": 100.0,
            },
            "in_xs_universe": s not in XS_EXCLUDED,
            "xs_exclusion_reason": XS_EXCLUDED.get(s),
            "carry_missing": CARRY_MISSING.get(s, "sin nota"),
        }

    report = {
        "contract": "CTR-QUANT-CONSTITUTION-001",
        "task": "E1 xasset book infrastructure",
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        "trials_opened": 0,
        "attestation": (
            "0 trials — ningun OOS abierto. SQL corta en <= 2024-12-31 y un assert "
            "verifica que nada post-cutoff entro en memoria. NINGUNA metrica de "
            "performance fue computada sobre 2025 ni 2026. Cualquier estudio futuro "
            "requiere pre-registro en HYPOTHESIS-REGISTRY + aprobacion del operador."
        ),
        "design_cutoff": str(DESIGN_CUTOFF.date()),
        "max_week_in_panel": str(panel["date"].max().date()),
        "alignment": "W-FRI weekly (mixed-calendar safe: BTC 24/7 vs FX 24/5 vs equity)",
        "signals": {
            "trend": f"TSMOM votes {TREND_LOOKBACKS_W} weeks (3/6/12m), shift(1), XS z-score",
            "value": f"-z(log price vs rolling {VALUE_WINDOW_W}w mean, min {VALUE_MIN_W}w), "
                     "shift(1), XS z-score",
            "carry": "NULL placeholder — ver carry_missing por simbolo",
        },
        "xs_universe": xs_cols,
        "xs_excluded": XS_EXCLUDED,
        "usdclp_note": "USD/CLP NO esta en asset_daily_ohlcv (solo como columna macro "
                       "FXRT_SPOT_USDCLP_CHL_D_USDCLP en MACRO_DAILY_CLEAN, sin OHLC)",
        "per_symbol": per_symbol,
    }
    return panel, report


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel, report = build()

    # línea roja, capa 3 (cinturón final antes de persistir): ninguna fila con
    # label posterior al cutoff — la semana parcial de fin de año se descartó.
    assert panel["date"].max() <= DESIGN_CUTOFF, "post-cutoff en panel"

    pq = OUT_DIR / "xasset_signals_weekly.parquet"
    panel.to_parquet(pq, index=False)
    js = OUT_DIR / "availability_report.json"
    with open(js, "w", encoding="utf-8") as f:
        safe_json_dump(report, f)

    print(f"[OK] {pq}  rows={len(panel)}  symbols={panel['symbol'].nunique()}")
    print(f"[OK] {js}")
    print(f"[OK] max week = {panel['date'].max().date()}  (cutoff {DESIGN_CUTOFF.date()})")
    for s, meta in report["per_symbol"].items():
        print(f"  {s:9s} {meta['first_week']} -> {meta['last_week']} "
              f"({meta['years_of_history']}y, {meta['n_weeks']}w) "
              f"trend_z NaN {meta['nan_pct']['trend_z']}% / value_z NaN {meta['nan_pct']['value_z']}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
