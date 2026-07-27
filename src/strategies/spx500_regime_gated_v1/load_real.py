"""Real-data loader for the SPX500 engine — official S&P 500 index (Investing).

Contract: CTR-SPX500-REALDATA-001 (v2, 2026-07-27)

**Directiva del operador (2026-07-27): "borrar SPY y usar solo el S&P 500 oficial".**
La fuente es el índice S&P 500 de Investing.com (id 166) — la SSOT canónica que el
PLAN-RENTABILIDAD §1 ya declaraba — ingerido a diario por el DAG
``asset_spx500_pipeline_weekly`` (stage l0_ingest, fail-closed) y persistido en
``seeds/latest/spx500_daily_ohlcv.parquet`` + ``asset_daily_ohlcv`` (1995-01-03 →).
El snapshot SPY (Yahoo, adj_close) queda RETIRADO de este track.

Column contract (SDD-006) — sin cambios para regimes/policies/gates:

    open_to_open_return : realized return open_t → open_{t+1}   (PnL)
    close               : price level                            (signals)
    vix                 : risk state (proxy declarado)
    macro_stress        : exogenous macro driver (proxy declarado)

## Qué es honesto aquí y qué no

**Honesto:** la serie es el índice OFICIAL con 31 años de historia — incluye los osos
2000-02 y 2008 que el snapshot SPY (2020→) no tenía.

**Declarado y NO negociable:** es un **price index SIN dividendos** (plan §1: "no
llamar total-return"). El PnL publicado es price-return; subestima el retorno total de
equity en ~1.8-2 pp/año. La señal (MA200/TSMOM sobre precio) no se ve afectada.

**Proxies (sin cambio, declarados):** `vix` = vol realizada 21d shift(1);
`macro_stress` = z252 del proxy. Las variables reales (VIX/NFCI/HY-OAS, migración 067)
existen en DB pero cablearlas al modelo = variante nueva = +1 trial pre-registrado.

**`available_at` reconstruido** (cierre + 1 día), no vintage del proveedor: el status
máximo alcanzable sigue siendo `research_validated`, nunca `production`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SEED = ROOT / "seeds" / "latest" / "spx500_daily_ohlcv.parquet"

# Realized-vol window used for the VIX proxy. 21 trading days ~ one month, matching VIX's
# 30-calendar-day horizon. Declared here as a prior, not tuned.
VOL_WINDOW = 21
ANNUALIZER = np.sqrt(252.0)


class RealDataUnavailable(RuntimeError):
    """Raised instead of falling back to synthetic data.

    A silent fallback is how a demo scaffold ends up producing numbers someone quotes.
    """


def load_real(*, seed: Path | None = None) -> pd.DataFrame:
    path = seed or SEED
    if not path.is_file():
        raise RealDataUnavailable(
            f"No official S&P 500 seed at {path}. Refusing to fall back to synthetic "
            "data or any other source (operator directive 2026-07-27: official index "
            "only). Run the ingest first: "
            "python scripts/data/ingest_asset_ohlcv.py --asset spx500 --skip-intraday"
        )

    d = pd.read_parquet(path).sort_values("time").reset_index(drop=True)
    for col in ("time", "open", "close"):
        if col not in d.columns:
            raise RealDataUnavailable(f"seed lacks required column `{col}`")
    if len(d) < 2500:
        raise RealDataUnavailable(
            f"seed has only {len(d)} rows — the official series carries 1995->; a "
            "short seed means the ingest lost history (fail-closed, do not publish)"
        )

    out = pd.DataFrame(index=range(len(d)))
    out["timestamp"] = pd.to_datetime(d["time"])
    # available_at reconstruido: cierre + 1 dia (conservador, NO vintage)
    out["available_at"] = out["timestamp"] + pd.Timedelta(days=1)

    # Price level drives the signals. PRICE-RETURN declarado (sin dividendos).
    out["close"] = d["close"].astype(float).to_numpy()

    # PnL is open-to-open: enter at tomorrow's open on today's close signal. Computing it any
    # other way would let a signal act on a price it could not have traded at.
    op = d["open"].astype(float).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        o2o = np.append(op[1:] / op[:-1] - 1.0, np.nan)
    out["open_to_open_return"] = np.nan_to_num(o2o, nan=0.0)

    # PROXY, not VIX. Realized vol of closes, annualized, shifted one day so a
    # bar never sees its own volatility.
    ret = pd.Series(out["close"]).pct_change()
    out["vix"] = (ret.rolling(VOL_WINDOW).std() * ANNUALIZER * 100.0).shift(1).bfill().to_numpy()

    # PROXY, not NFCI/HY-OAS. Z-score of the vol proxy: a crude "is stress elevated" driver.
    v = pd.Series(out["vix"])
    out["macro_stress"] = ((v - v.rolling(252).mean()) / v.rolling(252).std()).shift(1) \
        .fillna(0.0).to_numpy()

    out.attrs["data_class"] = "real_official_index_price_return"
    out.attrs["source"] = "investing.com S&P 500 index (id 166) via asset ingest, seed parquet"
    out.attrs["price_convention"] = "PRICE-RETURN (sin dividendos; plan SPX §1)"
    out.attrs["proxies"] = {
        "vix": f"realized vol {VOL_WINDOW}d annualized (VIX real en DB, no cableado: seria +1 trial)",
        "macro_stress": "252d z-score of the vol proxy (NFCI/HY-OAS reales en DB, idem)",
    }
    out.attrs["point_in_time"] = False
    out.attrs["pit_note"] = (
        "available_at reconstruido (cierre + 1d), no vintage del proveedor; "
        "max status: research_validated"
    )
    return out


def load(*, real: bool = True) -> pd.DataFrame:
    """Entry point for the pipeline. `real=False` must be an explicit, visible choice."""
    if real:
        return load_real()
    from src.strategies.spx500_regime_gated_v1 import datagen
    df = datagen.generate()
    df.attrs["data_class"] = "SYNTHETIC"
    return df
