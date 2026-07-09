"""H-COP-CORE-01 — carry condicionado por riesgo vs NULL-A (OLA 4 C4, pre-registrado 2026-07-07).

PRIOR EX-ANTE (congelado en _strategy-history.md §5 ANTES de correr):
  short USDCOP la semana t  SII  carry_ok(t-1) AND NOT risk_off(t-1), con la mecánica
  TP/HS v11 CONGELADA (idéntica a NULL-A — la única variable es el gate económico).
    carry_ok  = (IBR − prime_US) > 2.0 pp          [prior; sens. {1.5, 2.5}]
    risk_off  = z252(VIX) > 1.5  OR  z252(ΔEMBI_20d) > 1.5   [prior; sens. {1.0, 2.0}]
  Todo T-1 (asof del viernes/lunes-1). Evaluación: 2025 completo (metodología fija).
  Juez: block-bootstrap ΔCalmar(CORE, NULL-A); las 9 celdas se REPORTAN, no se elige.

Usage: POSTGRES_PASSWORD=... python -m scripts.analysis.cop_core_hypothesis
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.cop_null_suite import (  # noqa: E402
    WEEKS_PER_YEAR, _ann_return_dd_calmar, block_bootstrap_delta_calmar,
    load_data, weekly_short_returns,
)


def load_macro() -> pd.DataFrame:
    import psycopg2

    conn = psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
    )
    try:
        m = pd.read_sql(
            "SELECT fecha, finc_rate_ibr_overnight_col_d_ibr AS ibr, "
            "polr_prime_rate_usa_d_prime AS prime, volt_vix_usa_d_vix AS vix, "
            "crsk_spread_embi_col_d_embi AS embi "
            "FROM macro_indicators_daily ORDER BY fecha", conn)
    finally:
        conn.close()
    m["fecha"] = pd.to_datetime(m["fecha"])
    return m.set_index("fecha").sort_index().ffill()


def build_gates(m: pd.DataFrame) -> pd.DataFrame:
    g = pd.DataFrame(index=m.index)
    g["carry"] = m["ibr"] - m["prime"]
    zwin = 252
    g["z_vix"] = (m["vix"] - m["vix"].rolling(zwin, min_periods=60).mean()) / \
        m["vix"].rolling(zwin, min_periods=60).std()
    d_embi = m["embi"].diff(20)
    g["z_dembi"] = (d_embi - d_embi.rolling(zwin, min_periods=60).mean()) / \
        d_embi.rolling(zwin, min_periods=60).std()
    return g


def main() -> int:
    df, _ = load_data()
    gates = build_gates(load_macro())

    print("=" * 72)
    print("H-COP-CORE-01 — carry condicionado vs NULL-A (mecánica v11 congelada, 2025)")
    print("=" * 72)

    null_a = weekly_short_returns(df, "2025-01-01", "2025-12-31", use_stops=True)
    sa = _ann_return_dd_calmar(null_a.values, WEEKS_PER_YEAR)
    print(f"\nNULL-A (bar): ann={sa['ann_return_pct']}%  MaxDD={sa['max_dd_pct']}%  "
          f"Calmar={sa['calmar']}  Sharpe={sa['sharpe']}  semanas={len(null_a)}")

    prior = (2.0, 1.5)
    cells = [(c, r) for c in (1.5, 2.0, 2.5) for r in (1.0, 1.5, 2.0)]
    print(f"\n{'carry>':>7} {'risk_z':>7} {'sem_on':>7} {'ann%':>8} {'MaxDD%':>8} "
          f"{'Calmar':>8} {'Sharpe':>7}  nota")
    results = {}
    for c_thr, r_thr in cells:
        gated = {}
        for monday, ret in null_a.items():
            asof = monday - pd.Timedelta(days=1)
            row = gates.asof(asof)
            if row is None or pd.isna(row.get("carry")):
                continue
            carry_ok = row["carry"] > c_thr
            risk_off = (row.get("z_vix", 0) or 0) > r_thr or (row.get("z_dembi", 0) or 0) > r_thr
            gated[monday] = ret if (carry_ok and not risk_off) else 0.0
        s = pd.Series(gated)
        st_ = _ann_return_dd_calmar(s.values, WEEKS_PER_YEAR)
        on = int((s != 0).sum())
        tag = "<< PRIOR" if (c_thr, r_thr) == prior else ""
        results[(c_thr, r_thr)] = (s, st_)
        print(f"{c_thr:>7} {r_thr:>7} {on:>7} {st_['ann_return_pct']:>8} "
              f"{st_['max_dd_pct']:>8} {st_['calmar']:>8} {st_['sharpe']:>7}  {tag}")

    s_prior, st_prior = results[prior]
    common = s_prior.index.intersection(null_a.index)
    boot = block_bootstrap_delta_calmar(s_prior[common].values, null_a[common].values)
    print(f"\n[juez] ΔCalmar(CORE prior, NULL-A) = {boot.get('delta_calmar')}  "
          f"IC95 = {boot.get('ci_95')}  -> "
          f"{'RECHAZA H0 (CORE supera)' if (boot.get('ci_95') or [0, 0])[0] > 0 else 'NO RECHAZA H0 — el gate no supera a estar-corto'}")
    print("\n[nota] gate de ejecución de toda la tesis carry sigue siendo H-COP-CARRY-00 "
          "(swap real del broker). +1 trial prior, +8 celdas sensibilidad (todas reportadas).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
