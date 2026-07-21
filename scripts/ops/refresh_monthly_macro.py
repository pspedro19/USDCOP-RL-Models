#!/usr/bin/env python3
"""Refresh macro_indicators_monthly end-to-end from ALL live sources (Fase 1, plan 2026-07-22).

Root causes this closes (verified live before writing this):
  - The DAG-side extractors delegated to scraper modules under data/pipeline/02_scrapers/
    (gitignored path) that were DELETED in d47253e — restored from d47253e^ and force-added.
  - Colombian monthly series dark since Oct-2025 because nobody could run those scrapers.
  - infl_cpi_core_usa_m_cpilfesl / fxrt_reer_bilateral_usa_col_m_itcr_usa 100% NULL:
    sources work (FRED CPILFESL, SUAMECA REST serie 219) — they were simply never pulled.

Sources per series (SSOT config/macro_variables_ssot.yaml extraction blocks):
  FRED       : fedfunds, cpiaucsl, cpilfesl, pcepi, unrate, indpro, m2sl, umcsent
  SUAMECA REST: itcr (234), itcr_usa (219)
  DANE excel : expusd, impusd
  Fedesarrollo PDF: cci, ici (+ infexp/EOF if the scraper exposes it)
  SUAMECA Selenium (best-effort): ipccol (100002), tot (4180), resint (15051)

Rows are normalized to month-start and upserted with COALESCE-merge (a refresh never
blanks a series). publication_date stays NULL (unknown-honest; the wide view's
conservative bound governs model joins). After the DB upsert, MACRO_MONTHLY_CLEAN.parquet
is regenerated FROM the DB so parquet consumers and the DB never diverge again.

Run: POSTGRES_HOST=localhost POSTGRES_PASSWORD=... FRED_API_KEY=... \
     python scripts/ops/refresh_monthly_macro.py [--skip-selenium]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "airflow" / "dags"))

START = datetime(2015, 1, 1)
END = datetime.now()

FRED_SERIES = {
    "polr_fed_funds_usa_m_fedfunds": "FEDFUNDS",
    "infl_cpi_all_usa_m_cpiaucsl": "CPIAUCSL",
    "infl_cpi_core_usa_m_cpilfesl": "CPILFESL",
    "infl_pce_usa_m_pcepi": "PCEPI",
    "labr_unemployment_usa_m_unrate": "UNRATE",
    "prod_industrial_usa_m_indpro": "INDPRO",
    "mnys_m2_supply_usa_m_m2sl": "M2SL",
    "sent_consumer_usa_m_umcsent": "UMCSENT",
}
SUAMECA_REST = {
    "fxrt_reer_bilateral_col_m_itcr": 234,
    "fxrt_reer_bilateral_usa_col_m_itcr_usa": 219,
}
SUAMECA_SELENIUM = {
    "infl_cpi_total_col_m_ipccol": (100002, "https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/100002/ipc"),
    "ftrd_terms_trade_col_m_tot": (4180, "https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/4180/indice_terminos_intercambio_bienes"),
    "rsbp_reserves_international_col_m_resint": (15051, "https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/15051/reservas_internacionales"),
}
DANE = {"ftrd_exports_total_col_m_expusd": "obtener_exportaciones",
        "ftrd_imports_total_col_m_impusd": "obtener_importaciones"}
FEDES = {"crsk_sentiment_cci_col_m_cci": "obtener_cci",
         "crsk_sentiment_ici_col_m_ici": "obtener_ici",
         "infl_exp_eof_col_m_infexp": "obtener_eof_inflacion"}


def collect(skip_selenium: bool) -> dict[str, pd.Series]:
    """Returns {column_name: Series indexed by month-start} for every source that answered."""
    out: dict[str, pd.Series] = {}

    def keep(name: str, df: pd.DataFrame) -> None:
        s = df.set_index("fecha")[name].dropna()
        s.index = pd.to_datetime(s.index).to_period("M").to_timestamp()
        out[name] = s.groupby(level=0).last()
        print(f"  OK {name}: {len(s)} obs, ultimo={s.index.max().date()} -> {s.iloc[-1]}")

    from extractors.fred_extractor import FredExtractor
    fred = FredExtractor({"api_key": os.environ.get("FRED_API_KEY"),
                          "variables": [{"name": n, "series_id": sid}
                                        for n, sid in FRED_SERIES.items()]})
    for name in FRED_SERIES:
        r = fred.extract(name, START, END)
        keep(name, r.data) if r.success else print(f"  FALLO {name}: {r.error}")

    from extractors.suameca_extractor import SuamecaExtractor
    sua = SuamecaExtractor({"variables": [
        {"name": n, "serie_id": sid, "method": "rest_api"} for n, sid in SUAMECA_REST.items()]})
    for name in SUAMECA_REST:
        r = sua.extract(name, START, END)
        keep(name, r.data) if r.success else print(f"  FALLO {name}: {r.error}")

    from extractors.dane_extractor import DaneExtractor
    dane = DaneExtractor({"variables": [{"name": n, "function": f} for n, f in DANE.items()]})
    for name in DANE:
        r = dane.extract(name, START, END)
        keep(name, r.data) if r.success else print(f"  FALLO {name}: {r.error}")

    from extractors.fedesarrollo_extractor import FedesarrolloExtractor
    fed = FedesarrolloExtractor({"variables": [{"name": n, "function": f}
                                               for n, f in FEDES.items()]})
    for name in FEDES:
        try:
            r = fed.extract(name, START, END)
            keep(name, r.data) if r.success else print(f"  FALLO {name}: {r.error}")
        except Exception as e:  # noqa: BLE001
            print(f"  FALLO {name}: {str(e)[:120]}")

    if not skip_selenium:
        sel = SuamecaExtractor({"variables": [
            {"name": n, "serie_id": sid, "method": "selenium", "url": url}
            for n, (sid, url) in SUAMECA_SELENIUM.items()]})
        for name in SUAMECA_SELENIUM:
            try:
                r = sel.extract(name, START, END)
                keep(name, r.data) if r.success else print(f"  FALLO {name}: {r.error}")
            except Exception as e:  # noqa: BLE001
                print(f"  FALLO {name} (selenium): {str(e)[:120]}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-selenium", action="store_true")
    args = ap.parse_args()

    print("== extrayendo ==", flush=True)
    series = collect(args.skip_selenium)
    if not series:
        print("nada extraido")
        return 1

    import psycopg2
    conn = psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", ""))
    cur = conn.cursor()

    print("== upsert a macro_indicators_monthly ==", flush=True)
    months = sorted({m for s in series.values() for m in s.index})
    n_writes = 0
    for m in months:
        cols, vals = [], []
        for name, s in series.items():
            if m in s.index and pd.notna(s.loc[m]):
                cols.append(name)
                vals.append(float(s.loc[m]))
        if not cols:
            continue
        updates = ", ".join(f"{c}=COALESCE(EXCLUDED.{c}, macro_indicators_monthly.{c})"
                            for c in cols)
        cur.execute(
            f"INSERT INTO macro_indicators_monthly (fecha, {', '.join(cols)}, is_complete) "
            f"VALUES (%s, {', '.join(['%s'] * len(vals))}, TRUE) "
            f"ON CONFLICT (fecha) DO UPDATE SET {updates}, updated_at=now()",
            [m.date(), *vals])
        n_writes += 1
    conn.commit()
    print(f"  {n_writes} meses upsert, {len(series)} series")

    # Regenerate the CLEAN parquet FROM the DB (single source going forward)
    df = pd.read_sql("SELECT * FROM macro_indicators_monthly ORDER BY fecha", conn)
    keep_cols = [c for c in df.columns
                 if c not in ("created_at", "updated_at", "is_complete", "ffill_count",
                              "source_date", "publication_date")]
    clean = df[keep_cols].set_index("fecha")
    clean.columns = [c.upper() for c in clean.columns]
    out = REPO / "data/pipeline/04_cleaning/output/MACRO_MONTHLY_CLEAN.parquet"
    clean.to_parquet(out)
    print(f"  regenerado {out.name}: {clean.shape}")

    cur.execute("""SELECT count(*), min(fecha), max(fecha),
                   count(infl_cpi_total_col_m_ipccol), count(crsk_sentiment_cci_col_m_cci),
                   count(infl_cpi_core_usa_m_cpilfesl), count(fxrt_reer_bilateral_usa_col_m_itcr_usa)
                   FROM macro_indicators_monthly""")
    print("  estado final:", cur.fetchone())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
