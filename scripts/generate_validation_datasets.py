"""
Generate 3 validation CSVs for macro_indicators_daily.
Row 1: friendly column name
Row 2: source + reference link
Row 3+: actual data values

Output:
  outputs/macro_datasets/VALIDACION_DIARIO.csv      (17 daily cols, ALL trading days)
  outputs/macro_datasets/VALIDACION_MENSUAL.csv      (17 monthly cols, one row per month)
  outputs/macro_datasets/VALIDACION_TRIMESTRAL.csv   (4 quarterly/annual cols, one row per quarter)
"""

import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv(
    Path(__file__).parent.parent / "data/pipeline/05_resampling/output/MACRO_DAILY_CONSOLIDATED.csv"
)
df["fecha"] = pd.to_datetime(df["fecha"])

OUT = Path(__file__).parent.parent / "outputs" / "macro_datasets"
OUT.mkdir(parents=True, exist_ok=True)

# =========================================================================
# Column classification + source metadata
# =========================================================================
DAILY_COLS = {
    "FXRT_INDEX_DXY_USA_D_DXY":         ("dxy",    "investing | https://www.investing.com/indices/usdollar-historical-data | fallback: FRED DTWEXBGS"),
    "VOLT_VIX_USA_D_VIX":               ("vix",    "investing | https://www.investing.com/indices/volatility-s-p-500-historical-data | fallback: FRED VIXCLS"),
    "FINC_BOND_YIELD10Y_USA_D_UST10Y":  ("ust10y", "investing | https://www.investing.com/rates-bonds/u.s.-10-year-bond-yield-historical-data | fallback: FRED DGS10"),
    "FINC_BOND_YIELD2Y_USA_D_DGS2":     ("ust2y",  "investing | https://www.investing.com/rates-bonds/u.s.-2-year-bond-yield-historical-data | fallback: FRED DGS2"),
    "FXRT_SPOT_USDMXN_MEX_D_USDMXN":   ("usdmxn", "investing | https://es.investing.com/currencies/usd-mxn-historical-data | fallback: TwelveData USD/MXN"),
    "FXRT_SPOT_USDCLP_CHL_D_USDCLP":    ("usdclp", "investing | https://es.investing.com/currencies/usd-clp-historical-data | fallback: TwelveData USD/CLP"),
    "COMM_OIL_WTI_GLB_D_WTI":           ("wti",    "investing | https://www.investing.com/commodities/crude-oil-historical-data"),
    "COMM_OIL_BRENT_GLB_D_BRENT":       ("brent",  "investing | https://www.investing.com/commodities/brent-oil-historical-data"),
    "COMM_METAL_GOLD_GLB_D_GOLD":       ("gold",   "investing | https://www.investing.com/commodities/gold-historical-data"),
    "COMM_AGRI_COFFEE_GLB_D_COFFEE":    ("coffee", "investing | https://www.investing.com/commodities/us-coffee-c-historical-data"),
    "CRSK_SPREAD_EMBI_COL_D_EMBI":      ("embi",   "bcrp | https://estadisticas.bcrp.gob.pe/estadisticas/series/diarias/resultados/PD04715XD/html/"),
    "FINC_BOND_YIELD10Y_COL_D_COL10Y":  ("col10y", "investing | https://www.investing.com/rates-bonds/colombia-10-year-bond-yield-historical-data"),
    "FINC_BOND_YIELD5Y_COL_D_COL5Y":    ("col5y",  "investing | https://www.investing.com/rates-bonds/colombia-5-year-bond-yield-historical-data"),
    "FINC_RATE_IBR_OVERNIGHT_COL_D_IBR": ("ibr",    "banrep | https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/241"),
    "POLR_POLICY_RATE_COL_D_TPM":       ("tpm_d",  "banrep | https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/59"),
    "POLR_PRIME_RATE_USA_D_PRIME":      ("prime",  "fred | DPRIME | https://fred.stlouisfed.org/series/DPRIME"),
    "EQTY_INDEX_COLCAP_COL_D_COLCAP":   ("colcap", "investing | https://www.investing.com/indices/colcap-historical-data"),
}

MONTHLY_COLS = {
    "POLR_FED_FUNDS_USA_M_FEDFUNDS":             ("fedfunds", "fred | FEDFUNDS | https://fred.stlouisfed.org/series/FEDFUNDS"),
    "POLR_POLICY_RATE_COL_M_TPM":                ("tpm_m",    "banrep (legacy CSV) | serie 59 mensual"),
    "INFL_CPI_ALL_USA_M_CPIAUCSL":               ("cpi_usa",  "fred | CPIAUCSL | https://fred.stlouisfed.org/series/CPIAUCSL"),
    "INFL_CPI_CORE_USA_M_CPILFESL":              ("core_cpi", "fred | CPILFESL | https://fred.stlouisfed.org/series/CPILFESL"),
    "INFL_CPI_TOTAL_COL_M_IPCCOL":               ("cpi_col",  "banrep | https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/100002"),
    "INFL_PCE_USA_M_PCEPI":                      ("pce",      "fred | PCEPI | https://fred.stlouisfed.org/series/PCEPI"),
    "LABR_UNEMPLOYMENT_USA_M_UNRATE":            ("unemp",    "fred | UNRATE | https://fred.stlouisfed.org/series/UNRATE"),
    "PROD_INDUSTRIAL_USA_M_INDPRO":              ("indpro",   "fred | INDPRO | https://fred.stlouisfed.org/series/INDPRO"),
    "MNYS_M2_SUPPLY_USA_M_M2SL":                 ("m2",       "fred | M2SL | https://fred.stlouisfed.org/series/M2SL"),
    "SENT_CONSUMER_USA_M_UMCSENT":               ("umcsent",  "fred | UMCSENT | https://fred.stlouisfed.org/series/UMCSENT"),
    "FXRT_REER_BILATERAL_COL_M_ITCR":            ("itcr",     "banrep | https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/4170"),
    "FTRD_TERMS_TRADE_COL_M_TOT":                ("tot",      "banrep | https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/4180"),
    "RSBP_RESERVES_INTERNATIONAL_COL_M_RESINT":  ("reserves", "banrep | https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/15051"),
    "FTRD_EXPORTS_TOTAL_COL_M_EXPUSD":           ("exports",  "dane | scraper balanza comercial"),
    "FTRD_IMPORTS_TOTAL_COL_M_IMPUSD":           ("imports",  "dane | scraper balanza comercial"),
    "CRSK_SENTIMENT_CCI_COL_M_CCI":              ("cci",      "fedesarrollo | encuesta consumidor"),
    "CRSK_SENTIMENT_ICI_COL_M_ICI":              ("ici",      "fedesarrollo | encuesta industrial"),
}

QUARTERLY_COLS = {
    "GDPP_REAL_GDP_USA_Q_GDP_Q":              ("gdp_usa", "fred | GDP | https://fred.stlouisfed.org/series/GDP"),
    "RSBP_CURRENT_ACCOUNT_COL_Q_CACCT_Q":     ("cacct",   "banrep_bop | catalogo cuenta corriente trimestral"),
    "RSBP_FDI_INFLOW_COL_Q_FDIIN_Q":          ("fdi_in",  "GHOST COLUMN - NO existe en DB real, solo legacy CSV"),
    "RSBP_FDI_OUTFLOW_COL_Q_FDIOUT_Q":        ("fdi_out", "GHOST - DB real: rsbp_fdi_outflow_col_a_idce (anual)"),
}


# =========================================================================
# 1. DAILY: ALL trading days (full history)
# =========================================================================
def build_daily():
    available = {k: v for k, v in DAILY_COLS.items() if k in df.columns}
    df_f = df.sort_values("fecha")

    friendly_names = ["fecha"] + [v[0] for v in available.values()]
    source_row = ["FUENTE ->"] + [v[1] for v in available.values()]

    data_rows = []
    for _, row in df_f.iterrows():
        vals = [row["fecha"].strftime("%Y-%m-%d")]
        for col_db in available:
            val = row[col_db]
            vals.append("" if pd.isna(val) else val)
        data_rows.append(vals)

    out_df = pd.DataFrame([source_row] + data_rows, columns=friendly_names)
    path = OUT / "VALIDACION_DIARIO.csv"
    out_df.to_csv(path, index=False)
    print(f"VALIDACION_DIARIO.csv: {len(data_rows)} rows, {len(available)} cols -> {path}")


# =========================================================================
# 2. MONTHLY: one row per month (last non-null value)
# =========================================================================
def build_monthly():
    available = {k: v for k, v in MONTHLY_COLS.items() if k in df.columns}
    df2 = df.copy()
    df2["ym"] = df2["fecha"].dt.to_period("M")

    months = sorted(df2["ym"].unique())
    friendly_names = ["fecha"] + [v[0] for v in available.values()]
    source_row = ["FUENTE ->"] + [v[1] for v in available.values()]

    data_rows = []
    for ym in months:
        month_data = df2[df2["ym"] == ym]
        vals = [str(ym)]
        for col_db in available:
            col_data = month_data[col_db].dropna()
            vals.append(col_data.iloc[-1] if len(col_data) > 0 else "")
        data_rows.append(vals)

    out_df = pd.DataFrame([source_row] + data_rows, columns=friendly_names)
    path = OUT / "VALIDACION_MENSUAL.csv"
    out_df.to_csv(path, index=False)
    print(f"VALIDACION_MENSUAL.csv: {len(data_rows)} months, {len(available)} cols -> {path}")


# =========================================================================
# 3. QUARTERLY: one row per quarter
# =========================================================================
def build_quarterly():
    available = {k: v for k, v in QUARTERLY_COLS.items() if k in df.columns}
    df2 = df.copy()
    df2["yq"] = df2["fecha"].dt.to_period("Q")

    quarters = sorted(df2["yq"].unique())
    friendly_names = ["fecha"] + [v[0] for v in available.values()]
    source_row = ["FUENTE ->"] + [v[1] for v in available.values()]

    data_rows = []
    for yq in quarters:
        q_data = df2[df2["yq"] == yq]
        vals = [str(yq)]
        for col_db in available:
            col_data = q_data[col_db].dropna()
            vals.append(col_data.iloc[-1] if len(col_data) > 0 else "")
        data_rows.append(vals)

    out_df = pd.DataFrame([source_row] + data_rows, columns=friendly_names)
    path = OUT / "VALIDACION_TRIMESTRAL.csv"
    out_df.to_csv(path, index=False)
    print(f"VALIDACION_TRIMESTRAL.csv: {len(data_rows)} quarters, {len(available)} cols -> {path}")


# =========================================================================
# Run
# =========================================================================
build_daily()
build_monthly()
build_quarterly()
print("\nDone! Files in:", OUT)
