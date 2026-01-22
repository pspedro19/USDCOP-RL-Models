# Scraping Web de Variables Macroeconomicas

Coleccion completa de scrapers para obtener datos macroeconomicos de multiples fuentes.

## Estructura de Carpetas

```
scraping_web_macroeconomicas/
├── 01_banrep_selenium/          # Scrapers para Banco de la Republica (SUAMECA)
├── 02_apis_externas/            # APIs externas (TwelveData, FRED, datos.gov.co)
├── 03_scrapers_custom/          # Scrapers web para fuentes especificas
├── 04_tests_verificacion/       # Scripts de testing y verificacion
├── 05_dags_airflow/             # DAGs de Airflow para automatizacion
├── 06_fallbacks_manual/         # Scrapers de respaldo con archivos manuales
└── README.md                    # Esta documentacion
```

---

## 01_banrep_selenium/ - Scrapers Banco de la Republica

### scraper_banrep_selenium.py
**Scraper Multi-Fuente Principal** con bypass de Radware Bot Manager

**Fuentes soportadas:**
1. **datos.gov.co** - API publica (TRM)
2. **Investing.com** - Cloudscraper (COLCAP, TES, Gold, Coffee)
3. **SUAMECA BanRep** - Selenium con undetected-chromedriver

**Variables BanRep (SUAMECA):**
| Variable | Serie | Columna DB | Frecuencia | Descripcion |
|----------|-------|------------|------------|-------------|
| IBR | 241 | finc_rate_ibr_overnight_col_d_ibr | Diaria | Indicador Bancario de Referencia |
| TPM | 59 | polr_policy_rate_col_d_tpm | Diaria | Tasa de Politica Monetaria |
| ITCR | 4170 | fxrt_reer_bilateral_col_m_itcr | Mensual | Indice Tasa de Cambio Real |
| TOT | 4180 | ftrd_terms_trade_col_m_tot | Mensual | Terminos de Intercambio |
| RESINT | 15051 | rsbp_reserves_international_col_m_resint | Mensual | Reservas Internacionales |
| IPCCOL | 100002 | infl_cpi_total_col_m_ipccol | Mensual | IPC Colombia |

**Instalacion:**
```bash
pip install cloudscraper beautifulsoup4 pandas psycopg2-binary
pip install undetected-chromedriver selenium seleniumbase  # Para bypass
```

**Uso:**
```bash
python scraper_banrep_selenium.py
INCLUDE_INVESTING=true python scraper_banrep_selenium.py
```

### scraper_ibr_banrep.py
**Scraper especifico para IBR** desde datos.gov.co Socrata API

**Fuente:** https://www.datos.gov.co/resource/b8fs-cx24.json

**Uso:**
```bash
python scraper_ibr_banrep.py
```

---

## 02_apis_externas/ - APIs Externas

### verify_twelvedata_macro.py
**Verificador de disponibilidad TwelveData API**

**Variables verificadas:**
- WTI Crude Oil (CL)
- US Dollar Index (DXY)

**Requisitos:**
```bash
export TWELVEDATA_API_KEY_G1=tu_api_key
```

### upload_macro_manual.py
**Uploader manual desde CSV de Investing.com**

**Uso:**
```bash
python upload_macro_manual.py --file ~/Downloads/WTI_Historical_Data.csv --symbol WTI
python upload_macro_manual.py --file ~/Downloads/DXY_Historical_Data.csv --symbol DXY
```

**Pasos para obtener datos:**
1. WTI: https://www.investing.com/commodities/crude-oil-historical-data
2. DXY: https://www.investing.com/indices/usdollar-historical-data
3. Seleccionar rango de fechas
4. Descargar CSV
5. Ejecutar script

---

## 03_scrapers_custom/ - Scrapers Web Especificos

### scraper_cpi_investing_calendar.py
**CPI MoM USA** desde Economic Calendar de Investing.com

**URL:** https://www.investing.com/economic-calendar/cpi-69

### scraper_dane_balanza.py
**Balanza Comercial Colombia** desde DANE

**Fuente:** https://www.dane.gov.co/files/operaciones/BCOM/

**Datos:**
- Exportaciones mensuales (USD millones)
- Importaciones mensuales (USD millones)

### scraper_embi_bcrp.py
**EMBI Colombia** (Riesgo Pais) desde BCRP Peru

**URL:** https://estadisticas.bcrp.gob.pe/estadisticas/series/diarias/resultados/PD04715XD/html/

### scraper_fedesarrollo.py
**Indices de Confianza** desde Fedesarrollo

**Variables:**
- CCI (Indice de Confianza del Consumidor)
- ICI (Indice de Confianza Industrial/Comercial)

**Uso:**
```python
from scraper_fedesarrollo import obtener_cci, obtener_ici

df_cci = obtener_cci(n=3)  # Ultimos 3 valores
df_ici = obtener_ici(n=3)
```

### scraper_investing.py
**Scraper generico Investing.com** para commodities y forex

**Variables soportadas:**
| Variable | URL |
|----------|-----|
| WTI | /commodities/crude-oil-historical-data |
| BRENT | /commodities/brent-oil-historical-data |
| COAL | /commodities/newcastle-coal-futures-historical-data |
| GOLD | /commodities/gold-historical-data |
| COFFEE | /commodities/us-coffee-c-historical-data |
| USDCLP | /currencies/usd-clp-historical-data |
| USDMXN | /currencies/usd-mxn-historical-data |
| DXY | /indices/usdollar-historical-data |
| COLCAP | /indices/colcap-historical-data |
| UST10Y | /rates-bonds/u.s.-10-year-bond-yield-historical-data |

### scraper_ipc_col_calendar.py
**IPC Colombia** desde Economic Calendar de Investing.com

**URL:** https://www.investing.com/economic-calendar/colombian-cpi-1502

---

## 04_tests_verificacion/ - Tests y Verificacion

### test_atrasadas_scrapers.py
**Test de scrapers para las 7 variables "atrasadas":**
1. Gold - Investing.com / Yahoo (GC=F)
2. Coffee - Investing.com / Yahoo (KC=F)
3. COLCAP - Investing.com
4. TES_10Y - Investing.com
5. TES_5Y - Investing.com
6. IBR - BanRep SUAMECA
7. Prime_Rate - FRED (DPRIME)

### test_banrep_local.py
**Test local de conexion a SUAMECA BanRep**

Prueba 3 metodos:
1. HTTP Request basico
2. Cloudscraper
3. Selenium (undetected-chromedriver)

### test_colombia_features.py
**Test de Colombia Feature Builder V17**

---

## 05_dags_airflow/ - DAGs de Airflow

### l0_macro_unified.py
**DAG Unificado para todas las 37 variables macro**

**Schedule:** 50 12 * * 1-5 (7:50am COT, Mon-Fri)

**Fuentes:**
- FRED API: 12 variables
- TwelveData API: 4 variables
- BanRep SUAMECA: 6 variables
- Investing.com: 7 variables
- Forward-fill para restantes

### l0_ohlcv_realtime.py
**DAG de OHLCV Realtime** desde TwelveData

**Schedule:** */5 13-17 * * 1-5 (cada 5 min, horario de mercado)

---

## 06_fallbacks_manual/ - Fallbacks Manuales

### scraper_cuenta_corriente_manual_fallback.py
**Fallback para Cuenta Corriente** cuando los scrapers automaticos fallan

**Uso:**
1. Ir a: https://suameca.banrep.gov.co/estadisticas-economicas/informacionSerie/414001
2. Click en "Vista tabla"
3. Descargar Excel
4. Guardar como `cuenta_corriente_manual.csv` o `.xlsx`

---

## Resumen de Variables por Fuente

### FRED API (12 variables)
| Serie | Columna | Frecuencia |
|-------|---------|------------|
| DTWEXBGS | fxrt_index_dxy_usa_d_dxy | Diaria |
| VIXCLS | volt_vix_usa_d_vix | Diaria |
| DGS10 | finc_bond_yield10y_usa_d_ust10y | Diaria |
| DGS2 | finc_bond_yield2y_usa_d_dgs2 | Diaria |
| DPRIME | polr_prime_rate_usa_d_prime | Diaria |
| FEDFUNDS | polr_fed_funds_usa_m_fedfunds | Mensual |
| CPIAUCSL | infl_cpi_all_usa_m_cpiaucsl | Mensual |
| CPILFESL | infl_cpi_core_usa_m_cpilfesl | Mensual |
| PCEPI | infl_pce_usa_m_pcepi | Mensual |
| UNRATE | labr_unemployment_usa_m_unrate | Mensual |
| INDPRO | prod_industrial_usa_m_indpro | Mensual |
| M2SL | mnys_m2_supply_usa_m_m2sl | Mensual |

### TwelveData API (4 variables)
| Simbolo | Columna |
|---------|---------|
| USD/MXN | fxrt_spot_usdmxn_mex_d_usdmxn |
| USD/CLP | fxrt_spot_usdclp_chl_d_usdclp |
| CL | comm_oil_wti_glb_d_wti |
| BZ | comm_oil_brent_glb_d_brent |

### Investing.com (7+ variables)
- COLCAP, TES_10Y, TES_5Y, Gold, Coffee, WTI, DXY, etc.

### BanRep SUAMECA (6 variables)
- IBR, TPM, ITCR, TOT, RESINT, IPCCOL

### DANE Colombia
- Exportaciones, Importaciones

### Fedesarrollo
- CCI (Confianza Consumidor)
- ICI (Confianza Industrial)

### BCRP Peru
- EMBI Colombia (Riesgo Pais)

---

## Dependencias

```bash
# Core
pip install requests pandas psycopg2-binary

# Web Scraping
pip install cloudscraper beautifulsoup4

# Selenium (para BanRep SUAMECA)
pip install undetected-chromedriver selenium seleniumbase

# APIs
pip install fredapi yfinance

# PDF Processing
pip install PyPDF2

# MinIO (storage)
pip install minio pyarrow
```

---

## Variables de Entorno Requeridas

```bash
# Base de datos
POSTGRES_HOST=usdcop-postgres-timescale
POSTGRES_PORT=5432
POSTGRES_DB=usdcop_trading
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin123

# APIs
FRED_API_KEY=tu_api_key
TWELVEDATA_API_KEY_1=tu_api_key

# MinIO (opcional)
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
```
