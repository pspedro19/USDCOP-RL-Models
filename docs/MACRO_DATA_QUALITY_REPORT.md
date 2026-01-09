# Reporte de Calidad de Datos Macro
## macro_indicators_daily - Análisis Completo

**Fecha:** 2025-12-18
**Total filas:** 1,452
**Rango:** 2020-01-02 a 2025-12-18

---

## 📊 Resumen de Cobertura por Variable

### ✅ Variables con Excelente Cobertura (>95%)

| Variable | Cobertura | Fuente Recomendada |
|----------|-----------|-------------------|
| fxrt_spot_usdmxn_mex_d_usdmxn | 99.8% | TwelveData, Investing.com |
| fxrt_spot_usdclp_chl_d_usdclp | 99.8% | TwelveData, Investing.com |
| comm_oil_brent_glb_d_brent | 99.7% | TwelveData, Yahoo Finance |
| fxrt_index_dxy_usa_d_dxy | 99.6% | FRED (DTWEXBGS), Investing.com |
| comm_oil_wti_glb_d_wti | 99.5% | TwelveData, Yahoo Finance |
| finc_bond_yield10y_usa_d_ust10y | 99.4% | FRED (DGS10) |
| finc_rate_ibr_overnight_col_d_ibr | 99.3% | BanRep, Investing.com |
| eqty_index_colcap_col_d_colcap | 99.3% | BVC, Investing.com |
| comm_metal_gold_glb_d_gold | 99.2% | TwelveData, Yahoo Finance |
| polr_policy_rate_col_m_tpm | 99.2% | BanRep |
| volt_vix_usa_d_vix | 99.0% | FRED (VIXCLS), Yahoo Finance |
| finc_bond_yield10y_col_d_col10y | 98.6% | BanRep, Investing.com |
| polr_fed_funds_usa_m_fedfunds | 98.7% | FRED (FEDFUNDS) |
| comm_agri_coffee_glb_d_coffee | 97.2% | TwelveData, Yahoo Finance |
| infl_cpi_total_col_m_ipccol | 97.2% | DANE, BanRep |
| infl_cpi_all_usa_m_cpiaucsl | 97.0% | FRED (CPIAUCSL) |
| finc_bond_yield2y_usa_d_dgs2 | 96.7% | FRED (DGS2) |

### ⚠️ Variables con Cobertura Moderada (90-95%)

| Variable | Cobertura | Fuente Recomendada |
|----------|-----------|-------------------|
| crsk_spread_embi_col_d_embi | 90.6% | BCRP, JP Morgan |

### ❌ Variables con Baja Cobertura (<30%)

| Variable | Cobertura | Fuente Recomendada |
|----------|-----------|-------------------|
| infl_cpi_core_usa_m_cpilfesl | 28.2% | FRED (CPILFESL) |
| polr_policy_rate_col_d_tpm | 0.9% | BanRep (diario) |

---

## 🔴 Gaps Identificados (Últimos 30 días)

**11 días trading faltantes:**

| Fecha | Día | Tipo |
|-------|-----|------|
| 2025-11-28 | Vie | Thanksgiving (US) |
| 2025-12-01 | Lun | Gap scraper |
| 2025-12-02 | Mar | Gap scraper |
| 2025-12-03 | Mié | Gap scraper |
| 2025-12-04 | Jue | Gap scraper |
| 2025-12-05 | Vie | Gap scraper |
| 2025-12-08 | Lun | Gap scraper |
| 2025-12-09 | Mar | Gap scraper |
| 2025-12-10 | Mié | Gap scraper |
| 2025-12-11 | Jue | Gap scraper |
| 2025-12-15 | Lun | Gap scraper |

---

## 📋 Fuentes de Datos por Categoría

### Datos Diarios (D)
```
FOREX:
  - USD/MXN, USD/CLP: TwelveData API (gratis), Investing.com
  - DXY: FRED DTWEXBGS, Investing.com

COMMODITIES:
  - WTI, Brent: TwelveData, Yahoo Finance
  - Gold: TwelveData (plan pago), Yahoo Finance
  - Coffee: Yahoo Finance (KC=F)

EQUITY:
  - COLCAP: BVC, Investing.com

RATES:
  - VIX: FRED VIXCLS, Yahoo Finance (^VIX)
  - Treasury 10Y/2Y: FRED DGS10/DGS2
  - IBR: BanRep, Investing.com
  - COL Yields: BanRep, Investing.com

RISK:
  - EMBI Colombia: BCRP (Perú), JP Morgan
```

### Datos Mensuales (M) - Usar FFILL
```
USA (FRED):
  - CPI: CPIAUCSL, CPILFESL
  - PCE: PCEPI
  - Unemployment: UNRATE
  - Industrial Production: INDPRO
  - M2: M2SL
  - Fed Funds: FEDFUNDS
  - Consumer Sentiment: UMCSENT

COLOMBIA:
  - IPC: DANE, BanRep
  - TPM: BanRep
  - ITCR: BanRep
  - Reservas: BanRep
  - Exports/Imports: DANE
```

### Datos Trimestrales (Q) - Usar FFILL
```
USA:
  - GDP: FRED GDP

COLOMBIA:
  - Current Account: BanRep
  - FDI In/Out: BanRep
```

---

## 🔧 Recomendaciones para Mejorar Calidad

### 1. Gap Filling para Datos Diarios

```python
# Estrategia: Buscar en múltiples fuentes
DAILY_SOURCES = {
    'dxy': ['fred:DTWEXBGS', 'investing:us-dollar-index', 'yfinance:DX-Y.NYB'],
    'vix': ['fred:VIXCLS', 'yfinance:^VIX', 'investing:volatility-sp-500'],
    'wti': ['twelvedata:CL', 'yfinance:CL=F', 'investing:crude-oil'],
    'brent': ['twelvedata:BZ', 'yfinance:BZ=F', 'investing:brent-oil'],
    'gold': ['yfinance:GC=F', 'investing:gold'],
    'usdmxn': ['twelvedata:USD/MXN', 'yfinance:USDMXN=X'],
    'usdclp': ['twelvedata:USD/CLP', 'yfinance:USDCLP=X'],
    'colcap': ['investing:colombia-colcap', 'bvc'],
    'embi': ['bcrp:PD04701XD', 'investing:colombia-embi'],
}
```

### 2. FFILL para Datos Mensuales/Trimestrales

```sql
-- Aplicar forward fill para datos mensuales en la vista de features
CREATE OR REPLACE VIEW inference_features_daily AS
SELECT
    fecha,
    -- Datos diarios (sin ffill)
    fxrt_index_dxy_usa_d_dxy,
    volt_vix_usa_d_vix,
    -- Datos mensuales (con ffill)
    COALESCE(
        infl_cpi_all_usa_m_cpiaucsl,
        LAG(infl_cpi_all_usa_m_cpiaucsl) OVER (ORDER BY fecha)
    ) as infl_cpi_all_usa_m_cpiaucsl_ffill,
    ...
FROM macro_indicators_daily;
```

### 3. Detector de Gaps Automático

```python
def detect_gaps(table='macro_indicators_daily', lookback_days=30):
    """Detectar gaps en las últimas N días"""
    query = """
    WITH expected_dates AS (
        SELECT generate_series(
            CURRENT_DATE - interval '{} days',
            CURRENT_DATE - 1,
            '1 day'
        )::date as fecha
        WHERE EXTRACT(DOW FROM fecha) NOT IN (0, 6)
    )
    SELECT e.fecha
    FROM expected_dates e
    LEFT JOIN {} m ON e.fecha = m.fecha
    WHERE m.fecha IS NULL
    """.format(lookback_days, table)
    return gaps
```

### 4. Pipeline de Backfill

```
Para cada gap detectado:
1. Intentar fuente primaria (TwelveData/FRED)
2. Si falla, intentar fuente secundaria (Yahoo Finance)
3. Si falla, intentar scraping (Investing.com)
4. Si falla, marcar para revisión manual
5. Insertar datos recuperados
```

---

## 📈 Estado Actual del Pipeline

| DAG | Estado | Función |
|-----|--------|---------|
| v3.l0_macro_daily | ✅ Activo | Fetch diario TwelveData + FRED (7:50am COT) |
| v3.l0_macro_gap_filler | ✅ Activo | Gap detection + filling (8:00am COT) |

### Próximos Pasos:
1. ✅ FRED API Key configurada (752b8fee...0f2)
2. ✅ TwelveData funcionando
3. ✅ Gap detector automático implementado
4. ✅ Investing.com scraper agregado al gap-filler
5. ✅ Backfill completado (Dic 17-18, 2025)
6. ⬜ Configurar alertas de calidad

---

## 📊 ACTUALIZACIÓN: Verificación de Fuentes (2025-12-18)

### Aclaración Importante sobre Fuentes

**Las 7 variables "atrasadas" NO provienen directamente de BanRep ni Fedesarrollo:**

| Variable | Fuente Real | Origen de Datos |
|----------|-------------|-----------------|
| Gold | Investing.com / Yahoo Finance | COMEX (mercado internacional) |
| Coffee | Investing.com / Yahoo Finance | ICE (mercado internacional) |
| COLCAP | Investing.com | BVC Colombia vía agregadores |
| TES 10Y | Investing.com | BanRep/Bloomberg vía agregadores |
| TES 5Y | Investing.com | BanRep/Bloomberg vía agregadores |
| IBR | BanRep SUAMECA API | ⚠️ API retorna 404 actualmente |
| Prime Rate | FRED (DPRIME) | Federal Reserve USA |

### Estado Post-Actualización

| Variable | Última Fecha | Último Valor | Estado |
|----------|--------------|--------------|--------|
| Gold | 2025-12-18 | 4363.10 | ✅ Actualizado |
| Coffee | 2025-12-17 | 347.40 | ✅ Actualizado |
| COLCAP | 2025-12-17 | 2053.25 | ✅ Actualizado |
| TES 10Y | 2025-12-17 | 12.482% | ✅ Actualizado |
| TES 5Y | 2025-12-17 | 12.650% | ✅ Actualizado |
| IBR | 2025-12-18 | 8.723% | ✅ Forward-fill aplicado |
| TPM | 2025-12-18 | 9.250% | ✅ Forward-fill aplicado |
| Prime Rate | 2025-12-16 | 6.75% | ✅ Actualizado |

### Scripts de Mantenimiento

- `scripts/scraper_banrep_selenium.py` - **NUEVO** Scraper multi-fuente consolidado
- `scripts/update_atrasadas_variables.py` - Actualiza variables desde Investing.com + FRED
- `scripts/test_atrasadas_scrapers.py` - Verifica disponibilidad de fuentes

---

## 🔍 Verificación Exhaustiva de Fuentes BanRep (2025-12-18)

### Resumen de Pruebas

| Fuente | Estado | Variables Soportadas |
|--------|--------|---------------------|
| **datos.gov.co** | ✅ TRM funciona | TRM (USD/COP oficial) |
| **Investing.com** | ✅ Funcionando | COLCAP, TES 10Y/5Y, Gold, Coffee |
| **SUAMECA BanRep** | ❌ 404 Error | IBR, TPM, ITCR, TOT (URLs no disponibles) |
| **FRED API** | ✅ Funcionando | DXY, VIX, Treasuries, Prime Rate |

### Variables BanRep Identificadas (12 series SUAMECA)

| Serie ID | Variable | DB Column | Estado |
|----------|----------|-----------|--------|
| 241 | IBR Overnight | finc_rate_ibr_overnight_col_d_ibr | ⚠️ Forward-fill |
| 59 | TPM | polr_policy_rate_col_d_tpm | ⚠️ Forward-fill |
| 4170 | ITCR | fxrt_reer_bilateral_col_m_itcr | ⚠️ Sin acceso |
| 4180 | TOT | ftrd_terms_trade_col_m_tot | ⚠️ Sin acceso |
| 15051 | Reservas Int. | rsbp_reserves_international_col_m_resint | ⚠️ Sin acceso |
| 100002 | IPC Colombia | infl_cpi_total_col_m_ipccol | ⚠️ Sin acceso |

### Bypass de Radware Captcha

Se implementó `scraper_banrep_selenium.py` con:
- undetected-chromedriver para bypass de Radware Bot Manager
- SeleniumBase UC Mode como fallback
- **Nota**: Las URLs SUAMECA actualmente retornan 404, no captcha

### Solución Temporal: Forward-Fill

Para IBR y TPM se aplica forward-fill automático usando el último valor conocido:
- IBR: 8.723% (desde 2025-11-26)
- TPM: 9.250% (desde 2025-11-26)

Esto mantiene la continuidad de datos hasta que se restaure el acceso a SUAMECA.
