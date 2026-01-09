# USD/COP Data Pipeline v4.0

Pipeline de datos para modelos de Reinforcement Learning aplicados al par USD/COP.

**Ultima actualizacion:** 2025-12-15
**Version:** 4.0 - Estructura reorganizada con prefijos numericos

---

## Estructura del Pipeline

```
pipeline/
│
├── 00_config/                      # Configuracion centralizada
│   ├── DICCIONARIO_MACROECONOMICOS.csv
│   └── .env.example
│
├── 01_sources/                     # Datos crudos de fuentes externas
│   ├── 01_commodities/             # WTI, Brent, Gold, Coffee
│   ├── 02_exchange_rates/          # DXY, USDMXN, USDCLP
│   ├── 03_fixed_income/            # Bonos COL/USA, IBR
│   ├── 04_inflation/               # CPI USA, PCE, CPI COL
│   ├── 05_policy_rates/            # Fed Funds, TPM
│   ├── 06_country_risk/            # EMBI, ICC
│   ├── 07_economic_growth/         # PIB
│   ├── 08_foreign_trade/           # Balanza comercial
│   ├── 09_reserves_bop/            # Reservas, IED
│   ├── 10_labor_market/            # Desempleo USA
│   ├── 11_money_supply/            # M2 USA
│   ├── 12_sentiment/               # Consumer Sentiment
│   ├── 13_volatility/              # VIX
│   ├── 14_equities/                # COLCAP
│   ├── 15_production/              # Produccion Industrial
│   └── 16_usdcop_historical/       # OHLC diario USD/COP
│
├── 02_scrapers/                    # Scrapers para actualizacion
│   ├── 01_orchestrator/            # Script principal de scraping
│   ├── 02_custom/                  # Scrapers especificos por fuente
│   ├── 03_fallbacks/               # Scrapers de respaldo
│   └── utils/                      # Utilidades compartidas
│
├── 03_fusion/                      # Fusion de datos historicos
│   ├── run_fusion.py               # Script principal
│   └── output/                     # DATASET_MACRO_*.csv
│
├── 04_cleaning/                    # Limpieza y normalizacion
│   ├── run_clean.py                # Script principal
│   └── output/                     # MACRO_*_CLEAN.csv
│
├── 05_resampling/                  # Resampleo temporal
│   ├── run_resample.py             # Script principal
│   └── output/                     # MACRO_5MIN_CONSOLIDATED.csv
│
├── 06_rl_dataset_builder/          # Generacion de datasets RL
│   ├── 01_build_5min_datasets.py   # Datasets intraday
│   ├── 02_build_daily_datasets.py  # Datasets diarios
│   └── 03_analyze_datasets.py      # Analisis estadistico
│
├── 07_output/                      # DATASETS FINALES
│   ├── datasets_5min/              # 10 datasets 5min (~250MB)
│   ├── datasets_daily/             # 10 datasets diarios (~5MB)
│   ├── analysis/                   # Reportes estadisticos
│   └── docs/                       # Documentacion de datasets
│
├── run_pipeline.py                 # Script principal de ejecucion
└── README.md                       # Este archivo
```

---

## Inicio Rapido

```bash
# Ver estado del pipeline
python run_pipeline.py --check

# Ver diagrama del flujo
python run_pipeline.py --diagram

# Ejecutar pipeline completo (pasos 3-6)
python run_pipeline.py

# Ejecutar solo un paso especifico
python run_pipeline.py --step 3

# Ejecutar desde un paso en adelante
python run_pipeline.py --from 4
```

---

## Flujo de Ejecucion

| Paso | Carpeta | Script | Descripcion | Output |
|------|---------|--------|-------------|--------|
| 1 | `01_sources/` | - | Datos crudos manuales | CSV/XLSX |
| 2 | `02_scrapers/` | `actualizador_hpc_v3.py` | Actualizacion web | datos_*.csv |
| 3 | `03_fusion/` | `run_fusion.py` | Fusion de historicos | DATASET_MACRO_*.csv |
| 4 | `04_cleaning/` | `run_clean.py` | Limpieza y merge | MACRO_*_CLEAN.csv |
| 5 | `05_resampling/` | `run_resample.py` | Resampleo 5min | MACRO_5MIN_CONSOLIDATED.csv |
| 6 | `06_rl_builder/` | `01_build*.py` | Generacion RL | RL_DS*_*.csv |
| 7 | `07_output/` | - | Datasets finales | 10 datasets |

---

## Diagrama del Flujo

```
[01_sources]          Datos crudos de fuentes externas
     |                (commodities, exchange_rates, etc.)
     v
[02_scrapers]         Actualizacion automatica via web
     |                (investing.com, FRED, DANE, etc.)
     v
[03_fusion]           Fusion de todos los datos historicos
     |                -> DATASET_MACRO_*.csv
     v
[04_cleaning]         Limpieza y normalizacion
     |                -> MACRO_*_CLEAN.csv
     v
[05_resampling]       Resampleo a 5min + filtro festivos
     |                -> MACRO_5MIN_CONSOLIDATED.csv
     v
[06_rl_builder]       Generacion de 10 datasets RL
     |                (DS1-DS10 para diferentes estrategias)
     v
[07_output]           DATASETS FINALES LISTOS PARA RL
                      datasets_5min/ (250MB)
                      datasets_daily/ (5MB)
```

---

## Datasets Generados

### Datasets 5 Minutos (Intraday)

| Dataset | Features | Descripcion | Uso Recomendado |
|---------|----------|-------------|-----------------|
| `RL_DS1_MINIMAL` | 10 | Baseline minimo | Validar pipeline |
| `RL_DS2_TECHNICAL_MTF` | 14 | Multi-timeframe tecnico | Trend-following |
| `RL_DS3_MACRO_CORE` | 19 | Macro + tecnico | **PRODUCCION** |
| `RL_DS4_COST_AWARE` | 16 | Anti-overtrading | Reducir costos |
| `RL_DS5_REGIME` | 25 | Deteccion regimen | Transformers |
| `RL_DS6_CARRY_TRADE` | 18 | Tasas de interes | Carry trade |
| `RL_DS7_COMMODITY_BASKET` | 17 | Commodities COL | Export plays |
| `RL_DS8_RISK_SENTIMENT` | 21 | Risk-On/Off | Crisis detection |
| `RL_DS9_FED_WATCH` | 17 | Politica Fed | Ciclos monetarios |
| `RL_DS10_FLOWS` | 14 | Balanza pagos | Swing trading |

### Datasets Diarios

Los mismos 10 datasets pero agregados a frecuencia diaria (~1,500 filas cada uno).

---

## Variables Macro por Frecuencia

### Diarias (17 variables)
- DXY, USD/MXN, USD/CLP
- VIX, EMBI Colombia
- Brent, WTI, Gold, Coffee
- UST 2Y/10Y, Colombia 5Y/10Y
- IBR, Prime Rate, COLCAP

### Mensuales (15 variables)
- CPI USA, PCE, CPI Colombia
- Fed Funds, TPM Colombia
- Desempleo USA
- M2 USA, Consumer Sentiment
- Exportaciones/Importaciones Colombia
- ITCR, Reservas Internacionales

### Trimestrales (4 variables)
- IED Entrada/Salida
- Cuenta Corriente
- PIB USA

---

## Principios de Datos v4.0

1. **SIN FFILL para gaps**: Los datos se mantienen tal cual vienen de la fuente
2. **Filtro de Festivos Colombia**: Usando libreria `colombian-holidays`
3. **Horario de Mercado**: 8:00am - 12:55pm COT
4. **Sin Look-ahead Bias**: `merge_asof(direction='backward')`

---

## Dependencias

```bash
pip install pandas numpy requests cloudscraper beautifulsoup4
pip install colombian-holidays openpyxl xlrd python-dotenv
```

---

## Orden de Experimentacion RL

```
FASE 1 - CORE:
  1. DS1_MINIMAL        -> Sharpe > 0.3?  -> Continuar
  2. DS3_MACRO_CORE     -> Sharpe > 0.5?  -> USAR EN PRODUCCION
  3. DS6_CARRY_TRADE    -> Alto impacto esperado

FASE 2 - ESPECIALIZACION:
  4. DS8_RISK_SENTIMENT -> Regimenes volatiles
  5. DS7_COMMODITY      -> Correlacion Brent-COP
  6. DS9_FED_WATCH      -> Ciclos de Fed

FASE 3 - AVANZADO:
  7. DS4_COST_AWARE     -> Solo si overtrading
  8. DS5_REGIME         -> Arquitecturas attention
  9. DS10_FLOWS         -> Swing trading

FASE 4 - ENSEMBLE:
  Combinar DS3 + DS6 + DS8 con votacion ponderada
```

---

## Troubleshooting

### Error: "colombian-holidays not found"
```bash
pip install colombian-holidays
```

### Error: "No API keys found"
- Verificar archivo `00_config/.env.example`
- Crear `.env` con tus API keys

### Error: "OHLCV backup not found"
- Verificar `backups/database/usdcop_m5_ohlcv_*.csv.gz`

---

**Licencia:** Proyecto interno - Uso exclusivo para trading algoritmico USD/COP.
