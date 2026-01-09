# PROPUESTA DE REORGANIZACION - Data Pipeline USD/COP

## Nueva Estructura con Prefijos Numericos

```
data/
├── pipeline/                              # PIPELINE PRINCIPAL
│   │
│   ├── 00_raw_sources/                    # FUENTES CRUDAS (antes: Macros/)
│   │   ├── commodities/
│   │   ├── country_risk/
│   │   ├── economic_growth/
│   │   ├── equities/
│   │   ├── exchange_rates/
│   │   ├── fixed_income/
│   │   ├── foreign_trade/
│   │   ├── inflation/
│   │   ├── labor_market/
│   │   ├── money_supply/
│   │   ├── policy_rates/
│   │   ├── production/
│   │   ├── reserves_bop/
│   │   ├── sentiment/
│   │   └── volatility/
│   │
│   ├── 01_fusion/                         # FUSION DE DATOS (antes: Fuse Macro/)
│   │   ├── config/
│   │   │   └── DICCIONARIO_VARIABLES.csv
│   │   ├── scripts/
│   │   │   └── macro_data_fusion.py
│   │   └── output/
│   │       ├── MACRO_DAILY_RAW.csv
│   │       ├── MACRO_MONTHLY_RAW.csv
│   │       └── MACRO_QUARTERLY_RAW.csv
│   │
│   ├── 02_update_scrapers/                # SCRAPERS ACTUALIZACION (antes: data_daily_update_scapper/)
│   │   ├── config/
│   │   │   └── DICCIONARIO_VARIABLES.csv
│   │   ├── orchestrators/
│   │   │   └── actualizador_hpc.py
│   │   ├── scrapers/
│   │   │   ├── custom/                    # Scrapers especificos
│   │   │   ├── suameca/                   # Scrapers SUAMECA
│   │   │   └── fallbacks/                 # Scrapers de respaldo
│   │   ├── utils/
│   │   │   └── format_normalizer.py
│   │   └── output/
│   │       ├── datos_diarios_hpc.csv
│   │       ├── datos_mensuales_hpc.csv
│   │       └── datos_trimestrales_hpc.csv
│   │
│   ├── 03_processing/                     # PROCESAMIENTO (antes: PASS/)
│   │   ├── scripts/
│   │   │   ├── 01_clean_macro_data.py
│   │   │   ├── 02_resample_consolidated.py
│   │   │   ├── 03_create_rl_datasets.py
│   │   │   └── 04_analyze_rl_datasets.py
│   │   ├── input/
│   │   │   ├── historical/                # De 01_fusion/output
│   │   │   └── recent/                    # De 02_update_scrapers/output
│   │   └── intermediate/
│   │       ├── MACRO_DAILY_CLEAN.csv
│   │       ├── MACRO_5MIN_CONSOLIDATED.csv
│   │       └── MACRO_DAILY_CONSOLIDATED.csv
│   │
│   ├── 04_rl_datasets/                    # DATASETS FINALES RL
│   │   ├── datasets/
│   │   │   ├── RL_DS01_MINIMAL.csv
│   │   │   ├── RL_DS02_TECHNICAL_MTF.csv
│   │   │   ├── RL_DS03_MACRO_CORE.csv
│   │   │   ├── RL_DS04_COST_AWARE.csv
│   │   │   ├── RL_DS05_REGIME.csv
│   │   │   ├── RL_DS06_CARRY_TRADE.csv
│   │   │   ├── RL_DS07_COMMODITY_BASKET.csv
│   │   │   ├── RL_DS08_RISK_SENTIMENT.csv
│   │   │   ├── RL_DS09_FED_WATCH.csv
│   │   │   └── RL_DS10_FLOWS_FUNDAMENTALS.csv
│   │   ├── analysis/
│   │   │   └── STATS_*.csv
│   │   └── docs/
│   │       ├── DICCIONARIO_VARIABLES.md
│   │       └── README_DATASETS.md
│   │
│   └── 05_sentiment/                      # ANALISIS SENTIMENT (antes: investing_news_scraper_pro/)
│       ├── scripts/
│       │   ├── scrape_historical.py
│       │   ├── sentiment_analyzer.py
│       │   └── generate_sentiment_hourly.py
│       └── output/
│           ├── noticias_historicas.csv
│           └── sentiment_usdcop_hourly.csv
│
├── config/                                # CONFIGURACION GLOBAL
│   ├── DICCIONARIO_MACROECONOMICOS_MASTER.csv
│   └── Data_Engineering-Dictionary.xlsx
│
├── archive/                               # ARCHIVOS HISTORICOS/DEPRECADOS
│   ├── backups/
│   └── legacy_scripts/
│
└── README.md                              # Documentacion del pipeline
```

## Flujo de Ejecucion

| Paso | Carpeta | Script Principal | Input | Output |
|------|---------|------------------|-------|--------|
| 0 | 00_raw_sources/ | N/A (manual) | Descargas manuales | CSVs crudos |
| 1 | 01_fusion/ | macro_data_fusion.py | 00_raw_sources/ | MACRO_*_RAW.csv |
| 2 | 02_update_scrapers/ | actualizador_hpc.py | APIs/Web | datos_*_hpc.csv |
| 3 | 03_processing/ | 01→02→03→04 | 01+02 outputs | RL datasets |
| 4 | 04_rl_datasets/ | N/A (output) | 03_processing/ | Datasets finales |
| S | 05_sentiment/ | sentiment_analyzer.py | Investing.com | sentiment.csv |

## Componentes a ELIMINAR (no necesarios en produccion)

1. **data_daily_update_scapper/.cache/** - Cache de desarrollo
2. **data_daily_update_scapper/Actualizados/scrapers/suameca/debug_*.png** - ~400 screenshots de debug
3. **Fuse Macro/.claude/** - Configuracion local de Claude Code
4. **Scripts duplicados**:
   - `actualizador_completo_v2.py` (usar v3)
   - `actualizador_hpc_v3.py` (en raiz, usar el de Actualizados/)
   - `proyecto/` carpeta completa (versiones antiguas)

## Componentes OPCIONALES

1. **investing_news_scraper_pro/** - Si no usas sentiment en el modelo RL, puedes archivarlo
2. **Macros/POLICY RATES/** y **Macros/POLICY_RATES/** - Duplicados (uno con espacio, otro con _)

## Acciones Recomendadas

1. Crear estructura `pipeline/` con subcarpetas numeradas
2. Mover y renombrar archivos segun mapeo
3. Limpiar screenshots de debug (~40MB de imagenes .png)
4. Limpiar cache .pkl (~36 archivos)
5. Actualizar rutas en scripts (CONFIG paths)
6. Crear README principal
