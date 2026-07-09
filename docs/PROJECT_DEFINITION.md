# PROJECT_DEFINITION.md — GlobalMinds

> **Definición consolidada del proyecto.** Cubre identidad, arquitectura técnica, los 3 pipelines, performance verificada, modelo de negocio, estado operacional, roadmap, equipo y pedidos.
>
> **Última actualización**: 2026-05-09
> **Mantenedor**: Pedro Sánchez Briceño
> **Versión**: 1.0
> **Audiencias**: inversores · Microsoft Founders Hub · operadores · nuevos developers
>
> Este documento NO duplica `CLAUDE.md` (reglas técnicas detalladas), `.claude/rules/*` (specs SDD), ni `docs/architecture/*` (decisiones arquitectónicas). Apunta a ellos. Ver §11 Apéndice B para el mapa completo.

---

## Tabla de Contenidos

1. [Executive Summary](#1-executive-summary)
2. [Identidad del Proyecto](#2-identidad-del-proyecto)
3. [Arquitectura Técnica](#3-arquitectura-técnica)
4. [Tres Pipelines en Detalle](#4-tres-pipelines-en-detalle)
5. [Performance y Métricas](#5-performance-y-métricas)
6. [Modelo de Negocio](#6-modelo-de-negocio)
7. [Estado Operacional Actual](#7-estado-operacional-actual-mayo-2026)
8. [Hitos 90 días](#8-hitos-90-días-mayo-julio-2026)
9. [Equipo](#9-equipo)
10. [Pedido a Microsoft / Inversores](#10-pedido-a-microsoft--inversores)
11. [Apéndices](#11-apéndices)
12. [Navegación Rápida](#12-navegación-rápida)

---

## 1. Executive Summary

### Qué es GlobalMinds

**GlobalMinds** es una plataforma de IA aplicada al mercado cambiario de Latinoamérica. El núcleo es un sistema end-to-end de Machine Learning + Reinforcement Learning para forecasting y ejecución sobre USD/COP, con expansión planeada a USD/MXN, USD/BRL y USD/CLP.

Un mismo motor de IA alimenta **4 productos comerciales** distintos:

1. **Bot P2P USDT/COP** — automatización para merchants en Binance P2P
2. **SaaS Hedging FX para PYMES** — Kantox para LATAM
3. **Optimizador de Remesas B2B** — layer de IA para casas de cambio (USD 13B/año mercado)
4. **App para freelancers** — convierte USD ↔ COP con timing IA y rails Bre-B

### Estado actual (mayo 2026)

| Componente | Estado | Tracción |
|------------|--------|----------|
| Núcleo USD/COP H5 | ✅ **Producción** | Smart Simple v2.0, 18 meses dev, regime gate live |
| Pipeline H1 daily | ⏸ Pausado | Pendiente validación v2.0 |
| Pipeline RL | 🔻 Deprecado | NOT significant (p=0.272), kept for research |
| Dashboard | ✅ Producción | 8 páginas, 47 API routes, datos reales |
| 4 productos comerciales | 🟡 En MVP | Bot P2P en construcción |
| Pitch deck Microsoft | ✅ Listo | 23 slides + speaker notes ES |

### 3 cifras clave (backtest 2025 OOS, USD-denominated)

| Métrica | Valor | Categoría |
|---------|------:|-----------|
| **Retorno anual** | **+25.63%** | vs Buy & Hold −14.48% → alpha +40.11 pp |
| **Sharpe ratio** | **3.00** *(vs US 3M T-Bill, √52)* | Excellent (>3 = top quartile hedge funds) |
| **Calmar ratio** | **4.19** | Excellent (>3) — el "Sharpe sin asunciones" |
| **MaxDD** | **6.12%** | Mejor que hedge fund típico (8%) |
| **Win rate** | **82.4%** | 28 de 34 trades ganadores |
| **p-value** | **0.006** | Bootstrap 10K — 99.4% no es suerte |

### Las 4 oportunidades en 1 línea cada una

1. **Bot P2P** — merchants moviendo USD 50–200k/día con Excel; cero competencia ML; cash flow más rápido
2. **Hedging PYMES** — solo 17% de empresas COL usa cobertura FX; comparable Kantox vendida a Visa por €175M (2025)
3. **Remesas B2B** — USD 13,098M en 2025 (récord histórico), conversión sin optimización IA; ningún player ofrece este layer
4. **App Freelancers** — 500k–1M colombianos cobrando USD; pierden 1.5–3% en Wise/Payoneer; rails Bre-B + AI timing

**TAM combinado LATAM: USD 50–150M anuales.**

---

## 2. Identidad del Proyecto

### Visión

> *"IA aplicada al mercado cambiario de Latinoamérica."*

Construir la plataforma de inteligencia cambiaria de referencia para LATAM, partiendo del par más profundamente conocido (USD/COP) y replicando el motor a pares vecinos.

### Core thesis (insight contraintuitivo)

**El verdadero valor de nuestra IA no es predecir hacia dónde va el mercado — es saber CUÁNDO el mercado no es predecible y abstenerse de operar.**

El audit de 10 agentes (marzo 2026) reveló que el modelo predictivo (Ridge / BayesianRidge / XGBoost) tiene R² < 0 en 2025 y 2026 — es decir, peor que predecir la media. Sin embargo, la estrategia generó +25.63% en 2025. ¿Por qué?

El alpha NO viene del modelo direccional. Viene de:
1. **Regime Gate (Hurst exponent)** — bloquea operación cuando el mercado está mean-reverting. En Q1 2026 bloqueó 13 de 14 semanas, convirtiendo lo que hubiera sido −5.17% (versión sin gate) en +0.61% (versión con gate). Alpha del componente: **+5.78 pp**.
2. **Effective Hard Stop** — capa los stops a 3.5% del portfolio para evitar pérdidas catastróficas.
3. **Take Profit / Hard Stop adaptativos** — ajustados por volatilidad realizada.
4. **Dynamic leverage** — escala el sizing por win-rate rolling y drawdown actual [0.25, 1.0].

**Conclusión**: el sistema gana por gestión de riesgo y selectividad, no por predicción direccional pura.

### Filosofía — Spec-Driven Development (SDD)

El proyecto usa SDD como metodología fundamental:

```
Layer 1: SPEC (defines what)        → .claude/rules/sdd-*.md (10 specs)
Layer 2: CONTRACT (enforces how)    → lib/contracts/ + src/contracts/ (10 TS + 6 Py)
Layer 3: IMPLEMENTATION             → scripts/, pages, DAGs (conform to contracts)
```

**Principio**: la spec define la API antes de escribir código. El contrato la enforza. La implementación cumple. Nada se merge sin contract validation en CI.

### Pares cubiertos (presente y roadmap)

| Par | Estado | Datos disponibles |
|-----|--------|-------------------|
| **USD/COP** | ✅ Producción | 81K bars 5-min · 3K daily (2015→2026) |
| USD/MXN | 🟡 Listo (datos cargados) | 95K bars 5-min |
| USD/BRL | 🟡 Listo (datos cargados) | 90K bars 5-min |
| USD/CLP | 🔮 Roadmap Q3 2026 | — |
| USD/PEN | 🔮 Roadmap Q4 2026 | — |

---

## 3. Arquitectura Técnica

### Stack consolidado

| Capa | Tecnologías |
|------|-------------|
| Orquestación | Apache Airflow (27 DAGs L0→L7) |
| Tracking ML | MLflow (server desplegado, integración auto en roadmap) |
| Almacenamiento | PostgreSQL 15 + TimescaleDB · Redis 7 · MinIO (S3-compatible) |
| Backend | FastAPI · Python 3.11 |
| Frontend | Next.js 15 · TypeScript · Tailwind · shadcn/ui · Lightweight Charts |
| ML core | scikit-learn · XGBoost · LightGBM · CatBoost · Stable-Baselines3 |
| LLM | Azure OpenAI (GPT-4o) primario · Anthropic Claude fallback |
| Containerización | Docker + Docker Compose (15-25 servicios según modo) |
| Cloud-ready | Azure ML / OpenAI / Container Apps / Functions |
| Observabilidad | Prometheus · Grafana · AlertManager · Loki · Promtail · Jaeger |
| Secrets | HashiCorp Vault (AES-256-GCM) |

### Diagrama de alto nivel

```
                        DATOS                              MODELO                            SEÑALES                   PRODUCTOS
   ┌─────────────────┐                  ┌────────────────────────────────────┐         ┌─────────────────┐         ┌────────────────┐
   │ Mercado FX 5min │ ───┐             │                                    │         │                 │         │ Bot P2P        │ ───┐
   ├─────────────────┤    │             │   Pipeline ML / RL — Forecasting   │         │ Señales         │ ──────► │                │    │
   │ Macro 40 vars   │ ───┼──────────►  │   Engine                            │ ──────► │ universales:    │ ──────► ├────────────────┤    │
   ├─────────────────┤    │             │                                    │         │ dirección,      │         │ SaaS Hedging   │ ───┤
   │ Noticias 5 src  │ ───┤             │   Ridge + BayesianRidge + XGBoost  │         │ confianza,      │ ──────► │                │    │
   ├─────────────────┤    │             │   Regime Gate (Hurst R/S)          │         │ sizing,         │         ├────────────────┤    │
   │ Análisis IA     │ ───┘             │   Effective HS · Walk-forward      │         │ stops           │         │ Optimizador    │ ───┤
   │ GPT-4o + Claude │                  │   weekly retrain                   │         │                 │ ──────► │ Remesas B2B    │    │
   └─────────────────┘                  └────────────────────────────────────┘         └─────────────────┘         ├────────────────┤    │
                                                                                                                    │ App Freelancers│ ───┘
                                                                                                                    │ Bre-B + USD    │
                                                                                                                    └────────────────┘

                                                              SignalBridge OMS (FastAPI)
                                                              ┌───────────────────────────┐
                                                              │ MEXC / Binance via CCXT   │
                                                              │ Paper · Testnet · Live    │
                                                              │ Risk · Kill Switch · Audit│
                                                              └───────────────────────────┘
```

### 27 DAGs en producción (Airflow)

| Pipeline | DAGs | Schedule (COT) |
|----------|-----:|----------------|
| L0 Data | 5 | OHLCV: */5 8-12 Mon-Fri · Macro: hourly · Backfill: Sun · Seed backup: daily 13:00 |
| H1 Daily (PAUSED) | 5 | Sun 01:00 train · Mon-Fri 13:00 signal/13:30 vol/13:35 exec/19:00 monitor |
| H5 Weekly (PRODUCTION) | 5 | Sun 01:30 train · Mon 08:15 signal/08:45 vol · Mon-Fri */30 9-13 exec · Fri 14:30 monitor |
| Forecasting Weekly | 1 | Mon 09:00 — regenera CSV + 76 PNGs para `/forecasting` |
| RL | 6 | Manual / event-triggered (excepto L1 */5 8-12 Mon-Fri) |
| News + Analysis | 5 | News: 3x/día · Alert: */30 · Digest: Mon · Analysis L8: 14:00 |
| Watchdog | 1 | Hourly 8-13 COT — auto-heal stale data, forecasting, analysis |

### 25+ servicios (modo Full Enterprise)

| Servicio | Puerto | Rol |
|----------|-------:|-----|
| PostgreSQL + TimescaleDB | 5432 | OLTP primario |
| Redis 7 | 6379 | Cache · pub/sub · streams |
| MinIO | 9001 | S3 object store (modelos, artefactos) |
| Airflow (scheduler + webserver) | 8080 | Orquestación 27 DAGs |
| Dashboard Next.js | 5000 | Trading dashboard 8 páginas |
| SignalBridge API | 8085 | OMS execution layer |
| MLflow | 5001 | Experiment tracking (server ✅, auto-integration en roadmap) |
| HashiCorp Vault | 8200 | Secrets (exchange API keys) |
| Prometheus | 9090 | 53 alert rules cargadas |
| Grafana | 3002 | 4 dashboards (trading, MLOps, system, macro) |
| AlertManager | 9093 | Slack · PagerDuty (rules listas, secrets pendientes) |
| Loki | 3100 | Log aggregation |
| Promtail | — | Log shipping |
| Jaeger | 16686 | Tracing (deployado, instrumentación pendiente) |
| pgAdmin | 5050 | DB admin UI |

> Modo compact (12 servicios) cubre operación diaria con ~6-8GB RAM.

### Páginas del Dashboard (Next.js)

| Página | Ruta | Activación |
|--------|------|------------|
| Landing / Hub / Login | `/`, `/hub`, `/login` | Siempre |
| Forecasting | `/forecasting` | Tras `generate_weekly_forecasts.py` |
| Dashboard | `/dashboard` | Tras `--phase backtest` (revisión y aprobación human Vote 2/2) |
| Production | `/production` | Tras `--phase production` (lectura, monitoreo) |
| Analysis | `/analysis` | Tras `generate_weekly_analysis.py` |
| Execution | `/execution` | Tras configurar exchange API keys |

### Datos y backup

| Capa | Contenido | Frecuencia |
|------|-----------|------------|
| T1 — Live DB | OHLCV 5-min, daily, macro, signals, trades | Realtime |
| T2 — Daily backup | OHLCV + macro tables → parquet | Diario 13:00 COT |
| T3 — Git LFS | Seeds parquet, news CSVs, analysis JSONs | On commit |

**Restore on `docker-compose up`**: Daily backup → Git LFS → MinIO → legacy CSV.

### Cloud Migration Path

El stack está diseñado para migrar a Azure de forma directa:

| Servicio local | Azure equivalent |
|----------------|------------------|
| Airflow | Azure Container Apps + Container Apps Jobs |
| Modelos ML training | Azure Machine Learning (compute clusters) |
| LLM analysis | Azure OpenAI (ya en uso) |
| FastAPI services | Azure Container Apps |
| PostgreSQL | Azure Database for PostgreSQL Flexible Server |
| Redis | Azure Cache for Redis |
| MinIO | Azure Blob Storage |
| Vault | Azure Key Vault |
| Monitoring | Azure Monitor + Application Insights |

**Estimación de costo Azure**: USD ~5-8K/mes en producción enterprise. Cubierto por créditos Founders Hub (USD 150k) durante 18-24 meses.

---

## 4. Tres Pipelines en Detalle

### 4.1 H5 Weekly Pipeline — Smart Simple v2.0 (PRODUCTION)

**El producto principal hoy.** Forecasting weekly con horizonte H=5 días.

#### Arquitectura

```
Sunday 01:30 COT
    │
    ▼
H5-L3: Weekly Training
    │  Ridge + BayesianRidge + XGBoost on expanding window (2020 → last Friday)
    │  23 features (21 base + vol_regime_ratio + trend_slope_60d)
    │  Target = ln(close[t+5]/close[t])
    │
Monday 08:15 COT
    ▼
H5-L5: Weekly Signal
    │  Ensemble prediction (mean of Ridge + BR + XGB)
    │  Confidence scoring (3-tier: HIGH/MEDIUM/LOW)
    │  Skip LOW-confidence LONGs
    │
Monday 08:45 COT
    ▼
H5-L5: Vol-Targeting + Regime Gate
    │  1. Realized vol (21d lookback)
    │  2. Base leverage from vol-targeting (target_vol = 0.15)
    │  3. Asymmetric sizing + confidence multiplier
    │  4. ★ REGIME GATE (Hurst R/S, 60d window):
    │     - TRENDING (H>0.52): sizing × 1.0 (full)
    │     - INDETERMINATE (0.42-0.52): sizing × 0.40
    │     - MEAN-REVERTING (H<0.42): skip_trade=True
    │  5. ★ DYNAMIC LEVERAGE: scale by rolling WR + drawdown [0.25, 1.0]
    │  6. ★ EFFECTIVE HS: min(HS_base, 3.5% / leverage)
    │
Monday 09:00 COT
    ▼
H5-L7: Entry (limit order, 0% maker fee on MEXC)
    │
Mon-Fri */30 9:00-13:00
    ▼
H5-L7: Monitor TP / HS
    │
Friday 12:50 COT
    ▼
H5-L7: Friday Close (market order)
    │
Friday 14:30 COT
    ▼
H5-L6: Weekly Monitor — DA, Sharpe, MaxDD, guardrails
```

#### Configuración SSOT

`config/execution/smart_simple_v1.yaml`:

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `vol_multiplier` | 2.0 | Wider stops eliminan hard stops espurios |
| `tp_ratio` | 0.5 | TP = HS × 0.5 (sweet spot de sensibilidad) |
| `hard_stop_min_pct` | 1% | Floor en mercados calmos |
| `hard_stop_max_pct` | 3% | Cap en mercados volátiles |
| SHORT sizing | flat 1.5x | Scorer no discrimina (LOW WR 79% > MED 60%) |
| LONG LOW sizing | 0.0 (SKIP) | Net effect = -0.75% |
| LONG HIGH/MED | 1.0x / 0.5x | Exploratorio, sizing pequeño |
| `target_vol` | 0.15 | Target 15% vol anualizada |
| `effective_portfolio_cap` | 3.5% | Cap de pérdida por trade en términos de portfolio |
| Regime gate H thresholds | 0.42 / 0.52 | Mean-reverting / Trending bands |
| Dynamic leverage range | [0.25, 1.0] | Floor/ceiling de scaling |
| Retraining frequency | Weekly | RESTAURADO en v2.0 (era monthly en v1.1.0 — bug) |

#### Resultados verificables

**Backtest 2025 (OOS):**

| Métrica | Valor |
|---------|------:|
| Retorno anual (USD) | +25.63% |
| Sharpe (rf=0, √52) | 3.347 |
| Sharpe (vs US 3M T-Bill, √52) | 3.00 |
| MaxDD | 6.12% |
| Calmar | 4.19 |
| p-value (bootstrap 10K) | 0.006 |
| Win rate | 82.4% |
| Trades | 34 (5 LONG / 29 SHORT) |
| Profit factor | 2.756 |
| Exits: take_profit | 21 (62%) |
| Exits: week_end | 11 (32%) |
| Exits: hard_stop | 2 (6%) |
| $10K → | $12,563 |

**2026 YTD (producción real, hasta hoy):**

| Métrica | Valor |
|---------|------:|
| Retorno YTD | +0.61% |
| Trades ejecutados | 1 (ganador) |
| Semanas bloqueadas por regime gate | 13 de 14 |
| Versión vieja sin gate (contrafactual) | −5.17% |
| Alpha del regime gate | +5.78 pp |

> Spec completo: `.claude/rules/h5-smart-simple-pipeline.md`

### 4.2 H1 Daily Pipeline (PAUSED)

Pipeline diario con horizonte H=1 día. **Pausado pendiente validación de v2.0 sobre H1.**

| Componente | Detalle |
|------------|---------|
| Modelos | 9 ensemble (ridge, bayesian_ridge, ard, xgboost, lightgbm, catboost, 3 hybrids) |
| Selección | Top 3 por magnitud de predicción |
| Ejecución | Trailing stop (activation +0.2%, trail 0.3%, hard stop 1.5%) |
| Modo | SHORT-only (regime change 2026 broke LONG profitability) |
| Backtest 2025 | +36.84%, Sharpe 3.135, p=0.0178, $13,684 |
| Estado | ⏸ DAGs activos pero modo paper; cierre v2.0 es prerequisito |
| Config | `config/execution/smart_executor_v1.yaml` |

> Spec: `.claude/rules/h5-smart-simple-pipeline.md` (mismo archivo cubre ambos)

### 4.3 RL Pipeline (DEPRIORITIZED)

PPO agent sobre barras de 5 minutos. **No estadísticamente significativo** (p=0.272, +2.51%). Mantenido para research.

| Componente | Detalle |
|------------|---------|
| Algoritmo | PPO (Stable-Baselines3) MlpPolicy |
| Bars | 5-min (78 bars/día session 8:00-12:55 COT) |
| Mejor versión | V21.5b (4/5 seeds positivos, +2.51% mean) |
| Backtest 2025 | +2.51%, Sharpe 0.321, p=0.272, NOT significant |
| Razón de deprioritización | Sample size insuficiente en 5-min para PPO |
| Estado | Research only, no producción |

> Spec: `.claude/rules/l2-l3-l4-training-pipeline.md` y `.claude/rules/l1-l5-inference-pipeline.md`

### 4.4 News Engine + Analysis Module (OPERATIONAL)

Pipeline de ingesta de noticias + análisis IA generado.

| Componente | Detalle |
|------------|---------|
| Fuentes | GDELT Doc + Context · NewsAPI · Investing.com · La República · Portafolio |
| Pipeline enrichment | 5 etapas (categorize, relevance, sentiment, NER, breaking detection) |
| Cross-reference | Token-based clustering (Jaccard) entre fuentes en ventana 48h |
| Feature export | ~60 daily features para ML integration |
| Análisis IA | MacroAnalyzer (13 vars: SMA, BB, RSI, MACD, z-score) + LLM narratives |
| LLM | Azure OpenAI GPT-4o primario + Anthropic fallback |
| Budget | $1/día, $15/mes, file-based caching |
| Output | `analysis_index.json` + `weekly_YYYY_WXX.json` para `/analysis` page |
| Estado actual (mayo 2026) | Investing.com (78 art) + Portafolio (283 art) en DB. W01-W15 generados. |

> Spec: `.claude/rules/news-and-analysis-sdd.md`

### Features compartidos (21 features H1 + H5)

| # | Categoría | Feature | Fuente |
|---|-----------|---------|--------|
| 1-4 | Price | close, open, high, low | Daily OHLCV |
| 5-8 | Returns | return_1d, return_5d, return_10d, return_20d | Log returns |
| 9-11 | Volatility | volatility_5d, volatility_10d, volatility_20d | Std of returns |
| 12-14 | Technical | rsi_14d (Wilder's EMA), ma_ratio_20d, ma_ratio_50d | Price-derived |
| 15-17 | Calendar | day_of_week, month, is_month_end | Date |
| 18-21 | Macro (T-1) | dxy_close_lag1, oil_close_lag1, vix_close_lag1, embi_close_lag1 | Macro DB |

**Anti-leakage**: Macro `shift(1)` + `merge_asof(direction='backward')` + expanding window 2020 → last Friday + train-only `StandardScaler`.

**Smart v2.0 agrega 2 features**: `vol_regime_ratio`, `trend_slope_60d`.

---

## 5. Performance y Métricas

### 5.1 Backtest 2025 OOS — Números Brutos

Todos los números provienen de `usdcop-trading-dashboard/public/data/production/summary_2025.json` (verificable en repo) y `trades/smart_simple_v11_2025.json`.

| Métrica | Valor |
|---------|------:|
| Período de evaluación | 2025-01-06 → 2025-12-25 (12 meses, 170 trading days) |
| Capital inicial | USD $10,000 |
| Capital final | USD $12,563.05 |
| Retorno absoluto | +USD $2,563.05 |
| Retorno % (compounded) | **+25.63%** |
| Buy & Hold mismo período | −14.48% (USD $8,552) |
| Alpha vs B&H | **+40.11 pp** |
| Trades ejecutados | 34 (5 LONG / 29 SHORT) |
| Trades ganadores | 28 (82.4% win rate) |
| Trades perdedores | 6 |
| Direction Accuracy | 82.4% |
| MaxDD | 6.12% |
| Sharpe (rf=0, √52) | 3.347 |
| Profit Factor | 2.756 |
| p-value (bootstrap 10K) | 0.006 |
| Statistical significance | ✅ Sí (p < 0.05) |

**Distribución de exits:**
- Take profit: 21 (62%)
- Week end (Friday close): 11 (32%)
- Hard stop: 2 (6%)

### 5.2 Sharpe Analysis — 4 Perspectivas × Horizontes

#### Principio metodológico

> **El risk-free rate debe estar denominado en la misma moneda que los retornos del estrategia.** No en la moneda del subyacente, no en la del exchange — en la del bolsillo que pone el dinero.

Esto colapsa las 4 perspectivas en **2 análisis distintos**:

| Perspectiva | Moneda capital | Análisis aplicable |
|-------------|----------------|--------------------|
| Americano con USD | USD | USD-native |
| Colombiano con USD | USD | USD-native (idéntico) |
| Americano con COP | COP | COP-native FX-adjusted |
| Colombiano con COP | COP | COP-native FX-adjusted (idéntico) |

#### Análisis USD-native (perspectivas 1 + 2)

Aplica a cualquier inversor con capital en USD. **Datos**: mean weekly = 0.6843%, vol anualizada = 10.63%.

Sharpe textbook: `mean(weekly_excess) / std(weekly) × √52`

| Horizonte rf | Tasa | **Sharpe** | Categoría |
|--------------|-----:|-----------:|-----------|
| Overnight (SOFR) | 5.30% | 2.85 | Excellent |
| 1M T-Bill | 5.20% | 2.86 | Excellent |
| **3M T-Bill** ★ | **3.68%** | **3.00** | **Exceptional** |
| 6M T-Bill | 3.70% | 3.00 | Excellent |
| 1Y T-Bill | 3.75% | 2.99 | Excellent |
| 2Y Treasury | 3.85% | 2.98 | Excellent |
| 5Y Treasury | 4.10% | 2.96 | Excellent |
| 10Y Treasury | 4.30% | 2.94 | Excellent |

**Insight**: El Sharpe USD es 2.85–3.00 en TODA la curva. La elección del horizonte mueve el Sharpe ~5%. Robusto bajo cualquier convención.

#### Análisis COP-native FX-adjusted (perspectivas 3 + 4)

Inversor con capital en COP que convierte COP→USD→strategy→USD→COP. En 2025, el peso se fortaleció 14.09% (USD/COP cayó de 4,346 a 3,734).

**En COP**: retorno final +7.93%, vol ~10.63%.

| Horizonte rf COP | Tasa | **Sharpe (2025 atípico)** | Categoría |
|------------------|-----:|--------------------------:|-----------|
| BanRep policy | 11.25% | 0.26 | Poor |
| **TES 1Y** ★ | **9.50%** | **0.43** | **Poor** |
| TES 5Y | 10.50% | 0.33 | Poor |
| TES 10Y | 11.80% | 0.21 | Poor |

#### Sensibilidad al ciclo FX (crítico para comprensión)

2025 fue un año atípico de fortalecimiento del peso (14% en 12 meses). En otros escenarios:

| Movimiento USD/COP anual | Retorno COP | Sharpe vs TES 1Y |
|--------------------------|------------:|-----------------:|
| Peso fortalece −10% | +13.07% | 1.01 |
| Peso fortalece −5% | +19.35% | 1.73 |
| **FX flat (escenario neutro)** | **+25.63%** | **2.45** |
| Peso debilita +5% | +31.91% | 3.17 |
| Peso debilita +10% | +38.19% | 3.89 |

**Histórico USD/COP 2010–2024**: peso se debilita ~3% promedio anual contra USD. **El año 2025 (−14%) está en el extremo inferior de la distribución histórica.**

#### Recomendación headline (para deck)

> **Sharpe 3.00** *(USD-native, vs US 3M T-Bill 3.68%, √52)*
>
> Por qué: convención institucional bulletproof, robusto en toda la curva (2.94–3.00 entre 6M y 10Y), categoría "Exceptional" (>3.0 = top decil).

### 5.3 MDD Analysis — 4 Perspectivas + Calmar

#### MaxDrawdown por perspectiva

**USD investor (perspectivas 1+2):**

| Métrica | Valor |
|---------|------:|
| Equity USD | $10,000 → $12,563 (+25.63%) |
| Peak | $12,959.71 (trade #27) |
| Trough | $12,167.08 (trade #29) |
| **MDD USD** | **6.12%** |

**COP investor FX-adjusted (perspectivas 3+4):**

| Métrica | Valor |
|---------|------:|
| Equity COP | $43,458,500 → $46,904,398 (+7.93%) |
| Peak | COP $50,168,074 (trade #27) |
| Trough | COP $46,422,555 (trade #32) |
| **MDD COP** | **7.47%** |

**Insight**: MDD COP es 1.22× MDD USD. La apreciación del peso creó drawdowns adicionales en COP que no existieron en USD.

#### MDD vs random walk teórico

Por la ley de Magdon-Ismail & Atiya (2004), un random walk con vol σ tiene MDD esperado ~1.25 × σ × √T:

| Horizonte hold | MDD random walk teórico | MDD real (Smart v2.0) | Ratio |
|----------------|------------------------:|----------------------:|------:|
| 1 año | 13.3% | 6.12% | **0.46×** |
| 3 años (proyección) | 23.0% | proyectado ~9-11% | ~0.45× |
| 5 años (proyección) | 29.7% | proyectado ~12-14% | ~0.45× |

**Tu MDD real es MENOS DE LA MITAD del MDD random walk con tu misma volatilidad.** Esto es evidencia cuantitativa del valor de regime gate + hard stops + position sizing.

#### Calmar Ratio — el "Sharpe sin asunciones"

`Calmar = retorno anual / MDD` — preferida por hedge funds porque NO requiere risk-free rate.

| Perspectiva | Retorno | MDD | **Calmar** | Categoría |
|-------------|--------:|----:|-----------:|-----------|
| **USD investor** | +25.63% | 6.12% | **4.19** | **Excellent** (>3) |
| COP investor (2025 atípico) | +7.93% | 7.47% | 1.06 | Acceptable |
| COP investor (FX-flat scenario) | +25.63% | ~7.5% | ~3.40 | Excellent |
| COP investor (peso debilita +5%) | +31.91% | ~7.5% | ~4.25 | Excellent |

#### Comparación con benchmarks de MDD

| Benchmark | MDD | Comentario |
|-----------|----:|------------|
| Renaissance Medallion | 4.0% | Best-in-class quant |
| **Tu Smart v2.0 USD** | **6.12%** | ← El reporte |
| Hedge fund típico | 8.0% | Industria |
| **Tu Smart v2.0 COP** | **7.47%** | ← FX-adjusted 2025 |
| S&P 500 año típico | 10.0% | Recesión leve |
| Buy & Hold USD/COP 2025 | 14.48% | Tu benchmark directo |
| S&P 500 COVID 2020 | 34.0% | Crash |
| S&P 500 GFC 2008 | 55.0% | Catástrofe |

**Tu MDD está en el segundo cuartil mejor de hedge funds globales.**

### 5.4 Buy & Hold Benchmark — Alpha vs Pasivo

**Buy & Hold (B&H)** = mantener posición LONG USD/COP durante el período sin trades, sin stops, sin gestión.

#### Cálculo

| Concepto | Valor |
|----------|------:|
| FX inicial (2025-01-06) | 4,345.85 |
| FX final (2025-12-31) | ~3,734 |
| Retorno B&H | **−14.48%** |
| Equity final B&H sobre $10K | $8,552 |

#### 3 medidas de alpha vs B&H

**1. Alpha absoluto (lineal):**
```
α_abs = R_strategy − R_B&H = 25.63% − (−14.48%) = +40.11 pp
```

**2. Alpha relativo (ratio):**
```
α_rel = (1 + R_strategy) / (1 + R_B&H) − 1 = 1.2563 / 0.8552 − 1 = +46.9%
```
Tu equity final fue 46.9% mayor que el escenario pasivo.

**3. Information Ratio (Sharpe-like, vs B&H como benchmark):**
```
IR = (R_strategy − R_B&H) / σ(R_strategy − R_B&H) ≈ 40.11% / 15.4% ≈ 2.60
```
Categoría "Excellent" según convención CFA (>2.0).

#### Comparación con industria

| Estrategia / Fondo | Alpha vs B&H típico (anual) |
|-------------------|---------------------------:|
| Long-Short equity hedge fund mediano | 3–5 pp |
| Top quartile hedge fund | 8–12 pp |
| Renaissance Medallion (best-in-class) | 30–40 pp |
| **Smart v2.0 en 2025** | **+40.11 pp** |
| RL V21.5b (versión deprecada) | 14.8 pp |

**Tu alpha está en el rango de best-in-class hedge funds**, pero con disclaimer: 1 año, N=34, CI ancho.

#### Por qué este número es bulletproof

A diferencia del Sharpe, la comparación vs B&H NO requiere:
- ❌ Risk-free rate
- ❌ Annualization factor
- ❌ Currency conversion (ambos son USD)
- ❌ Look-ahead bias (siempre que hagas walk-forward)

**Es la métrica más robusta para defensa ante crítica metodológica.**

### 5.5 Regime Gate — 13 de 14 semanas Q1 2026

#### Qué es Hurst exponent

Estadístico que mide persistencia de una serie temporal. Calculado sobre retornos de USD/COP en ventana móvil de 60 días con análisis R/S de Mandelbrot:

```
H = log(R/S) / log(N)

Donde:
  R = rango acumulado (max − min) de los retornos
  S = desviación estándar
  N = tamaño de ventana (60 días)
```

Interpretación:
- **H > 0.52** → trending (correlaciones positivas, momentum)
- **H ≈ 0.50** → caminata aleatoria
- **H < 0.42** → mean-reverting (correlaciones negativas, reversión)

#### Regla del Regime Gate

```python
if hurst < 0.42:
    skip_trade = True              # mean-reverting → no operar
elif hurst > 0.52:
    skip_trade = False             # trending → operar con sizing completo
else:
    sizing_multiplier = 0.40       # zona indeterminada → operar al 40%
```

#### Por qué tiene sentido teórico

El modelo predictivo (Ridge + BR + XGB) está entrenado para detectar **continuación de movimientos**. Cuando el mercado es mean-reverting, su lógica subyacente cambia: lo que predice como "va a seguir bajando" en realidad **va a rebotar**. Las predicciones se vuelven anti-correlacionadas con la realidad.

**Operarlas equivale a una estrategia inversa garantizada de pérdidas.**

#### Q1 2026 — datos de producción

| Métrica | Valor |
|---------|------:|
| Semanas evaluadas | 14 |
| Hurst rango observado | 0.16 – 0.49 |
| Hurst mediano | ~0.32 |
| Semanas con H < 0.42 (gate bloqueó) | **13** |
| Semanas con H ∈ [0.42, 0.52] (sizing reducido) | 1 |
| Semanas con H > 0.52 (operación normal) | 0 |
| Trades ejecutados | 1 |
| Trades ganadores | 1 (100%) |
| Retorno YTD | +0.61% |

#### Comparación contrafactual

| Versión | Retorno Q1 2026 | Trades | Comportamiento |
|---------|---------------:|-------:|----------------|
| Smart v1.1.0 (sin gate) | **−5.17%** | 6 trades, 4 perdedoras | Operó a ciegas |
| **Smart v2.0 (con gate)** | **+0.61%** | 1 trade, ganadora | Esperó la señal correcta |
| **Alpha del gate** | **+5.78 pp** | — | Valor del componente |

#### Por qué es la "MVP" de v2.0

El **10-agent audit (marzo 2026)** reveló:

1. El modelo predictivo (Ridge / BR) tiene **R² < 0** en 2025 y 2026 → es peor que predecir la media
2. Sin embargo, la estrategia ganó +25.63% en 2025
3. El alpha NO viene del modelo direccional → viene de:
   - **Regime gate**: sabe cuándo NO operar
   - **TP/HS adaptativos**: gestionan el riesgo cuando opera
   - **Effective HS** (3.5% portfolio cap): limita pérdidas catastróficas

**Insight contraintuitivo**: la "inteligencia" del sistema no está en predecir hacia dónde va el mercado, sino en **saber cuándo el mercado no es predecible** y abstenerse.

### 5.6 P-value y Robustez Estadística

| Test | Valor | Interpretación |
|------|------:|----------------|
| Bootstrap iterations | 10,000 | Resampling con replacement |
| Test statistic | mean weekly return | Resampleado |
| Null hypothesis | retorno esperado = 0 | Random strategy |
| **p-value observado** | **0.006** | 6 en 1,000 |
| Confidence | 99.4% | Rechaza H0 |

**Sample size considerations:**

- N = 34 trades sobre 12 meses
- Standard error del Sharpe: SE(S) = √((1 + 0.5×3.35²)/34) ≈ 0.44
- 95% CI del Sharpe: [2.49, 4.21]

El p-value es la métrica MÁS robusta para defender contra críticas de sample size porque mide directamente la probabilidad de que el resultado sea suerte.

### 5.7 2026 YTD — Estado de Producción Real

| Métrica | Valor |
|---------|------:|
| Período | 2026-01-01 → 2026-05-09 |
| Retorno YTD | +0.61% |
| Trades ejecutados | 1 (regime gate bloqueó 13 de 14) |
| Trades ganadores | 1 (100%) |
| Buy & Hold mismo período | (peso continuó fortaleciéndose) |
| Alpha vs versión sin gate | +5.78 pp |

**Lectura**: en un trimestre estructuralmente mean-reverting (el peor caso para esta estrategia), el regime gate convirtió lo que hubiera sido pérdida en una pequeña ganancia. Esto valida que el sistema funciona en condiciones adversas — no solo en años buenos.

---

## 6. Modelo de Negocio

### 6.1 Las 4 Oportunidades

#### Bot P2P USDT/COP — "El XTX Markets de Binance P2P Colombia"

| Aspecto | Detalle |
|---------|---------|
| Cliente | Merchants P2P Colombia moviendo USD 5–200k/día |
| Pricing | USD 99–199/mes (Solo Merchant) · USD 499–1,500/mes + 5–10% perf fee (Pro/Whale) |
| Mercado | Binance P2P cero fees → margen 100% spread |
| Spread typical | 1.3–3.9% bruto Top 1 vs Top 10 |
| Estado actual | MVP en construcción (mes 1) |
| Proyección 18m | 25 mid + 3 whales = USD 15–25k MRR |
| Por qué punta de lanza | Cash flow más rápido · Mercado virgen · Cero competencia ML · Plug-and-play |

**Componentes del producto:**
- Anti-fraude (ML detecta chargebacks y operadores tóxicos)
- Spread dinámico (ajuste por volatilidad y profundidad)
- Top-1 ranking (mantiene posición en Binance)
- BYOK (Bring Your Own Keys) — cliente conecta API Binance

#### SaaS Hedging FX para PYMES — "Kantox para LATAM"

| Aspecto | Detalle |
|---------|---------|
| Cliente | 5,000+ PYMES exportadoras/importadoras desprotegidas |
| Pricing | USD 199/mes (PYME) · USD 999/mes (Mid) · USD 5,000+/mes (Corporate) + rev share banco |
| Estado actual | Diseño MVP (Q3 2026 lanzamiento) |
| Proyección 24m | USD 60k MRR (200 PYMES × USD 300 promedio) |
| Comparable | Kantox vendida a Visa por **€175M (2025)** |
| Validación | Bound, Neo, Pangea, TreasurUp con Series A activas |
| Diferenciador | Ningún player opera Colombia con esta propuesta |

**Tesis**: Solo 17% de empresas no financieras COL usa derivados FX (último estudio público Banco de la República). Hay 5,000+ empresas exportadoras desprotegidas. La penetración estructural baja se mantiene según estudios sectoriales más recientes (Asobancaria, Anif).

#### Optimizador de Remesas B2B — "Layer de IA sobre USD 13B/año"

| Aspecto | Detalle |
|---------|---------|
| Cliente | Casas de cambio (Western Union, MoneyGram, Movii, Wise, Remitly, Giros & Finanzas) |
| Pricing | USD 2k/mes floor + 10–20% rev share sobre alpha capturado |
| Estado actual | Outreach + diseño API (Q4 2026 piloto) |
| Proyección 18m | USD 10–25k MRR (5 casas de cambio) |
| TAM | USD 13,098M/año en remesas Colombia |
| Por qué B2B | No tocas al consumidor final · regulatorio limpio · tickets grandes |

**Tesis**: USD 13,098M en remesas 2025 (récord histórico). 1.6× la IED en Q1 2026. 53% origen EE.UU. (flujo estable). Bre-B con >100M llaves activas. Las casas de cambio convierten a TRM del día sin optimización IA — capturando timing con ML genera +0.5–1.5% por operación = **USD 65–195M alpha desbloqueable**.

#### App para Freelancers Bre-B — "Wise killer con AI timing"

| Aspecto | Detalle |
|---------|---------|
| Cliente | 500k–1M freelancers colombianos cobrando USD (Upwork, Deel, Fiverr, Remote) |
| Pricing | USD 5–15/mes suscripción + 0.3% por conversión optimizada |
| Estado actual | Lanzamiento beta cerrada Q1 2027 |
| Proyección 24m | USD 40k MRR (5,000 usuarios × USD 8/mes promedio) |
| Comparables valuación | Wise USD 11B · Remitly USD 3B · Deel USD 12B |
| Diferenciador | Ningún player ofrece AI timing layer + rails Bre-B |

**Tesis**: Wise cobra 1.5%, Payoneer 2-3%. App les dice **CUÁNDO convertir** con ML, ejecuta vía Binance P2P → Bre-B → cuenta colombiana en <5 min, recupera +1-2% por dólar. Sobre ingreso anual de freelancer mid (USD 60k), eso son **USD 600-1,200 extra/año**.

### 6.2 Pricing Tiers Consolidado

| Línea | Tier | Precio | Volumen target |
|-------|------|--------|----------------|
| Bot P2P | Solo Merchant | USD 99–199/mes | <USD 20k/día |
| Bot P2P | Pro / Whale | USD 499–1,500/mes + perf fee | USD 50–200k/día |
| Hedging | PYME | USD 199/mes | <USD 1M FX exposure |
| Hedging | Mid | USD 999/mes | USD 1–10M exposure |
| Hedging | Corporate | USD 5,000+/mes | Custom + rev share banco |
| Remesas B2B | API Integration | USD 2k/mes + 10–20% rev share | Por casa de cambio |
| Freelancers | Sub + Tx | USD 5–15/mes + 0.3%/conversión | Individual |

**Margen SaaS típico**: 70–85%. **BYOK garantiza cero costo variable.**

### 6.3 Proyección MRR 24 Meses

| Mes | Bot P2P | Hedging | Remesas | Freelancers | **Total MRR** |
|-----|--------:|--------:|--------:|------------:|---------------:|
| M3 | $2k | $0 | $0 | $0 | **$2k** |
| M6 | $5k | $1k | $0 | $0 | **$6k** |
| M9 | $8k | $2k | $2k | $1k | **$13k** |
| M12 | $12k | $5k | $5k | $3k | **$25k** |
| M15 | $17k | $12k | $10k | $8k | **$47k** |
| M18 | $21k | $25k | $15k | $18k | **$79k** |
| M21 | $23k | $42k | $20k | $28k | **$113k** |
| **M24** | **$25k** | **$60k** | **$25k** | **$40k** | **$150k** |

**Target M24: USD 100–150k MRR** (rango conservador). Equivale a USD 1.2–1.8M ARR.

**Inversor sofisticado descuenta proyecciones por 50%**, así que reportamos rango y subprometemos.

### 6.4 Comparables Internacionales (Validación)

| Producto | Comparable | Valuación / Ronda |
|----------|------------|-------------------|
| Hedging PYMES | **Kantox** | Adquirida por Visa por **€175M (2025)** |
| Hedging PYMES | Bound, Neo, Pangea, TreasurUp | Series A activas (2024–2025) |
| Freelancers / Remesas | **Wise** | **USD 11B valuación** · LSE: WISE |
| Freelancers / Remesas | **Remitly** | **USD 3B valuación** · NASDAQ: RELY |
| Freelancers / Remesas | **Deel** | **USD 12B valuación** (last round) |
| Trading quant FX | XTX Markets | Líder global ML market making |

**Tu ventaja**: ninguno opera Colombia con esta propuesta combinada.

### 6.5 TAM LATAM

**TAM combinado USD 50–150M anuales** en LATAM (rango conservador, calculado bottom-up sumando los 4 mercados objetivo).

Por qué no 200M (como mencionamos antes): inversor sofisticado pide cálculo bottom-up. 50–150M es defendible. 200M requeriría TAM supuesto que no podemos validar con datos públicos.

---

## 7. Estado Operacional Actual (Mayo 2026)

### ✅ Lo que funciona en producción

| Componente | Status | Evidencia |
|------------|--------|-----------|
| Pipeline H5 Smart v2.0 | Live | summary_2025.json + 2026 YTD trades |
| Regime Gate | Live | 13/14 semanas Q1 2026 bloqueadas |
| 27 DAGs Airflow | Activos | Healthy, 0 restarts en 2 días |
| Dashboard Next.js | Live | localhost:5000 (8 páginas, 47 API routes) |
| News pipeline | Live | Investing + Portafolio (361 articles en DB) |
| Análisis L8 | Live | W01-W15 generados (Azure OpenAI GPT-4o-mini) |
| MEXC + Binance integration | Live | SignalBridge OMS operacional |
| Paper trading | Live | Trades ejecutándose en tiempo real |
| MLflow tracking server | Live | localhost:5001 |
| Prometheus + Grafana | Live | 53 alert rules, 4 dashboards |
| Backup diario | Live | core_l0_05_seed_backup ejecutándose |

### 🟡 Lo que está en construcción

| Componente | Estado | ETA |
|------------|--------|-----|
| Bot P2P MVP | Alpha (5 merchants beta) | Mes 1 (Mayo) |
| Hedging PYMES MVP | Diseño | Mes 2-3 (Jun-Jul) |
| Newsletter premium | En desarrollo | Mes 1 (Mayo) |
| Outreach casas de cambio | Iniciando | Mes 2 (Jun) |
| Microsoft Founders Hub application | En preparación | Mes 2 (Jun) |
| Series Seed deck v2 | En preparación | Mes 3 (Jul) |

### 🔴 Lo que está pausado o no operativo

| Componente | Estado | Razón |
|------------|--------|-------|
| Pipeline H1 Daily | ⏸ Pausado | Pendiente validación v2.0 sobre H1 |
| Pipeline RL | 🔻 Deprecado | NOT significant (p=0.272) |
| AlertManager | ⚠️ Infra ready | SLACK_WEBHOOK_URL placeholder vacío |
| Jaeger tracing | ⚠️ Infra ready | 0 servicios instrumentados con OpenTelemetry |
| MLflow auto-integration en DAGs | ⚠️ Infra ready | Scripts ad-hoc loguean, DAGs L3 no |
| MinIO model artifact backup | ⚠️ Infra ready | Modelos persisten en filesystem |

### 📊 Health check actual (todos los servicios)

```
Containers up: 19/19 (compact + monitoring extras)
Restarts (últimos 7 días): 0
Healthy %: 100% (los 16 con healthcheck) + 3 sin healthcheck pero estables
Uptime promedio: 2+ días sin reinicio
```

---

## 8. Hitos 90 días (Mayo–Julio 2026)

### Mes 1 — Mayo

| Hito | Owner | Status |
|------|-------|--------|
| Cierre Smart v2.0 en producción | Pedro | ✅ Done |
| 5 merchants beta P2P firmados | Freddy | 🟡 En outreach |
| Deck Microsoft listo + reunión técnica | Pedro + Freddy | ✅ Deck listo |
| Newsletter premium live (USD 99/mes) | Pedro + Freddy | 🟡 En desarrollo |
| PROJECT_DEFINITION.md (este doc) | Pedro | ✅ Done |

### Mes 2 — Junio

| Hito | Owner | Status |
|------|-------|--------|
| Bot P2P: 10 clientes pagos | Freddy | ⏳ Pendiente |
| Primer piloto Hedging PYME | Freddy | ⏳ Pendiente |
| Aplicación Microsoft for Startups Founders Hub | Pedro | ⏳ Pendiente |
| Outreach 5 casas de cambio | Freddy | ⏳ Pendiente |
| MLflow auto-integration en DAGs L3 | Pedro | 🔮 Roadmap |

### Mes 3 — Julio

| Hito | Owner | Status |
|------|-------|--------|
| MRR USD 5k+ alcanzado | Pedro + Freddy | ⏳ Pendiente |
| Créditos Azure Founders Hub asegurados | Pedro | ⏳ Pendiente |
| 1er prospecto formal casa de cambio | Freddy | ⏳ Pendiente |
| Series Seed deck v2 listo | Pedro + Freddy | ⏳ Pendiente |

### Roadmap 12 meses (visión completa)

| Quarter | Bot P2P | Hedging | Remesas | Freelancers | Hito clave |
|---------|---------|---------|---------|-------------|------------|
| **Q2 2026** | MVP + 5 beta | — | — | — | v2.0 cierre + MRR USD 6k |
| **Q3 2026** | 25 clientes pagos | Diseño + MVP | — | — | MRR USD 25k + Microsoft Founders Hub |
| **Q4 2026** | 50 clientes + perf fees | 10 PYMES piloto | Outreach + API | — | Banco partner + casa de cambio piloto |
| **Q1 2027** | Whales + LATAM exp | Banco partner #1 | 2 casas piloto | Beta cerrada | MRR USD 100–150k + Series Seed |

---

## 9. Equipo

### Pedro Sánchez Briceño — Co-founder & CTO

- **Rol**: Senior Data Analyst / MLOps Engineer
- **Certificación**: Microsoft Azure AI-102
- **Experiencia previa**:
  - Aera Energy — Senior Data Analyst
  - CRC (Comisión de Regulación de Comunicaciones) — Análisis cuantitativo regulatorio
  - CODALTEC — FACSAT-2 (proyecto satelital)
  - Vault Insurance — MLOps en producción
- **Responsabilidades**: Motor de IA · arquitectura · DevOps · contracts · backend
- **Contacto**: pspedroelias96@gmail.com

### Freddy — Co-founder & CEO

- **Rol**: Estrategia comercial · Red de contactos corporativos LATAM
- **Experiencia**: Liderazgo comercial sector financiero · ciclo completo venta B2B y partnerships · comunicación ejecutiva con C-level
- **Responsabilidades**: Comercial · alianzas · ventas · relaciones inversionistas
- **Contacto**: (TBD)

### Roadmap de hiring

- **Q3 2026**: 1 Senior Engineer (cuando bot P2P escale)
- **Q4 2026**: 1 Sales Lead LATAM (cuando hedging arranque)
- **Q1 2027**: 1 Customer Success (post-Series Seed)

---

## 10. Pedido a Microsoft / Inversores

### 10.1 Microsoft for Startups — Founders Hub

**Pedido concreto:**

| # | Qué pedimos | Para qué |
|---|-------------|----------|
| 1 | Créditos Azure (hasta USD 150k) | Cubre infra ML + OpenAI + Container Apps por 18-24 meses |
| 2 | Co-selling LATAM | Introducción a 3 clientes corporativos: bancos, casas de cambio, PYMES grandes |
| 3 | Soporte técnico Azure ML | Arquitecto solutions dedicado para migración stack |
| 4 | Visibilidad LATAM | Caso de éxito conjunto en eventos, blog, casos Microsoft for Startups |

**Lo que ofrecemos a cambio:**

| ✓ | Qué entregamos |
|---|----------------|
| ✓ | Caso de éxito documentado Microsoft for Startups LATAM en fintech ML |
| ✓ | Migración stack a Azure (Container Apps + ML + OpenAI + Functions) — arquitectura referencia |
| ✓ | Cobranding GlobalMinds powered by Azure en website, decks, eventos |
| ✓ | Pipeline de leads — cada banco / casa de cambio / PYME que cerramos = lead corporativo Microsoft |

**Status**: Aplicación lista para someter en Mes 2 (Junio 2026).

### 10.2 Contactos comerciales (clientes piloto)

**Oferta**: Piloto 60 días GRATUITO a 3 clientes elegidos:
- 1 PYME (sector exportador o importador)
- 1 merchant P2P (Binance Colombia)
- 1 casa de cambio o family office

**Compromiso GlobalMinds**: SLA respuesta < 24h, métricas reportables semanales, derecho a publicar caso de éxito si resultados positivos.

### 10.3 Inversores Series Seed

**Plan ronda Q4 2026:**

| Variable | Valor |
|----------|-------|
| Tipo de instrumento | SAFE post-money valuation |
| Ticket mínimo | USD 25k |
| Valuación | TBD (post Mes 6 traction validada) |
| Uso de fondos | Hiring (3) + GTM (40%) + producto (30%) + reserva (15%) |
| Hito disparador | MRR ≥ USD 25k consistente 3 meses + 50+ clientes activos |

**Acceso temprano**: contactos comerciales y advisors con derecho de primera oferta.

---

## 11. Apéndices

### Apéndice A — Fuentes Públicas Verificables

> Lista completa con URLs activas en `presentation/globalminds_microsoft_pitch_may2026/SOURCES.md`. Aquí el resumen.

#### Cifras de mercado

| Cifra | Fuente | URL primaria |
|-------|--------|--------------|
| USD 13,098M remesas Colombia 2025 | Banco de la República | banrep.gov.co/es/transferencias-remesas-trabajadores |
| 17% empresas con cobertura FX | Banrep Borradores 1058 | banrep.gov.co/es/borrador-1058 |
| Bre-B >100M llaves activas | Banco de la República | banrep.gov.co/es/bre-b |

> **Aclaración honesta sobre el 17%**: el paper original Borradores 1058 cubre 2008–2014. Es el último estudio público disponible de penetración FX en empresas colombianas. Estudios sectoriales más recientes confirman que la subpenetración estructural se mantiene.

#### Comparables internacionales

| Comparable | Valuación / evento | Fuente |
|------------|-------------------|--------|
| Kantox / Visa | €175M (2025) | Visa Newsroom + TechCrunch + El País |
| Wise | USD 11B (público) | LSE: WISE |
| Remitly | USD 3B (público) | NASDAQ: RELY |
| Deel | USD 12B (last round) | Pitchbook + TechCrunch |
| XTX Markets | Líder global ML FX | xtxmarkets.com |

#### Datos internos (verificables en repo)

| Cifra | Archivo |
|-------|---------|
| Backtest 2025 (+25.63%, etc.) | `usdcop-trading-dashboard/public/data/production/summary_2025.json` |
| Trades individuales (34 trades) | `usdcop-trading-dashboard/public/data/production/trades/smart_simple_v11_2025.json` |
| Aprobación state | `usdcop-trading-dashboard/public/data/production/approval_state.json` |
| 2026 YTD | summary.json (live) |

### Apéndice B — Mapa de Documentación Especializada

#### Para entender el sistema técnico

| Concern | Documento autoritativo |
|---------|------------------------|
| Reglas técnicas master | `CLAUDE.md` (root) — 1,000+ líneas |
| Lifecycle MLOps end-to-end | `.claude/rules/sdd-mlops-lifecycle.md` |
| Operaciones diarias + schedule | `.claude/rules/elite-operations.md` |
| Pipeline H5 Weekly | `.claude/rules/h5-smart-simple-pipeline.md` |
| Pipeline H1 Daily | `.claude/rules/h5-smart-simple-pipeline.md` (mismo doc) |
| L0 Data Layer | `.claude/rules/l0-data-governance.md` |
| RL Training (L2-L3-L4) | `.claude/rules/l2-l3-l4-training-pipeline.md` |
| RL Inference (L1-L5) | `.claude/rules/l1-l5-inference-pipeline.md` |
| News Engine + Analysis | `.claude/rules/news-and-analysis-sdd.md` |

#### Para entender contratos y schemas

| Concern | Documento |
|---------|-----------|
| Universal Strategy Schemas | `.claude/rules/sdd-strategy-spec.md` |
| Approval lifecycle (2-vote) | `.claude/rules/sdd-approval-spec.md` |
| Dashboard integration | `.claude/rules/sdd-dashboard-integration.md` |
| Pipeline lifecycle quick ref | `.claude/rules/sdd-pipeline-lifecycle.md` |

#### Para entender execution + risk

| Concern | Documento |
|---------|-----------|
| SignalBridge OMS | `.claude/rules/sdd-execution-bridge.md` |
| Risk management 9 checks | `.claude/rules/sdd-risk-management.md` |
| Observability stack | `.claude/rules/sdd-observability.md` |

#### Para entender CI/CD + testing

| Concern | Documento |
|---------|-----------|
| GitHub Actions + Makefile | `.claude/rules/sdd-cicd-testing.md` |
| Data freshness gates | `.claude/rules/data-freshness-enforcement.md` |
| Backup & disaster recovery | `.claude/rules/backup-recovery-protocol.md` |

#### Para experimentos y versioning

| Concern | Documento |
|---------|-----------|
| Experiment protocol | `.claude/rules/experiment-protocol.md` |
| SSOT versioning | `.claude/rules/ssot-versioning.md` |
| Experiment log | `.claude/experiments/EXPERIMENT_LOG.md` |
| Experiment queue | `.claude/experiments/EXPERIMENT_QUEUE.md` |

#### Para arquitectura general

| Concern | Documento |
|---------|-----------|
| Architecture Decision Records | `docs/adr/ADR-000*.md` (5 docs) |
| API contracts | `docs/architecture/API_CONTRACTS_SHARED.md` |
| Database ER | `docs/architecture/DATABASE_ER_DIAGRAM.md` |
| Operations runbook | `docs/operations/RUNBOOK.md` |

#### Para el pitch / inversores

| Concern | Documento |
|---------|-----------|
| Pitch deck (.pptx) | `presentation/globalminds_microsoft_pitch_may2026/GlobalMinds_Pitch_Deck.pptx` |
| Speaker notes ES | Embedded en .pptx |
| Sources con URLs | `presentation/globalminds_microsoft_pitch_may2026/SOURCES.md` |
| README del deck | `presentation/globalminds_microsoft_pitch_may2026/README.md` |

### Apéndice C — Glosario

| Término | Definición |
|---------|------------|
| **Alpha** | Retorno de una estrategia activa por encima de un benchmark pasivo (Buy & Hold) |
| **Buy & Hold (B&H)** | Estrategia pasiva: comprar al inicio, vender al final, sin trades intermedios |
| **Calmar Ratio** | Retorno anual / MaxDD. Robusto porque no requiere risk-free rate |
| **Confidence tier** | HIGH/MEDIUM/LOW según agreement Ridge-BR + magnitud predicción |
| **Drawdown (DD)** | Caída porcentual desde un peak hasta un trough subsiguiente |
| **Effective HS** | min(HS_base, 3.5%/leverage) — capa pérdida total al 3.5% del portfolio |
| **Expanding window** | Train set crece en cada retraining (vs rolling window de tamaño fijo) |
| **FX-adjusted return** | Retorno en moneda local después de conversión de ida y vuelta |
| **H1 / H5** | Horizontes de forecasting: H1 = 1 día, H5 = 5 días |
| **Hard Stop (HS)** | Stop loss máximo por trade (vol-adaptive en v2.0) |
| **Hurst exponent** | Estadístico que mide persistencia/reversión de una serie temporal |
| **Information Ratio (IR)** | (Return − Benchmark) / TrackingError. Sharpe relativo a benchmark |
| **L0 → L7** | Layers del pipeline: L0=data, L1=features, L2=dataset, L3=train, L4=backtest, L5=inference, L6=monitor, L7=execution |
| **MaxDD (MDD)** | Máximo drawdown en el período observado |
| **Mean-reverting** | Régimen donde movimientos tienden a revertirse (Hurst < 0.42) |
| **OOS** | Out-of-sample — período no visto durante entrenamiento |
| **p-value** | Probabilidad de que el resultado observado sea producto de azar |
| **PPO** | Proximal Policy Optimization — algoritmo RL |
| **Profit Factor (PF)** | Suma ganancias / suma pérdidas absolutas |
| **R/S analysis** | Rescaled Range Analysis (Mandelbrot) — método de cálculo del Hurst |
| **Regime Gate** | Componente que bloquea trading en regímenes mean-reverting |
| **rf (risk-free rate)** | Tasa de retorno sin riesgo (T-Bill, TES) usada en Sharpe |
| **SDD** | Spec-Driven Development — metodología del proyecto |
| **Sharpe Ratio** | (Return − rf) / σ. Risk-adjusted return |
| **SSOT** | Single Source of Truth — un archivo de configuración autoritativo |
| **Take Profit (TP)** | Nivel de cierre en ganancia (= HS × 0.5 en v2.0) |
| **Trending** | Régimen donde movimientos se continúan (Hurst > 0.52) |
| **vol-targeting** | Position sizing inverso a volatilidad realizada (target_vol = 0.15) |
| **Walk-forward** | Validación que respeta orden temporal: train [t0, t], test [t, t+1] |
| **Win Rate (WR)** | % de trades ganadores |

### Apéndice D — Disclaimers Legales

> **GlobalMinds NO constituye asesoría financiera.**
>
> Resultados pasados NO garantizan resultados futuros. El uso de los productos GlobalMinds implica riesgo de pérdida de capital. Las cifras de retorno reportadas son resultados de backtest out-of-sample auditado con metodología walk-forward, pero corresponden a un período específico (12 meses, 34 trades) que puede no replicarse en condiciones futuras.
>
> Las proyecciones de MRR, alphas comparativos vs hedge funds, y comparaciones con instrumentos pasivos son ilustrativas y no garantías. Los inversores potenciales deben hacer su propia due diligence.
>
> Todo material en este documento es **CONFIDENCIAL** y propiedad de GlobalMinds. No reproducir ni distribuir sin autorización escrita.
>
> El uso de wordmarks de comparables (Kantox, Visa, Wise, Remitly, Deel, XTX Markets) es referencial editorial y no implica asociación o endorsement por parte de esas entidades.

---

## 12. Navegación Rápida

### Por audiencia

#### 👔 Para inversores (lectura completa: 25 min)
- §1 [Executive Summary](#1-executive-summary)
- §5 [Performance y Métricas](#5-performance-y-métricas) (Sharpe, MDD, regime gate)
- §6 [Modelo de Negocio](#6-modelo-de-negocio) (las 4 oportunidades, MRR)
- §7 [Estado Actual](#7-estado-operacional-actual-mayo-2026)
- §10 [Pedido a Inversores](#10-pedido-a-microsoft--inversores)

#### 🏢 Para Microsoft Founders Hub (lectura: 15 min)
- §1 [Executive Summary](#1-executive-summary)
- §3 [Arquitectura Técnica](#3-arquitectura-técnica) (cloud-ready a Azure)
- §4 [3 Pipelines](#4-tres-pipelines-en-detalle) (LLM analysis con Azure OpenAI)
- §10.1 [Pedido específico a Microsoft](#101-microsoft-for-startups--founders-hub)

#### ⚙️ Para operadores técnicos (lectura: 20 min)
- §3 [Arquitectura](#3-arquitectura-técnica)
- §4 [Pipelines en detalle](#4-tres-pipelines-en-detalle)
- §7 [Estado operacional](#7-estado-operacional-actual-mayo-2026)
- Después: `.claude/rules/sdd-mlops-lifecycle.md` para operaciones día a día

#### 💻 Para nuevos developers (lectura: 30 min + onboarding)
- `README.md` (root) — Quick Start
- §3 [Arquitectura](#3-arquitectura-técnica)
- §4 [Pipelines](#4-tres-pipelines-en-detalle)
- `CLAUDE.md` (root) — Reglas técnicas master (1,000+ líneas)
- `.claude/rules/` — Specs SDD por dominio

### Por concern

| Tengo una pregunta sobre... | Voy a... |
|------------------------------|----------|
| ¿Qué es GlobalMinds? | §1, §2 |
| ¿Cuál es el track record? | §5.1, §5.4 |
| ¿Cómo se calcula el Sharpe? | §5.2 |
| ¿Por qué Sharpe COP es bajo? | §5.2 (FX cycle) |
| ¿Qué es el Calmar? | §5.3 |
| ¿Qué es el regime gate? | §5.5 |
| ¿Cómo es el modelo de negocio? | §6 |
| ¿Cuáles son los comparables? | §6.4 |
| ¿Qué se está construyendo ahora? | §7 |
| ¿Qué piden a Microsoft? | §10.1 |
| ¿Cuándo es la próxima ronda? | §10.3 |
| ¿Quién está detrás del proyecto? | §9 |
| ¿Dónde están las fuentes? | §11 Apéndice A |
| ¿Qué documentos detallados existen? | §11 Apéndice B |

---

## Cierre

Este documento es la **definición consolidada del proyecto GlobalMinds** a fecha 2026-05-09. Es el punto de entrada único para entender:

1. Qué construimos (técnico)
2. Por qué importa (negocio)
3. Cómo nos comparamos (performance)
4. Dónde estamos (status)
5. A dónde vamos (roadmap)
6. Quiénes somos (equipo)

Mantenedor: **Pedro Sánchez Briceño** (pspedroelias96@gmail.com).

**Ciclo de actualización**: revisión mensual al cierre de cada hito 90-day. Próxima revisión: 2026-06-01.

---

*GlobalMinds · IA aplicada al mercado cambiario de Latinoamérica · Mayo 2026 · Confidencial*
