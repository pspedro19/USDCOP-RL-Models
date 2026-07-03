# SOURCES — GlobalMinds Pitch Deck (Mayo 2026)

> Lista completa de fuentes públicas verificables para cada cifra y comparable
> citados en `GlobalMinds_Pitch_Deck.pptx`.
>
> Última verificación: 2026-05-01.

---

## 1. Cifras de mercado Colombia

### Remesas Colombia — USD 13,098 M en 2025 (récord histórico)
- **Fuente primaria**: Banco de la República — Estadísticas externas
- URL: https://www.banrep.gov.co/es/estadisticas/balanza-pagos
- URL alterna: https://www.banrep.gov.co/es/transferencias-remesas-trabajadores
- Slides citantes: 2, 3, 9, 13

### 17% de empresas no financieras usa cobertura FX
- **Fuente primaria**: Banco de la República — Borradores de Economía No. 1058
- URL: https://www.banrep.gov.co/es/borrador-1058
- **Aclaración honesta**: el paper original cubre el período 2008–2014. Es el último estudio público de penetración de derivados FX en empresas colombianas. Estudios sectoriales más recientes (Asobancaria, Anif) confirman que la tendencia se mantiene baja vs economías OECD.
- Slides citantes: 2, 9
- **Defensa en speaker notes**: "Es el dato más reciente publicado por Banrep. Reconocemos que el universo puede haber cambiado, pero la subpenetración estructural se mantiene."

### Bre-B — >100M de llaves activas
- **Fuente primaria**: Banco de la República — Sistema de Pagos
- URL: https://www.banrep.gov.co/es/bre-b
- URL alterna: https://www.banrep.gov.co/es/sistemas-pago
- Slides citantes: 9, 13

### Spread y volúmenes Binance P2P Colombia
- **Fuente**: Binance P2P página pública (snapshot de mercado en vivo)
- URL: https://p2p.binance.com/es-LA/trade/all-payments/USDT?fiat=COP
- **Calculadoras públicas que validan rangos**: MetaReporte, El Dorado calculadora
- **Aclaración**: el rango 1.3–3.9% (Top 1 vs Top 10) es observación recurrente de mercado, no número oficial de Binance. Vol diario por merchant top (USD 50–200k) es estimación validada con operadores activos.
- Slides citantes: 1, 2, 9, 13

---

## 2. Comparables internacionales

### Kantox — adquirida por Visa por €175M (2025)
- **Fuente primaria**: Visa Newsroom — anuncio oficial de adquisición (2025)
- URL: https://corporate.visa.com/en/sites/visa-perspectives.html
- Cobertura: TechCrunch, Diario El País Negocios (España)
- Slides citantes: 8, 10
- Búsqueda directa: "Visa acquires Kantox 175 million"

### Bound · Neo · Pangea · TreasurUp — Series A activas
- **Fuentes**:
  - Bound — Crunchbase, TechCrunch
  - Neo — Crunchbase (€5M Series A confirmada)
  - Pangea — sitio corporativo + cobertura PR
  - TreasurUp — partnership con Sparkassen Group (Alemania) — sitio corporativo
- URL agregadora: https://www.crunchbase.com/discover/saved-search/list/fx-hedging-fintech-saas
- Slides citantes: 10

### Wise — USD 11B valuación pública
- **Fuente primaria**: London Stock Exchange (ticker: WISE)
- URL: https://www.londonstockexchange.com/stock/WISE/wise-plc/company-page
- URL alterna (relaciones inversores): https://wise.com/gb/about/investor-relations
- Slides citantes: 10

### Remitly — USD 3B valuación pública
- **Fuente primaria**: NASDAQ (ticker: RELY)
- URL: https://www.nasdaq.com/market-activity/stocks/rely
- URL alterna: https://ir.remitly.com
- Slides citantes: 10

### Deel — USD 12B valuación última ronda
- **Fuente**: Pitchbook · TechCrunch · Forbes (cobertura de la ronda Series D ampliada / secundaria)
- URL ejemplo: https://techcrunch.com/?s=deel+12+billion
- Slides citantes: 10

### XTX Markets — líder global ML en FX market making
- **Fuente**: sitio corporativo + Financial Times research
- URL: https://www.xtxmarkets.com
- Slides citantes: 10

---

## 3. Tracción y datos internos GlobalMinds

### Backtest 2025 — +25.63% / Sharpe 3.35 / p-value 0.006
- **Fuente**: walk-forward backtest interno, modelo Smart Simple v2.0
- Archivo: `usdcop-trading-dashboard/public/data/production/summary_2025.json`
- Período: 2025-01 a 2025-12 (out-of-sample)
- Train window: 2020-01 a 2024-12 (expanding)
- Trades: 34 (5 LONG · 29 SHORT)
- Win rate: 82.4% · Profit factor: 2.756 · Max DD: 6.12%
- **Buy & Hold mismo período**: −14.48% (línea roja del chart)
- Slides citantes: 1, 4, 7, 14, 21

### Regime Gate — 13 de 14 semanas mean-reverting bloqueadas Q1 2026
- **Fuente**: log de producción `forecast_h5_signals` table (Smart v2.0 live)
- Hurst exponent calculado en 60-day rolling window
- Threshold mean-reverting: H < 0.42
- Slides citantes: 4, 9, 21

### Dashboard — 8 páginas / 47 API routes / 27 DAGs Airflow
- **Fuente**: inventario interno del repositorio
- Slides citantes: 6

---

## 4. Mercado de freelancers cobrando USD

### 500k–1M freelancers colombianos
- **Fuente**: estimación basada en datos de plataformas (Upwork, Deel, Fiverr) reportes públicos de país
- Crossreferencia: Upwork annual report (Colombia top 10 LATAM)
- **Honestidad**: rango aproximado, no cifra oficial DANE — se presenta como rango por esa razón
- Slides citantes: 3, 8

### Wise cobra 1.5%, Payoneer 2–3% en conversiones USD→COP
- **Fuente**: tarifarios públicos Wise (https://wise.com/es/pricing) y Payoneer
- Slides citantes: 9

---

## 5. Microsoft for Startups Founders Hub

### Hasta USD 150,000 en créditos Azure
- **Fuente primaria**: Microsoft for Startups oficial
- URL: https://www.microsoft.com/en-us/startups
- Slides citantes: 18

---

## Aclaraciones generales de buena fe

1. **Proyecciones financieras** (slides 8, 11, 14): son **escenarios conservadores** basados en pricing comparable y tasas de adopción típicas de SaaS B2B en LATAM. NO son garantía de retorno. El deck disclaimer en slide 22 es vinculante.

2. **TAM USD 50–150M LATAM**: rango calculado bottom-up sumando tamaños de los 4 mercados objetivo. Estimación con incertidumbre normal — por eso es rango y no número único.

3. **Logos / wordmarks de comparables** (slide 10): se usaron wordmarks tipográficos en lugar de logos PNG para evitar zona gris legal de uso comercial sin licencia. Uso editorial/referencial.

4. **Speaker notes con datos blindados**: las notas del orador en cada slide incluyen formulaciones cuidadosas para defender cada cifra ante preguntas de inversores sofisticados.

---

*Documento mantenido por: Pedro Sánchez Briceño. Última actualización: 2026-05-01.*
