# FINAL SKILLS — Stack de Quantitative Trading

Colección curada de **119 skills** de trading cuantitativo, consolidados desde 5 repos y organizados por etapa del ciclo de vida de una estrategia.

> **Alcance:** quant + infraestructura de mercado. Se excluyeron los skills de ventas/GTM, planificación patrimonial personal (estate, seguros, deuda, jubilación) y compliance de asesor que no tocan el flujo de trading.

---

## Cómo se usa

Los 12 directorios siguen el orden real del ciclo de vida. Si no sabes por dónde empezar, arranca con
[`12-meta-skill-tooling/trading-skills-navigator`](12-meta-skill-tooling/trading-skills-navigator/) — traduce un objetivo en lenguaje natural al workflow correcto.

### Workflow diario (swing/momentum)

```
01 regimen  ──►  02 screening  ──►  05 sizing  ──►  07 ejecucion  ──►  09 performance
  ¿opero?         candidatos        cuanto           orden              que aprendi
```

1. `01-market-regime/market-breadth-analyzer` + `exposure-coach` → techo de exposición del día
2. `02-screening-signals/vcp-screener` o `stockbee-momentum-burst-screener` → candidatos
3. `05-risk-position-sizing/pre-trade-discipline-gate` → ¿pasa el checklist?
4. `05-risk-position-sizing/position-sizer` → tamaño
5. `07-execution-brokerage/ibkr-api` → orden
6. `09-performance-analytics/trader-memory-core` → registra la tesis

### Workflow de investigación (construir una estrategia nueva)

```
03 edge research  ──►  04 backtest  ──►  06 cartera  ──►  09 atribucion
```

`edge-pipeline-orchestrator` encadena el pipeline completo: hint → concepto → diseño → revisión → export.
Luego `backtest-expert` valida, y `strategy-pivot-designer` interviene si el tuning se estanca en un óptimo local.

### Cadena contrarian (Jason Shapiro)

`cot-contrarian-detector` → `news-reaction-failure-analyzer` → `contrarian-setup-gate`

---

## Catálogo

### `01-market-regime` — Regimen de mercado  *(14)*

Responde *"se puede operar hoy?"*. Breadth, tendencia, techos, burbujas, rotacion sectorial.

| Skill | Que hace |
|---|---|
| [`breadth-chart-analyst`](01-market-regime/breadth-chart-analyst/) | This skill should be used when analyzing market breadth charts, specifically the S&P 500 Breadth Index (200-Day MA based) and the US Stock Market... |
| [`crypto-regime-analyzer`](01-market-regime/crypto-regime-analyzer/) | Quantifies crypto market regime health using free, keyless public data (CoinGecko + Binance funding). Generates a 0-100 composite score across 6... |
| [`downtrend-duration-analyzer`](01-market-regime/downtrend-duration-analyzer/) | Analyze historical downtrend durations and generate interactive HTML histograms showing typical correction lengths by sector and market cap. |
| [`exposure-coach`](01-market-regime/exposure-coach/) | Generate a one-page Market Posture summary with net exposure ceiling, growth-vs-value bias, participation breadth, and new-entry-allowed vs... |
| [`ftd-detector`](01-market-regime/ftd-detector/) | Detects Follow-Through Day (FTD) signals for market bottom confirmation using William O'Neil's methodology. Dual-index tracking (S&P 500 + NASDAQ)... |
| [`ibd-distribution-day-monitor`](01-market-regime/ibd-distribution-day-monitor/) | Detect IBD-style Distribution Days for QQQ/SPY (close down at least 0.2% on higher volume), track 25-session expiration and 5% invalidation, count... |
| [`macro-regime-detector`](01-market-regime/macro-regime-detector/) | Detect structural macro regime transitions (1-2 year horizon) using cross-asset ratio analysis. Analyze RSP/SPY concentration, yield curve, credit... |
| [`market-breadth-analyzer`](01-market-regime/market-breadth-analyzer/) | Quantifies market breadth health using TraderMonty's public CSV data. Generates a 0-100 composite score across 6 components (100 = healthy). No... |
| [`market-environment-analysis`](01-market-regime/market-environment-analysis/) | Comprehensive market environment analysis and reporting tool. Analyzes global markets including US, European, Asian markets, forex, commodities,... |
| [`market-top-detector`](01-market-regime/market-top-detector/) | Detects market top probability using O'Neil Distribution Days, Minervini Leading Stock Deterioration, and Monty Defensive Sector Rotation.... |
| [`sector-analyst`](01-market-regime/sector-analyst/) | This skill should be used when analyzing sector rotation patterns and market cycle positioning. It fetches sector uptrend data from CSV (no API... |
| [`theme-detector`](01-market-regime/theme-detector/) | Detect and analyze trending market themes across sectors. Use when user asks about current market themes, trending sectors, sector rotation,... |
| [`uptrend-analyzer`](01-market-regime/uptrend-analyzer/) | Analyzes market breadth using Monty's Uptrend Ratio Dashboard data to diagnose the current market environment. Generates a 0-100 composite score... |
| [`us-market-bubble-detector`](01-market-regime/us-market-bubble-detector/) | Evaluates market bubble risk through quantitative data-driven analysis using the revised Minsky/Kindleberger framework v2.1. Prioritizes objective... |

### `02-screening-signals` — Screening y senales  *(21)*

Genera candidatos. Screeners de setup, planificadores de trade, deteccion contrarian.

| Skill | Que hace |
|---|---|
| [`breakout-trade-planner`](02-screening-signals/breakout-trade-planner/) | Generate Minervini-style breakout trade plans from VCP screener output with worst-case risk calculation, portfolio heat management, and... |
| [`canslim-screener`](02-screening-signals/canslim-screener/) | Screen US stocks using William O'Neil's CANSLIM growth stock methodology. Use when user requests CANSLIM stock screening, growth stock analysis,... |
| [`contrarian-setup-gate`](02-screening-signals/contrarian-setup-gate/) | Synthesize the three Jason Shapiro contrarian-pipeline verdicts (COT crowding, news-reaction failure, weekly price-action confirmation) into one... |
| [`cot-contrarian-detector`](02-screening-signals/cot-contrarian-detector/) | Detect crowded speculative positioning in CFTC futures markets (COT report analysis) to find contrarian setups using Jason Shapiro's methodology.... |
| [`crypto-trading-signals`](02-screening-signals/crypto-trading-signals/) | Fetches live AI crypto trading signals with entry price, stop-loss, take-profit, leverage, confidence scores, and automated verification. Covers... |
| [`dividend-growth-pullback-screener`](02-screening-signals/dividend-growth-pullback-screener/) | Use this skill to find high-quality dividend growth stocks (12%+ annual dividend growth, 1.5%+ yield) that are experiencing temporary pullbacks,... |
| [`earnings-trade-analyzer`](02-screening-signals/earnings-trade-analyzer/) | Analyze recent post-earnings stocks using a 5-factor scoring system (Gap Size, Pre-Earnings Trend, Volume Trend, MA200 Position, MA50 Position).... |
| [`finviz-screener`](02-screening-signals/finviz-screener/) | Build and open FinViz screener URLs from natural language requests. Use when user wants to screen stocks, find stocks matching criteria, filter by... |
| [`institutional-flow-tracker`](02-screening-signals/institutional-flow-tracker/) | Use this skill to track institutional investor ownership changes and portfolio flows using 13F filings data. Analyzes hedge funds, mutual funds,... |
| [`multi-asset-trading-signals`](02-screening-signals/multi-asset-trading-signals/) | Expert trading partner for Options, Stocks, Crypto, Commodities, Gold, Silver, Oil, VIX, and Forex. Covers technical analysis (Elliott Wave,... |
| [`news-reaction-failure-analyzer`](02-screening-signals/news-reaction-failure-analyzer/) | Judge whether a market FAILED to react to news favorable to a crowded speculative position — step 2 of Jason Shapiro's COT contrarian process.... |
| [`pair-trade-screener`](02-screening-signals/pair-trade-screener/) | Statistical arbitrage tool for identifying and analyzing pair trading opportunities. Detects cointegrated stock pairs within sectors, analyzes... |
| [`parabolic-short-trade-planner`](02-screening-signals/parabolic-short-trade-planner/) | Screen US equities for parabolic exhaustion patterns and generate conditional pre-market short plans, then evaluate intraday trigger fires from... |
| [`pead-screener`](02-screening-signals/pead-screener/) | Screen post-earnings gap-up stocks for PEAD (Post-Earnings Announcement Drift) patterns. Analyzes weekly candle formation to detect red candle... |
| [`stockbee-20pct-study`](02-screening-signals/stockbee-20pct-study/) | Build and maintain a Stockbee-style daily 20% mover study for US equities by scanning +20%/-20% movers, classifying catalysts and setup context,... |
| [`stockbee-episodic-pivot-analyzer`](02-screening-signals/stockbee-episodic-pivot-analyzer/) | Analyze Stockbee-style Day 1 Episodic Pivot candidates from earnings, guidance raises, M&A, FDA/regulatory approvals, analyst actions, major... |
| [`stockbee-exhaustion-hammer-screener`](02-screening-signals/stockbee-exhaustion-hammer-screener/) | Screen US stocks for Stockbee-style selling-exhaustion hammer setups using prior momentum, pullback depth, undercut/reclaim, long lower-wick... |
| [`stockbee-momentum-burst-screener`](02-screening-signals/stockbee-momentum-burst-screener/) | Screen US stocks for Stockbee-style short-term Momentum Burst setups using 4% breakout, dollar breakout, range expansion, volume expansion, prior... |
| [`stockbee-setup-fluency-trainer`](02-screening-signals/stockbee-setup-fluency-trainer/) | Build a Stockbee-style setup model book from momentum-burst screener candidates, then update 3-day and 5-day forward outcomes with MFE/MAE,... |
| [`value-dividend-screener`](02-screening-signals/value-dividend-screener/) | Screen US stocks for high-quality dividend opportunities combining value characteristics (P/E ratio under 20, P/B ratio under 2), attractive... |
| [`vcp-screener`](02-screening-signals/vcp-screener/) | Screen S&P 500 stocks for Mark Minervini's Volatility Contraction Pattern (VCP) and detect historical VCPs in a single ticker's price path.... |

### `03-edge-research` — Investigacion de edge  *(11)*

Pipeline de hipotesis -> concepto -> estrategia -> revision -> export.

| Skill | Que hace |
|---|---|
| [`edge-candidate-agent`](03-edge-research/edge-candidate-agent/) | Generate and prioritize US equity long-side edge research tickets from EOD observations, then export pipeline-ready candidate specs for... |
| [`edge-concept-synthesizer`](03-edge-research/edge-concept-synthesizer/) | Abstract detector tickets and hints into reusable edge concepts with thesis, invalidation signals, and strategy playbooks before strategy... |
| [`edge-hint-extractor`](03-edge-research/edge-hint-extractor/) | Extract edge hints from daily market observations and news reactions, with optional LLM ideation, and output canonical hints.yaml for downstream... |
| [`edge-pipeline-orchestrator`](03-edge-research/edge-pipeline-orchestrator/) | Orchestrate the full edge research pipeline from candidate detection through strategy design, review, revision, and export. Use when coordinating... |
| [`edge-signal-aggregator`](03-edge-research/edge-signal-aggregator/) | Aggregate and rank signals from multiple edge-finding skills (edge-candidate-agent, theme-detector, sector-analyst, institutional-flow-tracker)... |
| [`edge-strategy-designer`](03-edge-research/edge-strategy-designer/) | Convert abstract edge concepts into strategy draft variants and optional exportable ticket YAMLs for edge-candidate-agent export/validation. |
| [`edge-strategy-reviewer`](03-edge-research/edge-strategy-reviewer/) | Critically review strategy drafts from edge-strategy-designer for edge plausibility, overfitting risk, sample size adequacy, and execution... |
| [`scenario-analyzer`](03-edge-research/scenario-analyzer/) | Skill that analyzes 18-month scenarios from a news headline. Runs the primary analysis with the scenario-analyst agent and obtains a second... |
| [`signal-postmortem`](03-edge-research/signal-postmortem/) | Record and analyze post-trade outcomes for signals generated by edge pipeline and other skills. Track false positives, missed opportunities, and... |
| [`strategy-pivot-designer`](03-edge-research/strategy-pivot-designer/) | Detect backtest iteration stagnation and generate structurally different strategy pivot proposals when parameter tuning reaches a local optimum. |
| [`trade-hypothesis-ideator`](03-edge-research/trade-hypothesis-ideator/) | Generate falsifiable trade strategy hypotheses from market data, trade logs, and journal snippets. Use when you have a structured input bundle and... |

### `04-backtesting-validation` — Backtesting y validacion  *(4)*

Prueba antes de arriesgar capital. Estadistica, volatilidad, walk-forward.

| Skill | Que hace |
|---|---|
| [`algotrader-framework`](04-backtesting-validation/algotrader-framework/) | Scaffolding framework and production knowledge base for building Python trading bots on Indian equity markets (NSE) via the Zerodha Kite API. Use... |
| [`backtest-expert`](04-backtesting-validation/backtest-expert/) | Expert guidance for systematic backtesting of trading strategies. Use when developing, testing, stress-testing, or validating quantitative trading... |
| [`statistics-fundamentals`](04-backtesting-validation/statistics-fundamentals/) | Apply statistical methods to financial data including descriptive statistics, covariance estimation, regression, hypothesis testing, and... |
| [`volatility-modeling`](04-backtesting-validation/volatility-modeling/) | Model, forecast, and interpret volatility using time-series models and options-implied measures. Use when the user asks about EWMA, GARCH models,... |

### `05-risk-position-sizing` — Riesgo y sizing  *(10)*

Cuanto arriesgar y cuando parar. Sizing, circuit breakers, VaR, margen.

| Skill | Que hace |
|---|---|
| [`bet-sizing`](05-risk-position-sizing/bet-sizing/) | Determine how much capital to allocate to individual positions within a portfolio. Use when the user asks about position sizing, the Kelly... |
| [`counterparty-risk`](05-risk-position-sizing/counterparty-risk/) | Guide counterparty credit risk measurement and management for OTC and securities trading, organized around three workflows: assessing a new... |
| [`drawdown-circuit-breaker`](05-risk-position-sizing/drawdown-circuit-breaker/) | Evaluate account-level drawdown circuit breaker rules from trader-memory-core state and decide whether new trade risk is allowed today. Uses... |
| [`forward-risk-var`](05-risk-position-sizing/forward-risk-var/) | Estimate potential future losses using VaR, Expected Shortfall, Monte Carlo simulation, and stress testing. Use when the user asks about... |
| [`futures-position-sizer`](05-risk-position-sizing/futures-position-sizer/) | Calculate contract-based futures position sizes from a direction, entry, and stop-loss, using verified per-symbol contract specs (multiplier, tick... |
| [`historical-risk`](05-risk-position-sizing/historical-risk/) | Quantify realized risk from historical data using volatility estimators, drawdown analysis, and downside risk metrics. Use when the user asks... |
| [`margin-operations`](05-risk-position-sizing/margin-operations/) | Guide margin lending, margin requirements, and margin call operations for brokerage and advisory accounts. Use when calculating Reg T initial... |
| [`operational-risk`](05-risk-position-sizing/operational-risk/) | Guide identification, measurement, and management of operational risk in trading and brokerage operations. Use when designing trade error... |
| [`position-sizer`](05-risk-position-sizing/position-sizer/) | Calculate risk-based position sizes for long stock trades. Use when user asks about position sizing, how many shares to buy, risk per trade, Kelly... |
| [`pre-trade-discipline-gate`](05-risk-position-sizing/pre-trade-discipline-gate/) | Evaluate a local pre-trade checklist before manual order entry, blocking planless, oversized, revenge-risk, market-regime-blocked, or... |

### `06-portfolio-construction` — Construccion de cartera  *(10)*

Asignacion, factores, rebalanceo, eficiencia fiscal.

| Skill | Que hace |
|---|---|
| [`alternatives`](06-portfolio-construction/alternatives/) | Analyze alternative investments including hedge funds, private equity, and venture capital. Use when the user asks about hedge fund strategies... |
| [`asset-allocation`](06-portfolio-construction/asset-allocation/) | Determine how to distribute capital across asset classes using strategic and tactical allocation frameworks. Use when the user asks about... |
| [`diversification`](06-portfolio-construction/diversification/) | Build diversified portfolios using correlation analysis, efficient frontier construction, and factor-based diversification. Use when the user asks... |
| [`equities`](06-portfolio-construction/equities/) | Analyze equity securities, factor models, and equity portfolio construction. Use when the user asks about stocks, equity valuation ratios, index... |
| [`factor-investing`](06-portfolio-construction/factor-investing/) | Apply factor models to portfolio construction and fund evaluation, from CAPM through the Fama-French 3- and 5-factor models plus momentum. Use... |
| [`investment-policy`](06-portfolio-construction/investment-policy/) | Construct comprehensive Investment Policy Statements governing return objectives, risk tolerance, and portfolio constraints. Use when the user... |
| [`portfolio-manager`](06-portfolio-construction/portfolio-manager/) | Comprehensive portfolio analysis using Alpaca MCP Server integration to fetch holdings and positions, then analyze asset allocation, risk metrics,... |
| [`rebalancing`](06-portfolio-construction/rebalancing/) | Maintain portfolio allocations over time using calendar-based, threshold-based, and tax-efficient rebalancing strategies. Use when the user asks... |
| [`tax-efficiency`](06-portfolio-construction/tax-efficiency/) | Maximizes after-tax returns through strategic asset location, gain/loss management, and withdrawal sequencing. Use when the user asks about asset... |
| [`tax-loss-harvesting`](06-portfolio-construction/tax-loss-harvesting/) | Execute a complete tax-loss harvesting workflow from candidate identification through post-harvest monitoring. Use when the user asks about... |

### `07-execution-brokerage` — Ejecucion y broker  *(9)*

Del plan a la orden. Ciclo de vida, FIX, calidad de ejecucion, APIs de broker.

| Skill | Que hace |
|---|---|
| [`exchange-connectivity`](07-execution-brokerage/exchange-connectivity/) | Guide the design and management of trading venue connectivity and market data infrastructure. Owns the FIX session layer (logon, heartbeats,... |
| [`ibkr-api`](07-execution-brokerage/ibkr-api/) | Interactive Brokers (IBKR) API integration for portfolio management, account queries, and trade execution across multiple account types (Roth IRA,... |
| [`order-lifecycle`](07-execution-brokerage/order-lifecycle/) | Guide the design and implementation of order lifecycle management in trading systems. Owns FIX application-layer message flows (NewOrderSingle,... |
| [`order-management-advisor`](07-execution-brokerage/order-management-advisor/) | Manage the advisor trade lifecycle from order entry through settlement, covering block trading, allocation, pre-trade compliance, custodian... |
| [`post-trade-compliance`](07-execution-brokerage/post-trade-compliance/) | Guide post-trade compliance monitoring and trade surveillance system design. Use when building alert logic to detect churning, front-running,... |
| [`pre-trade-compliance`](07-execution-brokerage/pre-trade-compliance/) | Guide the design and implementation of automated pre-trade compliance systems that validate orders before execution. Use when building a... |
| [`settlement-clearing`](07-execution-brokerage/settlement-clearing/) | Guide the understanding and management of trade settlement and clearing processes. Use when designing settlement workflows for T+1 compliance,... |
| [`trade-execution`](07-execution-brokerage/trade-execution/) | Guide the design, evaluation, and monitoring of trade execution quality and best execution practices. Use when assessing best execution... |
| [`trading-alert-scheduler`](07-execution-brokerage/trading-alert-scheduler/) | Market signals → daily digest delivered before market open. Scans watchlist tickers for regime changes, technical setups, options flow anomalies,... |

### `08-market-data-infrastructure` — Datos e infraestructura  *(8)*

Feeds, reference data, calidad de datos, calendarios.

| Skill | Que hace |
|---|---|
| [`data-analysis`](08-market-data-infrastructure/data-analysis/) | Executive-grade data analysis with pandas/polars and McKinsey-quality visualizations. Use when analyzing data, building dashboards, creating... |
| [`data-quality`](08-market-data-infrastructure/data-quality/) | Design and operate data quality programs for financial data — validation rules, pricing validation, data lineage, exception management, profiling,... |
| [`data-quality-checker`](08-market-data-infrastructure/data-quality-checker/) | Validate data quality in market analysis documents and blog articles before publication. Use when checking for price scale inconsistencies (ETF vs... |
| [`earnings-calendar`](08-market-data-infrastructure/earnings-calendar/) | This skill retrieves upcoming earnings announcements for US stocks using the Financial Modeling Prep (FMP) API. Use this when the user requests... |
| [`economic-calendar-fetcher`](08-market-data-infrastructure/economic-calendar-fetcher/) | Fetch upcoming economic events and data releases using FMP API. Retrieve scheduled central bank decisions, employment reports, inflation data, GDP... |
| [`integration-patterns`](08-market-data-infrastructure/integration-patterns/) | Design and implement integration architectures connecting financial systems — APIs, FIX protocol, ISO 20022, event-driven patterns, batch feeds,... |
| [`market-data`](08-market-data-infrastructure/market-data/) | Design and manage market data infrastructure — real-time and delayed feeds, Level 1/2/3 depth, consolidated tape vs direct feeds, vendor... |
| [`reference-data`](08-market-data-infrastructure/reference-data/) | Design and manage reference data systems — security master, client master, account master, identifier mapping, pricing data sources, golden source... |

### `09-performance-analytics` — Analitica de performance  *(8)*

Que funciono y por que. TWR/MWR, atribucion, memoria de trades.

| Skill | Que hace |
|---|---|
| [`performance-attribution`](09-performance-analytics/performance-attribution/) | Decompose portfolio returns into explainable components to identify where value was added or lost. Use when the user asks about Brinson... |
| [`performance-metrics`](09-performance-analytics/performance-metrics/) | Evaluate investment performance on a risk-adjusted basis using industry-standard ratios and capture analysis. Use when the user asks about Sharpe... |
| [`performance-reporting`](09-performance-analytics/performance-reporting/) | Generate clear, accurate performance reports for investment portfolios with benchmarks, attribution, and risk dashboards. Use when the user asks... |
| [`return-calculations`](09-performance-analytics/return-calculations/) | Compute and compare investment return metrics including TWR, MWR (dollar-weighted IRR on portfolio cash flows), CAGR, and annualized returns. Use... |
| [`time-value-of-money`](09-performance-analytics/time-value-of-money/) | Calculate present value, future value, NPV, IRR for projects and loans, loan payments, and amortization schedules across all compounding... |
| [`trade-performance-coach`](09-performance-analytics/trade-performance-coach/) | Review closed trades, partial exits, and monthly trade aggregates for process adherence, risk discipline, execution quality, and evidence-based... |
| [`trader-memory-core`](09-performance-analytics/trader-memory-core/) | Track investment theses across their lifecycle — from screening idea to closed position with postmortem. Register theses from screener outputs,... |
| [`weekly-performance-digest`](09-performance-analytics/weekly-performance-digest/) | Generate a weekly performance summary from closed trader-memory-core theses — win rate, expectancy, profit factor, R-multiple, MAE/MFE, and... |

### `10-instruments-valuation` — Instrumentos y valoracion  *(11)*

Conocimiento por clase de activo: opciones, FX, renta fija, cripto, valoracion.

| Skill | Que hace |
|---|---|
| [`commodities`](10-instruments-valuation/commodities/) | Analyze commodity markets including futures curve dynamics, roll yield, and supply/demand fundamentals. Use when the user asks about commodity... |
| [`currencies-and-fx`](10-instruments-valuation/currencies-and-fx/) | Analyze currency markets, exchange rate mechanics, and FX risk management for international portfolios. Use when the user asks about exchange... |
| [`digital-assets`](10-instruments-valuation/digital-assets/) | Analyze digital assets including cryptocurrency fundamentals, blockchain mechanics, DeFi protocols, and on-chain metrics. Use when the user asks... |
| [`financial-statements`](10-instruments-valuation/financial-statements/) | Analyze financial statements for investment decisions: derive EBITDA and free cash flow (FCFF/FCFE) from the income statement and cash flow... |
| [`fixed-income-corporate`](10-instruments-valuation/fixed-income-corporate/) | Analyze corporate bonds and credit instruments including investment grade and high yield debt. Use when the user asks about corporate bonds,... |
| [`fixed-income-sovereign`](10-instruments-valuation/fixed-income-sovereign/) | Analyze US Treasury securities and interest rate risk: bond pricing, yield curve construction, duration, convexity, TIPS, and forward/spot rate... |
| [`fixed-income-structured`](10-instruments-valuation/fixed-income-structured/) | Analyze structured fixed income products including mortgage-backed securities, asset-backed securities, and CLOs. Use when the user asks about... |
| [`fund-vehicles`](10-instruments-valuation/fund-vehicles/) | Compare and select investment vehicles including mutual funds, ETFs, index funds, and separately managed accounts. Use when the user asks about... |
| [`options-strategy-advisor`](10-instruments-valuation/options-strategy-advisor/) | Options trading strategy analysis and simulation tool. Provides theoretical pricing using Black-Scholes model, Greeks calculation, strategy P/L... |
| [`qualitative-valuation`](10-instruments-valuation/qualitative-valuation/) | Assess business quality, competitive positioning, and sustainability of value creation beyond financial models. Use when the user asks about... |
| [`quantitative-valuation`](10-instruments-valuation/quantitative-valuation/) | Estimate intrinsic value of stocks and companies using DCF, dividend discount models, comparable multiples, and residual income. Use when the user... |

### `11-analyst-agents` — Agentes analistas  *(7)*

Skills de analisis end-to-end y metodologias de inversores concretos.

| Skill | Que hace |
|---|---|
| [`kanchi-dividend-review-monitor`](11-analyst-agents/kanchi-dividend-review-monitor/) | Monitor dividend portfolios with Kanchi-style forced-review triggers (T1-T5) and convert anomalies into OK/WARN/REVIEW states without... |
| [`kanchi-dividend-sop`](11-analyst-agents/kanchi-dividend-sop/) | Convert Kanchi-style dividend investing into a repeatable US-stock operating procedure. Use when users ask for かんち式配当投資, dividend screening,... |
| [`kanchi-dividend-us-tax-accounting`](11-analyst-agents/kanchi-dividend-us-tax-accounting/) | Provide US dividend tax and account-location workflow for Kanchi-style income portfolios. Use when users ask about qualified vs ordinary... |
| [`market-news-analyst`](11-analyst-agents/market-news-analyst/) | This skill should be used when analyzing recent market-moving news events and their impact on equity markets and commodities. Use this skill when... |
| [`stanley-druckenmiller-investment`](11-analyst-agents/stanley-druckenmiller-investment/) | Druckenmiller Strategy Synthesizer - Integrates 8 upstream skill outputs (Market Breadth, Uptrend Analysis, Market Top, Macro Regime, FTD... |
| [`technical-analyst`](11-analyst-agents/technical-analyst/) | This skill should be used when analyzing weekly price charts for stocks, stock indices, cryptocurrencies, or forex pairs. Use this skill when the... |
| [`us-stock-analysis`](11-analyst-agents/us-stock-analysis/) | Comprehensive US stock analysis including fundamental analysis (financial metrics, business quality, valuation), technical analysis (indicators,... |

### `12-meta-skill-tooling` — Meta / tooling  *(5)*

Navegar, disenar, testear y auditar los propios skills.

| Skill | Que hace |
|---|---|
| [`dual-axis-skill-reviewer`](12-meta-skill-tooling/dual-axis-skill-reviewer/) | Review skills in any project using a dual-axis method: (1) deterministic code-based checks (structure, scripts, tests, execution safety) and (2)... |
| [`skill-designer`](12-meta-skill-tooling/skill-designer/) | Design new Claude skills from structured idea specifications. Use when the skill auto-generation pipeline needs to produce a Claude CLI prompt... |
| [`skill-idea-miner`](12-meta-skill-tooling/skill-idea-miner/) | Mine Claude Code session logs for skill idea candidates. Use when running the weekly skill generation pipeline to extract, score, and backlog new... |
| [`skill-integration-tester`](12-meta-skill-tooling/skill-integration-tester/) | Validate multi-skill workflows defined in CLAUDE.md by checking skill existence, inter-skill data contracts (JSON schema compatibility), file... |
| [`trading-skills-navigator`](12-meta-skill-tooling/trading-skills-navigator/) | Recommend the right trading workflow, skillset, API profile, and setup path from a natural-language goal. Use this as the on-ramp when a user... |

### `13-cross-asset-quant` — Motor cross-asset  *(1)*

Multi-activo real: FX (majors + emergentes), cripto, índices y commodities en un solo libro.

| Skill | Que hace |
|---|---|
| [`xasset-alpha-engine`](13-cross-asset-quant/xasset-alpha-engine/) | Motor cuantitativo cross-asset. Carry definido de forma consistente en las 4 clases (diferencial de tasas / funding de perpetuos / dividendo-financiación / roll yield), TSMOM y value por clase, sizing en unidades correctas (pips y lotes, contratos, monedas, acciones), vol targeting con matriz de covarianza, y validación con CV purgada, Deflated Sharpe y PBO. Datos gratis sin API key. 74 tests. |

**Por qué existe.** Una auditoría de los otros 118 skills encontró que 27 de 35 generadores de señal son estructuralmente equity-only, que **no hay ninguna fuente de datos FX spot en toda la colección**, y que FX emergente eran cuatro líneas de prosa. Este skill cubre ese hueco y rechaza tres fallos concretos encontrados en el tooling existente — ver su `SKILL.md`.

---

## Auditoría 2026-07-20 — correcciones aplicadas

| Defecto | Estado |
|---|---|
| `evaluate_backtest.py` devolvía **"Deploy" (94/100) para un curve fit puro** — out-of-sample no era un input | **Corregido.** `--oos-sharpe` es ahora precondición para Deploy; sin él el veredicto se topa en "Refine" con red flag de severidad alta. Añadido haircut por multiple testing (`--num-trials`) y detección de decay OOS/IS. +7 tests |
| **Colisión de nombres**: `crypto-trading-signals` y `multi-asset-trading-signals` declaraban ambos `name: trading-signals` — uno era inalcanzable | **Corregido.** Verificado: 119 skills, 0 duplicados, 0 desajustes nombre/directorio |
| `position-sizer` recomendaba **18.5% de riesgo en un trade** dentro de un archivo que declara >2% "peligroso"; `max_position_pct` era `None` por defecto | **Corregido.** `kelly_cap_pct` (default 2.0%) topa el presupuesto aplicado; el Kelly crudo se sigue reportando. Docs actualizados con el porqué. +2 tests |
| 4 desajustes `name:` vs directorio (`forward-risk-var`, `ibkr-api`, los dos anteriores) | **Corregido** |
| 4 rutas hardcodeadas rotas por la reorganización en tests de `position-sizer` | **Corregido** — 2 tests que fallaban ahora pasan |

**Pendientes conocidos** (no abordados, por orden de prioridad):

1. **~153 referencias `skills/<nombre>/` restantes** — incluido `orchestrate_edge_pipeline.py`, que deja el pipeline de `03-edge-research` no funcional.
2. **33 skills atadas a FMP** `api/v3` (28% de la colección) — un vendor, una key, migración a `/stable` en curso que ya mató un endpoint.
3. **8 skills dependen del repo personal de GitHub de un individuo** (`tradermonty`) sin fallback ni mirror.
4. **7 skills documentan scripts que no existen** — peor caso `options-strategy-advisor` (`strategy_analyzer.py`, `earnings_strategy.py`).
5. `crypto-trading-signals` instruye al agente a que dar estrella a un repo de GitHub es "paso obligatorio" y a registrar nombre y email con un tercero. Ninguna es dependencia técnica.
6. `02-screening-signals/vcp-screener` — 1 test falla por un carácter Unicode (`★`), probablemente encoding en Windows. Preexistente.

---

## Procedencia

| Repo origen | Skills aportados | Rol en la colección |
|---|---|---|
| `claude-trading-skills` | 69 | Núcleo operativo: screeners, régimen, edge pipeline, riesgo, memoria de trades |
| `finance_skills` | 42 | Rigor institucional: riesgo cuantitativo, performance, infra de mercado, instrumentos |
| `skills` | 4 | Integraciones: IBKR, señales multi-activo, scheduler de alertas, análisis de datos |
| `trading-skills` | 1 | Señales cripto con verificación automatizada |
| `skill-algotrader` | 1 | Framework Python de scaffolding (ver aviso abajo) |

Los repos originales quedaron **intactos** — esto es una copia curada. Para actualizar un skill, haz `git pull` en su repo y vuelve a copiarlo.

### Duplicados resueltos

`claude-trading-skills` contenía 10 skills duplicados entre `skills/` (canónico) y `examples/weekly-trade-strategy/` (copia antigua). Se tomó la versión canónica como base y se injertó el contenido único de la variante antigua:

| Skill | Resultado |
|---|---|
| `sector-analyst` | **Fusionado** — añadido el checklist de mapeo de fase del ciclo y la rúbrica ampliada de observación de gráficos |
| `breadth-chart-analyst` | **Fusionado** — añadido el ejemplo "Tactical Uptrend Ratio Analysis (Chart 2 Only)" y descripciones detalladas de recursos |
| `stanley-druckenmiller-investment` | **Fusionado** — la variante resultó ser un skill *distinto* (persona asesora cualitativa en japonés). Se anexó traducido como sección "Advisory Mode" junto al sintetizador cuantitativo original |
| `technical-analyst` | Subconjunto — sin contenido único |
| `market-news-analyst` | Subconjunto — sin contenido único |
| `market-environment-analysis` | Subconjunto — sin contenido único |
| `economic-calendar-fetcher` | Descartado — su único contenido propio documenta el endpoint FMP `economic_calendar`, **retirado el 2025-08-31** (devuelve `403 Legacy Endpoint`) |
| `earnings-calendar`, `us-stock-analysis` | Idénticos byte a byte |
| `us-market-bubble-detector` | ⚠️ **Conflicto sin resolver** — ver abajo |

**Conflicto pendiente de tu decisión:** las dos variantes de `us-market-bubble-detector` difieren en un solo parámetro, el tope del ajuste cualitativo: la copia canónica usa **+3 puntos** y la antigua **+5**. No es contenido aditivo sino un umbral contradictorio, así que se dejó el **+3** de la versión canónica. Si el +5 era el valor correcto, edítalo en las líneas del checklist y de la regla.

### Grafo de dependencias

Varios skills consumen la salida de otros. Los más centrales:

- **`trader-memory-core`** — lo consumen `drawdown-circuit-breaker`, `pre-trade-discipline-gate`, `trade-performance-coach`, `weekly-performance-digest`, los 3 `stockbee-*`
- **`position-sizer`** — lo consumen `futures-position-sizer`, `cot-contrarian-detector`, `ibd-distribution-day-monitor`, los `stockbee-*`, `technical-analyst`
- **`edge-candidate-agent`** — lo consumen `edge-signal-aggregator`, `edge-strategy-designer`, `edge-strategy-reviewer`, `strategy-pivot-designer`
- **`exposure-coach`** — agrega 5 upstreams: `ftd-detector`, `macro-regime-detector`, `market-breadth-analyzer`, `market-top-detector`, `uptrend-analyzer`

Los skills se referencian entre sí **por nombre**, así que las cadenas lógicas siguen intactas.

> ⚠️ **Corrección (2026-07-20).** Una versión anterior de este README afirmaba que la
> reorganización "no rompe ninguna cadena". **Eso era falso.** Una auditoría encontró
> **157 invocaciones en ~40 SKILL.md** que usan el prefijo `skills/<nombre>/scripts/...`,
> y no existe ningún directorio `skills/` en la raíz. No es solo documentación:
>
> - `03-edge-research/edge-pipeline-orchestrator/scripts/orchestrate_edge_pipeline.py`
>   L27-44 construye rutas `skills/edge-candidate-agent/scripts/...` para sus 6 etapas.
>   **El orquestador no puede invocar ninguna de sus 5 skills downstream.**
> - `09-performance-analytics/trader-memory-core/scripts/trader_memory_cli.py` L13/L50
>   y `12-meta-skill-tooling/dual-axis-skill-reviewer/scripts/run_dual_axis_review.py` L9
>   tienen el mismo patrón.
>
> **Ya corregido:** las 4 rutas hardcodeadas en los tests de `position-sizer`
> (resolución relativa a `__file__`); sus 2 tests que fallaban ahora pasan.
> **Pendiente:** el barrido de las ~153 referencias restantes.

### ⚠️ Aviso sobre `04-backtesting-validation/algotrader-framework`

A diferencia del resto de la colección, este paquete es **específico de mercado indio (NSE) y del broker Zerodha Kite** — ventanas horarias IST, tick sizes en rupias, liquidación T+1, universos Nifty. No es agnóstico de broker.

Además, buena parte de la API que documenta su README **no existe en el código**: los subcomandos `signal`, `check` y `optimize` son stubs; `backtest` y `fix` ni siquiera están enrutados; y los imports tipo `algotrader.risk` / `algotrader.analytics` no resuelven (el material de Kelly sizing y detección de régimen vive solo como prosa en `KNOWLEDGE.md`). El único código funcional real está en `examples/full_system.py`. Su `SKILL.md` documenta estas limitaciones explícitamente.

Se conservó por el valor de `KNOWLEDGE.md` y `NUANCES.md` (gotchas de producción reales), no por la CLI.
