---
kind: as-built
status: IMPLEMENTED
contract: CTR-QUANT-LIBRARY-001
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - .claude/skills/quant-algo-trading/SKILL.md
---

# Procedencia - libreria de skills quant

Origen: `ALGO TRADING/SKILLS/FINAL SKILLS` (fuera del repo, read-only).
Importado: 2026-07-20 - **119 skills**, 13 categorias.

**`vendor/` NO es descubierto por Claude Code.** Solo lo promovido a `.claude/skills/` se
carga en cada sesion. Lo vendored es referencia de lectura, no fuente de evidencia:
la mayoria requiere APIs de pago de renta variable US y ninguna guarda snapshots
point-in-time, asi que cualquier backtest construido sobre ellas usaria datos restated.

| Veredicto | Significado |
|---|---|
| `A-PROMOTED` | Vive en `.claude/skills/`, revisada contra la constitucion |
| `QUARANTINED` | Contradice `quant-constitution.md` - nunca promover |
| `C-VENDORED` | Referencia; no aplica a COP/Gold/BTC o exige APIs US |

## Resumen

- **A-PROMOTED**: 11
- **C-VENDORED**: 102
- **QUARANTINED**: 6

## constitution_reviewed

Las 13 skills promovidas se revisaron contra
`.claude/rules/quant-constitution.md` el 2026-07-20.
Promover cualquier otra exige anadir su fila aqui con fecha de revision.

## Inventario completo

| Categoria | Skill | Veredicto | Proposito |
|---|---|---|---|
| 01-market-regime | `breadth-chart-analyst` | C-VENDORED | This skill should be used when analyzing market breadth charts, specifically the S&P 500 Breadth Index (200-Day MA based) and the US Stock M |
| 01-market-regime | `crypto-regime-analyzer` | C-VENDORED | Quantifies crypto market regime health using free, keyless public data (CoinGecko + Binance funding). Generates a 0-100 composite score acro |
| 01-market-regime | `downtrend-duration-analyzer` | C-VENDORED | Analyze historical downtrend durations and generate interactive HTML histograms showing typical correction lengths by sector and market cap. |
| 01-market-regime | `exposure-coach` | C-VENDORED | Generate a one-page Market Posture summary with net exposure ceiling, growth-vs-value bias, participation breadth, and new-entry-allowed vs  |
| 01-market-regime | `ftd-detector` | C-VENDORED | Detects Follow-Through Day (FTD) signals for market bottom confirmation using William O'Neil's methodology. Dual-index tracking (S&P 500 + N |
| 01-market-regime | `ibd-distribution-day-monitor` | C-VENDORED | Detect IBD-style Distribution Days for QQQ/SPY (close down at least 0.2% on higher volume), track 25-session expiration and 5% invalidation, |
| 01-market-regime | `macro-regime-detector` | C-VENDORED | Detect structural macro regime transitions (1-2 year horizon) using cross-asset ratio analysis. Analyze RSP/SPY concentration, yield curve,  |
| 01-market-regime | `market-breadth-analyzer` | C-VENDORED | Quantifies market breadth health using TraderMonty's public CSV data. Generates a 0-100 composite score across 6 components (100 = healthy). |
| 01-market-regime | `market-environment-analysis` | C-VENDORED | Comprehensive market environment analysis and reporting tool. Analyzes global markets including US, European, Asian markets, forex, commodit |
| 01-market-regime | `market-top-detector` | C-VENDORED | Detects market top probability using O'Neil Distribution Days, Minervini Leading Stock Deterioration, and Monty Defensive Sector Rotation. G |
| 01-market-regime | `sector-analyst` | C-VENDORED | This skill should be used when analyzing sector rotation patterns and market cycle positioning. It fetches sector uptrend data from CSV (no  |
| 01-market-regime | `theme-detector` | C-VENDORED | Detect and analyze trending market themes across sectors. Use when user asks about current market themes, trending sectors, sector rotation, |
| 01-market-regime | `uptrend-analyzer` | C-VENDORED | Analyzes market breadth using Monty's Uptrend Ratio Dashboard data to diagnose the current market environment. Generates a 0-100 composite s |
| 01-market-regime | `us-market-bubble-detector` | C-VENDORED | Evaluates market bubble risk through quantitative data-driven analysis using the revised Minsky/Kindleberger framework v2.1. Prioritizes obj |
| 02-screening-signals | `breakout-trade-planner` | C-VENDORED | Generate Minervini-style breakout trade plans from VCP screener output with worst-case risk calculation, portfolio heat management, and Alpa |
| 02-screening-signals | `canslim-screener` | C-VENDORED | Screen US stocks using William O'Neil's CANSLIM growth stock methodology. Use when user requests CANSLIM stock screening, growth stock analy |
| 02-screening-signals | `contrarian-setup-gate` | C-VENDORED | Synthesize the three Jason Shapiro contrarian-pipeline verdicts (COT crowding, news-reaction failure, weekly price-action confirmation) into |
| 02-screening-signals | `cot-contrarian-detector` | C-VENDORED | Detect crowded speculative positioning in CFTC futures markets (COT report analysis) to find contrarian setups using Jason Shapiro's methodo |
| 02-screening-signals | `crypto-trading-signals` | QUARANTINED | "Fetches live AI crypto trading signals with entry price, stop-loss, take-profit, leverage, confidence scores, and automated verification. C |
| 02-screening-signals | `dividend-growth-pullback-screener` | C-VENDORED | Use this skill to find high-quality dividend growth stocks (12%+ annual dividend growth, 1.5%+ yield) that are experiencing temporary pullba |
| 02-screening-signals | `earnings-trade-analyzer` | C-VENDORED | Analyze recent post-earnings stocks using a 5-factor scoring system (Gap Size, Pre-Earnings Trend, Volume Trend, MA200 Position, MA50 Positi |
| 02-screening-signals | `finviz-screener` | C-VENDORED | Build and open FinViz screener URLs from natural language requests. Use when user wants to screen stocks, find stocks matching criteria, fil |
| 02-screening-signals | `institutional-flow-tracker` | C-VENDORED | Use this skill to track institutional investor ownership changes and portfolio flows using 13F filings data. Analyzes hedge funds, mutual fu |
| 02-screening-signals | `multi-asset-trading-signals` | C-VENDORED | "Expert trading partner for Options, Stocks, Crypto, Commodities, Gold, Silver, Oil, VIX, and Forex. Covers technical analysis (Elliott Wave |
| 02-screening-signals | `news-reaction-failure-analyzer` | C-VENDORED | Judge whether a market FAILED to react to news favorable to a crowded speculative position — step 2 of Jason Shapiro's COT contrarian proces |
| 02-screening-signals | `pair-trade-screener` | C-VENDORED | Statistical arbitrage tool for identifying and analyzing pair trading opportunities. Detects cointegrated stock pairs within sectors, analyz |
| 02-screening-signals | `parabolic-short-trade-planner` | C-VENDORED | Screen US equities for parabolic exhaustion patterns and generate conditional pre-market short plans, then evaluate intraday trigger fires f |
| 02-screening-signals | `pead-screener` | C-VENDORED | Screen post-earnings gap-up stocks for PEAD (Post-Earnings Announcement Drift) patterns. Analyzes weekly candle formation to detect red cand |
| 02-screening-signals | `stockbee-20pct-study` | C-VENDORED | Build and maintain a Stockbee-style daily 20% mover study for US equities by scanning +20%/-20% movers, classifying catalysts and setup cont |
| 02-screening-signals | `stockbee-episodic-pivot-analyzer` | C-VENDORED | Analyze Stockbee-style Day 1 Episodic Pivot candidates from earnings, guidance raises, M&A, FDA/regulatory approvals, analyst actions, major |
| 02-screening-signals | `stockbee-exhaustion-hammer-screener` | C-VENDORED | Screen US stocks for Stockbee-style selling-exhaustion hammer setups using prior momentum, pullback depth, undercut/reclaim, long lower-wick |
| 02-screening-signals | `stockbee-momentum-burst-screener` | C-VENDORED | Screen US stocks for Stockbee-style short-term Momentum Burst setups using 4% breakout, dollar breakout, range expansion, volume expansion,  |
| 02-screening-signals | `stockbee-setup-fluency-trainer` | C-VENDORED | Build a Stockbee-style setup model book from momentum-burst screener candidates, then update 3-day and 5-day forward outcomes with MFE/MAE,  |
| 02-screening-signals | `value-dividend-screener` | C-VENDORED | Screen US stocks for high-quality dividend opportunities combining value characteristics (P/E ratio under 20, P/B ratio under 2), attractive |
| 02-screening-signals | `vcp-screener` | C-VENDORED | Screen S&P 500 stocks for Mark Minervini's Volatility Contraction Pattern (VCP) and detect historical VCPs in a single ticker's price path.  |
| 03-edge-research | `edge-candidate-agent` | C-VENDORED | Generate and prioritize US equity long-side edge research tickets from EOD observations, then export pipeline-ready candidate specs for trad |
| 03-edge-research | `edge-concept-synthesizer` | C-VENDORED | Abstract detector tickets and hints into reusable edge concepts with thesis, invalidation signals, and strategy playbooks before strategy de |
| 03-edge-research | `edge-hint-extractor` | C-VENDORED | Extract edge hints from daily market observations and news reactions, with optional LLM ideation, and output canonical hints.yaml for downst |
| 03-edge-research | `edge-pipeline-orchestrator` | C-VENDORED | Orchestrate the full edge research pipeline from candidate detection through strategy design, review, revision, and export. Use when coordin |
| 03-edge-research | `edge-signal-aggregator` | QUARANTINED | Aggregate and rank signals from multiple edge-finding skills (edge-candidate-agent, theme-detector, sector-analyst, institutional-flow-track |
| 03-edge-research | `edge-strategy-designer` | C-VENDORED | Convert abstract edge concepts into strategy draft variants and optional exportable ticket YAMLs for edge-candidate-agent export/validation. |
| 03-edge-research | `edge-strategy-reviewer` | C-VENDORED | > Critically review strategy drafts from edge-strategy-designer for edge plausibility, overfitting risk, sample size adequacy, and execution |
| 03-edge-research | `scenario-analyzer` | C-VENDORED | / Skill that analyzes 18-month scenarios from a news headline. Runs the primary analysis with the scenario-analyst agent and obtains a secon |
| 03-edge-research | `signal-postmortem` | QUARANTINED | Record and analyze post-trade outcomes for signals generated by edge pipeline and other skills. Track false positives, missed opportunities, |
| 03-edge-research | `strategy-pivot-designer` | QUARANTINED | Detect backtest iteration stagnation and generate structurally different strategy pivot proposals when parameter tuning reaches a local opti |
| 03-edge-research | `trade-hypothesis-ideator` | C-VENDORED | > Generate falsifiable trade strategy hypotheses from market data, trade logs, and journal snippets. Use when you have a structured input bu |
| 04-backtesting-validation | `algotrader-framework` | C-VENDORED | Scaffolding framework and production knowledge base for building Python trading bots on Indian equity markets (NSE) via the Zerodha Kite API |
| 04-backtesting-validation | `backtest-expert` | QUARANTINED | Expert guidance for systematic backtesting of trading strategies. Use when developing, testing, stress-testing, or validating quantitative t |
| 04-backtesting-validation | `statistics-fundamentals` | A-PROMOTED | "Apply statistical methods to financial data including descriptive statistics, covariance estimation, regression, hypothesis testing, and re |
| 04-backtesting-validation | `volatility-modeling` | A-PROMOTED | "Model, forecast, and interpret volatility using time-series models and options-implied measures. Use when the user asks about EWMA, GARCH m |
| 05-risk-position-sizing | `bet-sizing` | A-PROMOTED | "Determine how much capital to allocate to individual positions within a portfolio. Use when the user asks about position sizing, the Kelly  |
| 05-risk-position-sizing | `counterparty-risk` | C-VENDORED | "Guide counterparty credit risk measurement and management for OTC and securities trading, organized around three workflows: assessing a new |
| 05-risk-position-sizing | `drawdown-circuit-breaker` | C-VENDORED | DEMOTED 2026-07-20: consumer with no producer - reads `trader-memory-core` thesis YAML, and that skill was not promoted. | Evaluate account-level drawdown circuit breaker rules from trader-memory-core state and decide whether new trade risk is allowed today. Uses |
| 05-risk-position-sizing | `forward-risk-var` | A-PROMOTED | "Estimate potential future losses using VaR, Expected Shortfall, Monte Carlo simulation, and stress testing. Use when the user asks about Va |
| 05-risk-position-sizing | `futures-position-sizer` | C-VENDORED | Calculate contract-based futures position sizes from a direction, entry, and stop-loss, using verified per-symbol contract specs (multiplier |
| 05-risk-position-sizing | `historical-risk` | A-PROMOTED | "Quantify realized risk from historical data using volatility estimators, drawdown analysis, and downside risk metrics. Use when the user as |
| 05-risk-position-sizing | `margin-operations` | C-VENDORED | "Guide margin lending, margin requirements, and margin call operations for brokerage and advisory accounts. Use when calculating Reg T initi |
| 05-risk-position-sizing | `operational-risk` | C-VENDORED | "Guide identification, measurement, and management of operational risk in trading and brokerage operations. Use when designing trade error d |
| 05-risk-position-sizing | `position-sizer` | C-VENDORED | Calculate risk-based position sizes for long stock trades. Use when user asks about position sizing, how many shares to buy, risk per trade, |
| 05-risk-position-sizing | `pre-trade-discipline-gate` | C-VENDORED | Evaluate a local pre-trade checklist before manual order entry, blocking planless, oversized, revenge-risk, market-regime-blocked, or circui |
| 06-portfolio-construction | `alternatives` | C-VENDORED | "Analyze alternative investments including hedge funds, private equity, and venture capital. Use when the user asks about hedge fund strateg |
| 06-portfolio-construction | `asset-allocation` | QUARANTINED | "Determine how to distribute capital across asset classes using strategic and tactical allocation frameworks. Use when the user asks about p |
| 06-portfolio-construction | `diversification` | C-VENDORED | "Build diversified portfolios using correlation analysis, efficient frontier construction, and factor-based diversification. Use when the us |
| 06-portfolio-construction | `equities` | C-VENDORED | "Analyze equity securities, factor models, and equity portfolio construction. Use when the user asks about stocks, equity valuation ratios,  |
| 06-portfolio-construction | `factor-investing` | C-VENDORED | "Apply factor models to portfolio construction and fund evaluation, from CAPM through the Fama-French 3- and 5-factor models plus momentum.  |
| 06-portfolio-construction | `investment-policy` | C-VENDORED | "Construct comprehensive Investment Policy Statements governing return objectives, risk tolerance, and portfolio constraints. Use when the u |
| 06-portfolio-construction | `portfolio-manager` | C-VENDORED | Comprehensive portfolio analysis using Alpaca MCP Server integration to fetch holdings and positions, then analyze asset allocation, risk me |
| 06-portfolio-construction | `rebalancing` | C-VENDORED | "Maintain portfolio allocations over time using calendar-based, threshold-based, and tax-efficient rebalancing strategies. Use when the user |
| 06-portfolio-construction | `tax-efficiency` | C-VENDORED | "Maximizes after-tax returns through strategic asset location, gain/loss management, and withdrawal sequencing. Use when the user asks about |
| 06-portfolio-construction | `tax-loss-harvesting` | C-VENDORED | "Execute a complete tax-loss harvesting workflow from candidate identification through post-harvest monitoring. Use when the user asks about |
| 07-execution-brokerage | `exchange-connectivity` | C-VENDORED | "Guide the design and management of trading venue connectivity and market data infrastructure. Owns the FIX session layer (logon, heartbeats |
| 07-execution-brokerage | `ibkr-api` | C-VENDORED | Interactive Brokers (IBKR) API integration for portfolio management, account queries, and trade execution across multiple account types (Rot |
| 07-execution-brokerage | `order-lifecycle` | C-VENDORED | "Guide the design and implementation of order lifecycle management in trading systems. Owns FIX application-layer message flows (NewOrderSin |
| 07-execution-brokerage | `order-management-advisor` | C-VENDORED | "Manage the advisor trade lifecycle from order entry through settlement, covering block trading, allocation, pre-trade compliance, custodian |
| 07-execution-brokerage | `post-trade-compliance` | C-VENDORED | "Guide post-trade compliance monitoring and trade surveillance system design. Use when building alert logic to detect churning, front-runnin |
| 07-execution-brokerage | `pre-trade-compliance` | C-VENDORED | "Guide the design and implementation of automated pre-trade compliance systems that validate orders before execution. Use when building a co |
| 07-execution-brokerage | `settlement-clearing` | C-VENDORED | "Guide the understanding and management of trade settlement and clearing processes. Use when designing settlement workflows for T+1 complian |
| 07-execution-brokerage | `trade-execution` | C-VENDORED | "Guide the design, evaluation, and monitoring of trade execution quality and best execution practices. Use when assessing best execution obl |
| 07-execution-brokerage | `trading-alert-scheduler` | C-VENDORED | "Market signals → daily digest delivered before market open. Scans watchlist tickers for regime changes, technical setups, options flow anom |
| 08-market-data-infrastructure | `data-analysis` | C-VENDORED | "Executive-grade data analysis with pandas/polars and McKinsey-quality visualizations. Use when analyzing data, building dashboards, creatin |
| 08-market-data-infrastructure | `data-quality` | C-VENDORED | "Design and operate data quality programs for financial data — validation rules, pricing validation, data lineage, exception management, pro |
| 08-market-data-infrastructure | `data-quality-checker` | C-VENDORED | Validate data quality in market analysis documents and blog articles before publication. Use when checking for price scale inconsistencies ( |
| 08-market-data-infrastructure | `earnings-calendar` | C-VENDORED | This skill retrieves upcoming earnings announcements for US stocks using the Financial Modeling Prep (FMP) API. Use this when the user reque |
| 08-market-data-infrastructure | `economic-calendar-fetcher` | C-VENDORED | "Fetch upcoming economic events and data releases using FMP API. Retrieve scheduled central bank decisions, employment reports, inflation da |
| 08-market-data-infrastructure | `integration-patterns` | C-VENDORED | "Design and implement integration architectures connecting financial systems — APIs, FIX protocol, ISO 20022, event-driven patterns, batch f |
| 08-market-data-infrastructure | `market-data` | C-VENDORED | "Design and manage market data infrastructure — real-time and delayed feeds, Level 1/2/3 depth, consolidated tape vs direct feeds, vendor se |
| 08-market-data-infrastructure | `reference-data` | C-VENDORED | "Design and manage reference data systems — security master, client master, account master, identifier mapping, pricing data sources, golden |
| 09-performance-analytics | `performance-attribution` | C-VENDORED | "Decompose portfolio returns into explainable components to identify where value was added or lost. Use when the user asks about Brinson att |
| 09-performance-analytics | `performance-metrics` | A-PROMOTED | "Evaluate investment performance on a risk-adjusted basis using industry-standard ratios and capture analysis. Use when the user asks about  |
| 09-performance-analytics | `performance-reporting` | C-VENDORED | "Generate clear, accurate performance reports for investment portfolios with benchmarks, attribution, and risk dashboards. Use when the user |
| 09-performance-analytics | `return-calculations` | A-PROMOTED | "Compute and compare investment return metrics including TWR, MWR (dollar-weighted IRR on portfolio cash flows), CAGR, and annualized return |
| 09-performance-analytics | `time-value-of-money` | C-VENDORED | "Calculate present value, future value, NPV, IRR for projects and loans, loan payments, and amortization schedules across all compounding co |
| 09-performance-analytics | `trade-performance-coach` | C-VENDORED | >- Review closed trades, partial exits, and monthly trade aggregates for process adherence, risk discipline, execution quality, and evidence |
| 09-performance-analytics | `trader-memory-core` | C-VENDORED | Track investment theses across their lifecycle — from screening idea to closed position with postmortem. Register theses from screener outpu |
| 09-performance-analytics | `weekly-performance-digest` | C-VENDORED | Generate a weekly performance summary from closed trader-memory-core theses — win rate, expectancy, profit factor, R-multiple, MAE/MFE, and  |
| 10-instruments-valuation | `commodities` | A-PROMOTED | "Analyze commodity markets including futures curve dynamics, roll yield, and supply/demand fundamentals. Use when the user asks about commod |
| 10-instruments-valuation | `currencies-and-fx` | A-PROMOTED | "Analyze currency markets, exchange rate mechanics, and FX risk management for international portfolios. Use when the user asks about exchan |
| 10-instruments-valuation | `digital-assets` | A-PROMOTED | "Analyze digital assets including cryptocurrency fundamentals, blockchain mechanics, DeFi protocols, and on-chain metrics. Use when the user |
| 10-instruments-valuation | `financial-statements` | C-VENDORED | "Analyze financial statements for investment decisions: derive EBITDA and free cash flow (FCFF/FCFE) from the income statement and cash flow |
| 10-instruments-valuation | `fixed-income-corporate` | C-VENDORED | "Analyze corporate bonds and credit instruments including investment grade and high yield debt. Use when the user asks about corporate bonds |
| 10-instruments-valuation | `fixed-income-sovereign` | C-VENDORED | "Analyze US Treasury securities and interest rate risk: bond pricing, yield curve construction, duration, convexity, TIPS, and forward/spot  |
| 10-instruments-valuation | `fixed-income-structured` | C-VENDORED | "Analyze structured fixed income products including mortgage-backed securities, asset-backed securities, and CLOs. Use when the user asks ab |
| 10-instruments-valuation | `fund-vehicles` | C-VENDORED | "Compare and select investment vehicles including mutual funds, ETFs, index funds, and separately managed accounts. Use when the user asks a |
| 10-instruments-valuation | `options-strategy-advisor` | C-VENDORED | Options trading strategy analysis and simulation tool. Provides theoretical pricing using Black-Scholes model, Greeks calculation, strategy  |
| 10-instruments-valuation | `qualitative-valuation` | C-VENDORED | "Assess business quality, competitive positioning, and sustainability of value creation beyond financial models. Use when the user asks abou |
| 10-instruments-valuation | `quantitative-valuation` | C-VENDORED | "Estimate intrinsic value of stocks and companies using DCF, dividend discount models, comparable multiples, and residual income. Use when t |
| 11-analyst-agents | `kanchi-dividend-review-monitor` | C-VENDORED | Monitor dividend portfolios with Kanchi-style forced-review triggers (T1-T5) and convert anomalies into OK/WARN/REVIEW states without auto-s |
| 11-analyst-agents | `kanchi-dividend-sop` | C-VENDORED | Convert Kanchi-style dividend investing into a repeatable US-stock operating procedure. Use when users ask for かんち式配当投資, dividend screening, |
| 11-analyst-agents | `kanchi-dividend-us-tax-accounting` | C-VENDORED | Provide US dividend tax and account-location workflow for Kanchi-style income portfolios. Use when users ask about qualified vs ordinary div |
| 11-analyst-agents | `market-news-analyst` | C-VENDORED | This skill should be used when analyzing recent market-moving news events and their impact on equity markets and commodities. Use this skill |
| 11-analyst-agents | `stanley-druckenmiller-investment` | C-VENDORED | Druckenmiller Strategy Synthesizer - Integrates 8 upstream skill outputs (Market Breadth, Uptrend Analysis, Market Top, Macro Regime, FTD De |
| 11-analyst-agents | `technical-analyst` | C-VENDORED | This skill should be used when analyzing weekly price charts for stocks, stock indices, cryptocurrencies, or forex pairs. Use this skill whe |
| 11-analyst-agents | `us-stock-analysis` | C-VENDORED | Comprehensive US stock analysis including fundamental analysis (financial metrics, business quality, valuation), technical analysis (indicat |
| 12-meta-skill-tooling | `dual-axis-skill-reviewer` | C-VENDORED | DEMOTED 2026-07-20: duplicates this repo's own knowledge gates (409 frontmatter tests, inventory, links) and the `spec-auditor` agent. | "Review skills in any project using a dual-axis method: (1) deterministic code-based checks (structure, scripts, tests, execution safety) an |
| 12-meta-skill-tooling | `skill-designer` | C-VENDORED | Design new Claude skills from structured idea specifications. Use when the skill auto-generation pipeline needs to produce a Claude CLI prom |
| 12-meta-skill-tooling | `skill-idea-miner` | C-VENDORED | Mine Claude Code session logs for skill idea candidates. Use when running the weekly skill generation pipeline to extract, score, and backlo |
| 12-meta-skill-tooling | `skill-integration-tester` | C-VENDORED | Validate multi-skill workflows defined in CLAUDE.md by checking skill existence, inter-skill data contracts (JSON schema compatibility), fil |
| 12-meta-skill-tooling | `trading-skills-navigator` | C-VENDORED | >- Recommend the right trading workflow, skillset, API profile, and setup path from a natural-language goal. Use this as the on-ramp when a  |
| 13-cross-asset-quant | `xasset-alpha-engine` | A-PROMOTED | "Cross-asset quantitative engine for FX (majors and emerging markets), crypto, equity indices and commodities. Builds trend/carry/value sign |
