# Slide 6/7 — INTELLIGENCE: News + AI Analysis + Dashboard Generation

> 6 DAGs | 3 ACTIVE + 3 PAUSED | 5 news sources, LLM analysis, 9x7 forecasting output
> "What does the market say? Generate daily intelligence + populate dashboard pages."

```mermaid
flowchart TB
    subgraph NEWS_IN["NEWS INGESTION (5 Sources)"]
        PORT["Portafolio.co<br/>RSS + feedparser<br/>486 articles"]
        INVS["Investing.com<br/>Search API + cloudscraper<br/>123 articles"]
        GDELT["GDELT 2.0<br/>Rate limited<br/>Currently inactive"]
        LREP["La Republica<br/>RSS + feedparser<br/>Low match rate"]
        NAPI["NewsAPI.org<br/>No API key<br/>Disabled"]
    end

    subgraph NEWS_DAGS["NEWS PIPELINE DAGs"]
        NDP["<b>news_daily_pipeline</b><br/>3x/day: 02:00, 07:00, 13:00 COT | ACTIVE<br/><br/>1. Ingest from 5 sources (SourceRegistry)<br/>2. Enrich: categorize 9 categories<br/>   monetary_policy, fx_market, commodities<br/>   inflation, fiscal, risk_premium, etc<br/>3. Relevance score 0-1<br/>   keyword 60% + source 20% + recency 20%<br/>4. Sentiment -1 to +1<br/>   GDELT tone primary, VADER fallback<br/>5. NER + keywords extraction<br/>6. Cross-reference clustering (Jaccard)<br/>7. Export ~60 daily features"]
        NAM["<b>news_alert_monitor</b><br/>every 5 min market hours | PAUSED<br/>GDELT crisis keyword scan"]
        NWD["<b>news_weekly_digest</b><br/>Friday 18:00 COT | PAUSED<br/>Weekly text summary"]
        NMT["<b>news_maintenance</b><br/>Sunday | PAUSED<br/>Cleanup + vacuum"]
    end

    subgraph ANALYSIS["L8 AI ANALYSIS DAG"]
        L8["<b>analysis_l8_daily_generation</b><br/>14:00 COT Mon-Fri | ACTIVE<br/><br/>Sensor: waits for news_daily_pipeline<br/><br/>DAILY (Mon-Fri):<br/>  1. MacroAnalyzer computes<br/>     SMA-5/10/20/50, Bollinger,<br/>     RSI Wilders, MACD, ROC, z-score<br/>     for 8 macro variables<br/>  2. LLM generates daily headline<br/>     + sentiment + analysis (Spanish)<br/>     Azure OpenAI primary, Claude fallback<br/>  3. Update weekly JSON incrementally<br/><br/>FRIDAY ONLY:<br/>  4. Generate full weekly summary<br/>  5. Generate 8 macro chart PNGs<br/>  6. Rebuild analysis_index.json<br/><br/>Cost: ~$0.004/week"]
    end

    subgraph FORECAST_GEN["FORECASTING DASHBOARD GENERATION"]
        FWG["<b>forecast_weekly_generation</b><br/>Monday 09:00 COT | ACTIVE<br/><br/>Subprocess: generate_weekly_forecasts.py<br/><br/>For EACH of 19 weeks:<br/>  For EACH of 9 models:<br/>    For EACH of 7 horizons:<br/>      1. target = ln(close_t+H / close_t)<br/>      2. Walk-forward validation<br/>      3. Train final on all data<br/>      4. Predict future return<br/><br/>= 63 backtest + 63 forward per week<br/>= 1,260 CSV rows + 310 PNGs<br/><br/>Runtime: ~90 min (19 weeks)"]
    end

    subgraph NEWS_DB["NEWS DB TABLES"]
        ART[("news_articles<br/>609 rows")]
        FEAT[("news_feature_snapshots<br/>23 rows, ~60 features/day")]
        DIG[("news_daily_digests<br/>23 rows")]
    end

    subgraph ANALYSIS_OUT["ANALYSIS OUTPUT (JSON files)"]
        WEEKLY_JSON["weekly_2026_W01.json<br/>... through ...<br/>weekly_2026_W16.json<br/>(16 weeks)"]
        INDEX["analysis_index.json"]
        CHARTS["charts/macro_*_WNN.png<br/>(8 vars x 16 weeks)"]
    end

    subgraph FORECAST_OUT["FORECASTING OUTPUT"]
        CSV["bi_dashboard_unified.csv<br/>1,260 rows (63 BT + 1197 FF)"]
        PNGS["310 PNGs:<br/>63 backtest + 247 forward<br/>(13 per week x 19 weeks)"]
    end

    subgraph DASHBOARD["DASHBOARD PAGES"]
        ANAL_PAGE["/analysis<br/>WeekSelector + DailyTimeline<br/>MacroChartGrid + SignalCards<br/>FloatingChatWidget"]
        FORE_PAGE["/forecasting<br/>MetricsRankingPanel<br/>BacktestView + ForwardView<br/>Consensus + Ensembles"]
    end

    PORT & INVS & GDELT & LREP & NAPI --> NDP
    NDP --> ART & FEAT & DIG
    NDP -.->|"ExternalTaskSensor"| L8

    L8 --> WEEKLY_JSON & INDEX & CHARTS
    FWG --> CSV & PNGS

    WEEKLY_JSON --> ANAL_PAGE
    CHARTS --> ANAL_PAGE
    CSV --> FORE_PAGE
    PNGS --> FORE_PAGE

    style NEWS_DAGS fill:#1e3a5f,stroke:#3b82f6,color:#dbeafe
    style ANALYSIS fill:#4a1d96,stroke:#8b5cf6,color:#ede9fe
    style FORECAST_GEN fill:#064e3b,stroke:#10b981,color:#d1fae5
    style DASHBOARD fill:#7c2d12,stroke:#ea580c,color:#fed7aa
```

## Forward Forecast PNG Anatomy

```
Each model PNG shows predicted USDCOP price at 7 future dates:

Price ($)
  4,400 |
        |        .--- H=10: $4,328
  4,350 |   .----'
        |  /         .--- H=20: $4,290
  4,300 |/----------'
        |    '--- H=5: $4,343     '--- H=30: $4,265
  4,250 |
        +----+----+----+----+----+----+----+---
         Today H=1 H=5 H=10 H=15 H=20 H=25 H=30
              1d   1w   2w   3w   1m   5w   6w

  pred_price[H] = base_price x exp(model.predict(features))
```

## 13 PNGs Per Week

| # | PNG | Content |
|---|-----|---------|
| 1-9 | `forward_{model}_{week}.png` | Price curve for each of 9 models |
| 10 | `forward_consensus_{week}.png` | Average of all 9 models |
| 11 | `forward_ensemble_top_3_{week}.png` | Top 3 by direction accuracy |
| 12 | `forward_ensemble_best_of_breed_{week}.png` | Best linear + best boosting + best hybrid |
| 13 | `forward_ensemble_top_6_mean_{week}.png` | Top 6 averaged |

## LLM Configuration

| Setting | Value |
|---------|-------|
| Primary | Azure OpenAI GPT-4o-mini |
| Fallback | Anthropic Claude Sonnet |
| Language | Spanish |
| Cache | File-based, TTL 24h |
| Budget | $1/day, $15/month |
| Actual cost | ~$0.004/week |
