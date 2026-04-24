# Slide 1/7 — L0 DATA OPS: Data Foundation

> 5 DAGs | ALL ACTIVE | Shared by every pipeline track
> "Everything starts here: 3 FX pairs every 5 min + 40 macro variables hourly"

```mermaid
flowchart TB
    subgraph EXT["EXTERNAL SOURCES"]
        TD["TwelveData API<br/>3 FX: COP / MXN / BRL<br/>8 API keys rotation"]
        MACRO_API["FRED + Investing.com<br/>BanRep + BCRP<br/>DANE + Fedesarrollo<br/>BanRep BOP"]
    end

    subgraph RT["L0 REALTIME — Market Hours 8:00-12:55 COT, Mon-Fri"]
        direction TB
        ORT["<b>core_l0_02_ohlcv_realtime</b><br/>every 5 min | 3 FX pairs<br/>TradingCalendar + CircuitBreaker<br/>BRL: fetch UTC, convert COT<br/><i>Freshness: less than 5 min</i>"]
        MUP["<b>core_l0_04_macro_update</b><br/>hourly 8-12 COT<br/>7 extractors via ExtractorRegistry<br/>Rewrites last 15 records/variable<br/><i>Freshness: less than 1 hour</i>"]
    end

    subgraph BATCH["L0 BATCH — Weekly + Manual"]
        OBF["<b>core_l0_01_ohlcv_backfill</b><br/>MANUAL trigger<br/>Seed restore if DB empty<br/>Gap detection MIN to MAX<br/>Exports updated seeds"]
        MBF["<b>core_l0_03_macro_backfill</b><br/>Sunday 23:00 COT<br/>7 sources full extraction<br/>Generates 9 MASTER files<br/>Regenerates MACRO_DAILY_CLEAN<br/>Runtime ~2.5h"]
    end

    subgraph BKP["L0 BACKUP — Daily Durability"]
        SBK["<b>core_l0_05_seed_backup</b><br/>Daily 15:00 COT<br/>DB to parquet + MinIO upload<br/>Atomic writes + manifest hash"]
    end

    subgraph DB["PostgreSQL + TimescaleDB"]
        OHLCV[("usdcop_m5_ohlcv<br/>286,839 rows<br/>3 FX pairs<br/>2020-01 to today")]
        MACRO[("macro_indicators_daily<br/>10,819 rows | 40 vars<br/>1954 to today")]
    end

    subgraph FS["Seed Files - Host"]
        SEEDS["seeds/latest/*.parquet<br/>MACRO_DAILY_CLEAN.parquet<br/>data/backups/seeds/*"]
    end

    subgraph S3["MinIO s3://"]
        MINIO["99-common-trading-backups<br/>seeds/YYYY-MM-DD/<br/>7.3 MB OHLCV + 263 KB macro"]
    end

    TD --> ORT & OBF
    MACRO_API --> MUP & MBF

    ORT -->|"UPSERT (time,symbol)"| OHLCV
    OBF -->|"restore + gap fill"| OHLCV
    MUP -->|"FrequencyRouter"| MACRO
    MBF -->|"full extract"| MACRO
    MBF --> SEEDS
    OBF --> SEEDS
    SBK -->|"DB to parquet"| SEEDS
    SBK -->|"boto3 put_object"| MINIO

    OHLCV -.->|"consumed by"| DOWNSTREAM["L3 Training + L1 Features + L8 Analysis"]
    MACRO -.->|"consumed by"| DOWNSTREAM
    SEEDS -.->|"consumed by"| DOWNSTREAM

    style RT fill:#064e3b,stroke:#10b981,color:#d1fae5
    style BATCH fill:#1e3a5f,stroke:#3b82f6,color:#dbeafe
    style BKP fill:#4a1d96,stroke:#8b5cf6,color:#ede9fe
    style DB fill:#7c2d12,stroke:#ea580c,color:#fed7aa
    style S3 fill:#065f46,stroke:#34d399,color:#d1fae5
```

## Data Freshness Thresholds

| Data | Max Staleness | Gate Type | Consequence |
|------|---------------|-----------|-------------|
| OHLCV 5-min | 3 days | BLOCKING | Training halted |
| Macro daily | 7 days | BLOCKING | Training halted |
| Model artifacts | 10 days | WARNING | Inference continues |
| News articles | 24 hours | NONE | Analysis lacks context |

## Key Tables

| Table | PK | Update Frequency | Consumers |
|-------|-----|-----------------|-----------|
| `usdcop_m5_ohlcv` | `(time, symbol)` | Every 5 min | L1, L3 H1/H5, L8 |
| `macro_indicators_daily` | `fecha` | Hourly | L3 H1/H5, L8 Analysis |
| `macro_indicators_monthly` | `fecha` | Weekly | L3 (macro features) |
