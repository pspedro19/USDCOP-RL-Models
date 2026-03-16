# USDCOP Trading System - Architecture Diagrams

**Version:** 2.0.0
**Date:** October 22, 2025

This document contains visual architecture diagrams using Mermaid for the USDCOP Trading System.

---

## System Overview

```mermaid
graph TB
    subgraph "Client Layer"
        Browser[Web Browser]
        Mobile[Mobile App]
        External[External Clients]
    end

    subgraph "Load Balancer"
        NGINX[NGINX Reverse Proxy<br/>Port 80/443]
    end

    subgraph "Application Layer"
        Dashboard[Dashboard<br/>Next.js<br/>Port 5000]
        TradingAPI[Trading API<br/>FastAPI<br/>Port 8000]
        AnalyticsAPI[Analytics API<br/>FastAPI<br/>Port 8001]
        PipelineAPI[Pipeline API<br/>FastAPI<br/>Port 8002]
        ComplianceAPI[Compliance API<br/>FastAPI<br/>Port 8003]
        RTOrch[RT Orchestrator<br/>Port 8085]
        WSService[WebSocket Service<br/>Port 8082]
    end

    subgraph "Orchestration Layer"
        Airflow[Apache Airflow<br/>Scheduler + Webserver<br/>Port 8080]
    end

    subgraph "Storage Layer"
        Postgres[(PostgreSQL<br/>TimescaleDB<br/>Port 5432)]
        Redis[(Redis<br/>Cache + Pub/Sub<br/>Port 6379)]
        MinIO[(MinIO<br/>S3 Storage<br/>Port 9000)]
    end

    subgraph "External Services"
        TwelveData[TwelveData API<br/>16 API Keys]
    end

    Browser -->|HTTPS| NGINX
    Mobile -->|HTTPS| NGINX
    External -->|HTTPS| NGINX

    NGINX --> Dashboard
    NGINX --> TradingAPI
    NGINX --> AnalyticsAPI

    Dashboard --> TradingAPI
    Dashboard --> AnalyticsAPI
    Dashboard --> PipelineAPI
    Dashboard --> ComplianceAPI
    Dashboard -.->|WebSocket| WSService

    TradingAPI --> Postgres
    AnalyticsAPI --> Postgres
    PipelineAPI --> Postgres
    PipelineAPI --> MinIO
    ComplianceAPI --> Postgres

    RTOrch --> Postgres
    RTOrch --> Redis
    RTOrch --> TwelveData

    WSService --> Redis

    Airflow --> Postgres
    Airflow --> MinIO
    Airflow --> TwelveData

    style Dashboard fill:#4CAF50
    style TradingAPI fill:#2196F3
    style AnalyticsAPI fill:#2196F3
    style PipelineAPI fill:#2196F3
    style ComplianceAPI fill:#2196F3
    style RTOrch fill:#FF9800
    style WSService fill:#9C27B0
    style Airflow fill:#00BCD4
```

---

## Data Pipeline Flow (L0-L6)

```mermaid
graph LR
    subgraph "L0: Raw Ingestion"
        API[TwelveData API] -->|OHLCV Data| L0[L0 Pipeline<br/>Intelligent Gap Detection]
        L0 --> PG0[(PostgreSQL)]
        L0 --> M0[(MinIO Bucket<br/>00-raw-usdcop)]
    end

    subgraph "L1: Standardization"
        M0 --> L1[L1 Pipeline<br/>Data Cleaning]
        L1 --> PG1[(PostgreSQL)]
        L1 --> M1[(MinIO Bucket<br/>01-l1-standardize)]
    end

    subgraph "L2: Preparation"
        M1 --> L2[L2 Pipeline<br/>Technical Indicators]
        L2 --> PG2[(PostgreSQL)]
        L2 --> M2[(MinIO Bucket<br/>02-l2-prepare)]
    end

    subgraph "L3: Feature Engineering"
        M2 --> L3[L3 Pipeline<br/>Feature Creation]
        L3 --> M3[(MinIO Bucket<br/>03-l3-feature)]
    end

    subgraph "L4: RL-Ready"
        M3 --> L4[L4 Pipeline<br/>Episode Generation]
        L4 --> M4[(MinIO Bucket<br/>04-l4-rlready)]
    end

    subgraph "L5: Serving"
        M4 --> L5[L5 Pipeline<br/>Model Deployment]
        L5 --> M5[(MinIO Bucket<br/>05-l5-serving)]
    end

    subgraph "L6: Backtest"
        M5 --> L6[L6 Pipeline<br/>Performance Analysis]
        L6 --> M6[(MinIO Bucket<br/>06-l6-backtest)]
        L6 --> PG6[(PostgreSQL<br/>Results)]
    end

    style L0 fill:#f44336
    style L1 fill:#FF5722
    style L2 fill:#FF9800
    style L3 fill:#FFC107
    style L4 fill:#4CAF50
    style L5 fill:#2196F3
    style L6 fill:#9C27B0
```

---

## Real-Time Data Flow

```mermaid
sequenceDiagram
    participant Market as 🏦 Market Opens<br/>08:00 AM COT
    participant Airflow as Apache Airflow
    participant L0 as L0 Pipeline
    participant DB as PostgreSQL
    participant RTOrch as RT Orchestrator
    participant Redis as Redis Pub/Sub
    participant WS as WebSocket Service
    participant Client as 💻 Dashboard

    Market->>Airflow: Trigger scheduled DAG
    Airflow->>L0: Execute L0 task
    L0->>DB: Fetch & store OHLCV bar
    L0->>DB: Update pipeline_status table
    L0-->>Airflow: Task complete

    Note over RTOrch: Polling pipeline_status

    RTOrch->>DB: Check L0 completion
    DB-->>RTOrch: L0 completed ✓
    RTOrch->>RTOrch: Start RT collection

    loop Every 5 seconds
        RTOrch->>Market: Fetch current price
        Market-->>RTOrch: Price data
        RTOrch->>DB: Upsert data
        RTOrch->>Redis: PUBLISH market_data
    end

    Redis->>WS: Subscriber receives
    WS->>Client: WebSocket push
    Client->>Client: Update chart
```

---

## WebSocket Architecture

```mermaid
graph TB
    subgraph "RT Orchestrator"
        Collector[Data Collector<br/>5-second polling]
        DBWriter[Database Writer<br/>Asyncpg]
        Publisher[Redis Publisher]
    end

    subgraph "Redis"
        Channel[market_data_channel]
    end

    subgraph "WebSocket Service"
        Subscriber[Redis Subscriber]
        ConnMgr[Connection Manager]
        WS1[WebSocket Connection 1]
        WS2[WebSocket Connection 2]
        WS3[WebSocket Connection N]
    end

    subgraph "Clients"
        Client1[Dashboard 1]
        Client2[Dashboard 2]
        Client3[Mobile App]
    end

    Collector --> DBWriter
    Collector --> Publisher
    Publisher --> Channel
    Channel --> Subscriber
    Subscriber --> ConnMgr
    ConnMgr --> WS1
    ConnMgr --> WS2
    ConnMgr --> WS3
    WS1 --> Client1
    WS2 --> Client2
    WS3 --> Client3

    style Collector fill:#FF9800
    style Publisher fill:#F44336
    style Channel fill:#E91E63
    style Subscriber fill:#9C27B0
    style ConnMgr fill:#673AB7
```

---

## Deployment Architecture

```mermaid
graph TB
    subgraph "Docker Host"
        subgraph "Network: usdcop-trading-network"
            subgraph "Frontend Tier"
                Dashboard[usdcop-dashboard]
            end

            subgraph "API Tier"
                TradingAPI[usdcop-trading-api]
                AnalyticsAPI[usdcop-analytics-api]
                PipelineAPI[usdcop-pipeline-api]
                ComplianceAPI[usdcop-compliance-api]
            end

            subgraph "Processing Tier"
                RTOrch[usdcop-realtime-orchestrator]
                AirflowScheduler[usdcop-airflow-scheduler]
                AirflowWeb[usdcop-airflow-webserver]
                WSService[usdcop-websocket]
            end

            subgraph "Data Tier"
                Postgres[usdcop-postgres-timescale]
                Redis[usdcop-redis]
                MinIO[usdcop-minio]
            end
        end
    end

    subgraph "Volumes"
        PGData[postgres_data]
        RedisData[redis_data]
        MinIOData[minio_data]
        AirflowLogs[airflow_logs]
    end

    Postgres -.-> PGData
    Redis -.-> RedisData
    MinIO -.-> MinIOData
    AirflowScheduler -.-> AirflowLogs

    style Dashboard fill:#4CAF50
    style TradingAPI fill:#2196F3
    style AnalyticsAPI fill:#2196F3
    style PipelineAPI fill:#2196F3
    style ComplianceAPI fill:#2196F3
    style RTOrch fill:#FF9800
    style AirflowScheduler fill:#00BCD4
    style AirflowWeb fill:#00BCD4
    style WSService fill:#9C27B0
    style Postgres fill:#336791
    style Redis fill:#DC382D
    style MinIO fill:#C72E49
```

---

## Database Schema (TimescaleDB)

```mermaid
erDiagram
    usdcop_m5_ohlcv ||--o{ pipeline_status : "tracked by"
    usdcop_m5_ohlcv ||--o{ rl_metrics : "generates"

    usdcop_m5_ohlcv {
        timestamptz time PK
        text symbol PK
        numeric open
        numeric high
        numeric low
        numeric close
        numeric volume
        text source
        numeric ema_20
        numeric ema_50
        numeric rsi
        numeric macd
        numeric bb_upper
        numeric bb_middle
        numeric bb_lower
    }

    pipeline_status {
        serial id PK
        text pipeline_name
        text pipeline_type
        text status
        timestamptz started_at
        timestamptz completed_at
        integer records_processed
        text error_message
    }

    rl_metrics {
        serial id PK
        date metric_date
        text symbol
        numeric win_rate
        numeric sharpe_ratio
        numeric sortino_ratio
        numeric max_drawdown
        integer total_trades
        numeric total_pnl
    }
```

---

## API Request Flow

```mermaid
sequenceDiagram
    participant Client as Dashboard
    participant NGINX as NGINX
    participant API as Trading API
    participant Pool as Connection Pool
    participant DB as PostgreSQL
    participant Cache as Redis

    Client->>NGINX: GET /api/candlesticks/USDCOP
    NGINX->>API: Forward request

    API->>Cache: Check cache
    alt Cache Hit
        Cache-->>API: Return cached data
        API-->>NGINX: 200 OK + JSON
        NGINX-->>Client: Response
    else Cache Miss
        API->>Pool: Get connection
        Pool->>DB: Query hypertable
        DB-->>Pool: Return rows
        Pool-->>API: Result set
        API->>Cache: Store in cache (TTL 60s)
        API-->>NGINX: 200 OK + JSON
        NGINX-->>Client: Response
    end
```

---

## Airflow DAG Dependencies

```mermaid
graph LR
    Start[DAG Start] --> L0[L0: Raw Ingestion]
    L0 --> L1[L1: Standardization]
    L1 --> L2[L2: Preparation]
    L2 --> L3[L3: Feature Engineering]
    L3 --> L4[L4: RL-Ready]
    L4 --> L5[L5: Serving]
    L5 --> L6[L6: Backtest]
    L6 --> End[DAG End]

    L0 -.->|On Failure| Retry1[Retry with<br/>exponential backoff]
    Retry1 -.->|Max 3 attempts| Alert1[Send alert]

    L1 -.->|On Failure| Alert2[Send alert]
    L2 -.->|On Failure| Alert3[Send alert]

    style L0 fill:#f44336
    style L1 fill:#FF5722
    style L2 fill:#FF9800
    style L3 fill:#FFC107
    style L4 fill:#4CAF50
    style L5 fill:#2196F3
    style L6 fill:#9C27B0
```

---

## Monitoring & Observability

```mermaid
graph TB
    subgraph "Services"
        TradingAPI[Trading API]
        Dashboard[Dashboard]
        Airflow[Airflow]
        DB[(PostgreSQL)]
    end

    subgraph "Metrics Collection"
        Prometheus[Prometheus<br/>Port 9090]
    end

    subgraph "Visualization"
        Grafana[Grafana<br/>Port 3002]
    end

    subgraph "Logging"
        Docker[Docker Logs]
        Files[Log Files]
    end

    subgraph "Alerting"
        Alerts[Alert Manager]
        Slack[Slack Notifications]
        Email[Email Alerts]
    end

    TradingAPI -->|Metrics| Prometheus
    Dashboard -->|Metrics| Prometheus
    Airflow -->|Metrics| Prometheus
    DB -->|Metrics| Prometheus

    Prometheus --> Grafana
    Prometheus --> Alerts

    Alerts --> Slack
    Alerts --> Email

    TradingAPI --> Docker
    Dashboard --> Docker
    Airflow --> Files

    style Prometheus fill:#E6522C
    style Grafana fill:#F46800
    style Alerts fill:#FF4081
```

---

## Security Architecture

```mermaid
graph TB
    subgraph "External"
        Internet[Internet]
    end

    subgraph "DMZ"
        Firewall[Firewall<br/>Port Filtering]
        WAF[WAF<br/>Web Application Firewall]
        NGINX[NGINX<br/>SSL Termination]
    end

    subgraph "Application Network"
        Dashboard[Dashboard]
        APIs[API Services]
    end

    subgraph "Data Network"
        DB[(PostgreSQL<br/>No external access)]
        Redis[(Redis<br/>Password protected)]
        MinIO[(MinIO<br/>Private access)]
    end

    Internet -->|HTTPS only| Firewall
    Firewall --> WAF
    WAF --> NGINX
    NGINX -->|Internal network| Dashboard
    NGINX -->|Internal network| APIs

    Dashboard -->|Private network| APIs
    APIs -->|Private network| DB
    APIs -->|Private network| Redis
    APIs -->|Private network| MinIO

    style Firewall fill:#F44336
    style WAF fill:#E91E63
    style NGINX fill:#9C27B0
```

---

## Disaster Recovery Flow

```mermaid
graph TB
    Start[Disaster Detected] --> Assess[Assess Impact]
    Assess --> Severity{Severity Level}

    Severity -->|P0: Critical| Immediate[Immediate Action<br/>0-5 min]
    Severity -->|P1: High| Urgent[Urgent Action<br/>5-15 min]
    Severity -->|P2: Medium| Scheduled[Scheduled Action<br/>1-4 hours]

    Immediate --> StopServices[Stop Services]
    StopServices --> LoadBackup[Load Latest Backup]
    LoadBackup --> RestoreDB[Restore Database]
    RestoreDB --> StartServices[Start Services]
    StartServices --> Verify[Verify System Health]

    Urgent --> Diagnose[Diagnose Issue]
    Diagnose --> Fix[Apply Fix]
    Fix --> Test[Test Fix]
    Test --> Deploy[Deploy Fix]
    Deploy --> Verify

    Scheduled --> Plan[Plan Maintenance]
    Plan --> Notify[Notify Stakeholders]
    Notify --> Execute[Execute Fix]
    Execute --> Verify

    Verify --> Success{System Healthy?}
    Success -->|Yes| PostMortem[Post-Mortem Analysis]
    Success -->|No| Escalate[Escalate to CTO]
    Escalate --> Immediate

    PostMortem --> End[Document & Close]

    style Start fill:#F44336
    style Immediate fill:#FF5722
    style Urgent fill:#FF9800
    style Scheduled fill:#FFC107
    style Success fill:#4CAF50
    style Escalate fill:#F44336
```

---

## Capacity Planning

```mermaid
graph LR
    subgraph "Current Capacity (V2.0)"
        CPU[CPU: 4 cores]
        RAM[RAM: 16 GB]
        Disk[Disk: 100 GB]
        API_RPS[API: 100 req/sec]
        WS_Conn[WebSocket: 100 clients]
    end

    subgraph "Growth Projections"
        Year1[Year 1<br/>+50% traffic]
        Year2[Year 2<br/>+100% traffic]
        Year3[Year 3<br/>+200% traffic]
    end

    subgraph "Scaling Options"
        Vertical[Vertical Scaling<br/>Bigger machines]
        Horizontal[Horizontal Scaling<br/>More instances]
        Cloud[Cloud Migration<br/>Auto-scaling]
    end

    CPU --> Year1
    RAM --> Year1
    Disk --> Year1
    API_RPS --> Year1
    WS_Conn --> Year1

    Year1 --> Year2
    Year2 --> Year3

    Year1 -.->|Option A| Vertical
    Year2 -.->|Option B| Horizontal
    Year3 -.->|Option C| Cloud

    style Year1 fill:#FFC107
    style Year2 fill:#FF9800
    style Year3 fill:#F44336
```

---

## Technology Stack

```mermaid
mindmap
  root((USDCOP<br/>Trading<br/>System))
    Frontend
      Next.js 15.5.2
      React 19
      TypeScript
      Tailwind CSS
      TanStack Query
      Lightweight Charts
    Backend
      Python 3.11
      FastAPI
      Uvicorn
      Asyncpg
      Pydantic
    Data Processing
      Apache Airflow
      Pandas
      NumPy
      TA-Lib
    Storage
      PostgreSQL 15
      TimescaleDB
      Redis 7
      MinIO (S3)
    Infrastructure
      Docker
      Docker Compose
      NGINX
      Prometheus
      Grafana
    External
      TwelveData API
      MLflow
```

---

## Future Enhancements

```mermaid
timeline
    title USDCOP Trading System Roadmap
    section Q1 2026
      Kubernetes Migration : Multi-node cluster
      Advanced Monitoring : Distributed tracing
      API Gateway : Kong integration
    section Q2 2026
      Multi-Symbol Support : USDBRL, USDMXN
      ML Model Registry : MLflow production
      A/B Testing : Strategy comparison
    section Q3 2026
      Cloud Deployment : AWS/Azure migration
      Auto-Scaling : Dynamic resources
      Multi-Region : Disaster recovery
    section Q4 2026
      Mobile App : React Native
      Advanced Analytics : Predictive models
      Multi-Tenancy : User workspaces
```

---

**For detailed architecture documentation, see:** `docs/ARCHITECTURE.md`
**For API specifications, see:** `docs/API_REFERENCE_V2.md`
**For operations, see:** `docs/RUNBOOK.md`
