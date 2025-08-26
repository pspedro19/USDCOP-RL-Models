# DIAGRAMA DE ARQUITECTURA DE PIPELINES - USDCOP TRADING
================================================================================

## ARQUITECTURA COMPLETA

```mermaid
graph TB
    %% Fuentes de Datos
    subgraph "DATA SOURCES"
        MT5[MT5 API<br/>26,669 records]
        TD[TwelveData API<br/>258,583 records]
    end

    %% Capa Bronze
    subgraph "BRONZE LAYER - Ingesta"
        B1[bronze_pipeline_smart.py<br/>Detección inteligente]
        B2[bronze_pipeline_enhanced.py<br/>Consolidación]
        B3[bronze_pipeline_utc.py<br/>Normalización UTC]
        
        MT5 --> B1
        TD --> B1
        B1 --> B2
        B2 --> B3
        B3 --> BronzeData[(Bronze Storage<br/>258,583 records)]
    end

    %% Capa Silver
    subgraph "SILVER LAYER - Limpieza"
        S1[silver_pipeline_enhanced.py<br/>Limpieza general]
        S2[silver_pipeline_premium_only.py<br/>Premium Session Filter]
        
        BronzeData --> S2
        S2 --> SilverData[(Silver Storage<br/>86,272 records<br/>90.9% complete)]
    end

    %% Capa Gold
    subgraph "GOLD LAYER - Features"
        G1[gold_pipeline.py<br/>PENDIENTE]
        
        SilverData -.-> G1
        G1 -.-> GoldData[(Gold Storage<br/>50+ features)]
    end

    %% Capa Diamond
    subgraph "DIAMOND LAYER - ML Ready"
        D1[diamond_stage.py<br/>Preparación básica]
        D2[diamond_stage_enhanced.py<br/>Validaciones]
        D3[diamond_stage_final.py<br/>Production ready]
        D4[diamond_stage_optimized.py<br/>Optimizado]
        
        GoldData -.-> D3
        D3 --> DiamondData[(Diamond Storage<br/>ML Ready)]
    end

    %% ML y Trading
    subgraph "ML & TRADING"
        ML[ML Models<br/>PPO/A2C/DQN]
        SG[Signal Generator]
        OE[Order Executor]
        
        DiamondData --> ML
        ML --> SG
        SG --> OE
    end

    %% Orquestación
    subgraph "ORCHESTRATION"
        MP[master_pipeline.py<br/>Coordinador principal]
        PO[pipeline_orchestrator.py<br/>Patrón Saga]
        AF[Apache Airflow<br/>3 DAGs]
        
        MP --> B1
        MP --> S2
        MP --> D3
        AF --> MP
    end

    %% Tiempo Real
    subgraph "REALTIME"
        RT[start_realtime_pipeline.py<br/>Streaming <100ms]
        
        MT5 --> RT
        RT --> ML
    end

    %% Servicios
    subgraph "INFRASTRUCTURE"
        Docker[Docker<br/>16 containers]
        Kafka[Kafka<br/>Streaming]
        MinIO[MinIO<br/>Object Storage]
        MLflow[MLflow<br/>Tracking]
        DB[(PostgreSQL<br/>Database)]
        
        Docker --> Kafka
        Docker --> MinIO
        Docker --> MLflow
        Docker --> DB
    end

    %% Dashboards
    subgraph "DASHBOARDS"
        D1B[Premium Dashboard<br/>:8082]
        D2B[Backtest Dashboard<br/>:8083]
        D3B[Analysis Dashboard<br/>:8084]
        D4B[ML Dashboard<br/>:8085]
        D5B[Trading Dashboard<br/>:8086]
        
        SilverData --> D1B
        DiamondData --> D2B
        ML --> D4B
        OE --> D5B
    end

    %% Monitoring
    subgraph "MONITORING"
        Health[Health Checks]
        Trace[FULL_PIPELINE_TRACE.py]
        
        MP --> Trace
        Docker --> Health
    end

    classDef source fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef bronze fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef silver fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef gold fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef diamond fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef ml fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef infra fill:#e0f2f1,stroke:#004d40,stroke-width:2px
    classDef pending fill:#ffebee,stroke:#b71c1c,stroke-width:2px,stroke-dasharray: 5 5

    class MT5,TD source
    class B1,B2,B3,BronzeData bronze
    class S1,S2,SilverData silver
    class G1,GoldData pending
    class D1,D2,D3,D4,DiamondData diamond
    class ML,SG,OE ml
    class Docker,Kafka,MinIO,MLflow,DB infra
```

## FLUJO DE PROCESAMIENTO DETALLADO

```mermaid
sequenceDiagram
    participant API as APIs (MT5/Twelve)
    participant Bronze as Bronze Pipeline
    participant Silver as Silver Pipeline
    participant Gold as Gold Pipeline
    participant Diamond as Diamond Pipeline
    participant ML as ML Models
    participant Trade as Trading System

    API->>Bronze: Raw data (293K)
    Bronze->>Bronze: Smart detection
    Bronze->>Bronze: Consolidation
    Bronze->>Bronze: UTC normalization
    Bronze->>Silver: 258K records

    Silver->>Silver: Filter Premium only
    Silver->>Silver: Clean anomalies
    Silver->>Silver: Impute gaps
    Silver->>Gold: 86K clean records

    Gold-->>Gold: Feature engineering (PENDING)
    Gold-->>Diamond: 50+ features

    Diamond->>Diamond: Train/test split
    Diamond->>Diamond: Normalization
    Diamond->>ML: ML-ready dataset

    ML->>Trade: Predictions
    Trade->>Trade: Generate signals
    Trade->>Trade: Execute orders
```

## COMPONENTES POR CAPA

### 🥉 BRONZE (3 pipelines)
```
┌─────────────────────────────────────┐
│  1. bronze_pipeline_smart.py        │ ← Detección inteligente
│  2. bronze_pipeline_enhanced.py     │ ← Consolidación
│  3. bronze_pipeline_utc.py          │ ← Normalización UTC
└─────────────────────────────────────┘
Output: 258,583 registros consolidados
```

### 🥈 SILVER (2 pipelines)
```
┌─────────────────────────────────────┐
│  4. silver_pipeline_enhanced.py     │ ← Limpieza general
│  5. silver_pipeline_premium_only.py │ ← PREMIUM ONLY ⭐
└─────────────────────────────────────┘
Output: 86,272 registros (90.9% completos)
```

### 🥇 GOLD (0 pipelines - PENDIENTE)
```
┌─────────────────────────────────────┐
│  6. gold_pipeline.py                │ ← Por implementar
└─────────────────────────────────────┘
Output esperado: Dataset con 50+ features
```

### 💎 DIAMOND (4 pipelines)
```
┌─────────────────────────────────────┐
│  7. diamond_stage.py                │ ← Básico
│  8. diamond_stage_enhanced.py       │ ← Con validaciones
│  9. diamond_stage_final.py          │ ← Production ready ⭐
│  10. diamond_stage_optimized.py     │ ← Optimizado 3x
└─────────────────────────────────────┘
Output: Dataset ML-ready
```

### 🔄 ORCHESTRATION (3 pipelines)
```
┌─────────────────────────────────────┐
│  11. master_pipeline.py             │ ← Coordinador principal
│  12. pipeline_orchestrator.py       │ ← Patrón Saga
│  13. pipeline.py (usdcop)           │ ← Específico USDCOP
└─────────────────────────────────────┘
```

### 📡 REALTIME (2 pipelines)
```
┌─────────────────────────────────────┐
│  14. start_realtime_pipeline.py     │ ← Streaming <100ms
│  15. FULL_PIPELINE_TRACE.py         │ ← Debugging
└─────────────────────────────────────┘
```

## INFRAESTRUCTURA DOCKER

```yaml
Services (16 containers):
  ├── postgresql (DB)
  ├── redis (Cache)
  ├── minio (S3 Storage)
  ├── kafka (Streaming)
  ├── zookeeper (Kafka coord)
  ├── mlflow (ML tracking)
  ├── airflow-webserver
  ├── airflow-scheduler
  ├── airflow-worker
  ├── airflow-init
  ├── airflow-triggerer
  ├── dashboard-premium (:8082)
  ├── dashboard-backtest (:8083)
  ├── dashboard-analysis (:8084)
  ├── dashboard-ml (:8085)
  └── dashboard-trading (:8086)
```

## CALIDAD POR ETAPA

```
Bronze:  ⭐⭐⭐     (75% completitud, datos crudos)
Silver:  ⭐⭐⭐⭐⭐   (90.9% completitud, Premium only)
Gold:    PENDIENTE  (Feature engineering)
Diamond: ⭐⭐⭐⭐⭐   (100% ML-ready)
```

## DECISIONES CLAVE

1. **Solo Premium Session**: 08:00-14:00 COT por 91.4% completitud
2. **Descarte de London/Afternoon**: <60% completitud inaceptable
3. **Pipeline Smart**: Evita re-descargas innecesarias
4. **Imputación conservadora**: Solo gaps <30 minutos
5. **16 servicios Docker**: Todo accesible vía web

---

*Arquitectura optimizada para calidad de datos y eficiencia de procesamiento*