# USDCOP Trading System — Final MLOps Project

**Course**: _MLOps / Advanced ML Engineering_ (placeholder)
**Team size**: Up to 4 members
**Presentation date**: 2026-04-23
**Repository due**: 2026-04-30
**Repository**: `USDCOP-RL-Models`

---

## Table of Contents

1. [Objective](#1-objective)
2. [Course Requirements Compliance](#2-course-requirements-compliance)
3. [Architecture Overview](#3-architecture-overview)
4. [Tech Stack Detail](#4-tech-stack-detail)
5. [Course Technologies in Depth](#5-course-technologies-in-depth)
6. [How to Run](#6-how-to-run)
7. [Live Demo Script](#7-live-demo-script-10-min)
8. [Repository Layout](#8-repository-layout)
9. [Evaluation Self-Check](#9-evaluation-self-check)
10. [Team Members](#10-team-members)
11. [Presentation](#11-presentation)

---

## 1. Objective

Implement a complete, production-grade **MLOps lifecycle** for a USDCOP (Colombian Peso) directional trading model. The project prioritizes **infrastructure and operational maturity over model precision** — the toy forecasting model (Ridge + BayesianRidge ensemble with a regime gate) is simply the vehicle to exercise the full MLOps stack: containerized services, Airflow orchestration, MLflow experiment tracking, gRPC microservices for low-latency inference, Kafka streaming for decoupled signal distribution, observability (Prometheus + Grafana + Loki), and a Next.js dashboard for human-in-the-loop approval. The goal is to demonstrate that the _system around the model_ is what enables safe, reproducible, and observable deployment — not the model itself.

---

## 2. Course Requirements Compliance

| Requirement | Our implementation | Evidence |
|---|---|---|
| ≥2 course techs (non-REST) | **gRPC + Kafka** | `services/grpc_predictor/`, `services/kafka_bridge/` |
| Docker | 15+ containerized services | `docker-compose.compact.yml` |
| Orchestration | Airflow (27 DAGs across L0–L8) | `airflow/dags/` |
| MLflow | Tracking server deployed | `http://localhost:5001` |
| Live demo | Scripted end-to-end flow | `make course-demo` |
| Cloud deployment | _Optional — not done (local Docker)_ | N/A |
| Federated learning | _Optional — not done_ | N/A |

---

## 3. Architecture Overview

```
+------------------+     +------------------+     +----------------------+
|   L0: Data       | --> |  L1: Features    | --> |  L3: Train (Airflow) |
|   TwelveData +   |     |  Canonical       |     |  + MLflow tracking   |
|   macro sources  |     |  feature builder |     |  (Ridge + BR + XGB)  |
+------------------+     +------------------+     +----------+-----------+
                                                             |
                                                             v
+-----------------------------------------------+  +----------------------+
|  L5: Signal generation DAG                     |  |  MLflow Registry    |
|  forecast_h5_l5_weekly_signal                  |  |  :5001              |
|                                                |  +----------------------+
|   +------------------+   +-------------------+ |
|   | Kafka producer   |   | gRPC Predict()    | |  --- COURSE TECHS ---
|   | topic signals.h5 |   | port 50051        | |  [1] gRPC  (low-lat RPC)
|   +------------------+   +-------------------+ |  [2] Kafka (streaming)
+---------+---------------------+----------------+
          |                     |
          v                     v
+----------------------+  +----------------------+
|  Kafka consumer      |  |  SignalBridge OMS    |
|  services/           |  |  (FastAPI, :8085)    |
|  kafka_bridge/       |  |  Risk + exchange     |
+----------+-----------+  +----------+-----------+
           |                         |
           +------------+------------+
                        v
            +------------------------------+
            |  Next.js Dashboard (:5000)    |
            |  /forecasting /dashboard      |
            |  /production /analysis        |
            +------------------------------+

Observability: Prometheus :9090 -> Grafana :3002 -> Loki :3100 (logs)
Storage:       PostgreSQL+TimescaleDB :5432, Redis :6379, MinIO :9001
```

Slides (7) are already rendered under [`docs/slides/`](./slides/). They mirror the layers above and are the backbone of the 23-apr presentation.

---

## 4. Tech Stack Detail

| Service | Purpose | Port | Tech |
|---|---|---|---|
| PostgreSQL + TimescaleDB | OHLCV (5-min/daily), macro, signals, executions | 5432 | Postgres 15 + TimescaleDB |
| Redis | Cache + streams + Celery broker | 6379 | Redis 7-alpine |
| MinIO | Object storage (MLflow artifacts, seeds) | 9001 | MinIO |
| Airflow (webserver + scheduler) | Orchestration: 27 DAGs | 8080 | Airflow 2.x |
| MLflow | Experiment tracking + model registry | 5001 | MLflow (SQLite + MinIO) |
| **Redpanda** | Kafka-compatible streaming broker | 9092, 8088 (Console) | Redpanda (single-binary) |
| **gRPC Predictor** | Low-latency `Predict(features)` service | 50051 | Python + `grpcio` |
| **Kafka Bridge** | Consumer that ingests `signals.h5` | — | Python + `confluent-kafka` |
| SignalBridge OMS | Order routing + risk + exchange adapters | 8085 | FastAPI + CCXT |
| Prometheus | Metrics scraping + alerting | 9090 | Prometheus |
| Grafana | Dashboards (4 provisioned) | 3002 | Grafana |
| Loki + Promtail | Log aggregation from Docker socket | 3100 | Loki 2.9 |
| Next.js Dashboard | `/forecasting`, `/dashboard`, `/production`, `/analysis`, `/execution` | 5000 | Next.js 14 |

---

## 5. Course Technologies in Depth

### 5.1 gRPC — Low-latency model inference

**What.** A dedicated gRPC microservice at [`services/grpc_predictor/`](../services/grpc_predictor/) exposes a single RPC:

```proto
// services/grpc_predictor/proto/predictor.proto
service Predictor {
  rpc Predict (FeaturesRequest) returns (PredictionResponse);
}
message FeaturesRequest  { repeated double features = 1; string model_id = 2; }
message PredictionResponse { int32 direction = 1; double confidence = 2; string model_id = 3; }
```

**Why gRPC instead of REST.** The REST SignalBridge API already exists as the OMS baseline, but model inference has hard latency requirements (p99 < 100ms per `sdd-observability.md` SLA). gRPC gives us: (1) HTTP/2 multiplexing over a single persistent connection, (2) Protobuf binary encoding (smaller + faster than JSON), (3) strongly-typed service contracts generated from `.proto` files. This lets downstream Airflow tasks and the SignalBridge call the predictor with single-digit millisecond overhead.

**How.** The predictor loads the latest approved model from MLflow at startup and serves requests on `:50051`. Example client usage:

```python
import grpc
from services.grpc_predictor.proto import predictor_pb2, predictor_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
stub = predictor_pb2_grpc.PredictorStub(channel)
response = stub.Predict(predictor_pb2.FeaturesRequest(
    features=[0.12, -0.05, 1.3, ...],  # 23-dim feature vector
    model_id="smart_simple_v2",
))
print(response.direction, response.confidence)  # -1, 0.82
```

### 5.2 Kafka (via Redpanda) — Signal streaming

**What.** The L5 signal DAG publishes each generated signal to Kafka topic `signals.h5`, which is consumed by the Kafka Bridge service (and optionally by external subscribers like audit loggers or alternative OMS backends).

**Why Redpanda.** Redpanda is Kafka API–compatible but ships as a **single Go binary** — no Zookeeper, no JVM, ~100MB memory footprint. Perfect for a course project where we need real Kafka semantics but don't want to run a 3-node ZK cluster locally. Kafka buys us **producer/consumer decoupling**: the Airflow DAG does not need to know who is listening, and consumers can replay history by resetting offsets.

**Topic schema (`signals.h5`):**

```json
{
  "signal_id": "h5_2026-W17",
  "strategy_id": "smart_simple_v2",
  "signal_date": "2026-04-20",
  "direction": -1,
  "confidence": 0.82,
  "leverage": 1.5,
  "hard_stop_pct": 2.81,
  "take_profit_pct": 1.41,
  "regime": "TRENDING",
  "hurst": 0.58,
  "timestamp": "2026-04-20T13:15:00-05:00"
}
```

**Producer**: [`airflow/dags/forecast_h5_l5_weekly_signal.py`](../airflow/dags/forecast_h5_l5_weekly_signal.py) — publishes after the signal is persisted to Postgres.
**Consumer**: [`services/kafka_bridge/consumer.py`](../services/kafka_bridge/consumer.py) — consumer group `kafka_bridge`, auto-offset-reset `latest`, logs each signal and forwards eligible ones to the SignalBridge REST OMS.

The Redpanda Console at `:8088` provides a browser UI to inspect topics, partitions, and consumer lag live during the demo.

---

## 6. How to Run

```bash
git clone <repo-url>
cd USDCOP-RL-Models

# Start all 15+ services
docker compose -f docker-compose.compact.yml up -d

# Wait ~60s for services to become healthy
docker compose -f docker-compose.compact.yml ps

# Run the end-to-end demo (8-10 minutes)
make course-demo
```

### Service URLs

| Service | URL |
|---|---|
| Airflow | http://localhost:8080 |
| MLflow | http://localhost:5001 |
| Grafana | http://localhost:3002 |
| Redpanda Console | http://localhost:8088 |
| Next.js Dashboard | http://localhost:5000 |
| SignalBridge API | http://localhost:8085/docs |
| gRPC Predictor | `localhost:50051` (use `grpcurl` or Python client) |

---

## 7. Live Demo Script (~10 min)

Run in this exact order on 2026-04-23:

| Step | Action | Duration |
|---|---|---|
| 1 | `docker compose ps` — show all green healthy services | 30s |
| 2 | Open **Airflow UI** (`:8080`) — show `forecast_h5_l3_weekly_training` DAG graph + most recent successful run | 1 min |
| 3 | Open **MLflow UI** (`:5001`) — show `h5_smart_simple` experiment, click a run, show logged params + metrics + artifacts | 1 min |
| 4 | `make course-grpc` — invokes Python client against the gRPC predictor, prints `direction=-1, confidence=0.82` | 1 min |
| 5 | `make course-kafka` — produces a test signal and shows the consumer receiving it live; open **Redpanda Console** (`:8088`) and show topic `signals.h5` with messages + consumer group lag=0 | 2 min |
| 6 | Open **Grafana** (`:3002`) — open "Trading Performance" dashboard, show P&L, Sharpe, inference latency panels | 1 min |
| 7 | Open **Next.js dashboard** (`:5000`) — walk through `/forecasting` (model zoo), then `/production` (2026 YTD trades) | 2 min |
| 8 | Show compliance checklist slide — recap ✅ gRPC + Kafka + Docker + Airflow + MLflow + Demo | 1 min |

Total: ~9.5 min. Buffer for Q&A.

---

## 8. Repository Layout

```
USDCOP-RL-Models/
├── airflow/dags/              # 27 DAGs (L0 data, L3 train, L5 signal, L6 monitor, L7 execute, L8 analysis)
│   ├── forecast_h5_l3_weekly_training.py
│   ├── forecast_h5_l5_weekly_signal.py    # <-- Kafka producer
│   └── ...
├── services/
│   ├── signalbridge_api/      # FastAPI OMS (existing REST baseline — does not count for course)
│   ├── grpc_predictor/        # COURSE TECH #1 — gRPC predictor
│   │   ├── proto/predictor.proto
│   │   ├── server.py
│   │   └── client.py
│   └── kafka_bridge/          # COURSE TECH #2 — Kafka consumer
│       └── consumer.py
├── docker-compose.compact.yml # 15 services (compact mode)
├── Makefile                   # course-demo, course-grpc, course-kafka targets
├── docs/
│   ├── COURSE_PROJECT.md      # THIS FILE
│   └── slides/                # 7 architecture slides for 23-apr presentation
├── scripts/
│   ├── demo/                  # Shell scripts orchestrating the live demo
│   └── log_training_to_mlflow.py
└── CLAUDE.md                  # Project instructions (SDD spec references)
```

For deep-dives into any layer, see the SDD specs under [`.claude/rules/`](../.claude/rules/) — they are the authoritative technical documentation.

---

## 9. Evaluation Self-Check

- ✅ ≥2 non-REST course technologies (**gRPC + Kafka**)
- ✅ Docker containerization (15+ services in `docker-compose.compact.yml`)
- ✅ Orchestration (Airflow — 27 DAGs spanning training, signal, execution, monitoring)
- ✅ MLflow experiment tracking (server at `:5001`, runs logged via `scripts/log_training_to_mlflow.py`)
- ✅ Functional live demo (`make course-demo` end-to-end, ~10 min)
- ✅ Git repository with clear structure and documentation
- ✅ Observability stack (Prometheus + Grafana + Loki — not required but included)
- ⬜ Cloud deployment _(optional — not implemented; all local Docker)_
- ⬜ Federated learning _(optional — not implemented)_

---

## 10. Team Members

| Name | Role | Email |
|---|---|---|
| _TBD_ | Infrastructure & Docker | _TBD_ |
| _TBD_ | Airflow DAGs & MLflow | _TBD_ |
| _TBD_ | gRPC microservice | _TBD_ |
| _TBD_ | Kafka streaming & dashboard | _TBD_ |

---

## 11. Presentation

Slides are exported to PPTX from [`docs/slides/`](./slides/) prior to the presentation.
Link: _`<presentation-link-placeholder>`_

---

_Last updated: 2026-04-23_
