"""Generate highly visual architecture diagrams for the MLOps course project presentation.

All diagrams use a consistent dark theme and are saved to docs/slides/course_diagrams/*.png
at 1920x1080 (16:9) for seamless PPTX insertion.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
from pathlib import Path

OUT = Path("docs/slides/course_diagrams")
OUT.mkdir(parents=True, exist_ok=True)

# ── Theme ─────────────────────────────────────────────────────────────
BG = "#0b1220"
TITLE_BAR = "#1e293b"
TEXT = "#ffffff"
TEXT_DIM = "#94a3b8"
GRID = "#1e293b"

# Tech category colors
DATA = ("#1e3a5f", "#3b82f6")        # fill, border
PIPELINE = ("#064e3b", "#10b981")
ML = ("#422006", "#eab308")
MLOPS = ("#3b0764", "#a855f7")
INFRA = ("#1e293b", "#475569")
GRPC = ("#7c2d12", "#ea580c")        # course tech highlight 1
KAFKA = ("#450a0a", "#ef4444")       # course tech highlight 2
UI = ("#164e63", "#06b6d4")
OUTPUT = ("#064e3b", "#22c55e")

DPI = 140

def setup(title, subtitle=None, w=16, h=9):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    # title bar
    ax.add_patch(FancyBboxPatch((0, 92), 100, 8, boxstyle="square,pad=0",
                                facecolor=TITLE_BAR, edgecolor="#334155", linewidth=1.5))
    ax.text(50, 96.5, title, ha="center", va="center", fontsize=22, fontweight="bold", color=TEXT)
    if subtitle:
        ax.text(50, 93.2, subtitle, ha="center", va="center", fontsize=12, color=TEXT_DIM)
    return fig, ax


def box(ax, x, y, w, h, label, colors, fontsize=11, sublabel=None, rot=0):
    fill, border = colors
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", mutation_aspect=0.5,
                       facecolor=fill, edgecolor=border, linewidth=2.2)
    ax.add_patch(b)
    cx = x + w/2; cy = y + h/2
    if sublabel:
        ax.text(cx, cy + 1.2, label, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=TEXT, rotation=rot)
        ax.text(cx, cy - 1.8, sublabel, ha="center", va="center", fontsize=fontsize-2, color=TEXT_DIM, rotation=rot)
    else:
        ax.text(cx, cy, label, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=TEXT, rotation=rot)


def arrow(ax, x1, y1, x2, y2, color="#94a3b8", lw=1.8, style="->"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color,
                        linewidth=lw, mutation_scale=15, connectionstyle="arc3,rad=0")
    ax.add_patch(a)


def curved_arrow(ax, x1, y1, x2, y2, color="#94a3b8", lw=1.8, rad=0.2, label=None):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", color=color,
                        linewidth=lw, mutation_scale=15, connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2 + rad*8
        ax.text(mx, my, label, ha="center", va="center", fontsize=9, color=TEXT_DIM,
                bbox=dict(facecolor=BG, edgecolor="none", pad=1))


def legend(ax, items, x=2, y=1, gap=12):
    for i, (color, label) in enumerate(items):
        fill, border = color
        ax.add_patch(Rectangle((x + i*gap, y), 1.5, 1.5, facecolor=fill, edgecolor=border))
        ax.text(x + i*gap + 2, y + 0.75, label, va="center", fontsize=9, color=TEXT_DIM)


def save(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=DPI, facecolor=BG, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  saved {path}")


# ──────────────────────────────────────────────────────────────────────
# Diagram 1 — END-TO-END MLOPS ARCHITECTURE (hero shot)
# ──────────────────────────────────────────────────────────────────────
def diagram_endtoend():
    fig, ax = setup("Arquitectura MLOps End-to-End",
                    "Flujo completo: Datos → Features → Entrenamiento → Tracking → Serving → Streaming → UI")

    # Row 1: Data sources
    box(ax, 2, 78, 14, 8, "TwelveData\nAPI", DATA, 10, "OHLCV realtime")
    box(ax, 18, 78, 14, 8, "FRED / BanRep\nBCRP / DANE", DATA, 10, "Macro")
    box(ax, 34, 78, 14, 8, "PostgreSQL +\nTimescaleDB", INFRA, 10, "Time-series DB")

    # Row 2: Airflow orchestration (wide)
    box(ax, 2, 64, 46, 9, "AIRFLOW", PIPELINE, 14, "27 DAGs: L0→L1→L3→L5→L6→L7")

    # Row 3: Training
    box(ax, 2, 50, 14, 9, "L3 Training\nDAG", PIPELINE, 11, "Sun 01:30 COT")
    box(ax, 18, 50, 14, 9, "Ridge + BR\n+ XGBoost", ML, 11, "H5 ensemble")
    box(ax, 34, 50, 14, 9, "Regime Gate\n(Hurst R/S)", ML, 11, "Trending filter")

    # Row 4: MLflow + artifacts
    box(ax, 2, 36, 14, 9, "MLflow\nTracker", MLOPS, 11, ":5001")
    box(ax, 18, 36, 14, 9, "MinIO\n(S3)", MLOPS, 11, "artifact store")
    box(ax, 34, 36, 14, 9, "Model .pkl\n+ norm_stats", INFRA, 11, "promoted")

    # Row 5: Serving layer — TWO COURSE TECHS
    box(ax, 2, 22, 22, 9, "gRPC PREDICTOR", GRPC, 14, "port 50051")
    box(ax, 26, 22, 22, 9, "KAFKA (Redpanda)", KAFKA, 14, "topic signals.h5")

    # Right column: UI + consumers
    box(ax, 60, 78, 34, 8, "Dashboard Next.js\n/forecasting • /dashboard • /production • /analysis", UI, 11, ":5000")
    box(ax, 60, 64, 34, 8, "Grafana + Prometheus", UI, 11, ":3002 — metrics + alerts")
    box(ax, 60, 50, 34, 8, "SignalBridge OMS\n(FastAPI)", UI, 11, ":8085 — CCXT MEXC/Binance")
    box(ax, 60, 36, 34, 8, "Redpanda Console", UI, 11, ":8088 — topic UI")
    box(ax, 60, 22, 16, 9, "pgAdmin", INFRA, 11, ":5050")
    box(ax, 78, 22, 16, 9, "MinIO UI", INFRA, 11, ":9001")

    # Bottom: Docker bar
    box(ax, 2, 8, 92, 8, "Docker Compose — 21 contenedores en red usdcop-trading-network", INFRA, 13,
        "PostgreSQL • Redis • Airflow • MLflow • Redpanda • gRPC • Kafka Bridge • Grafana • Prometheus • SignalBridge • Dashboard")

    # Arrows
    arrow(ax, 9, 77, 9, 73)
    arrow(ax, 25, 77, 25, 73)
    arrow(ax, 41, 77, 41, 73)
    arrow(ax, 9, 63, 9, 59)
    arrow(ax, 25, 63, 25, 59)
    arrow(ax, 41, 63, 41, 59)
    arrow(ax, 9, 49, 9, 45)
    arrow(ax, 25, 49, 25, 45)
    arrow(ax, 41, 49, 41, 45)
    # From artifacts to serving
    arrow(ax, 20, 35, 13, 31, color="#ea580c", lw=2.5)
    arrow(ax, 28, 35, 37, 31, color="#ef4444", lw=2.5)
    # Serving → UI
    arrow(ax, 24, 26, 60, 52, color="#06b6d4", lw=1.8)
    arrow(ax, 48, 26, 60, 68, color="#ef4444", lw=1.8)
    arrow(ax, 48, 30, 60, 40, color="#ef4444", lw=1.5)

    legend(ax, [
        (DATA, "Data"),
        (PIPELINE, "Airflow / Pipeline"),
        (ML, "ML Models"),
        (MLOPS, "MLOps Tracking"),
        (GRPC, "gRPC (course)"),
        (KAFKA, "Kafka (course)"),
        (UI, "UI / Observability"),
    ], x=2, y=2.5, gap=13)
    save(fig, "01_endtoend_architecture")


# ──────────────────────────────────────────────────────────────────────
# Diagram 2 — gRPC DETAIL
# ──────────────────────────────────────────────────────────────────────
def diagram_grpc():
    fig, ax = setup("gRPC Predictor — Detalle de Contrato + Flujo",
                    "Protocol Buffers + HTTP/2 — substituye llamadas REST por invocaciones tipadas de alta performance")

    # LEFT: proto definition box
    ax.add_patch(FancyBboxPatch((3, 18), 40, 62, boxstyle="round,pad=0.02",
                                facecolor="#0f172a", edgecolor="#64748b", linewidth=1.5))
    ax.text(23, 77, ".proto definition", ha="center", fontsize=13, fontweight="bold", color=TEXT)
    proto = """syntax = "proto3";
package predictor;

service PredictorService {
  rpc Predict(PredictRequest)
      returns (PredictResponse);
  rpc HealthCheck(HealthRequest)
      returns (HealthResponse);
}

message PredictRequest {
  map<string, double> features = 1;
}

message PredictResponse {
  string direction        = 1;
  double confidence       = 2;
  double ensemble_return  = 3;
  string model_version    = 4;
}

message HealthRequest  {}
message HealthResponse {
  bool   ready        = 1;
  string model_loaded = 2;
}"""
    ax.text(5, 73, proto, fontsize=10, color="#cbd5e1", family="monospace", va="top")

    # RIGHT: flow
    # Client
    box(ax, 55, 70, 35, 10, "Python Client", UI, 12, "grpcio stub")
    # server
    box(ax, 55, 45, 35, 10, "gRPC Server :50051", GRPC, 13, "services/grpc_predictor/server.py")
    # model
    box(ax, 55, 25, 35, 10, "Ridge + BR Ensemble", ML, 12, "joblib.load from outputs/")

    # Arrows (client ↔ server)
    arrow(ax, 72, 69, 72, 56, color="#ea580c", lw=2.5)
    ax.text(74, 62, "Predict({features})", fontsize=10, color=TEXT, fontweight="bold")
    arrow(ax, 72, 56, 72, 69, color="#22c55e", lw=2.5, style="<-")
    ax.text(74, 66, "PredictResponse", fontsize=10, color=TEXT, fontweight="bold")

    # server ↔ model
    arrow(ax, 72, 44, 72, 35, color="#eab308", lw=2)
    arrow(ax, 72, 35, 72, 44, color="#eab308", lw=2, style="<-")

    # Footer — why gRPC
    ax.add_patch(FancyBboxPatch((3, 4), 87, 10, boxstyle="round,pad=0.02",
                                facecolor="#0f172a", edgecolor=GRPC[1], linewidth=1.5))
    ax.text(46.5, 11, "¿Por qué gRPC en lugar de REST?", ha="center", fontsize=12, fontweight="bold", color=GRPC[1])
    ax.text(46.5, 7, "✓ Contratos tipados (proto3)   ✓ HTTP/2 multiplexado   ✓ Codificación binaria 3-10× más rápida que JSON   ✓ Streaming bidireccional",
            ha="center", fontsize=10, color=TEXT_DIM)

    save(fig, "02_grpc_detail")


# ──────────────────────────────────────────────────────────────────────
# Diagram 3 — KAFKA FLOW
# ──────────────────────────────────────────────────────────────────────
def diagram_kafka():
    fig, ax = setup("Kafka / Redpanda — Stream de Signals H5",
                    "Airflow DAG → topic signals.h5 → consumer(s) — desacoplamiento asíncrono productor/consumidor")

    # Producer side
    box(ax, 3, 65, 22, 14, "Airflow DAG\nforecast_h5_l5", PIPELINE, 12, "weekly signal job")
    box(ax, 3, 45, 22, 14, "Kafka Bridge\nProducer", KAFKA, 13, "services/kafka_bridge/\nproducer.py")
    arrow(ax, 14, 64, 14, 60, color="#ef4444", lw=2.5)

    # Broker (center)
    ax.add_patch(FancyBboxPatch((35, 40), 30, 35, boxstyle="round,pad=0.02",
                                facecolor="#450a0a", edgecolor=KAFKA[1], linewidth=3))
    ax.text(50, 73, "REDPANDA", ha="center", fontsize=15, fontweight="bold", color=TEXT)
    ax.text(50, 70, "(Kafka-compatible)", ha="center", fontsize=10, color=TEXT_DIM)
    # Topic
    for i, off in enumerate(range(3, 9)):
        ax.add_patch(Rectangle((38 + i*4, 55), 3.5, 8, facecolor="#7f1d1d", edgecolor="#fca5a5"))
        ax.text(39.75 + i*4, 59, f"msg\n{off}", ha="center", va="center", fontsize=8, color=TEXT)
    ax.text(50, 52, "topic: signals.h5", ha="center", fontsize=11, fontweight="bold", color="#fca5a5")
    ax.text(50, 49, "partition 0 / 1 replica", ha="center", fontsize=9, color=TEXT_DIM)
    ax.text(50, 44, "Consumer group:\nsignalbridge-consumer", ha="center", fontsize=9, color=TEXT_DIM)

    # Consumer side
    box(ax, 75, 65, 22, 14, "Kafka Bridge\nConsumer", KAFKA, 13, "consumer.py")
    box(ax, 75, 45, 22, 14, "SignalBridge\n(FastAPI OMS)", UI, 12, "future: exec orders")

    arrow(ax, 25, 52, 35, 57, color="#ef4444", lw=2.5)
    arrow(ax, 65, 57, 75, 72, color="#ef4444", lw=2.5)
    arrow(ax, 86, 64, 86, 60, color="#06b6d4", lw=2)

    # Message schema
    ax.add_patch(FancyBboxPatch((5, 8), 90, 28, boxstyle="round,pad=0.02",
                                facecolor="#0f172a", edgecolor="#64748b", linewidth=1.5))
    ax.text(50, 33, "JSON Message Schema", ha="center", fontsize=12, fontweight="bold", color=TEXT)
    msg = """{
  "week": "2026-W17",         "direction": "SHORT",        "confidence": 0.85,
  "ensemble_return": -0.012,  "skip_trade": false,         "hard_stop_pct": 2.81,
  "take_profit_pct": 1.41,    "adjusted_leverage": 1.5,    "regime": "trending",
  "hurst": 0.55,              "timestamp": "2026-04-23T14:00:00-05:00",
  "source": "airflow_h5_l5"
}"""
    ax.text(50, 21, msg, ha="center", fontsize=10, color="#cbd5e1", family="monospace")

    save(fig, "03_kafka_flow")


# ──────────────────────────────────────────────────────────────────────
# Diagram 4 — STACK LAYERS
# ──────────────────────────────────────────────────────────────────────
def diagram_stack():
    fig, ax = setup("Stack Tecnológico — Capas del Sistema",
                    "De infraestructura a UI, destacando las 2 tecnologías del curso (no-REST)")

    layers = [
        # (y, h, title, items_str, color)
        (78, 11, "UI / Dashboard", "Next.js 14 + React + Recharts  •  /forecasting • /dashboard • /production • /analysis", UI),
        (66, 11, "Observability", "Prometheus (:9090) + Grafana (:3002) + Loki + AlertManager  •  53 alert rules", UI),
        (54, 11, "Course Tech #1 — gRPC", ".proto contract  •  grpcio server :50051  •  PredictorService.Predict()", GRPC),
        (42, 11, "Course Tech #2 — Kafka (Redpanda)", "broker :19092  •  topic signals.h5  •  producer + consumer + Console :8088", KAFKA),
        (30, 11, "MLOps", "Airflow (27 DAGs)  •  MLflow (:5001, SQLite + MinIO artifacts)  •  SignalBridge FastAPI (:8085)", MLOPS),
        (18, 11, "ML Models", "Ridge + BayesianRidge + XGBoost ensemble  •  Regime Gate (Hurst R/S)  •  Dynamic Leverage", ML),
        (6, 11, "Data & Storage", "PostgreSQL + TimescaleDB  •  Redis  •  MinIO S3 (11 buckets)  •  27 FX + macro sources", DATA),
    ]

    for y, h, title, items, col in layers:
        fill, border = col
        ax.add_patch(FancyBboxPatch((3, y), 94, h, boxstyle="round,pad=0.02",
                                    facecolor=fill, edgecolor=border, linewidth=2.2))
        ax.text(8, y + h/2 + 1, title, fontsize=14, fontweight="bold", color=TEXT, va="center")
        ax.text(8, y + h/2 - 2.2, items, fontsize=10, color=TEXT_DIM, va="center")

    save(fig, "04_stack_layers")


# ──────────────────────────────────────────────────────────────────────
# Diagram 5 — DEMO FLOW (step by step)
# ──────────────────────────────────────────────────────────────────────
def diagram_demo():
    fig, ax = setup("Flujo de la Demo en Vivo",
                    "8 pasos, ~10 minutos — `make course-demo` ejecuta todo en secuencia")

    steps = [
        (1, "docker\ncompose ps", "21 servicios\nhealthy", (5, 70)),
        (2, "Airflow UI", "27 DAGs\nL3 training", (22, 70)),
        (3, "MLflow UI", "4 experiments\nruns + metrics", (39, 70)),
        (4, "make\ncourse-grpc", "Predict(features)\n→ SHORT 0.85", (56, 70)),
        (5, "make\ncourse-kafka", "producer →\ntopic → consumer", (73, 70)),
        (6, "Redpanda\nConsole", "topic signals.h5\n6+ messages", (5, 35)),
        (7, "Grafana", "Trading Perf\ndashboard", (22, 35)),
        (8, "Next.js\nDashboard", "/forecasting\n+ /production", (39, 35)),
    ]

    for num, head, sub, (x, y) in steps:
        # circle with number
        c = plt.Circle((x + 2, y + 15), 3, facecolor="#a855f7", edgecolor="#e9d5ff", linewidth=2, zorder=3)
        ax.add_patch(c)
        ax.text(x + 2, y + 15, str(num), ha="center", va="center", fontsize=13, fontweight="bold", color=TEXT)
        # card
        ax.add_patch(FancyBboxPatch((x, y), 14, 14, boxstyle="round,pad=0.02",
                                    facecolor="#1e293b", edgecolor="#64748b", linewidth=1.5, zorder=2))
        ax.text(x + 7, y + 9, head, ha="center", fontsize=11, fontweight="bold", color=TEXT)
        ax.text(x + 7, y + 4, sub, ha="center", fontsize=9, color=TEXT_DIM)

    # Arrows between steps
    for i in range(len(steps)-1):
        x1 = steps[i][3][0] + 14; y1 = steps[i][3][1] + 7
        x2 = steps[i+1][3][0]; y2 = steps[i+1][3][1] + 7
        if steps[i][3][1] == steps[i+1][3][1]:
            arrow(ax, x1, y1, x2, y2, color="#a855f7", lw=2)
        else:
            curved_arrow(ax, x1, y1, x2, y2, color="#a855f7", rad=-0.4)

    # Compliance footer
    ax.add_patch(FancyBboxPatch((55, 20), 42, 30, boxstyle="round,pad=0.02",
                                facecolor="#064e3b", edgecolor="#22c55e", linewidth=2))
    ax.text(76, 46, "✅ CUMPLIMIENTO", ha="center", fontsize=14, fontweight="bold", color="#d1fae5")
    compliance = [
        "✅ ≥2 tecnologías no-REST: gRPC + Kafka",
        "✅ Docker + orquestación (Airflow + MLflow)",
        "✅ Demo funcional `make course-demo`",
        "✅ Repositorio Git con 21 servicios",
        "✅ 15+ tests de integración automatizados",
        "✅ Dashboard Next.js con 8 páginas",
    ]
    for i, line in enumerate(compliance):
        ax.text(57, 41 - i*3, line, fontsize=11, color="#a7f3d0", va="center")

    save(fig, "05_demo_flow")


# ──────────────────────────────────────────────────────────────────────
# Diagram 6 — COURSE COMPLIANCE MATRIX
# ──────────────────────────────────────────────────────────────────────
def diagram_compliance():
    fig, ax = setup("Matriz de Cumplimiento — Requisitos del Curso",
                    "Comparación entre requisitos del curso y evidencia implementada")

    headers = ["Requisito", "Implementación", "Evidencia", "Status"]
    rows = [
        ("Tecnología #1 no-REST", "gRPC (Protocol Buffers)", "services/grpc_predictor/\nproto/predictor.proto", "✅"),
        ("Tecnología #2 no-REST", "Kafka (Redpanda)", "services/kafka_bridge/\ntopic signals.h5", "✅"),
        ("Docker", "docker-compose.compact.yml", "21 servicios en red\nusdcop-trading-network", "✅"),
        ("Orquestación (mejor nota)", "Airflow + MLflow", "27 DAGs\n+ 4 experiments", "✅"),
        ("Modelo ML", "Ridge + BR + XGBoost", "H5 Smart Simple v2.0\n+25.63% backtest", "✅"),
        ("Demo funcional", "make course-demo", "scripts/demo/demo_full.sh\n8 pasos, 10 min", "✅"),
        ("Repositorio Git", "GitHub repo", "Dockerfiles + docs\n+ tests", "✅"),
        ("Cloud (opcional)", "No implementado", "Todo local Docker", "⬜"),
        ("Federated learning (opc.)", "No implementado", "—", "⬜"),
    ]

    # Header
    header_y = 82
    col_x = [4, 26, 52, 88]
    col_w = [22, 26, 36, 9]
    ax.add_patch(Rectangle((3, header_y - 1), 94, 5, facecolor="#334155", edgecolor=TEXT_DIM))
    for x, w, h in zip(col_x, col_w, headers):
        ax.text(x + w/2, header_y + 1.5, h, ha="center", va="center", fontsize=12, fontweight="bold", color=TEXT)

    # Rows
    row_h = 7.5
    for i, row in enumerate(rows):
        y = header_y - 2 - (i+1) * row_h
        bg_color = "#1e293b" if i % 2 == 0 else "#0f172a"
        ax.add_patch(Rectangle((3, y), 94, row_h, facecolor=bg_color, edgecolor="#334155"))
        for x, w, val in zip(col_x, col_w, row):
            color = TEXT
            fs = 10
            if val == "✅":
                color = "#22c55e"; fs = 18
            elif val == "⬜":
                color = "#64748b"; fs = 16
            ax.text(x + w/2, y + row_h/2, val, ha="center", va="center", fontsize=fs, color=color, wrap=True)

    # Score summary
    ax.text(50, 6, "7 de 7 requisitos obligatorios cumplidos  •  0 de 2 opcionales (cloud + federated)",
            ha="center", fontsize=12, fontweight="bold", color="#22c55e")
    save(fig, "06_compliance_matrix")


def main():
    print("Generating course architecture diagrams...")
    diagram_endtoend()
    diagram_grpc()
    diagram_kafka()
    diagram_stack()
    diagram_demo()
    diagram_compliance()
    print(f"\nAll diagrams saved to {OUT}/")


if __name__ == "__main__":
    main()
