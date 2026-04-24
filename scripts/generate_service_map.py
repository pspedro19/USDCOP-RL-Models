"""Generate a highly visual service access map — every service as a visual card with icon."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from pathlib import Path

OUT = Path("docs/slides/course_diagrams")
OUT.mkdir(parents=True, exist_ok=True)

BG = "#0b1220"
TITLE_BAR = "#1e293b"
TEXT = "#ffffff"
TEXT_DIM = "#94a3b8"
DIM_BORDER = "#334155"


def setup(title, subtitle=None, w=16, h=9):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 92), 100, 8, boxstyle="square,pad=0",
                                facecolor=TITLE_BAR, edgecolor="#334155", linewidth=1.5))
    ax.text(50, 96.5, title, ha="center", va="center", fontsize=22, fontweight="bold", color=TEXT)
    if subtitle:
        ax.text(50, 93.2, subtitle, ha="center", va="center", fontsize=12, color=TEXT_DIM)
    return fig, ax


def card(ax, x, y, w, h, *, icon, name, url, creds, color, icon_color=None):
    """Big service card with icon circle + name + URL + credentials."""
    fill = color[0]; border = color[1]
    # background card
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                facecolor=fill, edgecolor=border, linewidth=2.2))
    # icon circle
    ic_color = icon_color or border
    icon_r = min(w, h) * 0.18
    cx = x + 3.5; cy = y + h - 3.5
    ax.add_patch(Circle((cx, cy), icon_r, facecolor=border, edgecolor="#ffffff", linewidth=1.5, zorder=3))
    ax.text(cx, cy, icon, ha="center", va="center", fontsize=16, fontweight="bold", color="#ffffff", zorder=4)
    # name
    ax.text(x + 7, y + h - 2.2, name, fontsize=12, fontweight="bold", color=TEXT, va="center")
    # URL
    ax.text(x + 1, y + h - 7, url, fontsize=10, color="#e2e8f0", family="monospace", va="center")
    # creds
    ax.text(x + 1, y + h - 10, creds, fontsize=9, color=TEXT_DIM, family="monospace", va="center", wrap=True)


# Colors — tied to slide theme
UI_UP = ("#164e63", "#06b6d4")      # cyan (main UI)
MLOPS = ("#3b0764", "#a855f7")       # purple
GRPC = ("#7c2d12", "#ea580c")        # orange (course)
KAFKA = ("#450a0a", "#ef4444")       # red (course)
PIPE = ("#064e3b", "#10b981")        # green (pipeline)
DATA = ("#422006", "#eab308")        # yellow (data)
INFRA = ("#1e293b", "#64748b")       # gray (infra)


def service_map():
    fig, ax = setup("Mapa de Acceso a Servicios — Demo en Vivo",
                    "Cada servicio con puerto, URL y credenciales — clickeable para la presentación")

    # Grid: 4 cols x 3 rows of cards
    #       card size ~21 x 25 with gaps
    cw, ch = 22, 23
    gap_x, gap_y = 2.5, 2
    start_x = 3
    start_y = 65

    cards = [
        # row 1 — COURSE TECHS highlighted
        (0, 0, "1", "gRPC Predictor", "localhost:50051", "service: PredictorService\nRPC: Predict() + HealthCheck()", GRPC),
        (0, 1, "2", "Kafka (Redpanda)", "localhost:19092", "topic: signals.h5\n(broker Kafka-compatible)", KAFKA),
        (0, 2, "UI", "Redpanda Console", "http://localhost:8088", "(sin autenticacion)\nUI para inspeccionar topics", KAFKA),
        (0, 3, "📊", "MLflow Tracking", "http://localhost:5001", "(sin autenticacion)\n4 experiments logueados", MLOPS),
        # row 2
        (1, 0, "⚙", "Airflow UI", "http://localhost:8080", "admin / admin123\n27 DAGs (L0->L7)", PIPE),
        (1, 1, "📈", "Grafana", "http://localhost:3002", "admin / admin\n4 dashboards", UI_UP),
        (1, 2, "🔔", "Prometheus", "http://localhost:9090", "(sin autenticacion)\n53 reglas de alerta", UI_UP),
        (1, 3, "🖥", "Dashboard Next.js", "http://localhost:5000", "(sin autenticacion)\n8 paginas operacionales", UI_UP),
        # row 3
        (2, 0, "🔀", "SignalBridge OMS", "http://localhost:8085/docs", "JWT via /api/auth/login\nFastAPI Swagger", MLOPS),
        (2, 1, "📦", "MinIO Console", "http://localhost:9001", "admin / admin123\n11 buckets S3", DATA),
        (2, 2, "🐘", "pgAdmin", "http://localhost:5050", "admin@admin.com\nadmin123", DATA),
        (2, 3, "🗄", "PostgreSQL", "localhost:5432", "admin / admin123\ndb: usdcop_trading", INFRA),
    ]

    for (row, col, icon, name, url, creds, color) in cards:
        x = start_x + col * (cw + gap_x)
        y = start_y - row * (ch + gap_y)
        card(ax, x, y, cw, ch, icon=icon, name=name, url=url, creds=creds, color=color)

    # Legend at bottom
    legend_y = 4
    ax.add_patch(FancyBboxPatch((3, legend_y), 94, 10, boxstyle="round,pad=0.02",
                                facecolor="#0f172a", edgecolor=DIM_BORDER, linewidth=1))
    ax.text(50, legend_y + 8, "Leyenda de colores", ha="center", fontsize=11, fontweight="bold", color=TEXT)
    legend_items = [
        (GRPC, "gRPC (curso)"),
        (KAFKA, "Kafka (curso)"),
        (PIPE, "Orquestacion"),
        (MLOPS, "MLOps / OMS"),
        (UI_UP, "UI / Monitoring"),
        (DATA, "Data / Storage"),
        (INFRA, "Infraestructura"),
    ]
    step = 92 / len(legend_items)
    for i, (c, label) in enumerate(legend_items):
        x = 5 + i * step
        ax.add_patch(Rectangle((x, legend_y + 3), 2, 2.5, facecolor=c[0], edgecolor=c[1], linewidth=1.5))
        ax.text(x + 2.5, legend_y + 4.25, label, fontsize=10, color=TEXT_DIM, va="center")

    fig.savefig(OUT / "07_service_map.png", dpi=140, facecolor=BG, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  saved {OUT/'07_service_map.png'}")


# Second visual: Terminal/CLI commands reference (visual cheat sheet)
def commands_reference():
    fig, ax = setup("Comandos de Demo — Referencia Rapida",
                    "Ejecutables en vivo durante la presentacion — copy/paste friendly")

    # 4 big command cards
    commands = [
        ("▶", "Estado General", "$ docker compose ps\n$ bash scripts/verify_course_delivery.sh",
         "21 servicios corriendo\n38/41 checks OK", PIPE),
        ("🔌", "gRPC Predict()", "$ docker exec usdcop-grpc-predictor \\\n    python client_example.py",
         "-> direction=LONG, confidence=0.11\n-> model_version=...", GRPC),
        ("📡", "Kafka Roundtrip", "$ docker exec usdcop-kafka-producer \\\n    python producer.py --demo\n$ docker logs usdcop-kafka-consumer",
         "3/3 mensajes publicados\n3/3 consumidos con JSON completo", KAFKA),
        ("🎬", "Demo Completa", "$ make course-demo",
         "Orquesta 8 pasos: ~10 minutos\nAbre Airflow, MLflow, Grafana,\nRedpanda Console, Dashboard", MLOPS),
    ]

    for i, (icon, title, cmd, result, color) in enumerate(commands):
        row = i // 2; col = i % 2
        x = 4 + col * 47
        y = 70 - row * 32
        w, h = 45, 28
        fill, border = color
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    facecolor=fill, edgecolor=border, linewidth=2.5))
        # Icon circle
        ax.add_patch(Circle((x + 4, y + h - 3), 2.2, facecolor=border, edgecolor="#ffffff", linewidth=1.5))
        ax.text(x + 4, y + h - 3, icon, ha="center", va="center", fontsize=14, color="#ffffff")
        # Title
        ax.text(x + 8, y + h - 3, title, fontsize=14, fontweight="bold", color=TEXT, va="center")
        # Command block
        ax.add_patch(FancyBboxPatch((x + 1, y + 10), w - 2, h - 17, boxstyle="round,pad=0.01",
                                    facecolor="#020617", edgecolor="#1e293b", linewidth=1))
        ax.text(x + 2, y + h - 9, cmd, fontsize=10, color="#22c55e", family="monospace", va="top")
        # Result
        ax.text(x + 2, y + 3, "resultado esperado:", fontsize=8, color=TEXT_DIM, va="center", style="italic")
        ax.text(x + 2, y + 1, result, fontsize=9, color="#cbd5e1", family="monospace", va="bottom")

    fig.savefig(OUT / "08_commands_reference.png", dpi=140, facecolor=BG, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  saved {OUT/'08_commands_reference.png'}")


def data_flow():
    """Large, clean end-to-end data flow diagram — more polished version."""
    fig, ax = setup("Flujo de Datos — De Mercado a Trade",
                    "Desde la ingesta 5-min hasta la orden ejecutada via gRPC / Kafka")

    # Horizontal pipeline
    stages = [
        # (x, y, w, h, title, sub, colors)
        (2, 55, 15, 20, "MERCADO", "TwelveData API\nFRED / BanRep", DATA),
        (19, 55, 15, 20, "L0 Ingesta", "OHLCV realtime\ncada 5 min", DATA),
        (36, 55, 15, 20, "TimescaleDB", "PostgreSQL\n+ TimescaleDB", INFRA),
        (53, 55, 15, 20, "L1-L3 Airflow", "Features +\nEntrenamiento\n(Ridge+BR+XGB)", PIPE),
        (70, 55, 15, 20, "L5 Signal", "Ensemble +\nRegime Gate\n(Hurst)", PIPE),
        (87, 55, 11, 20, "MLflow", "Tracking\n+ Artifacts", MLOPS),
    ]
    for x, y, w, h, t, s, c in stages:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    facecolor=c[0], edgecolor=c[1], linewidth=2))
        ax.text(x + w/2, y + h - 3, t, ha="center", fontsize=11, fontweight="bold", color=TEXT)
        ax.text(x + w/2, y + h/2 - 1, s, ha="center", fontsize=9, color=TEXT_DIM, wrap=True)

    # Arrows top row
    for i in range(len(stages) - 1):
        x1 = stages[i][0] + stages[i][2]
        x2 = stages[i+1][0]
        y = stages[i][1] + stages[i][3]/2
        a = FancyArrowPatch((x1, y), (x2, y), arrowstyle="->", color="#94a3b8",
                            linewidth=2.5, mutation_scale=18)
        ax.add_patch(a)

    # Split to 2 parallel branches: gRPC and Kafka
    # From L5 Signal down
    a1 = FancyArrowPatch((60, 55), (30, 40), arrowstyle="->", color=GRPC[1], linewidth=3, mutation_scale=20)
    ax.add_patch(a1)
    a2 = FancyArrowPatch((60, 55), (77, 40), arrowstyle="->", color=KAFKA[1], linewidth=3, mutation_scale=20)
    ax.add_patch(a2)

    # gRPC branch
    ax.add_patch(FancyBboxPatch((10, 23), 40, 17, boxstyle="round,pad=0.02",
                                facecolor=GRPC[0], edgecolor=GRPC[1], linewidth=3))
    ax.text(30, 37, "TECH #1 — gRPC PREDICTOR", ha="center", fontsize=13, fontweight="bold", color=TEXT)
    ax.text(30, 33, "localhost:50051", ha="center", fontsize=11, color="#fed7aa", family="monospace")
    ax.text(30, 30, "PredictorService.Predict(features)", ha="center", fontsize=10, color=TEXT_DIM, family="monospace")
    ax.text(30, 27, "-> direction + confidence + model_version", ha="center", fontsize=9, color="#22c55e", family="monospace")
    ax.text(30, 24.5, "[serving sincrono de baja latencia]", ha="center", fontsize=9, style="italic", color=TEXT_DIM)

    # Kafka branch
    ax.add_patch(FancyBboxPatch((55, 23), 40, 17, boxstyle="round,pad=0.02",
                                facecolor=KAFKA[0], edgecolor=KAFKA[1], linewidth=3))
    ax.text(75, 37, "TECH #2 — KAFKA (REDPANDA)", ha="center", fontsize=13, fontweight="bold", color=TEXT)
    ax.text(75, 33, "topic: signals.h5", ha="center", fontsize=11, color="#fecaca", family="monospace")
    ax.text(75, 30, "producer (Airflow) -> broker -> consumer(s)", ha="center", fontsize=10, color=TEXT_DIM, family="monospace")
    ax.text(75, 27, "-> JSON message persistido con offset", ha="center", fontsize=9, color="#22c55e", family="monospace")
    ax.text(75, 24.5, "[streaming asincrono desacoplado]", ha="center", fontsize=9, style="italic", color=TEXT_DIM)

    # Arrows down to UI
    a3 = FancyArrowPatch((30, 23), (30, 15), arrowstyle="->", color="#06b6d4", linewidth=2, mutation_scale=15)
    ax.add_patch(a3)
    a4 = FancyArrowPatch((75, 23), (75, 15), arrowstyle="->", color="#06b6d4", linewidth=2, mutation_scale=15)
    ax.add_patch(a4)

    # Bottom: UI + OMS + Exchange
    ax.add_patch(FancyBboxPatch((3, 3), 94, 12, boxstyle="round,pad=0.02",
                                facecolor=UI_UP[0], edgecolor=UI_UP[1], linewidth=2))
    ax.text(50, 12, "UI + CONSUMIDORES + EXECUTION", ha="center", fontsize=13, fontweight="bold", color=TEXT)
    ax.text(15, 8, "Dashboard :5000\n/forecasting", ha="center", fontsize=9, color=TEXT_DIM)
    ax.text(32, 8, "Grafana :3002\nmetrics", ha="center", fontsize=9, color=TEXT_DIM)
    ax.text(50, 8, "SignalBridge :8085\nFastAPI OMS", ha="center", fontsize=9, color=TEXT_DIM)
    ax.text(67, 8, "Redpanda Console :8088\ntopic UI", ha="center", fontsize=9, color=TEXT_DIM)
    ax.text(85, 8, "MEXC / Binance\n(via CCXT)", ha="center", fontsize=9, color=TEXT_DIM)
    ax.text(50, 4.5, "5-min bars -> features -> signal -> [gRPC SYNC | KAFKA ASYNC] -> OMS -> Exchange",
            ha="center", fontsize=10, color="#e2e8f0", fontweight="bold")

    fig.savefig(OUT / "09_data_flow.png", dpi=140, facecolor=BG, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  saved {OUT/'09_data_flow.png'}")


if __name__ == "__main__":
    service_map()
    commands_reference()
    data_flow()
    print("\nDone.")
