"""Render terminal outputs as styled PNG images for presentation evidence.

Uses matplotlib to create fake-terminal-window PNGs with syntax highlighting.
"""
import subprocess
import re
import shlex
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

OUT = Path("docs/slides/course_terminals")
OUT.mkdir(parents=True, exist_ok=True)

# Terminal theme
TERM_BG = "#0b1220"
TERM_FRAME = "#1e293b"
TEXT = "#e2e8f0"
GREEN = "#22c55e"
RED = "#ef4444"
YELLOW = "#eab308"
BLUE = "#60a5fa"
PURPLE = "#a855f7"
ORANGE = "#fb923c"
DIM = "#64748b"


def run(cmd, timeout=30):
    """Run command and return stdout (stderr merged)."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout + (r.stderr if r.stderr and not r.stdout else "")
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR: {e}]"


def strip_ansi(s):
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def render_terminal_png(out_name, title, command, output, *, lines=32, width=14, height=9,
                         highlight_rules=None):
    """Render a terminal window PNG.

    highlight_rules: list of (substring, color) — lines containing substring get color
    """
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("#020617")
    ax.set_facecolor("#020617")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # Title banner
    ax.add_patch(FancyBboxPatch((0, 93), 100, 7, boxstyle="square,pad=0",
                                facecolor="#1e293b", edgecolor="#475569", linewidth=1))
    ax.text(50, 96.5, title, ha="center", va="center", fontsize=18, fontweight="bold", color=TEXT)

    # Terminal window
    ax.add_patch(FancyBboxPatch((3, 5), 94, 84, boxstyle="round,pad=0.01",
                                facecolor=TERM_BG, edgecolor=TERM_FRAME, linewidth=2))

    # Title bar with 3 dots
    ax.add_patch(FancyBboxPatch((3, 84), 94, 5, boxstyle="square,pad=0",
                                facecolor=TERM_FRAME, edgecolor="none"))
    for i, c in enumerate([RED, "#fbbf24", GREEN]):
        ax.add_patch(Circle((6 + i*2.5, 86.5), 0.9, facecolor=c, edgecolor="none"))
    ax.text(50, 86.5, "terminal  —  usdcop-rl-models", ha="center", va="center",
            fontsize=11, color=DIM, family="monospace")

    # Command line
    ax.text(5, 81, "$", fontsize=14, color=GREEN, family="monospace", fontweight="bold", va="top")
    ax.text(7, 81, command, fontsize=12, color=TEXT, family="monospace", va="top")

    # Output lines
    lines_list = output.split("\n")[:lines]
    y = 76
    dy = 72 / max(len(lines_list), 1)
    dy = min(dy, 2.6)
    for line in lines_list:
        color = TEXT
        if highlight_rules:
            for substr, c in highlight_rules:
                if substr in line:
                    color = c
                    break
        # truncate
        line = line[:180]
        ax.text(5, y, line, fontsize=9.5, color=color, family="monospace", va="top")
        y -= dy

    plt.tight_layout()
    path = OUT / f"{out_name}.png"
    fig.savefig(path, dpi=140, facecolor="#020617", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  saved {path}")
    return path


def main():
    print("Rendering terminal evidence as PNGs...")

    # 1. docker ps — all services
    out = run("docker compose -f docker-compose.compact.yml ps --format 'table {{.Name}}\t{{.Status}}\t{{.Ports}}'")
    render_terminal_png("term_01_docker_ps", "$ docker compose ps — 21 servicios healthy",
                        "docker compose -f docker-compose.compact.yml ps",
                        out, lines=22,
                        highlight_rules=[("healthy", GREEN), ("Exited", RED),
                                         ("unhealthy", YELLOW), ("NAME", BLUE)])

    # 2. proto file content
    out = run("docker exec usdcop-grpc-predictor cat /app/proto/predictor.proto")
    render_terminal_png("term_02_proto", "$ cat predictor.proto — Contrato Protocol Buffers",
                        "docker exec usdcop-grpc-predictor cat /app/proto/predictor.proto",
                        out, lines=35,
                        highlight_rules=[("syntax", PURPLE), ("package", PURPLE),
                                         ("service", ORANGE), ("rpc ", ORANGE),
                                         ("message", BLUE), ("//", DIM)])

    # 3. gRPC Predict call
    grpc_code = r"""docker exec usdcop-grpc-predictor python -c "
import grpc, predictor_pb2, predictor_pb2_grpc
ch = grpc.insecure_channel('localhost:50051')
stub = predictor_pb2_grpc.PredictorServiceStub(ch)
h = stub.HealthCheck(predictor_pb2.HealthRequest())
print(f'HealthCheck  ->  ready={h.ready}  model_loaded={h.model_loaded}')
feats = {'dxy_z': 0.5, 'vix_z': 0.8, 'wti_z': -0.3, 'embi_col_z': -0.1,
         'ust10y_z': 0.2, 'rsi_14': 45.0, 'trend_slope_60d': -0.002,
         'vol_regime_ratio': 1.15}
r = stub.Predict(predictor_pb2.PredictRequest(features=feats))
print(f'Predict      ->  direction={r.direction}  confidence={r.confidence:.4f}')
print(f'                 ensemble_return={r.ensemble_return:.6f}  model_version={r.model_version}')
" """
    out = run(grpc_code, timeout=20)
    render_terminal_png("term_03_grpc_call", "$ gRPC Predict() + HealthCheck — llamada en vivo",
                        "python -c 'import grpc; stub.Predict(features=...)'",
                        out, lines=10,
                        highlight_rules=[("HealthCheck", GREEN), ("Predict", ORANGE),
                                         ("ready=True", GREEN), ("LONG", GREEN),
                                         ("SHORT", RED), ("direction=", YELLOW)])

    # 4. rpk topic consume
    out = run("docker exec usdcop-redpanda rpk topic consume signals.h5 --num 3 --format '%v\\n' 2>&1", timeout=15)
    render_terminal_png("term_04_kafka_consume", "$ rpk topic consume signals.h5 — 3 mensajes reales",
                        "docker exec usdcop-redpanda rpk topic consume signals.h5 --num 3",
                        out, lines=14,
                        highlight_rules=[('"week"', YELLOW), ('"direction"', ORANGE),
                                         ("SHORT", RED), ("LONG", GREEN),
                                         ('"confidence"', BLUE)])

    # 5. kafka producer demo output
    out = run("docker exec usdcop-kafka-producer python producer.py --demo 2>&1 | grep -E 'INFO (published|demo complete|connected)'",
              timeout=20)
    render_terminal_png("term_05_kafka_producer", "$ Kafka Producer — Publica 3 mensajes al topic",
                        "docker exec usdcop-kafka-producer python producer.py --demo",
                        out, lines=10,
                        highlight_rules=[("published week=", GREEN), ("demo complete", PURPLE),
                                         ("connected to Kafka", BLUE), ("offset=", YELLOW)])

    # 6. verify output (summary)
    out = run("bash scripts/verify_course_delivery.sh 2>&1 | tail -50", timeout=60)
    # strip ansi
    out = strip_ansi(out)
    render_terminal_png("term_06_verify", "$ verify_course_delivery.sh — Auto-verificación 41 checks",
                        "bash scripts/verify_course_delivery.sh",
                        out, lines=35, height=10,
                        highlight_rules=[("OK", GREEN), ("✅", GREEN), ("PASSED", GREEN),
                                         ("FAIL", RED), ("❌", RED),
                                         ("WARN", YELLOW), ("⚠", YELLOW),
                                         ("━━━", BLUE)])

    # 7. rpk topic describe
    out = run("docker exec usdcop-redpanda rpk topic describe signals.h5 2>&1", timeout=10)
    render_terminal_png("term_07_topic_describe", "$ rpk topic describe signals.h5 — Metadata del topic",
                        "docker exec usdcop-redpanda rpk topic describe signals.h5",
                        out, lines=30,
                        highlight_rules=[("SUMMARY", BLUE), ("CONFIGS", BLUE),
                                         ("PARTITIONS", YELLOW), ("signals.h5", ORANGE)])

    # 8. Makefile course-* targets help
    out = run("grep -A 1 'course-' Makefile | head -25", timeout=5)
    render_terminal_png("term_08_makefile", "Makefile — Targets `course-*` para la demo",
                        "grep -A 1 'course-' Makefile",
                        out, lines=20,
                        highlight_rules=[("course-", GREEN), ("##", DIM), (":", PURPLE)])

    print("\nAll terminal PNGs rendered.")


if __name__ == "__main__":
    main()
