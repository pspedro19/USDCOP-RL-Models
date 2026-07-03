"""Generate all PNG assets for the GlobalMinds pitch deck."""
from __future__ import annotations
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
import numpy as np

# ─── Brand ────────────────────────────────────────────────────────────────────
BG_DARK = "#0B1426"
BG_PANEL = "#15233A"
WHITE = "#FFFFFF"
LIME = "#C5FF4A"
GRAY = "#94A3B8"
GRAY_DIM = "#64748B"
RED = "#F87171"
GREEN = "#34D399"
BLUE = "#60A5FA"
AMBER = "#FBBF24"

ASSETS = Path("presentation/globalminds_microsoft_pitch_may2026/assets")
CHARTS = ASSETS / "charts"
DIAGRAMS = ASSETS / "diagrams"
CHARTS.mkdir(parents=True, exist_ok=True)
DIAGRAMS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "DejaVu Sans", "Liberation Sans"],
    "axes.edgecolor": GRAY_DIM,
    "axes.labelcolor": WHITE,
    "xtick.color": GRAY,
    "ytick.color": GRAY,
    "text.color": WHITE,
})

W, H = 1920, 1080
DPI = 120
FIGSIZE = (W / DPI, H / DPI)


def _new_fig(facecolor: str = BG_DARK):
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI, facecolor=facecolor)
    return fig


def _save(fig, path: Path):
    fig.savefig(path, facecolor=fig.get_facecolor(), dpi=DPI, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  ✓ {path.name} ({path.stat().st_size // 1024} KB)")


# ─── 1. Hero (slide 1) ────────────────────────────────────────────────────────
def hero():
    fig = _new_fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # gradient background simulated with rectangles
    for i in range(40):
        alpha = i / 40
        ax.add_patch(patches.Rectangle((0, i * 9 / 40), 16, 9 / 40,
                                       color=BG_DARK, alpha=1 - alpha * 0.3, zorder=0))

    # subtle USD/COP-like silhouette
    rng = np.random.default_rng(7)
    x = np.linspace(0.5, 15.5, 200)
    base = 4.0 + np.cumsum(rng.standard_normal(200) * 0.06)
    base = (base - base.min()) / (base.max() - base.min()) * 3.5 + 1.5
    ax.plot(x, base, color=LIME, alpha=0.18, linewidth=2.5, zorder=1)
    ax.fill_between(x, base, 0, color=LIME, alpha=0.05)

    # GlobalMinds wordmark
    ax.text(8, 5.8, "GlobalMinds", fontsize=92, fontweight="bold",
            color=WHITE, ha="center", va="center", zorder=3)
    # lime dot
    ax.add_patch(Circle((11.65, 5.45), 0.18, color=LIME, zorder=4))

    # tagline
    ax.text(8, 4.4, "IA aplicada al mercado cambiario de Latinoamérica",
            fontsize=26, color=GRAY, ha="center", va="center", style="italic", zorder=3)

    # divider line
    ax.plot([6.5, 9.5], [3.6, 3.6], color=LIME, linewidth=2, zorder=3)

    # presenters / date
    ax.text(8, 2.9, "Pedro Sánchez Briceño  ·  Freddy",
            fontsize=22, color=WHITE, ha="center", va="center", zorder=3)
    ax.text(8, 2.2, "Mayo 2026", fontsize=18, color=GRAY, ha="center", va="center", zorder=3)

    _save(fig, DIAGRAMS / "01_hero.png")


# ─── 2. Architecture diagram (slide 5) ───────────────────────────────────────
def architecture():
    fig = _new_fig()
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.88])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    ax.text(50, 56.5, "Arquitectura: 1 motor de IA → 4 productos comerciales",
            fontsize=24, fontweight="bold", color=WHITE, ha="center")

    # Layer 1 — Data Sources
    sources = [
        ("Mercado FX\n5-min OHLCV", 8),
        ("Macro\n40 indicadores", 22),
        ("Noticias\n5 fuentes ES/EN", 36),
        ("Análisis IA\nGPT-4o + Claude", 50),
    ]
    for label, x in sources:
        box = FancyBboxPatch((x - 6, 42), 12, 7,
                             boxstyle="round,pad=0.3,rounding_size=0.6",
                             linewidth=1.4, edgecolor=BLUE, facecolor=BG_PANEL)
        ax.add_patch(box)
        ax.text(x, 45.5, label, fontsize=11, color=WHITE, ha="center", va="center")

    ax.text(70, 45.5, "Datos", fontsize=12, color=GRAY_DIM, fontweight="bold", ha="left", va="center")

    # arrows down to ML core
    for _, x in sources:
        ax.annotate("", xy=(x, 36), xytext=(x, 42),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.2))

    # Layer 2 — ML core
    core = FancyBboxPatch((6, 27), 50, 9,
                          boxstyle="round,pad=0.3,rounding_size=0.8",
                          linewidth=2, edgecolor=LIME, facecolor=BG_PANEL)
    ax.add_patch(core)
    ax.text(31, 33.5, "Pipeline ML / RL — Forecasting Engine",
            fontsize=15, fontweight="bold", color=LIME, ha="center")
    ax.text(31, 30, "Ridge + BayesianRidge + XGBoost  ·  Regime Gate (Hurst)  ·  Effective HS  ·  Walk-forward weekly retrain",
            fontsize=10, color=WHITE, ha="center")
    ax.text(70, 31.5, "Modelo", fontsize=12, color=GRAY_DIM, fontweight="bold", ha="left", va="center")

    # arrow down
    ax.annotate("", xy=(31, 21), xytext=(31, 27),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.4))

    # Layer 3 — Signals
    sig = FancyBboxPatch((18, 16), 26, 5,
                         boxstyle="round,pad=0.2,rounding_size=0.5",
                         linewidth=1.4, edgecolor=AMBER, facecolor=BG_PANEL)
    ax.add_patch(sig)
    ax.text(31, 18.5, "Señales universales: dirección · confianza · sizing · stops",
            fontsize=11, color=WHITE, ha="center", va="center")
    ax.text(70, 18.5, "Señales", fontsize=12, color=GRAY_DIM, fontweight="bold", ha="left", va="center")

    # Layer 4 — 4 products
    products = [
        ("Bot P2P\nUSDT/COP", 8, "#34D399"),
        ("SaaS Hedging\nPYMES", 22, "#60A5FA"),
        ("Optimizador\nRemesas", 36, "#FBBF24"),
        ("App Freelancers\nBre-B + USD", 50, "#F472B6"),
    ]
    for label, x, color in products:
        ax.annotate("", xy=(x, 9), xytext=(x, 16),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.2))
        box = FancyBboxPatch((x - 6.5, 2), 13, 7,
                             boxstyle="round,pad=0.3,rounding_size=0.6",
                             linewidth=1.6, edgecolor=color, facecolor=BG_PANEL)
        ax.add_patch(box)
        ax.text(x, 5.5, label, fontsize=11, fontweight="bold", color=WHITE, ha="center", va="center")
    ax.text(70, 5.5, "Productos", fontsize=12, color=GRAY_DIM, fontweight="bold", ha="left", va="center")

    # right column: cloud-ready
    ax.text(80, 49, "Stack", fontsize=12, fontweight="bold", color=LIME, ha="left")
    stack_items = ["Airflow", "MLflow", "PostgreSQL + TimescaleDB", "FastAPI", "Next.js / React",
                   "Docker / Kubernetes", "Azure ML / OpenAI ready"]
    for i, it in enumerate(stack_items):
        ax.text(80, 45 - i * 2.7, f"·  {it}", fontsize=11, color=WHITE, ha="left")

    _save(fig, DIAGRAMS / "05_architecture.png")


# ─── 3. Equity curve (slide 7) — REAL data from trades JSON ──────────────────
def equity_curve():
    trades = json.loads(Path("usdcop-trading-dashboard/public/data/production/trades/smart_simple_v11_2025.json").read_text())["trades"]
    timestamps = [t["timestamp"][:10] for t in trades]
    equity = [t["equity_at_exit"] for t in trades]
    timestamps = ["2025-01-01"] + timestamps
    equity = [10000.0] + equity

    # buy & hold synthetic: start 10000, end 8552 (linear)
    bh_x = list(range(len(equity)))
    bh_end = 8552.0
    bh_y = np.linspace(10000, bh_end, len(equity))

    fig = _new_fig()
    ax = fig.add_axes([0.08, 0.12, 0.86, 0.78])
    ax.set_facecolor(BG_DARK)
    for spine in ax.spines.values():
        spine.set_color(GRAY_DIM)

    x = np.arange(len(equity))
    ax.plot(x, equity, color=LIME, linewidth=3, label="GlobalMinds Smart v2.0", zorder=3)
    ax.fill_between(x, equity, 10000, where=(np.array(equity) >= 10000),
                    color=LIME, alpha=0.10, zorder=2)
    ax.plot(bh_x, bh_y, color=RED, linewidth=2.2, linestyle="--", label="Buy & Hold USD/COP", zorder=3)

    ax.axhline(10000, color=GRAY_DIM, linewidth=1, linestyle=":", zorder=1)

    # KPI badges
    ax.text(0.99, 0.97, "  +25.63%  ", transform=ax.transAxes,
            fontsize=20, color=BG_DARK, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc=LIME, ec="none"), zorder=10, fontweight="bold")
    ax.text(0.99, 0.86, "  Sharpe 3.35  ", transform=ax.transAxes,
            fontsize=14, color=WHITE, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc=BG_PANEL, ec=LIME, lw=1), zorder=10)
    ax.text(0.99, 0.78, "  p-value 0.006  ", transform=ax.transAxes,
            fontsize=14, color=WHITE, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc=BG_PANEL, ec=LIME, lw=1), zorder=10)
    ax.text(0.99, 0.70, "  Win rate 82.4%  ", transform=ax.transAxes,
            fontsize=14, color=WHITE, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc=BG_PANEL, ec=LIME, lw=1), zorder=10)

    ax.set_title("Backtest 2025 — $10,000 → $12,563 (vs Buy & Hold $8,552)",
                 fontsize=18, color=WHITE, fontweight="bold", pad=18)
    ax.set_xlabel("Trades secuenciales 2025 (34 trades, 5L / 29S)", fontsize=12, color=GRAY)
    ax.set_ylabel("Equity (USD)", fontsize=12, color=GRAY)
    ax.legend(loc="upper left", facecolor=BG_PANEL, edgecolor=GRAY_DIM, labelcolor=WHITE, fontsize=12)
    ax.grid(True, color=GRAY_DIM, alpha=0.2, linestyle=":")

    _save(fig, CHARTS / "07_equity_curve_2025.png")


# ─── 4. 4 oportunidades cards (slide 8) ──────────────────────────────────────
def opportunities():
    fig = _new_fig()
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.92])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    ax.text(50, 53, "Las 4 oportunidades validadas",
            fontsize=28, fontweight="bold", color=WHITE, ha="center")
    ax.text(50, 49.5, "1 motor de IA · 4 mercados sin atender · USD 50–150M TAM LATAM",
            fontsize=14, color=GRAY, ha="center", style="italic")

    cards = [
        {
            "n": "01", "color": "#34D399",
            "title": "Bot P2P USDT/COP",
            "subtitle": "El XTX Markets de Binance P2P Colombia",
            "client": "Merchants P2P (USD 5–200k/día)",
            "ticket": "USD 99–1,500/mes + perf fee",
            "mrr": "USD 15–25k MRR @ 18m",
        },
        {
            "n": "02", "color": "#60A5FA",
            "title": "SaaS Hedging FX PYMES",
            "subtitle": "Kantox para LATAM",
            "client": "5,000+ PYMES exportadoras/importadoras",
            "ticket": "USD 199–2,499/mes + rev share banco",
            "mrr": "USD 60k MRR @ 24m",
        },
        {
            "n": "03", "color": "#FBBF24",
            "title": "Optimizador Remesas B2B",
            "subtitle": "Layer de IA sobre USD 13B/año",
            "client": "Casas de cambio (WU, MG, Movii, Wise...)",
            "ticket": "USD 2k/mes + 10–20% rev share",
            "mrr": "USD 10–25k MRR @ 18m",
        },
        {
            "n": "04", "color": "#F472B6",
            "title": "App Freelancers Bre-B",
            "subtitle": "Wise killer con AI timing",
            "client": "500k–1M freelancers cobrando USD",
            "ticket": "USD 5–15/mes + 0.3% por conversión",
            "mrr": "USD 40k MRR @ 24m",
        },
    ]
    # Cards span y=2 to y=43 (height 41). Top of card = 43.
    card_top = 43
    card_height = 41
    card_w = 22.5
    starts = [2, 26, 50, 74]

    for card, x in zip(cards, starts):
        bottom = card_top - card_height
        # card body
        box = FancyBboxPatch((x, bottom), card_w, card_height,
                             boxstyle="round,pad=0.4,rounding_size=1.2",
                             linewidth=2, edgecolor=card["color"], facecolor=BG_PANEL)
        ax.add_patch(box)

        # number badge — large, centered horizontally near top
        badge_y = card_top - 4.5
        ax.add_patch(Circle((x + card_w / 2, badge_y), 2.6, color=card["color"]))
        ax.text(x + card_w / 2, badge_y, card["n"], fontsize=18, fontweight="bold",
                color=BG_DARK, ha="center", va="center")

        # title
        ax.text(x + card_w / 2, card_top - 10, card["title"],
                fontsize=14, fontweight="bold", color=WHITE, ha="center", va="center")
        # subtitle
        ax.text(x + card_w / 2, card_top - 13.5, card["subtitle"],
                fontsize=10.5, color=card["color"], ha="center", va="center", style="italic",
                wrap=True)

        # divider
        ax.plot([x + 2, x + card_w - 2], [card_top - 17, card_top - 17],
                color=GRAY_DIM, linewidth=0.8, alpha=0.7)

        # 3 detail blocks
        labels = [("CLIENTE", card["client"]), ("PRICING", card["ticket"]), ("PROYECCIÓN", card["mrr"])]
        block_top = card_top - 19
        block_height = 7
        for i, (k, v) in enumerate(labels):
            yy = block_top - i * block_height
            ax.text(x + 1.5, yy, k, fontsize=8.5, color=card["color"], fontweight="bold")
            ax.text(x + 1.5, yy - 2.5, v, fontsize=10, color=WHITE, wrap=True,
                    va="top")

    _save(fig, DIAGRAMS / "08_opportunities.png")


# ─── 5. Comparables wordmarks (slide 10) ─────────────────────────────────────
def comparables():
    fig = _new_fig()
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.9])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    ax.text(50, 56, "Comparables internacionales",
            fontsize=26, fontweight="bold", color=WHITE, ha="center")
    ax.text(50, 52.5, "Validación de mercado con valuaciones reales — ninguno opera Colombia con esta propuesta",
            fontsize=13, color=GRAY, ha="center", style="italic")

    rows = [
        # producto, comparable, valuación, color
        ("Hedging PYMES", "KANTOX", "Adquirida por VISA · €175M (2025)", "#60A5FA"),
        ("Hedging PYMES", "BOUND  ·  NEO  ·  PANGEA  ·  TREASURUP", "Series A activas (2024–2025)", "#60A5FA"),
        ("Freelancers / Remesas", "WISE", "USD 11B valuación · NYSE", "#FBBF24"),
        ("Freelancers / Remesas", "REMITLY", "USD 3B valuación · NASDAQ", "#FBBF24"),
        ("Freelancers / Remesas", "DEEL", "USD 12B valuación (last round)", "#FBBF24"),
        ("Trading quant FX", "XTX MARKETS", "Líder global ML en FX market making", "#34D399"),
    ]

    yt = 46
    for tag, brand, valuation, color in rows:
        # category badge
        ax.text(8, yt, tag.upper(), fontsize=10, color=color, fontweight="bold", ha="left", va="center")
        # wordmark
        ax.text(38, yt, brand, fontsize=20, fontweight="bold", color=WHITE, ha="left", va="center",
                family="sans-serif")
        # valuation
        ax.text(95, yt, valuation, fontsize=11, color=GRAY, ha="right", va="center")
        # divider
        ax.plot([6, 96], [yt - 2.5, yt - 2.5], color=GRAY_DIM, linewidth=0.6, alpha=0.5)
        yt -= 6.5

    # bottom callout
    callout = FancyBboxPatch((10, 1.5), 80, 5,
                             boxstyle="round,pad=0.4,rounding_size=0.8",
                             linewidth=1.4, edgecolor=LIME, facecolor=BG_PANEL)
    ax.add_patch(callout)
    ax.text(50, 4, "Tu ventaja: ninguno opera Colombia con esta propuesta combinada",
            fontsize=14, fontweight="bold", color=LIME, ha="center", va="center")

    _save(fig, DIAGRAMS / "10_comparables.png")


# ─── 6. 1 motor → 4 ingresos (slide 11) ──────────────────────────────────────
def revenue_streams():
    fig = _new_fig()
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.9])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    ax.text(50, 56, "4 motores de ingreso · 1 solo motor de IA",
            fontsize=26, fontweight="bold", color=WHITE, ha="center")
    ax.text(50, 52.5, "Mismo CAPEX  ·  4× revenue  ·  Diversificación de riesgo  ·  Cross-sell",
            fontsize=13, color=GRAY, ha="center", style="italic")

    # central engine
    cx, cy = 50, 28
    engine = FancyBboxPatch((cx - 14, cy - 6), 28, 12,
                            boxstyle="round,pad=0.4,rounding_size=1.2",
                            linewidth=2.5, edgecolor=LIME, facecolor=BG_PANEL)
    ax.add_patch(engine)
    ax.text(cx, cy + 1.5, "MOTOR ML / RL", fontsize=18, fontweight="bold", color=LIME, ha="center")
    ax.text(cx, cy - 2.5, "Ridge · BayesianRidge · XGBoost · Regime Gate",
            fontsize=10, color=WHITE, ha="center", style="italic")

    # 4 revenue streams radiating
    streams = [
        # angle deg, label, sublabel, color
        (135, "Bot P2P", "Sub USD 99–1,500/mes\n+ perf fee", "#34D399"),
        (45, "SaaS Hedging", "USD 199–2,499/mes\n+ rev share banco", "#60A5FA"),
        (-45, "Remesas B2B", "USD 2k/mes\n+ 10–20% rev share", "#FBBF24"),
        (-135, "App Freelancers", "USD 5–15/mes\n+ 0.3% por tx", "#F472B6"),
    ]

    for angle_deg, label, sub, color in streams:
        angle = math.radians(angle_deg)
        r = 22
        ex = cx + math.cos(angle) * r
        ey = cy + math.sin(angle) * r * 0.55
        # arrow from engine to product
        ax.annotate("", xy=(ex - math.cos(angle) * 7, ey - math.sin(angle) * 4),
                    xytext=(cx + math.cos(angle) * 13, cy + math.sin(angle) * 5.5),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2))
        # product box
        box = FancyBboxPatch((ex - 11, ey - 4.5), 22, 9,
                             boxstyle="round,pad=0.4,rounding_size=0.8",
                             linewidth=1.6, edgecolor=color, facecolor=BG_PANEL)
        ax.add_patch(box)
        ax.text(ex, ey + 1.8, label, fontsize=14, fontweight="bold", color=WHITE, ha="center", va="center")
        ax.text(ex, ey - 1.8, sub, fontsize=9.5, color=GRAY, ha="center", va="center")

    # bottom: aggregate
    ax.text(50, 4, "Margen SaaS típico 70–85%  ·  BYOK (cero costo variable)  ·  MRR proyectado USD 100–150k @ 24m",
            fontsize=13, color=LIME, ha="center", fontweight="bold")

    _save(fig, DIAGRAMS / "11_revenue_streams.png")


# ─── 7. Pricing tiers (slide 12) ─────────────────────────────────────────────
def pricing():
    fig = _new_fig()
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.9])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    ax.text(50, 56, "Pricing por línea de producto",
            fontsize=26, fontweight="bold", color=WHITE, ha="center")
    ax.text(50, 52.5, "Precios anclados en comparables internacionales · Tiered pricing por volumen",
            fontsize=13, color=GRAY, ha="center", style="italic")

    columns = [
        {"title": "Bot P2P", "color": "#34D399", "tiers": [
            ("Solo Merchant", "USD 99–199/mes", "Volumen <20k/día"),
            ("Pro / Whale", "USD 499–1,500/mes", "+5–10% performance fee"),
        ]},
        {"title": "Hedging PYMES", "color": "#60A5FA", "tiers": [
            ("PYME", "USD 199/mes", "<USD 1M FX exposure"),
            ("Mid", "USD 999/mes", "1–10M FX exposure"),
            ("Corporate", "USD 5,000+/mes", "Custom + rev share"),
        ]},
        {"title": "Remesas B2B", "color": "#FBBF24", "tiers": [
            ("API Integration", "USD 2k/mes floor", "+10–20% rev share alpha"),
        ]},
        {"title": "App Freelancers", "color": "#F472B6", "tiers": [
            ("Suscripción + Tx", "USD 5–15/mes", "+0.3% por conversión"),
        ]},
    ]
    xs = [4, 28, 52, 76]
    width = 20

    for col, x in zip(columns, xs):
        # header
        head = FancyBboxPatch((x, 44), width, 4.5,
                              boxstyle="round,pad=0.2,rounding_size=0.5",
                              linewidth=0, facecolor=col["color"])
        ax.add_patch(head)
        ax.text(x + width / 2, 46.3, col["title"], fontsize=14, fontweight="bold",
                color=BG_DARK, ha="center", va="center")

        # tiers
        ty = 41
        for tname, tprice, tdesc in col["tiers"]:
            box = FancyBboxPatch((x, ty - 8.5), width, 8,
                                 boxstyle="round,pad=0.3,rounding_size=0.5",
                                 linewidth=1, edgecolor=GRAY_DIM, facecolor=BG_PANEL)
            ax.add_patch(box)
            ax.text(x + width / 2, ty - 1.5, tname, fontsize=11, fontweight="bold",
                    color=col["color"], ha="center")
            ax.text(x + width / 2, ty - 4, tprice, fontsize=12, fontweight="bold",
                    color=WHITE, ha="center")
            ax.text(x + width / 2, ty - 6.7, tdesc, fontsize=9, color=GRAY, ha="center")
            ty -= 9.5

    ax.text(50, 4, "Margen 70–85% típico SaaS  ·  BYOK = cero costo variable  ·  Pricing escalable por uso",
            fontsize=12, color=LIME, ha="center", fontweight="bold")

    _save(fig, DIAGRAMS / "12_pricing.png")


# ─── 8. Bot P2P punta de lanza (slide 13) ────────────────────────────────────
def p2p_lead():
    fig = _new_fig()
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.9])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    ax.text(50, 56, "Bot P2P — Punta de lanza",
            fontsize=26, fontweight="bold", color=WHITE, ha="center")
    ax.text(50, 52.5, "Cash flow inmediato · Mercado virgen · Cero competencia ML",
            fontsize=14, color="#34D399", ha="center", style="italic")

    # left panel: 3 stat boxes stacked
    stats = [
        ("USD 50–200k", "Volumen diario por merchant top tier"),
        ("1.3–3.9%", "Spread bruto Top 1 vs Top 10"),
        ("0%", "Fees Binance P2P → margen 100% spread"),
    ]
    sy = 45
    for big, small in stats:
        box = FancyBboxPatch((4, sy - 9), 38, 9,
                             boxstyle="round,pad=0.3,rounding_size=0.8",
                             linewidth=1.6, edgecolor="#34D399", facecolor=BG_PANEL)
        ax.add_patch(box)
        ax.text(6, sy - 2, big, fontsize=28, fontweight="bold", color="#34D399", va="center")
        ax.text(6, sy - 6.5, small, fontsize=11, color=WHITE, va="center")
        sy -= 11

    # right panel: bullet points
    rx = 48
    ax.text(rx, 47, "¿Qué hace nuestro bot?", fontsize=15, fontweight="bold", color=LIME)
    bullets = [
        ("Anti-fraude", "ML detecta patrones de chargebacks y operadores tóxicos"),
        ("Spread dinámico", "Modelo ajusta spread según volatilidad y profundidad"),
        ("Top-1 ranking", "Optimiza ofertas para mantener posición Top 1 en Binance"),
        ("Plug-and-play", "Cliente conecta API Binance · bot opera solo · BYOK"),
        ("Validado", "Operadores actuales mueven 50–200k/día con Excel + 4 personas"),
    ]
    by = 42
    for title, desc in bullets:
        ax.add_patch(Circle((rx + 1, by), 0.6, color=LIME))
        ax.text(rx + 3, by, title, fontsize=12, fontweight="bold", color=WHITE, va="center")
        ax.text(rx + 3, by - 2.3, desc, fontsize=10, color=GRAY, va="center")
        by -= 6.5

    # bottom callout
    callout = FancyBboxPatch((4, 1.5), 92, 5,
                             boxstyle="round,pad=0.4,rounding_size=0.8",
                             linewidth=1.6, edgecolor=LIME, facecolor=BG_PANEL)
    ax.add_patch(callout)
    ax.text(50, 4, "Proyección 18 meses: 25 merchants mid + 3 whales = USD 15–25k MRR limpio",
            fontsize=14, fontweight="bold", color=LIME, ha="center", va="center")

    _save(fig, DIAGRAMS / "13_p2p_lead.png")


# ─── 9. MRR projection 24m (slide 14) — stacked bars ─────────────────────────
def mrr_projection():
    months = ["M3", "M6", "M9", "M12", "M15", "M18", "M21", "M24"]
    # values in USD k MRR — conservadora (techo 100, optimista 150)
    p2p = [2, 5, 8, 12, 17, 21, 23, 25]
    hedging = [0, 1, 2, 5, 12, 25, 42, 60]
    remesas = [0, 0, 2, 5, 10, 15, 20, 25]
    freelancers = [0, 0, 1, 3, 8, 18, 28, 40]

    fig = _new_fig()
    ax = fig.add_axes([0.10, 0.14, 0.85, 0.74])
    ax.set_facecolor(BG_DARK)
    for spine in ax.spines.values():
        spine.set_color(GRAY_DIM)

    x = np.arange(len(months))
    width = 0.6
    p1 = ax.bar(x, p2p, width, color="#34D399", label="Bot P2P", edgecolor=BG_DARK, linewidth=1)
    p2 = ax.bar(x, hedging, width, bottom=p2p, color="#60A5FA", label="Hedging PYMES", edgecolor=BG_DARK, linewidth=1)
    bottom2 = np.array(p2p) + np.array(hedging)
    p3 = ax.bar(x, remesas, width, bottom=bottom2, color="#FBBF24", label="Remesas B2B", edgecolor=BG_DARK, linewidth=1)
    bottom3 = bottom2 + np.array(remesas)
    p4 = ax.bar(x, freelancers, width, bottom=bottom3, color="#F472B6", label="App Freelancers", edgecolor=BG_DARK, linewidth=1)

    totals = np.array(p2p) + np.array(hedging) + np.array(remesas) + np.array(freelancers)
    for xi, total in zip(x, totals):
        ax.text(xi, total + 3, f"${total}k", fontsize=12, fontweight="bold",
                color=LIME, ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(months, fontsize=12, color=WHITE)
    ax.set_ylabel("MRR (USD miles)", fontsize=12, color=GRAY)
    ax.set_title("Proyección MRR por línea — 24 meses (escenario conservador)",
                 fontsize=18, fontweight="bold", color=WHITE, pad=20)
    ax.set_ylim(0, 175)
    ax.grid(True, axis="y", color=GRAY_DIM, alpha=0.2, linestyle=":")
    ax.legend(loc="upper left", facecolor=BG_PANEL, edgecolor=GRAY_DIM, labelcolor=WHITE, fontsize=11)

    # annotation: target band
    ax.axhspan(100, 150, color=LIME, alpha=0.06)
    ax.text(7.4, 125, "Target M24:\nUSD 100–150k", fontsize=11, color=LIME, ha="right", va="center",
            style="italic", fontweight="bold")

    _save(fig, CHARTS / "14_mrr_projection.png")


# ─── 10. Roadmap 12 meses (slide 16) ─────────────────────────────────────────
def roadmap():
    fig = _new_fig()
    ax = fig.add_axes([0.05, 0.06, 0.92, 0.86])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    ax.text(50, 56, "Roadmap 12 meses",
            fontsize=26, fontweight="bold", color=WHITE, ha="center")
    ax.text(50, 52.5, "Q2 2026 → Q1 2027 · 4 swimlanes paralelos",
            fontsize=13, color=GRAY, ha="center", style="italic")

    # Layout: reserved label column x=[0,17], timeline track x=[18,99] (81 units = 92 timeline units)
    LABEL_COL = 17.5
    TRACK_START = LABEL_COL + 0.5
    TRACK_END = 99
    TRACK_WIDTH = TRACK_END - TRACK_START

    def t2x(unit):
        return TRACK_START + (unit / 92) * TRACK_WIDTH

    quarters = ["Q2 2026", "Q3 2026", "Q4 2026", "Q1 2027"]
    # quarters represent [0-23, 23-46, 46-69, 69-92] in timeline units; center each
    quarter_units = [11.5, 34.5, 57.5, 80.5]
    for q, qu in zip(quarters, quarter_units):
        qx = t2x(qu)
        ax.text(qx, 47.5, q, fontsize=14, fontweight="bold", color=LIME, ha="center")
        ax.plot([qx, qx], [6, 45], color=GRAY_DIM, linewidth=0.6, alpha=0.4, linestyle=":")

    # vertical divider between labels and track
    ax.plot([LABEL_COL, LABEL_COL], [9, 45], color=GRAY_DIM, linewidth=0.6, alpha=0.5)

    swimlanes = [
        ("Bot P2P",          "#34D399",  [(0, 18, "MVP + 5 merchants beta"),
                                          (18, 23, "25 clientes pagos"),
                                          (41, 23, "50 clientes + perf fees"),
                                          (64, 23, "Whales + expansión LATAM")]),
        ("Hedging PYMES",    "#60A5FA",  [(18, 23, "Diseño + MVP"),
                                          (41, 23, "10 PYMES piloto"),
                                          (64, 23, "Banco partner")]),
        ("Remesas B2B",      "#FBBF24",  [(41, 23, "Outreach + API"),
                                          (64, 23, "2 casas de cambio piloto")]),
        ("App Freelancers",  "#F472B6",  [(64, 23, "Lanzamiento beta cerrada")]),
    ]
    lane_y = [38, 30, 22, 14]

    for (lane, color, blocks), ly in zip(swimlanes, lane_y):
        # lane label in reserved column
        ax.text(1, ly, lane, fontsize=12, fontweight="bold", color=color, va="center")
        # lane track
        ax.add_patch(FancyBboxPatch((TRACK_START, ly - 3), TRACK_WIDTH, 0.4,
                                    boxstyle="round,pad=0,rounding_size=0",
                                    linewidth=0, facecolor=GRAY_DIM, alpha=0.25))
        # blocks
        for start, span, label in blocks:
            xx = t2x(start)
            ww = (span / 92) * TRACK_WIDTH
            box = FancyBboxPatch((xx + 0.3, ly - 2.5), ww - 0.6, 5,
                                 boxstyle="round,pad=0.2,rounding_size=0.6",
                                 linewidth=1.3, edgecolor=color, facecolor=BG_PANEL)
            ax.add_patch(box)
            ax.text(xx + ww / 2, ly, label, fontsize=9.5, color=WHITE, ha="center", va="center")

    # milestones bottom
    ax.text(1, 6, "Hitos:", fontsize=12, fontweight="bold", color=LIME, va="center")
    msu = [11.5, 34.5, 57.5, 80.5]
    msl = ["v2.0 cierre + MRR USD 6k",
           "MRR USD 25k + Microsoft\nFounders Hub",
           "Banco partner + casa\nde cambio piloto",
           "MRR USD 100–150k\n+ Series Seed"]
    for u, l in zip(msu, msl):
        x = t2x(u)
        ax.add_patch(Circle((x, 6), 0.8, color=LIME))
        ax.text(x, 2.8, l, fontsize=9, color=GRAY, ha="center", va="top")

    _save(fig, DIAGRAMS / "16_roadmap.png")


# ─── 11. 90-day milestones (slide 17) ────────────────────────────────────────
def milestones_90():
    fig = _new_fig()
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    ax.text(50, 56, "Hitos críticos próximos 90 días",
            fontsize=26, fontweight="bold", color=WHITE, ha="center")
    ax.text(50, 52.5, "Mayo · Junio · Julio 2026",
            fontsize=14, color=LIME, ha="center", style="italic")

    months = [
        {"name": "MES 1 — MAYO", "color": "#34D399",
         "items": ["Cierre Smart v2.0 en producción",
                   "5 merchants beta P2P firmados",
                   "Deck Microsoft listo + reunión técnica",
                   "Newsletter premium live (USD 99/mes)"]},
        {"name": "MES 2 — JUNIO", "color": "#FBBF24",
         "items": ["Bot P2P: 10 clientes pagos",
                   "Primer piloto Hedging PYME",
                   "Aplicación Microsoft for Startups",
                   "Outreach 5 casas de cambio"]},
        {"name": "MES 3 — JULIO", "color": "#F472B6",
         "items": ["MRR USD 5k+ alcanzado",
                   "Créditos Azure asegurados",
                   "1er prospecto casa de cambio",
                   "Series Seed deck v2 listo"]},
    ]
    xs = [6, 36, 66]
    width = 28

    for col, x in zip(months, xs):
        # header
        head = FancyBboxPatch((x, 42), width, 5.5,
                              boxstyle="round,pad=0.2,rounding_size=0.6",
                              linewidth=0, facecolor=col["color"])
        ax.add_patch(head)
        ax.text(x + width / 2, 44.7, col["name"], fontsize=13, fontweight="bold",
                color=BG_DARK, ha="center", va="center")

        # body
        body = FancyBboxPatch((x, 6), width, 35,
                              boxstyle="round,pad=0.3,rounding_size=0.6",
                              linewidth=1.4, edgecolor=col["color"], facecolor=BG_PANEL)
        ax.add_patch(body)

        for i, item in enumerate(col["items"]):
            yy = 37 - i * 7
            ax.add_patch(Circle((x + 2.5, yy), 0.7, color=col["color"]))
            ax.text(x + 4.5, yy, item, fontsize=11, color=WHITE, va="center", wrap=True)

    _save(fig, DIAGRAMS / "17_milestones_90.png")


# ─── 12. Tracción (slide 21) ─────────────────────────────────────────────────
def traction():
    fig = _new_fig()
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.9])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    ax.text(50, 56, "Tracción y validación",
            fontsize=28, fontweight="bold", color=WHITE, ha="center")
    ax.text(50, 52, "18 meses construyendo · Producción real · 4 oportunidades validadas",
            fontsize=14, color=GRAY, ha="center", style="italic")

    big_stats = [
        ("18", "meses de\ndesarrollo", LIME),
        ("+25.63%", "backtest 2025\n(p = 0.006)", "#34D399"),
        ("3.35", "Sharpe ratio\n(institucional)", "#60A5FA"),
        ("82.4%", "win rate\n(34 trades OOS)", "#FBBF24"),
        ("4", "productos en\nroadmap activo", "#F472B6"),
    ]
    xs = [6, 25.5, 44, 61.5, 80]
    width = 14
    for (val, lbl, color), x in zip(big_stats, xs):
        box = FancyBboxPatch((x, 22), width, 22,
                             boxstyle="round,pad=0.4,rounding_size=1.0",
                             linewidth=1.8, edgecolor=color, facecolor=BG_PANEL)
        ax.add_patch(box)
        ax.text(x + width / 2, 36.5, val, fontsize=30, fontweight="bold",
                color=color, ha="center", va="center")
        ax.text(x + width / 2, 27.5, lbl, fontsize=11, color=WHITE, ha="center", va="center")

    # bottom row: extra validations
    extras = [
        "Backtest auditado walk-forward (2020–2024 train, 2025 OOS)",
        "Sistema en producción con datos reales (Airflow + MLflow + 25+ servicios)",
        "Comparables internacionales validan precio: €175M – USD 12B",
        "Regime gate (Hurst) bloqueó 13/14 semanas mean-reverting en Q1 2026 → +0.61% YTD",
    ]
    ey = 16
    for e in extras:
        ax.add_patch(Circle((6, ey), 0.5, color=LIME))
        ax.text(8, ey, e, fontsize=12, color=WHITE, va="center")
        ey -= 3.2

    _save(fig, DIAGRAMS / "21_traction.png")


def main():
    print("Generating PNG assets...")
    hero()
    architecture()
    equity_curve()
    opportunities()
    comparables()
    revenue_streams()
    pricing()
    p2p_lead()
    mrr_projection()
    roadmap()
    milestones_90()
    traction()
    print("Done.")


if __name__ == "__main__":
    main()
