#!/usr/bin/env python3
"""Generate 8 professional architecture diagram PNGs for USDCOP Trading System."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Global Theme ──────────────────────────────────────────────────────────────
BG = '#0f172a'
TEXT = '#ffffff'
TEXT_DIM = '#94a3b8'
GRID = '#1e293b'

# Status colors
ACTIVE_BORDER = '#10b981'
ACTIVE_FILL = '#064e3b'
PAUSED_BORDER = '#6b7280'
PAUSED_FILL = '#374151'
DB_BORDER = '#ea580c'
DB_FILL = '#431407'
FILE_BORDER = '#3b82f6'
FILE_FILL = '#1e3a5f'
EXT_BORDER = '#475569'
EXT_FILL = '#1e293b'
DECISION_BORDER = '#eab308'
DECISION_FILL = '#422006'
RISK_BORDER = '#ef4444'
RISK_FILL = '#450a0a'
HIGHLIGHT_BORDER = '#a855f7'
HIGHLIGHT_FILL = '#3b0764'

DPI = 150


def setup_fig(w_in, h_in, title, subtitle=None):
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    # Title bar
    ax.add_patch(FancyBboxPatch((0, 93), 100, 7, boxstyle="square,pad=0",
                                facecolor='#1e293b', edgecolor='#334155', linewidth=1.5))
    ax.text(50, 96.5, title, ha='center', va='center', fontsize=18,
            fontweight='bold', color=TEXT, family='sans-serif')
    if subtitle:
        ax.text(50, 93.8, subtitle, ha='center', va='center', fontsize=10,
                color=TEXT_DIM, family='sans-serif')
    return fig, ax


def box(ax, x, y, w, h, label, border, fill, fontsize=10, sublabel=None, radius=0.3):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={radius}",
                        facecolor=fill, edgecolor=border, linewidth=1.8)
    ax.add_patch(b)
    if sublabel:
        # Count lines in sublabel to compute spacing
        sub_lines = sublabel.count('\n') + 1
        title_lines = label.count('\n') + 1
        total_lines = title_lines + sub_lines
        line_h = min(2.2, h / (total_lines + 1.5))
        top = y + h - line_h * 0.8
        # Title
        ax.text(x + w/2, top - (title_lines - 1) * line_h * 0.4, label,
                ha='center', va='top',
                fontsize=fontsize, fontweight='bold', color=TEXT, family='sans-serif',
                linespacing=1.0)
        # Sublabel below
        sub_y = y + line_h * (sub_lines * 0.45 + 0.3)
        ax.text(x + w/2, sub_y, sublabel, ha='center', va='center',
                fontsize=max(fontsize - 2, 6), color=TEXT_DIM, family='sans-serif',
                linespacing=1.1)
    else:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=TEXT, family='sans-serif')
    return b


def arrow(ax, x1, y1, x2, y2, color='#64748b', style='->', lw=1.5):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, color=color, linewidth=lw,
                        mutation_scale=15, connectionstyle='arc3,rad=0')
    ax.add_patch(a)
    return a


def arrow_curved(ax, x1, y1, x2, y2, color='#64748b', rad=0.15, lw=1.5):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle='->', color=color, linewidth=lw,
                        mutation_scale=15, connectionstyle=f'arc3,rad={rad}')
    ax.add_patch(a)
    return a


def diamond(ax, cx, cy, w, h, label, border, fill, fontsize=9):
    pts = np.array([[cx, cy + h/2], [cx + w/2, cy], [cx, cy - h/2], [cx - w/2, cy]])
    poly = plt.Polygon(pts, facecolor=fill, edgecolor=border, linewidth=1.8, closed=True)
    ax.add_patch(poly)
    ax.text(cx, cy, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=TEXT, family='sans-serif')


def legend_item(ax, x, y, color, label):
    ax.add_patch(FancyBboxPatch((x, y), 2.5, 1.8, boxstyle="round,pad=0.1",
                                facecolor=color, edgecolor='#475569', linewidth=1))
    ax.text(x + 3.5, y + 0.9, label, va='center', fontsize=8, color=TEXT_DIM, family='sans-serif')


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 0: Overview
# ═══════════════════════════════════════════════════════════════════════════════
def slide0():
    fig, ax = setup_fig(16, 4, 'USDCOP Trading System — Architecture Overview',
                        '9 Layers  |  26 DAGs  |  25+ Docker Services  |  Smart Simple v2.0 Production')

    layers = [
        ('L0\nData Ops', '5 DAGs', ACTIVE_BORDER, ACTIVE_FILL),
        ('L1\nFeatures', '2 DAGs', PAUSED_BORDER, PAUSED_FILL),
        ('L2\nDatasets', '1 Script', PAUSED_BORDER, PAUSED_FILL),
        ('L3\nTraining', '3 DAGs', ACTIVE_BORDER, ACTIVE_FILL),
        ('L4\nValidation', '5 Gates', ACTIVE_BORDER, ACTIVE_FILL),
        ('L5\nSignals', '4 DAGs', ACTIVE_BORDER, ACTIVE_FILL),
        ('L7\nExecution', '2 DAGs', ACTIVE_BORDER, ACTIVE_FILL),
        ('L6\nMonitor', '2 DAGs', ACTIVE_BORDER, ACTIVE_FILL),
        ('L8\nIntelligence', '5 DAGs', ACTIVE_BORDER, ACTIVE_FILL),
    ]

    n = len(layers)
    bw = 8.5
    gap = 1.8
    total = n * bw + (n - 1) * gap
    sx = (100 - total) / 2
    by = 30
    bh = 28

    for i, (name, sub, border, fill) in enumerate(layers):
        x = sx + i * (bw + gap)
        box(ax, x, by, bw, bh, name, border, fill, fontsize=11, sublabel=sub)
        if i < n - 1:
            arrow(ax, x + bw + 0.2, by + bh/2, x + bw + gap - 0.2, by + bh/2,
                  color='#64748b', lw=2)

    # Bottom annotation
    ax.text(50, 15, 'H5 Weekly (PRODUCTION): Ridge+BR+XGB + Regime Gate + Effective HS  |  '
            '+25.63% (2025)  |  +0.61% YTD (2026, gate blocked 13/14 weeks)',
            ha='center', va='center', fontsize=10, color=ACTIVE_BORDER, family='sans-serif')
    ax.text(50, 8, 'H1 Daily (PAUSED): 9 models x 7 horizons  |  '
            'RL (DEPRIORITIZED): PPO, p=0.272, NOT significant',
            ha='center', va='center', fontsize=9, color=PAUSED_BORDER, family='sans-serif')

    # Legend
    legend_item(ax, 3, 1, ACTIVE_FILL, 'Active')
    legend_item(ax, 18, 1, PAUSED_FILL, 'Paused/RL')

    fig.savefig('/home/globalforex/Documents/USDCOP-RL/USDCOP-RL-Models/docs/slides/slide0_overview.png',
                dpi=DPI, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print('  slide0_overview.png')


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 1: Data Ops (L0)
# ═══════════════════════════════════════════════════════════════════════════════
def slide1():
    fig, ax = setup_fig(12, 6, 'Slide 1 — L0: Data Operations',
                        '5 DAGs  |  3 FX Pairs  |  40 Macro Variables  |  7 API Sources')

    # External sources
    box(ax, 3, 72, 16, 8, 'TwelveData API', EXT_BORDER, EXT_FILL, 10,
        'COP + MXN + BRL')
    box(ax, 3, 52, 16, 8, 'Macro APIs (7)', EXT_BORDER, EXT_FILL, 10,
        'FRED, BanRep, ...')

    # DAGs column
    box(ax, 30, 74, 20, 6, 'ohlcv_realtime', ACTIVE_BORDER, ACTIVE_FILL, 9,
        '*/5 min Mon-Fri')
    box(ax, 30, 65, 20, 6, 'ohlcv_backfill', ACTIVE_BORDER, ACTIVE_FILL, 9,
        'Manual / Gap-fill')
    box(ax, 30, 54, 20, 6, 'macro_update', ACTIVE_BORDER, ACTIVE_FILL, 9,
        'Hourly 8-12 COT')
    box(ax, 30, 45, 20, 6, 'macro_backfill', ACTIVE_BORDER, ACTIVE_FILL, 9,
        'Weekly Sun 04:00')
    box(ax, 30, 33, 20, 6, 'seed_backup', ACTIVE_BORDER, ACTIVE_FILL, 9,
        'Daily 13:00 COT')

    # DB
    box(ax, 62, 63, 18, 14, 'PostgreSQL\nTimescaleDB', DB_BORDER, DB_FILL, 11,
        'usdcop_m5_ohlcv\nmacro_indicators')

    # File outputs
    box(ax, 62, 42, 18, 10, 'Seed Parquets', FILE_BORDER, FILE_FILL, 10,
        'seeds/latest/\ndata/backups/')

    # MinIO
    box(ax, 62, 28, 18, 8, 'MinIO (S3)', EXT_BORDER, EXT_FILL, 10,
        'Tier 3 Backup')

    # Arrows: sources to DAGs
    arrow(ax, 19, 76, 29.5, 76, ACTIVE_BORDER, lw=1.8)
    arrow(ax, 19, 68, 29.5, 68, ACTIVE_BORDER, lw=1.8)
    arrow(ax, 19, 56, 29.5, 56, ACTIVE_BORDER, lw=1.8)
    arrow(ax, 19, 56, 29.5, 48, ACTIVE_BORDER, lw=1.8)

    # DAGs to DB
    arrow(ax, 50.5, 76, 61.5, 72, ACTIVE_BORDER, lw=1.8)
    arrow(ax, 50.5, 68, 61.5, 70, ACTIVE_BORDER, lw=1.8)
    arrow(ax, 50.5, 56, 61.5, 68, ACTIVE_BORDER, lw=1.8)
    arrow(ax, 50.5, 48, 61.5, 66, ACTIVE_BORDER, lw=1.8)

    # seed_backup to files
    arrow(ax, 50.5, 36, 61.5, 47, ACTIVE_BORDER, lw=1.8)
    arrow(ax, 50.5, 34, 61.5, 33, ACTIVE_BORDER, lw=1.8)

    # DB to seed_backup
    arrow_curved(ax, 62, 64, 50.5, 38, '#ea580c', rad=-0.2, lw=1.2)

    # Key constraints
    ax.text(50, 20, 'All timestamps: America/Bogota  |  Session: 8:00-12:55 COT Mon-Fri',
            ha='center', fontsize=9, color=TEXT_DIM)
    ax.text(50, 15, 'BRL quirk: fetch UTC, convert to COT  |  Macro T-1 anti-leakage',
            ha='center', fontsize=9, color=TEXT_DIM)

    # Freshness gates
    box(ax, 3, 28, 16, 10, 'Freshness Gates', RISK_BORDER, RISK_FILL, 9,
        'OHLCV <3d\nMacro <7d')

    legend_item(ax, 3, 4, ACTIVE_FILL, 'Active DAG')
    legend_item(ax, 18, 4, DB_FILL, 'Database')
    legend_item(ax, 33, 4, FILE_FILL, 'Files')
    legend_item(ax, 48, 4, EXT_FILL, 'External')

    fig.savefig('/home/globalforex/Documents/USDCOP-RL/USDCOP-RL-Models/docs/slides/slide1_data_ops.png',
                dpi=DPI, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print('  slide1_data_ops.png')


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 2: Feature Ops (L1/L2) — RL Track, all paused
# ═══════════════════════════════════════════════════════════════════════════════
def slide2():
    fig, ax = setup_fig(12, 6, 'Slide 2 — L1/L2: Feature Operations (RL Track)',
                        'PAUSED  |  CanonicalFeatureBuilder  |  Z-score Normalization  |  Anti-Leakage')

    # RL track label
    ax.add_patch(FancyBboxPatch((5, 82, ), 90, 5, boxstyle="round,pad=0.2",
                                facecolor='#1c1917', edgecolor=PAUSED_BORDER, linewidth=2,
                                linestyle='--'))
    ax.text(50, 84.5, 'RL TRACK (DEPRIORITIZED) — All DAGs Paused', ha='center',
            fontsize=12, fontweight='bold', color=PAUSED_BORDER)

    # L1 DAGs
    box(ax, 5, 60, 22, 10, 'L1: feature_refresh', PAUSED_BORDER, PAUSED_FILL, 10,
        '*/5 min  |  18 features\nZ-score + clip [-5,5]')
    box(ax, 5, 42, 22, 10, 'L1: model_promotion', PAUSED_BORDER, PAUSED_FILL, 10,
        'On approval trigger\nRecompute historical')

    # NRT table
    box(ax, 38, 55, 22, 12, 'inference_ready_nrt', DB_BORDER, DB_FILL, 10,
        'FLOAT[18] features\n+ feature_order_hash\n+ norm_stats_hash')

    # L2
    box(ax, 72, 60, 22, 10, 'L2: dataset_build', PAUSED_BORDER, PAUSED_FILL, 10,
        'SSOTDatasetBuilder\ntrain/val/test splits')

    # Outputs
    box(ax, 72, 42, 22, 10, 'Parquet Outputs', FILE_BORDER, FILE_FILL, 10,
        'DS_train.parquet\nDS_val.parquet\nnorm_stats.json')

    # Feature contract
    box(ax, 35, 28, 30, 8, 'Feature Contract (CTR-FEATURE-001)', HIGHLIGHT_BORDER, HIGHLIGHT_FILL, 10,
        'FEATURE_ORDER + SHA256 hash  |  20 features')

    # Arrows
    arrow(ax, 27.5, 65, 37.5, 63, PAUSED_BORDER, lw=1.5)
    arrow(ax, 27.5, 47, 37.5, 58, PAUSED_BORDER, lw=1.5)
    arrow(ax, 60.5, 61, 71.5, 65, PAUSED_BORDER, lw=1.5)
    arrow(ax, 83, 59.5, 83, 52.5, FILE_BORDER, lw=1.5)
    arrow(ax, 50, 55, 50, 36.5, HIGHLIGHT_BORDER, lw=1.2)

    # Anti-leakage annotations
    ax.text(50, 18, 'Anti-Leakage: norm_stats from TRAIN split ONLY  |  '
            'Macro merge_asof(backward) + shift(1)',
            ha='center', fontsize=9, color=TEXT_DIM)
    ax.text(50, 12, 'RSI: Wilder\'s EMA (NOT pandas ewm)  |  '
            'State features added at runtime by TradingEnvironment',
            ha='center', fontsize=9, color=TEXT_DIM)

    legend_item(ax, 3, 3, PAUSED_FILL, 'Paused (RL)')
    legend_item(ax, 20, 3, DB_FILL, 'NRT Table')
    legend_item(ax, 37, 3, FILE_FILL, 'Files')
    legend_item(ax, 52, 3, HIGHLIGHT_FILL, 'Contract')

    fig.savefig('/home/globalforex/Documents/USDCOP-RL/USDCOP-RL-Models/docs/slides/slide2_feature_ops.png',
                dpi=DPI, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print('  slide2_feature_ops.png')


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 3: Training & Validation (L3/L4)
# ═══════════════════════════════════════════════════════════════════════════════
def slide3():
    fig, ax = setup_fig(13.3, 7.3, 'Slide 3 — L3/L4: Training & Validation',
                        'H5 ACTIVE (Ridge+BR+XGB)  |  H1 PAUSED (9 models x 7 horizons)  |  5 Approval Gates')

    # ── H5 Training (ACTIVE) ──
    box(ax, 2, 68, 28, 12, 'H5-L3: Weekly Training', ACTIVE_BORDER, ACTIVE_FILL, 11,
        'Sun 01:30 COT  |  Expanding window\n'
        'Ridge + BayesianRidge + XGBoost\n'
        '23 features  |  target = ln(c[t+5]/c[t])')

    # ── H1 Training (PAUSED) ──
    box(ax, 2, 48, 28, 14, 'H1-L3: Weekly Training', PAUSED_BORDER, PAUSED_FILL, 11,
        'Sun 01:00 COT  |  PAUSED\n'
        '9 models x 7 horizons = 63 variants')

    # Model grid (9x7)
    models = ['Ridge', 'BR', 'ARD', 'XGB', 'LGBM', 'CatB', 'HybX', 'HybL', 'HybC']
    horizons = ['H1', 'H5', 'H10', 'H15', 'H20', 'H25', 'H30']
    gx, gy = 4, 28
    cw, ch = 2.8, 2.0
    for j, h in enumerate(horizons):
        ax.text(gx + j * cw + cw/2, gy + len(models) * ch + 0.8, h,
                ha='center', fontsize=6, color=TEXT_DIM)
    for i, m in enumerate(models):
        ax.text(gx - 0.5, gy + (len(models) - 1 - i) * ch + ch/2, m,
                ha='right', fontsize=6, color=TEXT_DIM)
        for j in range(7):
            c = ACTIVE_FILL if (m in ['Ridge', 'BR'] and h == 'H5') else '#1e293b'
            b = ACTIVE_BORDER if (m in ['Ridge', 'BR'] and j == 1) else '#334155'
            ax.add_patch(FancyBboxPatch((gx + j * cw, gy + (len(models) - 1 - i) * ch),
                                        cw - 0.3, ch - 0.3, boxstyle="round,pad=0.05",
                                        facecolor=c, edgecolor=b, linewidth=0.8))

    ax.text(gx + 3.5 * cw, gy - 2.5, '9 x 7 = 63 model variants (H1 track)',
            ha='center', fontsize=8, color=TEXT_DIM)

    # ── L4: Validation Gates ──
    box(ax, 40, 70, 25, 10, 'L4: Backtest Validation', ACTIVE_BORDER, ACTIVE_FILL, 11,
        'Walk-forward OOS (2025)\n5 automated gates')

    gates = [
        ('min_return', '> -15%', True),
        ('min_sharpe', '> 0.0', True),
        ('max_drawdown', '< 20%', True),
        ('min_trades', '>= 10', True),
        ('p_value', '< 0.05', True),
    ]
    for i, (name, thresh, passed) in enumerate(gates):
        bx = 40 + i * 5
        by_g = 58
        c_b = ACTIVE_BORDER if passed else RISK_BORDER
        c_f = ACTIVE_FILL if passed else RISK_FILL
        ax.add_patch(FancyBboxPatch((bx, by_g), 4.5, 6, boxstyle="round,pad=0.1",
                                    facecolor=c_f, edgecolor=c_b, linewidth=1.2))
        ax.text(bx + 2.25, by_g + 4, name, ha='center', fontsize=6, fontweight='bold',
                color=TEXT)
        ax.text(bx + 2.25, by_g + 1.5, thresh, ha='center', fontsize=6, color=TEXT_DIM)

    arrow(ax, 30.5, 74, 39.5, 76, ACTIVE_BORDER, lw=2)
    arrow(ax, 52.5, 69.5, 52.5, 64.5, ACTIVE_BORDER, lw=1.5)

    # ── Approval Flow ──
    box(ax, 72, 70, 22, 10, 'Vote 1/2\n(Automatic)', ACTIVE_BORDER, ACTIVE_FILL, 11,
        'Python gates')
    box(ax, 72, 52, 22, 10, 'Vote 2/2\n(Human)', HIGHLIGHT_BORDER, HIGHLIGHT_FILL, 11,
        '/dashboard review')
    box(ax, 72, 35, 22, 8, 'APPROVED', ACTIVE_BORDER, ACTIVE_FILL, 12)
    box(ax, 72, 22, 22, 8, '--phase production', FILE_BORDER, FILE_FILL, 10,
        'Retrain 2020-2025')

    arrow(ax, 65.5, 75, 71.5, 75, ACTIVE_BORDER, lw=2)
    arrow(ax, 83, 69.5, 83, 62.5, HIGHLIGHT_BORDER, lw=2)
    arrow(ax, 83, 51.5, 83, 43.5, ACTIVE_BORDER, lw=2)
    arrow(ax, 83, 34.5, 83, 30.5, FILE_BORDER, lw=2)

    # Results annotation
    ax.text(83, 14, 'v2.0 Results (2025 OOS):\n+25.63%  |  Sharpe 3.35  |  p=0.006\n'
            '34 trades  |  82.4% WR  |  $10K -> $12,563',
            ha='center', fontsize=9, color=ACTIVE_BORDER, style='italic',
            bbox=dict(facecolor='#0a1628', edgecolor=ACTIVE_BORDER, boxstyle='round,pad=0.5',
                      linewidth=1))

    legend_item(ax, 40, 3, ACTIVE_FILL, 'Active')
    legend_item(ax, 52, 3, PAUSED_FILL, 'Paused')
    legend_item(ax, 64, 3, HIGHLIGHT_FILL, 'Human Review')

    fig.savefig('/home/globalforex/Documents/USDCOP-RL/USDCOP-RL-Models/docs/slides/slide3_training_ops.png',
                dpi=DPI, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print('  slide3_training_ops.png')


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 4: Signal & Execution (L5/L7)
# ═══════════════════════════════════════════════════════════════════════════════
def slide4():
    fig, ax = setup_fig(13.3, 6.7, 'Slide 4 — L5/L7: Signal Generation & Execution',
                        'H5 Weekly Lifecycle  |  Regime Gate (Hurst)  |  TP/HS Adaptive Stops  |  No Trailing')

    # ── Top flow: signal pipeline ──
    box(ax, 2, 72, 14, 9, 'L3 Models', ACTIVE_BORDER, ACTIVE_FILL, 10,
        'Ridge + BR\n+ XGBoost')

    box(ax, 20, 72, 14, 9, 'L5: Signal', ACTIVE_BORDER, ACTIVE_FILL, 10,
        'Mon 08:15 COT\nEnsemble mean')

    box(ax, 38, 72, 14, 9, 'Confidence\nScorer', ACTIVE_BORDER, ACTIVE_FILL, 10,
        'HIGH/MED/LOW\n3-tier')

    arrow(ax, 16.5, 76.5, 19.5, 76.5, ACTIVE_BORDER, lw=2)
    arrow(ax, 34.5, 76.5, 37.5, 76.5, ACTIVE_BORDER, lw=2)

    # Vol targeting
    box(ax, 56, 72, 14, 9, 'Vol-Target', ACTIVE_BORDER, ACTIVE_FILL, 10,
        'Mon 08:45 COT\ntv=0.15 annual')
    arrow(ax, 52.5, 76.5, 55.5, 76.5, ACTIVE_BORDER, lw=2)

    # ── Regime Gate (diamond) ──
    diamond(ax, 80, 76.5, 16, 10, 'Regime\nGate', DECISION_BORDER, DECISION_FILL, 10)
    arrow(ax, 70.5, 76.5, 71.5, 76.5, DECISION_BORDER, lw=2)

    # Gate outcomes
    ax.text(92, 82, 'TRENDING', fontsize=8, color=ACTIVE_BORDER, fontweight='bold')
    ax.text(92, 80, 'H > 0.52: Full size', fontsize=7, color=TEXT_DIM)

    ax.text(92, 76.5, 'INDETERMINATE', fontsize=8, color=DECISION_BORDER, fontweight='bold')
    ax.text(92, 74.5, '0.42 < H < 0.52: 40%', fontsize=7, color=TEXT_DIM)

    ax.text(92, 71, 'MEAN-REVERT', fontsize=8, color=RISK_BORDER, fontweight='bold')
    ax.text(92, 69, 'H < 0.42: SKIP', fontsize=7, color=TEXT_DIM)

    # ── Middle: Execution timeline ──
    ax.add_patch(FancyBboxPatch((2, 47), 96, 16, boxstyle="round,pad=0.3",
                                facecolor='#0a1628', edgecolor='#334155', linewidth=1.2))
    ax.text(50, 61, 'Weekly Trade Lifecycle (H5-L7 Executor)', ha='center',
            fontsize=12, fontweight='bold', color=TEXT)

    # Timeline
    timeline_y = 52
    days = ['Mon 9:00', 'Tue', 'Wed', 'Thu', 'Fri 12:50']
    day_x = [10, 28, 46, 64, 82]

    ax.plot([8, 90], [timeline_y, timeline_y], color='#475569', linewidth=2, zorder=1)

    for dx, dl in zip(day_x, days):
        ax.plot(dx, timeline_y, 'o', color=ACTIVE_BORDER, markersize=8, zorder=2)
        ax.text(dx, timeline_y - 3, dl, ha='center', fontsize=8, color=TEXT_DIM)

    ax.text(10, timeline_y + 2.5, 'ENTRY\nLimit order', ha='center', fontsize=7,
            color=ACTIVE_BORDER, fontweight='bold')
    ax.text(46, timeline_y + 2.5, 'MONITOR */30\nTP/HS checks', ha='center', fontsize=7,
            color=DECISION_BORDER, fontweight='bold')
    ax.text(82, timeline_y + 2.5, 'CLOSE\nMarket order', ha='center', fontsize=7,
            color=FILE_BORDER, fontweight='bold')

    # ── Bottom: Stop mechanics ──
    box(ax, 2, 22, 20, 14, 'Adaptive Stops', ACTIVE_BORDER, ACTIVE_FILL, 10,
        'HS = vol * sqrt(5) * 2.0\nClamp [1%, 3%]\nTP = HS * 0.5')

    box(ax, 26, 22, 20, 14, 'Effective HS\n(v2.0)', DECISION_BORDER, DECISION_FILL, 10,
        'min(HS_base,\n3.5% / leverage)\nPortfolio cap')

    box(ax, 50, 22, 20, 14, 'Dynamic\nLeverage (v2.0)', HIGHLIGHT_BORDER, HIGHLIGHT_FILL, 10,
        'Scale [0.25, 1.0]\nRolling WR +\nDrawdown factor')

    box(ax, 74, 22, 22, 14, 'Sizing Rules', FILE_BORDER, FILE_FILL, 10,
        'SHORT: flat 1.5x\nLONG HIGH: 1.0x\nLONG MED: 0.5x\nLONG LOW: SKIP')

    # Exit reasons
    ax.text(50, 12, 'Exit Reasons: take_profit (62%)  |  week_end (32%)  |  hard_stop (6%)  |  '
            'circuit_breaker (0%)',
            ha='center', fontsize=8, color=TEXT_DIM)

    legend_item(ax, 3, 3, ACTIVE_FILL, 'Active')
    legend_item(ax, 18, 3, DECISION_FILL, 'Decision')
    legend_item(ax, 33, 3, HIGHLIGHT_FILL, 'v2.0 Addition')
    legend_item(ax, 50, 3, FILE_FILL, 'Config')

    fig.savefig('/home/globalforex/Documents/USDCOP-RL/USDCOP-RL-Models/docs/slides/slide4_signal_execution.png',
                dpi=DPI, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print('  slide4_signal_execution.png')


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 5: Monitoring & Risk (L6)
# ═══════════════════════════════════════════════════════════════════════════════
def slide5():
    fig, ax = setup_fig(12, 6.7, 'Slide 5 — L6: Monitoring & Risk Management',
                        '3-Layer Risk Architecture  |  9+7 Checks  |  4 Guardrails  |  Kill Switch')

    # ── Layer 1: Chain of Responsibility ──
    box(ax, 2, 68, 30, 14, 'Layer 1: Risk Check Chain', RISK_BORDER, RISK_FILL, 11,
        '9 checks in sequence\nFirst failure stops chain')

    checks = ['HOLD', 'Hours', 'CB', 'Cool', 'Conf', 'Loss', 'DD', 'Losses', 'MaxTr']
    for i, ch in enumerate(checks):
        cx = 4 + i * 3.1
        ax.add_patch(FancyBboxPatch((cx, 69), 2.8, 3.5, boxstyle="round,pad=0.05",
                                    facecolor='#1c0a0a', edgecolor='#dc2626', linewidth=0.8))
        ax.text(cx + 1.4, 70.7, ch, ha='center', fontsize=5.5, color=TEXT, fontweight='bold')

    # ── Layer 2: Risk Enforcer ──
    box(ax, 36, 68, 28, 14, 'Layer 2: Risk Enforcer', DECISION_BORDER, DECISION_FILL, 11,
        '7 pluggable rules\nALLOW / BLOCK / REDUCE')

    rules = ['Kill', 'DLoss', 'Trade', 'Cool', 'Short', 'Size', 'Conf']
    for i, rl in enumerate(rules):
        rx = 38 + i * 3.7
        ax.add_patch(FancyBboxPatch((rx, 69), 3.2, 3.5, boxstyle="round,pad=0.05",
                                    facecolor='#1c1006', edgecolor='#ca8a04', linewidth=0.8))
        ax.text(rx + 1.6, 70.7, rl, ha='center', fontsize=5.5, color=TEXT, fontweight='bold')

    # ── Kill Switch ──
    box(ax, 70, 68, 24, 14, 'Kill Switch', RISK_BORDER, RISK_FILL, 13,
        'DD >= 15%: ACTIVATE\nBlocks ALL trades\nManual reset only\n(confirm=True)')

    # Arrows between layers
    arrow(ax, 32.5, 75, 35.5, 75, RISK_BORDER, lw=2)
    arrow(ax, 64.5, 75, 69.5, 75, DECISION_BORDER, lw=2)

    # ── Guardrails row ──
    ax.text(50, 56, 'H5 Pipeline Guardrails (Strategic Level)', ha='center',
            fontsize=12, fontweight='bold', color=TEXT)

    guardrails = [
        ('Circuit Breaker', '5 losses OR\n12% drawdown', 'Pause + Alert', RISK_BORDER, RISK_FILL),
        ('Long Insistence', '>60% LONGs in\n8-week window', 'Alert Only', DECISION_BORDER, DECISION_FILL),
        ('Rolling DA\n(SHORT)', 'SHORT DA <55%\n16-week window', 'Pause SHORTs', RISK_BORDER, RISK_FILL),
        ('Rolling DA\n(LONG)', 'LONG DA <45%\n16-week window', 'Pause LONGs', RISK_BORDER, RISK_FILL),
    ]

    for i, (name, trigger, action, brd, fll) in enumerate(guardrails):
        gx = 4 + i * 24
        box(ax, gx, 38, 20, 12, name, brd, fll, 10)
        ax.text(gx + 10, 42, trigger, ha='center', fontsize=7, color=TEXT_DIM)
        ax.text(gx + 10, 39.5, action, ha='center', fontsize=7, color=ACTIVE_BORDER,
                fontweight='bold')

    # ── L6 DAGs ──
    box(ax, 10, 18, 24, 10, 'H5-L6: Weekly Monitor', ACTIVE_BORDER, ACTIVE_FILL, 10,
        'Fri 14:30 COT\nDA, Sharpe, MaxDD, guardrails')
    box(ax, 42, 18, 24, 10, 'H1-L6: Paper Monitor', PAUSED_BORDER, PAUSED_FILL, 10,
        'Mon-Fri 19:00 COT\nPaper trading log')

    # Command pattern
    box(ax, 74, 18, 20, 10, 'Command Pattern', HIGHLIGHT_BORDER, HIGHLIGHT_FILL, 10,
        '6 commands\nUndo/Redo + Audit')

    legend_item(ax, 3, 5, RISK_FILL, 'Risk / Block')
    legend_item(ax, 20, 5, DECISION_FILL, 'Decision / Warn')
    legend_item(ax, 40, 5, HIGHLIGHT_FILL, 'Ops Commands')
    legend_item(ax, 60, 5, ACTIVE_FILL, 'Active DAG')

    fig.savefig('/home/globalforex/Documents/USDCOP-RL/USDCOP-RL-Models/docs/slides/slide5_monitoring.png',
                dpi=DPI, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print('  slide5_monitoring.png')


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 6: Intelligence (L8 + Forecasting)
# ═══════════════════════════════════════════════════════════════════════════════
def slide6():
    fig, ax = setup_fig(13.3, 7.3, 'Slide 6 — L8: Intelligence & Analysis Module',
                        'News Engine (5 sources)  |  LLM Analysis (GPT-4o)  |  '
                        '/analysis + /forecasting pages')

    # ── News Engine Pipeline ──
    ax.text(50, 87, 'News Engine Pipeline', ha='center', fontsize=13, fontweight='bold',
            color=TEXT)

    sources = ['GDELT', 'NewsAPI', 'Investing', 'LaRepublica', 'Portafolio']
    src_status = [PAUSED_FILL, PAUSED_FILL, ACTIVE_FILL, ACTIVE_FILL, ACTIVE_FILL]
    src_border = [PAUSED_BORDER, PAUSED_BORDER, ACTIVE_BORDER, ACTIVE_BORDER, ACTIVE_BORDER]
    for i, (s, sf, sb) in enumerate(zip(sources, src_status, src_border)):
        sx = 3 + i * 11
        box(ax, sx, 75, 9.5, 6, s, sb, sf, 8)

    # Pipeline DAGs
    box(ax, 58, 75, 14, 6, 'news_daily\n3x/day', ACTIVE_BORDER, ACTIVE_FILL, 8)
    box(ax, 74, 75, 14, 6, 'news_alert\n*/30 min', ACTIVE_BORDER, ACTIVE_FILL, 8)

    for i in range(5):
        arrow(ax, 3 + i * 11 + 4.75, 75, 64, 81.5, '#475569', lw=0.8)

    # Enrichment
    box(ax, 20, 63, 26, 6, 'Enrichment Pipeline (5 stages)', HIGHLIGHT_BORDER, HIGHLIGHT_FILL, 9,
        'Categorize, Relevance, Sentiment, NER, Flags')

    arrow(ax, 65, 75, 33, 69.5, ACTIVE_BORDER, lw=1.5)

    # DB
    box(ax, 55, 63, 16, 6, 'news_articles\n8 tables', DB_BORDER, DB_FILL, 9)
    arrow(ax, 46.5, 66, 54.5, 66, ACTIVE_BORDER, lw=1.5)

    # Feature export
    box(ax, 78, 63, 16, 6, '~60 daily\nfeatures', FILE_BORDER, FILE_FILL, 9)
    arrow(ax, 71.5, 66, 77.5, 66, FILE_BORDER, lw=1.5)

    # ── Analysis Module ──
    ax.text(50, 55, 'Analysis Module (L8)', ha='center', fontsize=13, fontweight='bold',
            color=TEXT)

    box(ax, 3, 40, 16, 8, 'MacroAnalyzer', ACTIVE_BORDER, ACTIVE_FILL, 9,
        '13 vars: SMA, BB\nRSI, MACD, z-score')

    box(ax, 22, 40, 14, 8, 'LLMClient', HIGHLIGHT_BORDER, HIGHLIGHT_FILL, 9,
        'Azure GPT-4o\n+ Claude fallback')

    box(ax, 39, 40, 14, 8, 'L8 DAG', ACTIVE_BORDER, ACTIVE_FILL, 9,
        'Mon-Fri 14:00\nDaily + Fri weekly')

    box(ax, 56, 40, 16, 8, 'JSON Export', FILE_BORDER, FILE_FILL, 9,
        'weekly_YYYY_WXX.json\nanalysis_index.json')

    box(ax, 76, 40, 18, 8, '/analysis Page', ACTIVE_BORDER, ACTIVE_FILL, 9,
        '14 components\n4 API routes')

    arrow(ax, 19.5, 44, 21.5, 44, ACTIVE_BORDER, lw=1.5)
    arrow(ax, 36.5, 44, 38.5, 44, HIGHLIGHT_BORDER, lw=1.5)
    arrow(ax, 53.5, 44, 55.5, 44, ACTIVE_BORDER, lw=1.5)
    arrow(ax, 72.5, 44, 75.5, 44, FILE_BORDER, lw=1.5)

    # ── Forecasting Module ──
    ax.text(50, 32, 'Forecasting Module', ha='center', fontsize=13, fontweight='bold',
            color=TEXT)

    box(ax, 3, 14, 18, 10, 'L3: 9 Models\nx 7 Horizons', ACTIVE_BORDER, ACTIVE_FILL, 10,
        '= 63 variants\nWalk-forward')

    box(ax, 26, 14, 16, 10, 'generate_\nforecasts.py', FILE_BORDER, FILE_FILL, 9,
        'Backtest +\nForward views')

    box(ax, 47, 14, 16, 10, 'CSV Output', FILE_BORDER, FILE_FILL, 9,
        'bi_dashboard_\nunified.csv\n505+ rows')

    box(ax, 68, 14, 12, 10, '310+ PNGs', FILE_BORDER, FILE_FILL, 9,
        '63 backtest\n+ forward\n+ ensemble')

    box(ax, 83, 14, 14, 10, '/forecasting\nPage', ACTIVE_BORDER, ACTIVE_FILL, 10,
        'Model zoo\nMetrics rank')

    arrow(ax, 21.5, 19, 25.5, 19, ACTIVE_BORDER, lw=1.5)
    arrow(ax, 42.5, 19, 46.5, 19, FILE_BORDER, lw=1.5)
    arrow(ax, 63.5, 19, 67.5, 19, FILE_BORDER, lw=1.5)
    arrow(ax, 80.5, 19, 82.5, 19, ACTIVE_BORDER, lw=1.5)

    # Budget
    ax.text(50, 6, 'LLM Budget: $1/day, $15/month  |  File-based cache (TTL 24h)  |  '
            'W01-W15 generated  |  ~$0.01/week',
            ha='center', fontsize=8, color=TEXT_DIM)

    fig.savefig('/home/globalforex/Documents/USDCOP-RL/USDCOP-RL-Models/docs/slides/slide6_intelligence.png',
                dpi=DPI, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print('  slide6_intelligence.png')


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 7: System Ops (Infrastructure + Watchdog)
# ═══════════════════════════════════════════════════════════════════════════════
def slide7():
    fig, ax = setup_fig(12, 6.7, 'Slide 7 — System Operations & Infrastructure',
                        '25+ Docker Services  |  53 Alert Rules  |  4 Grafana Dashboards  |  Watchdog')

    # ── Watchdog center ──
    cx, cy = 50, 52
    circle = plt.Circle((cx, cy), 8, facecolor=ACTIVE_FILL, edgecolor=ACTIVE_BORDER,
                         linewidth=2.5)
    ax.add_patch(circle)
    ax.text(cx, cy + 1, 'WATCHDOG', ha='center', fontsize=11, fontweight='bold', color=TEXT)
    ax.text(cx, cy - 2, 'Health Hub', ha='center', fontsize=9, color=TEXT_DIM)

    # ── 8 Spokes ──
    spokes = [
        (25, 78, 'OHLCV Freshness', '<3 days', 'Trigger backfill'),
        (75, 78, 'Macro Freshness', '<7 days', 'Trigger backfill'),
        (10, 52, 'Model Freshness', '<10 days', 'Retrain L3'),
        (90, 52, 'News Pipeline', '<24 hours', 'Re-trigger DAG'),
        (25, 26, 'Feature Drift', 'KS p<0.01', 'Alert + retrain'),
        (75, 26, 'Seed Backup', 'Daily 13:00', 'Re-trigger backup'),
        (8, 68, 'DB Health', 'Connections', 'Scale pool'),
        (92, 68, 'Container\nResources', 'CPU/Mem/Disk', 'Alert + scale'),
    ]

    for sx, sy, name, monitor, heal in spokes:
        box(ax, sx - 8, sy - 4, 16, 8, name, EXT_BORDER, EXT_FILL, 8)
        ax.text(sx, sy - 1.5, monitor, ha='center', fontsize=6.5, color=TEXT_DIM)
        ax.text(sx, sy - 3, heal, ha='center', fontsize=6.5, color=ACTIVE_BORDER)

        # Spoke line to center
        dx = cx - sx
        dy = cy - sy
        dist = (dx**2 + dy**2)**0.5
        # Start from edge of box, end at edge of circle
        if dist > 0:
            nx, ny = dx / dist, dy / dist
            arrow(ax, sx + nx * 9, sy + ny * 5, cx - nx * 8.5, cy - ny * 8.5,
                  '#475569', lw=1.2)

    # ── Infrastructure ring (bottom) ──
    ax.add_patch(FancyBboxPatch((2, 4), 96, 12, boxstyle="round,pad=0.3",
                                facecolor='#0a1628', edgecolor='#334155', linewidth=1.2))
    ax.text(50, 14.5, 'Infrastructure Services', ha='center', fontsize=10,
            fontweight='bold', color=TEXT)

    services = [
        ('PostgreSQL\n:5432', ACTIVE_BORDER),
        ('Redis\n:6379', ACTIVE_BORDER),
        ('Airflow\n:8080', ACTIVE_BORDER),
        ('Dashboard\n:5000', ACTIVE_BORDER),
        ('SignalBridge\n:8085', ACTIVE_BORDER),
        ('MLflow\n:5001', ACTIVE_BORDER),
        ('Prometheus\n:9090', ACTIVE_BORDER),
        ('Grafana\n:3002', ACTIVE_BORDER),
        ('Loki\n:3100', ACTIVE_BORDER),
        ('AlertMgr\n:9093', ACTIVE_BORDER),
        ('MinIO\n:9001', ACTIVE_BORDER),
    ]

    sw = 7.8
    total_w = len(services) * sw
    start_x = (100 - total_w) / 2
    for i, (svc, clr) in enumerate(services):
        sx = start_x + i * sw
        ax.add_patch(FancyBboxPatch((sx, 5.5), sw - 0.8, 7, boxstyle="round,pad=0.1",
                                    facecolor='#0f2027', edgecolor=clr, linewidth=1))
        ax.text(sx + (sw - 0.8)/2, 9, svc, ha='center', fontsize=5.5, color=TEXT,
                fontweight='bold')

    fig.savefig('/home/globalforex/Documents/USDCOP-RL/USDCOP-RL-Models/docs/slides/slide7_system_ops.png',
                dpi=DPI, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print('  slide7_system_ops.png')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating USDCOP Architecture Slides...')
    slide0()
    slide1()
    slide2()
    slide3()
    slide4()
    slide5()
    slide6()
    slide7()
    print('\nAll 8 slides generated successfully.')
