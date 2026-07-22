"""
H-ENTRY-01 (D1.1) — Perfil intradia DESCRIPTIVO de USD/COP (M5) por bloque de 30 min.
=====================================================================================

DESCRIPTIVO, no comparativo: este script NO evalua ninguna regla de entrada.
Solo caracteriza el costo/friccion intradia de la sesion COP (8:00-12:55 COT)
para fundamentar el prior economico del pre-registro H-ENTRY-01.

LINEA ROJA (quant-constitution §1): corre SOLO sobre anos de DISENO 2020-2024.
Un assert duro rechaza cualquier intento de tocar 2025+ (el OOS/forward queda
virgen para la comparacion pre-registrada, que corre el operador tras sellar
el pre-registro).

Datos: usdcop_m5_ohlcv (symbol='USD/COP'), TIMESTAMPTZ en UTC -> convertido a
America/Bogota (data-governance golden rule). Sesion 8:00-12:55 COT, Lun-Vie.

ADVERTENCIA DE CALIDAD DE DATOS (verificada 2026-07-21 contra la DB):
  2020-2022: el 100% de las barras M5 son "flat" (high=low=open=close) — son
  snapshots de precio, no barras OHLC reales. 2023: ~90% flat; 2024: ~84% flat.
  => (high-low)/close por barra es DEGENERADO como proxy de spread pre-2023.
  Por eso el proxy primario de costo aqui es el RANGO DE CLOSES INTRA-BLOQUE
  (max(close)-min(close))/close_final por (dia, bloque), que si captura el
  movimiento realizado dentro de la media hora, y la volatilidad de retornos
  5m close-to-close. El (high-low)/close por barra se reporta igualmente,
  con su flat_share al lado, para que nadie lo lea sin el caveat.

Metricas por bloque de 30 min (8:00, 8:30, ..., 12:30; el ultimo cubre
12:30-12:55), para (a) todos los dias y (b) solo lunes (dia de entrada H5):
  - bar_range_bp_mean : mean( 1e4*(high-low)/close ) por barra  [con flat_share]
  - flat_share        : fraccion de barras high==low (caveat del anterior)
  - close_range_bp_*  : 1e4*(max(c)-min(c))/c_last por (dia,bloque) -> mean/median
  - ret5m_vol_bp      : std de log-returns 5m close-to-close (intra-dia) * 1e4
  - abs_ret5m_bp_mean : mean(|log-return 5m|) * 1e4
  - n_bars, n_days

Salida (JSON sin Infinity/NaN via sanitize + tabla legible):
  .claude/evidence/cop_entry_profile/2026-07-22/
    ├── cop_entry_profile.json
    ├── cop_entry_profile.txt
    └── generator_script.py   (copia de este script)

Uso:
  set -a; . ./.env; set +a
  python scripts/analysis/cop_entry_profile.py

@contract H-ENTRY-01 / TAREA D1.1
@date 2026-07-21
"""

import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Design-years hard line (quant-constitution §1)
# ---------------------------------------------------------------------------
DESIGN_START = "2020-01-01"
DESIGN_END_EXCL = "2025-01-01"   # exclusive — 2025+ is OOS/forward, NEVER touched here

EVIDENCE_DIR = PROJECT_ROOT / ".claude" / "evidence" / "cop_entry_profile" / "2026-07-22"

SESSION_START = "08:00"
SESSION_END = "12:55"            # inclusive last bar
BLOCKS = ["08:00", "08:30", "09:00", "09:30", "10:00",
          "10:30", "11:00", "11:30", "12:00", "12:30"]

# Ventanas de referencia del pre-registro (solo etiquetas descriptivas aqui)
FIRST_BLOCK = "08:00"                    # primera media hora
TWAP_BLOCKS = ["09:30", "10:00"]         # ventana TWAP candidata 9:30-10:30


def _sanitize(obj):
    """JSON safety (strategy-contract): Infinity/NaN -> None, numpy -> python."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def load_m5_design_years() -> pd.DataFrame:
    import psycopg2
    conn = psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
    )
    # Bounds anchored in COT instants; conversion to America/Bogota in-query.
    q = f"""
        SELECT (time AT TIME ZONE 'America/Bogota') AS ts_cot,
               open::float8, high::float8, low::float8, close::float8
        FROM usdcop_m5_ohlcv
        WHERE symbol = 'USD/COP'
          AND time >= '{DESIGN_START} 00:00:00-05'
          AND time <  '{DESIGN_END_EXCL} 00:00:00-05'
        ORDER BY time
    """
    df = pd.read_sql(q, conn)
    conn.close()

    df["ts_cot"] = pd.to_datetime(df["ts_cot"])
    df["date"] = df["ts_cot"].dt.date
    df["year"] = df["ts_cot"].dt.year
    df["dow"] = df["ts_cot"].dt.dayofweek
    df["hhmm"] = df["ts_cot"].dt.strftime("%H:%M")

    # Session filter 8:00-12:55 COT (bars outside session are a data bug upstream)
    df = df[(df["hhmm"] >= SESSION_START) & (df["hhmm"] <= SESSION_END)].copy()

    # HARD LINE: no bar beyond design years may survive into the profile.
    assert df["year"].max() <= 2024, (
        f"RED LINE VIOLATION: found year {df['year'].max()} > 2024 — "
        "this profile is design-years (2020-2024) ONLY."
    )
    assert df["year"].min() >= 2020, f"unexpected year {df['year'].min()} < 2020"

    # 30-min block label (floor)
    minute_block = (df["ts_cot"].dt.minute // 30) * 30
    df["block"] = (df["ts_cot"].dt.hour.astype(str).str.zfill(2) + ":" +
                   minute_block.astype(str).str.zfill(2))

    # Intra-day 5m log-returns (close-to-close), first bar of each day = NaN
    df = df.sort_values("ts_cot").reset_index(drop=True)
    df["log_close"] = np.log(df["close"])
    df["ret5m"] = df.groupby("date")["log_close"].diff()
    return df


def profile_blocks(df: pd.DataFrame) -> list:
    """Per-30min-block descriptive metrics for the given subset of days."""
    out = []
    for block in BLOCKS:
        b = df[df["block"] == block]
        if b.empty:
            out.append({"block": block, "n_bars": 0})
            continue
        # per-bar range proxy (degenerate pre-2023 — see flat_share)
        bar_range_bp = 1e4 * (b["high"] - b["low"]) / b["close"]
        flat_share = float((b["high"] == b["low"]).mean())
        # per-(day,block) close-range proxy
        g = b.groupby("date")["close"]
        close_range_bp = 1e4 * (g.max() - g.min()) / g.last()
        # 5m return stats (first bar of day excluded by construction)
        rets = b["ret5m"].dropna()
        out.append({
            "block": block,
            "n_bars": int(len(b)),
            "n_days": int(b["date"].nunique()),
            "bar_range_bp_mean": float(bar_range_bp.mean()),
            "flat_share": flat_share,
            "close_range_bp_mean": float(close_range_bp.mean()),
            "close_range_bp_median": float(close_range_bp.median()),
            "ret5m_vol_bp": float(rets.std(ddof=1) * 1e4) if len(rets) > 2 else None,
            "abs_ret5m_bp_mean": float(rets.abs().mean() * 1e4) if len(rets) else None,
        })
    return out


def first_vs_twap(df: pd.DataFrame) -> dict:
    """Headline ratio: first half-hour (8:00-8:30) vs TWAP window (9:30-10:30).

    Equal-granularity comparison: the close-range for the TWAP window is the
    average of its per-30min-block means (NOT the pooled 60-min range, which
    would mechanically inflate the window and understate the ratio).
    """
    def _stats(sub, blocks):
        # per-(day, 30min-block) close range, then mean — same horizon everywhere
        cr = (sub.groupby(["date", "block"])["close"]
                 .agg(["max", "min", "last"]))
        cr_bp = 1e4 * (cr["max"] - cr["min"]) / cr["last"]
        rets = sub["ret5m"].dropna()
        return {
            "close_range_bp_mean_per_30min": float(cr_bp.mean()) if len(cr_bp) else None,
            "ret5m_vol_bp": float(rets.std(ddof=1) * 1e4) if len(rets) > 2 else None,
            "n_days": int(sub["date"].nunique()),
            "blocks": blocks,
        }

    first = _stats(df[df["block"] == FIRST_BLOCK], [FIRST_BLOCK])
    twap = _stats(df[df["block"].isin(TWAP_BLOCKS)], TWAP_BLOCKS)
    ratios = {}
    for k in ("close_range_bp_mean_per_30min", "ret5m_vol_bp"):
        if first.get(k) and twap.get(k):
            ratios[f"{k}_ratio_first_over_twap"] = first[k] / twap[k]
    return {"first_0800_0830": first, "twap_0930_1030": twap, "ratios": ratios}


def fmt_table(rows: list, title: str) -> str:
    hdr = (f"{'block':>6} | {'n_bars':>6} | {'n_days':>6} | {'barRng_bp':>9} | "
           f"{'flat%':>6} | {'closeRng_bp':>11} | {'medRng_bp':>9} | "
           f"{'vol5m_bp':>8} | {'|r5m|_bp':>8}")
    lines = [title, "=" * len(hdr), hdr, "-" * len(hdr)]
    for r in rows:
        if r.get("n_bars", 0) == 0:
            lines.append(f"{r['block']:>6} | {'0':>6} | (no data)")
            continue
        def f(v, nd=2):
            return "n/a" if v is None else f"{v:.{nd}f}"
        lines.append(
            f"{r['block']:>6} | {r['n_bars']:>6} | {r['n_days']:>6} | "
            f"{f(r['bar_range_bp_mean']):>9} | {100*r['flat_share']:>5.1f}% | "
            f"{f(r['close_range_bp_mean']):>11} | {f(r['close_range_bp_median']):>9} | "
            f"{f(r['ret5m_vol_bp']):>8} | {f(r['abs_ret5m_bp_mean']):>8}"
        )
    return "\n".join(lines)


def main() -> int:
    print(f"[H-ENTRY-01 D1.1] Loading M5 USD/COP {DESIGN_START}..{DESIGN_END_EXCL} (excl)")
    df = load_m5_design_years()
    n_flat = int((df["high"] == df["low"]).sum())
    print(f"  bars={len(df)}, days={df['date'].nunique()}, "
          f"flat_bars={n_flat} ({100*n_flat/len(df):.1f}%)")

    mondays = df[df["dow"] == 0]

    result = {
        "hypothesis": "H-ENTRY-01",
        "task": "D1.1 descriptive intraday entry-cost profile (DESIGN YEARS ONLY)",
        "generated_at": pd.Timestamp.now(tz="America/Bogota").isoformat(),
        "design_window": {"start": DESIGN_START, "end_exclusive": DESIGN_END_EXCL},
        "session_cot": f"{SESSION_START}-{SESSION_END}",
        "source": "usdcop_m5_ohlcv (symbol='USD/COP'), TIMESTAMPTZ->America/Bogota",
        "data_quality_warning": (
            "2020-2022 M5 bars are 100% flat (high=low=open=close, price snapshots); "
            "2023 ~90% flat, 2024 ~84% flat. Per-bar (high-low)/close is degenerate as "
            "a spread proxy pre-2023 — primary cost proxies here are the intra-block "
            "close-range and 5m close-to-close return volatility."
        ),
        "totals": {
            "n_bars": int(len(df)),
            "n_days": int(df["date"].nunique()),
            "n_mondays": int(mondays["date"].nunique()),
            "flat_bar_share": float(n_flat / len(df)),
        },
        "profile_all_days": profile_blocks(df),
        "profile_mondays": profile_blocks(mondays),
        "first_vs_twap": {
            "all_days": first_vs_twap(df),
            "mondays": first_vs_twap(mondays),
            "mondays_by_year": {
                str(y): first_vs_twap(mondays[mondays["year"] == y])
                for y in sorted(mondays["year"].unique())
            },
        },
    }

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    json_path = EVIDENCE_DIR / "cop_entry_profile.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(result), f, indent=2, ensure_ascii=False, allow_nan=False)

    tables = [
        "H-ENTRY-01 — Perfil intradia descriptivo USD/COP (M5, DESIGN 2020-2024 ONLY)",
        f"Generado: {result['generated_at']}   Sesion: 8:00-12:55 COT",
        "",
        "CAVEAT: " + result["data_quality_warning"],
        "",
        fmt_table(result["profile_all_days"], "TODOS LOS DIAS (Lun-Vie)"),
        "",
        fmt_table(result["profile_mondays"],
                  f"SOLO LUNES (dia de entrada H5) — {result['totals']['n_mondays']} lunes"),
        "",
        "PRIMERA MEDIA HORA (8:00-8:30) vs VENTANA TWAP (9:30-10:30):",
        json.dumps(_sanitize(result["first_vs_twap"]), indent=2, ensure_ascii=False),
    ]
    txt_path = EVIDENCE_DIR / "cop_entry_profile.txt"
    txt_path.write_text("\n".join(tables), encoding="utf-8")

    shutil.copyfile(__file__, EVIDENCE_DIR / "generator_script.py")

    print(f"  [OK] {json_path}")
    print(f"  [OK] {txt_path}")
    print(f"  [OK] {EVIDENCE_DIR / 'generator_script.py'}")
    print()
    print("\n".join(tables[5:9]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
