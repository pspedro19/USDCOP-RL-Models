"""
H-ENTRY-01 (D1.2) — Comparacion PRE-REGISTRADA de regla de entrada del lunes (USD/COP H5).
==========================================================================================

*** ESTE SCRIPT NO SE EJECUTA HASTA SELLAR EL PRE-REGISTRO H-ENTRY-01. ***
*** CADA EJECUCION REAL = +1 TRIAL en HYPOTHESIS-REGISTRY (quant-constitution §2). ***
La corrida real exige el flag explicito --confirm-trial; sin el, el script se niega.
--dry-run solo valida sintaxis/imports/conexion (LIMIT 10) y NO computa ninguna metrica.

REGLA DE ENTRADA AS-IS DEL MOTOR (documentada leyendo
scripts/pipeline/train_and_export_smart_simple.py, 2026-07-21):

  - `_run_weekly_loop` (lineas ~377-386) y el loop de produccion (~625-634):
        # Entry = Monday close
        monday_row = df[df["date"] == monday_ts]
        if monday_row.empty:            # lunes festivo/sin dato ->
            monday_row = df[df["date"] >= monday_ts].iloc[:1]   # primer dia habil sig.
        entry = float(monday_row["close"].iloc[0])
    => La entrada es el CLOSE DIARIO del lunes (barra diaria = sesion 8:00-12:55 COT,
       o sea el cierre ~12:55 COT), tomado del dataset diario del engine
       (ForecastingDatasetLoader: DB usdcop_daily_ohlcv-first, seed parquet fallback).
  - Si el lunes no existe (festivo), entra al close del primer dia habil siguiente.
  - Costos modelados en `compute_pnl`: maker_fee (0 bps MEXC) + slippage (1 bps por
    lado) sobre esa entrada; NO se modela ningun spread intradia adicional.
  - Inconsistencia cosmetica documentada: el trade exportado lleva
    `timestamp = <lunes> 09:00 COT` (linea ~433) aunque el PRECIO usado es el close
    diario (~12:55 COT). El precio manda; el timestamp es solo presentacion.
  - En vivo, la senal H5-L5 sale Lun 08:15 COT y el executor corre */30 8:00-12:55;
    este script compara la regla del BACKTEST/export (la que produce los numeros de
    aprobacion), no el timing del executor vivo.

REGLA CANDIDATA: TWAP 9:30-10:30 COT = promedio simple de los closes M5 con
timestamp COT en [09:30, 10:25] (12 barras que cubren 9:30-10:30), del mismo lunes.

METRICA (pre-registrada):
  improvement_bp = direction * (entry_engine - entry_twap) / entry_engine * 1e4
  (>0 = la TWAP consigue mejor precio EN LA DIRECCION del trade; direction: LONG=+1,
  SHORT=-1, reproducida con la MISMA cadena del motor: regime gate Hurst +
  compute_momentum_signal sobre datos previos al lunes — Layers 1-2 as-is).
  Limitacion declarada: el circuit breaker (path-dependent en PnL) NO se reproduce
  para seleccionar semanas; se comparan todas las semanas que pasan gate+signal.
  Tambien se reporta la version incondicional (todos los lunes, ambas direcciones
  hipoteticas y |diff| bruto) como diagnostico secundario.

  Por año y pooled: media, IC95 bootstrap con remuestreo por BLOQUE SEMANAL
  (la observacion = 1 semana; 10,000 remuestras, seed=42, percentiles 2.5/97.5).

LINEA ROJA (quant-constitution §1): SOLO años de DISEÑO 2020-2024. Asserts duros
rechazan cualquier año > 2024 tanto en el query como en los datos cargados.
El juez de confirmacion es el forward (>= sello del pre-registro), NUNCA este OOS.

Salida (solo corrida real):
  .claude/evidence/cop_entry_compare/<YYYY-MM-DD>/
    ├── cop_entry_compare.json   (sin Infinity/NaN)
    ├── cop_entry_compare.txt
    └── generator_script.py

Uso:
  set -a; . ./.env; set +a
  python scripts/analysis/cop_entry_compare.py --dry-run          # smoke SIN computo
  python scripts/analysis/cop_entry_compare.py --confirm-trial    # corrida real (+1 trial)

@contract H-ENTRY-01 / TAREA D1.2
@date 2026-07-21
"""

import argparse
import json
import math
import os
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Design-years hard line (quant-constitution §1)
# ---------------------------------------------------------------------------
DESIGN_YEARS = [2020, 2021, 2022, 2023, 2024]
DESIGN_START = "2020-01-01"
DESIGN_END_EXCL = "2025-01-01"   # exclusive — 2025+ is OOS/forward, NEVER touched here

TWAP_HHMM_START = "09:30"
TWAP_HHMM_END = "10:25"          # inclusive bar stamps covering 9:30-10:30 COT
TWAP_MIN_BARS = 6                # else the Monday is skipped (recorded)

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42


def _assert_design_only(years) -> None:
    # raise explícito, NO assert: python -O elimina asserts y la barrera debe
    # sobrevivir a cualquier modo de ejecución (Codex verify D1 issue #2)
    years = [int(y) for y in years]
    if not years or max(years) > 2024 or min(years) < 2020:
        raise SystemExit(
            f"RED LINE VIOLATION: years {sorted(set(years))} outside design window "
            "2020-2024. This script REFUSES to touch 2025+ (OOS/forward). "
            "Confirmation judge is the forward period per the H-ENTRY-01 pre-registration."
        )


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


def _db_conn():
    import psycopg2
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
    )


def load_m5_design_years(limit: int | None = None) -> pd.DataFrame:
    """M5 bars, design years only, converted to America/Bogota (data-governance)."""
    conn = _db_conn()
    lim = f"LIMIT {int(limit)}" if limit else ""
    q = f"""
        SELECT (time AT TIME ZONE 'America/Bogota') AS ts_cot, close::float8
        FROM usdcop_m5_ohlcv
        WHERE symbol = 'USD/COP'
          AND time >= '{DESIGN_START} 00:00:00-05'
          AND time <  '{DESIGN_END_EXCL} 00:00:00-05'
        ORDER BY time
        {lim}
    """
    df = pd.read_sql(q, conn)
    conn.close()
    df["ts_cot"] = pd.to_datetime(df["ts_cot"])
    df["date"] = df["ts_cot"].dt.normalize()
    df["hhmm"] = df["ts_cot"].dt.strftime("%H:%M")
    if not limit:  # dry-run LIMIT 10 already inside bounds; full load re-asserted
        _assert_design_only(df["ts_cot"].dt.year.unique())
    return df


def reproduce_engine_weeks(year: int):
    """
    Reproduce, AS-IS, which Mondays the engine trades and in which direction
    (Layers 1-2 of _run_weekly_loop: regime gate + momentum signal), using the
    engine's own config loader, dataset loader and signal functions.
    Circuit breaker (Layer 3, path-dependent) intentionally NOT reproduced —
    declared limitation in the pre-registration.
    Returns (list of {monday, direction, engine_entry, engine_entry_date}, df_daily).
    """
    from scripts.pipeline.train_and_export_smart_simple import load_config, load_data
    from src.forecasting.regime_gate import classify_regime
    from src.forecasting.momentum_signal import compute_momentum_signal, SignalConfidence

    cfg = load_config()
    df, _feature_cols = load_data()
    # truncar a diseño INMEDIATAMENTE tras la carga: el loader trae la serie
    # completa (2025+ incluido) y ninguna línea posterior debe poder verla
    # (Codex verify D1 issue #1)
    df = df[df["date"] <= pd.Timestamp("2024-12-31")].copy()

    _assert_design_only([year])
    test = df[(df["date"] >= pd.Timestamp(f"{year}-01-01")) &
              (df["date"] <= pd.Timestamp(f"{year}-12-31"))].copy()
    test["dow"] = test["date"].dt.dayofweek
    mondays = sorted(test[test["dow"] == 0]["date"].unique())

    weeks = []
    for monday in mondays:
        monday_ts = pd.Timestamp(monday)
        prior = df[df["date"] < monday_ts]
        daily_rets = prior["return_1d"].dropna().values
        regime = classify_regime(daily_rets.tolist(), cfg["regime_gate"])
        if regime.sizing_factor <= 0:
            continue
        closes = prior["close"].dropna().values
        if len(closes) < 52:
            continue
        signal = compute_momentum_signal(closes, regime.state.value)
        if signal.direction is None or signal.confidence == SignalConfidence.SKIP:
            continue

        # Engine entry AS-IS: Monday DAILY close; holiday fallback = next bday close
        row = df[df["date"] == monday_ts]
        if row.empty:
            m2 = df["date"] >= monday_ts
            if not m2.any():
                continue
            row = df[m2].iloc[:1]
        weeks.append({
            "monday": monday_ts,
            "direction": int(signal.direction),
            "engine_entry": float(row["close"].iloc[0]),
            "engine_entry_date": pd.Timestamp(row["date"].iloc[0]),
        })
    return weeks


def twap_entry(m5: pd.DataFrame, day: pd.Timestamp):
    """TWAP 9:30-10:30 COT = simple mean of M5 closes stamped [09:30, 10:25]."""
    bars = m5[(m5["date"] == day.normalize()) &
              (m5["hhmm"] >= TWAP_HHMM_START) & (m5["hhmm"] <= TWAP_HHMM_END)]
    if len(bars) < TWAP_MIN_BARS:
        return None, int(len(bars))
    return float(bars["close"].mean()), int(len(bars))


def bootstrap_ci(values: np.ndarray, n_boot: int = BOOTSTRAP_N,
                 seed: int = BOOTSTRAP_SEED):
    """Weekly-block bootstrap: the observation unit IS the week (one entry per
    Monday), so resampling weeks i.i.d. == block bootstrap with weekly blocks."""
    if len(values) < 3:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_compare() -> dict:
    _assert_design_only(DESIGN_YEARS)
    m5 = load_m5_design_years()

    rows, skipped = [], []
    for year in DESIGN_YEARS:
        for w in reproduce_engine_weeks(year):
            entry_day = w["engine_entry_date"]
            twap, nbars = twap_entry(m5, entry_day)
            if twap is None:
                skipped.append({"monday": str(w["monday"].date()),
                                "reason": f"twap_window_bars={nbars} < {TWAP_MIN_BARS}"})
                continue
            eng = w["engine_entry"]
            rows.append({
                "year": year,
                "monday": str(w["monday"].date()),
                "entry_date": str(entry_day.date()),
                "direction": w["direction"],
                "engine_entry": eng,
                "twap_entry": twap,
                "improvement_bp": w["direction"] * (eng - twap) / eng * 1e4,
                "abs_diff_bp": abs(eng - twap) / eng * 1e4,
            })

    dfw = pd.DataFrame(rows)

    def _summary(sub: pd.DataFrame) -> dict:
        vals = sub["improvement_bp"].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(vals)
        return {
            "n_weeks": int(len(sub)),
            "improvement_bp_mean": float(vals.mean()) if len(vals) else None,
            "improvement_bp_ci95": [lo, hi],
            "ci95_excludes_zero": (lo is not None and (lo > 0 or hi < 0)),
            "abs_diff_bp_mean": float(sub["abs_diff_bp"].mean()) if len(sub) else None,
            "share_twap_better": float((sub["improvement_bp"] > 0).mean()) if len(sub) else None,
        }

    result = {
        "hypothesis": "H-ENTRY-01",
        "task": "D1.2 pre-registered entry-rule comparison (DESIGN YEARS ONLY)",
        "generated_at": pd.Timestamp.now(tz="America/Bogota").isoformat(),
        "design_window": {"start": DESIGN_START, "end_exclusive": DESIGN_END_EXCL},
        "rule_engine_asis": "Monday DAILY close (~12:55 COT session close); holiday -> next bday close",
        "rule_candidate": f"TWAP mean of M5 closes stamped [{TWAP_HHMM_START},{TWAP_HHMM_END}] COT",
        "metric": "direction * (entry_engine - entry_twap) / entry_engine * 1e4  (>0 = TWAP better)",
        "bootstrap": {"kind": "weekly-block (obs=week)", "n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED},
        "limitations": [
            "circuit breaker (path-dependent) not reproduced for week selection",
            "M5 2020-2022 bars are price snapshots (flat OHLC); closes are valid, ranges are not",
        ],
        "by_year": {str(y): _summary(dfw[dfw["year"] == y]) for y in DESIGN_YEARS},
        "pooled": _summary(dfw),
        "skipped_weeks": skipped,
        "weeks": rows,
        "trial_accounting": "THIS RUN = +1 trial for H-ENTRY-01 in "
                            ".claude/specs/assets/usdcop/HYPOTHESIS-REGISTRY.md — log it.",
    }
    return result


def dry_run() -> int:
    """Smoke: syntax/imports/DB connectivity with LIMIT 10. NO metrics computed."""
    print("[H-ENTRY-01 D1.2] DRY-RUN — validation only, NO comparison is computed.")
    _assert_design_only(DESIGN_YEARS)
    # imports of the engine chain must resolve
    from scripts.pipeline.train_and_export_smart_simple import load_config  # noqa: F401
    from src.forecasting.regime_gate import classify_regime                # noqa: F401
    from src.forecasting.momentum_signal import compute_momentum_signal    # noqa: F401
    cfg = load_config()
    assert "regime_gate" in cfg, "engine config missing regime_gate"
    df = load_m5_design_years(limit=10)
    assert len(df) == 10 and df["close"].notna().all(), "DB smoke failed"
    print(f"  [OK] engine config loaded (regime_gate present)")
    print(f"  [OK] DB reachable, LIMIT 10 rows in-bounds "
          f"({df['ts_cot'].min()} .. {df['ts_cot'].max()} COT)")
    print("  [OK] dry-run passed. Real run requires --confirm-trial "
          "(and counts +1 trial in the H-ENTRY-01 registry).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--dry-run", action="store_true",
                    help="validate syntax/imports/DB (LIMIT 10) without computing anything")
    ap.add_argument("--confirm-trial", action="store_true",
                    help="REQUIRED for a real run: acknowledges +1 trial in HYPOTHESIS-REGISTRY")
    args = ap.parse_args()

    if args.dry_run:
        return dry_run()

    if not args.confirm_trial:
        print("REFUSED: real run requires --confirm-trial.")
        print("Reason: the H-ENTRY-01 pre-registration must be sealed FIRST, and every")
        print("real execution counts as +1 trial (quant-constitution §2). Use --dry-run")
        print("for a computation-free smoke test.")
        return 2

    result = run_compare()

    out_dir = (PROJECT_ROOT / ".claude" / "evidence" / "cop_entry_compare" /
               date.today().isoformat())
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cop_entry_compare.json", "w", encoding="utf-8") as f:
        json.dump(_sanitize(result), f, indent=2, ensure_ascii=False, allow_nan=False)

    lines = [
        "H-ENTRY-01 — Engine Monday-close vs TWAP 9:30-10:30 (DESIGN 2020-2024 ONLY)",
        f"Generado: {result['generated_at']}",
        "",
        f"{'year':>6} | {'n':>4} | {'mean_bp':>8} | {'ci95_lo':>8} | {'ci95_hi':>8} | "
        f"{'excl0':>5} | {'|d|_bp':>7} | {'%twap+':>6}",
        "-" * 72,
    ]
    for y in [*map(str, DESIGN_YEARS), "pooled"]:
        s = result["pooled"] if y == "pooled" else result["by_year"][y]
        lo, hi = s["improvement_bp_ci95"]
        def f(v): return "n/a" if v is None else f"{v:.2f}"
        lines.append(f"{y:>6} | {s['n_weeks']:>4} | {f(s['improvement_bp_mean']):>8} | "
                     f"{f(lo):>8} | {f(hi):>8} | {str(s['ci95_excludes_zero']):>5} | "
                     f"{f(s['abs_diff_bp_mean']):>7} | {f(s['share_twap_better']):>6}")
    lines += ["", result["trial_accounting"]]
    (out_dir / "cop_entry_compare.txt").write_text("\n".join(lines), encoding="utf-8")
    shutil.copyfile(__file__, out_dir / "generator_script.py")

    print("\n".join(lines))
    print(f"\n[OK] evidence -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
