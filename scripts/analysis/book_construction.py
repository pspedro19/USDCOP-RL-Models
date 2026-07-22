"""TAREA A1 — Construccion del LIBRO multi-activo por Equal Risk Contribution (ERC).

0 trials: este script SOLO usa covarianzas (riesgo). JAMAS usa retornos esperados
(medias) en la optimizacion. Los pesos son una decision de riesgo, no una prediccion:
no hay claim de edge; el juez es el forward.

Sleeves (campeonas por activo, ver .claude/specs/assets/*/PLAN-RENTABILIDAD-2026-07.md):
  - cop_smart_simple_v12  : USD/COP Smart Simple v12 cap 1.5 — trades semanales (pnl_pct)
                            desde .claude/evidence/cop_monitor_2025_2026/2026-07-21/monitor.json
  - xau_gold_trend_simple : XAU/USD gold_trend_simple — daily_equity del bundle publicado
  - btc_hodl_b1           : BTC/USDT btc_hodl_b1 (HODL vol-targeted) — daily_equity del bundle

Series semanales 2025 (ISO year 2025, W01..W52):
  - COP: pnl_pct del trade asignado a la semana ISO de entrada (una operacion semanal
    Lun->Vie; la semana del trade ES la semana del PnL). Semanas sin trade = 0
    (gate bloqueado => flat, ese ES el P&L real del sleeve).
  - XAU/BTC: retorno semanal desde la curva daily_equity publicada en el bundle
    (ultimo equity de cada semana ISO vs. el de la semana anterior). Se usa la curva
    de equity y no el pnl_pct por-trade porque esos sleeves mantienen posiciones
    multi-semana: asignar todo el pnl a la semana de salida distorsionaria la covarianza.

Metodo:
  1. Alinear las 3 series por semana ISO (52 semanas de 2025).
  2. Covarianza Ledoit-Wolf (sklearn) — >=3 sleeves. (LW centra internamente para
     ESTIMAR la covarianza; la media no se usa en ninguna parte de la asignacion.)
  3. Pesos ERC via iteracion multiplicativa estandar (fixed-point sobre las
     contribuciones de riesgo).
  4. Escalado del libro a vol objetivo 10% anualizada (52 semanas).

Prohibido en este script: medias de retorno en la optimizacion, Sharpe, p-values,
ranking entre activos. Ningun JSON con Infinity/NaN (se convierte a null).

Uso:
    python scripts/analysis/book_construction.py
    (rutas y fecha de evidencia parametrizables via --help)
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import date, datetime
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- Fuentes (bundles publicados / evidencia congelada) -------------------------
COP_MONITOR_JSON = (
    REPO_ROOT / ".claude/evidence/cop_monitor_2025_2026/2026-07-21/monitor.json"
)
STRATEGIES_DIR = (
    REPO_ROOT / "usdcop-trading-dashboard/public/data/strategies"
)

SLEEVES = ["cop_smart_simple_v12", "xau_gold_trend_simple", "btc_hodl_b1"]

TARGET_ANNUAL_VOL = 0.10
WEEKS_PER_YEAR = 52
ISO_YEAR = 2025

DECLARATION = (
    "Pesos = riesgo, no retorno. Construccion 0-trials: solo covarianza "
    "(Ledoit-Wolf) sobre series semanales 2025; ninguna media de retorno entra "
    "en la optimizacion ni se reporta. Sin claims de edge; el juez es el forward."
)


# ---------------------------------------------------------------------------
# JSON safety (regla strategy-contract: nunca Infinity/NaN/undefined)
# ---------------------------------------------------------------------------

def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def safe_json_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_sanitize(obj), fh, indent=2, ensure_ascii=False, allow_nan=False)


# ---------------------------------------------------------------------------
# Series semanales
# ---------------------------------------------------------------------------

def iso_weeks_of_year(year: int) -> list[str]:
    """Semanas ISO del anio ISO `year` como 'YYYY-Wnn' (52 para 2025)."""
    last_week = date(year, 12, 28).isocalendar()[1]  # 28-dic siempre esta en la ultima semana ISO
    return [f"{year}-W{w:02d}" for w in range(1, last_week + 1)]


def _iso_key(d: date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def cop_weekly_returns(monitor_path: Path) -> tuple[dict[str, float], dict]:
    """Retornos semanales (decimal) de v12_cap15 2025 desde la lista de trades.

    pnl_pct del monitor esta en porciento. Un trade por semana (entrada lunes);
    si hubiera >1 en la misma semana ISO se componen.
    """
    with open(monitor_path, encoding="utf-8") as fh:
        monitor = json.load(fh)
    trades = monitor["results"]["v12_cap15"]["2025"]["trades"]
    weekly: dict[str, float] = {}
    excluded = []
    for t in trades:
        d = datetime.fromisoformat(t["timestamp"]).date()
        key = _iso_key(d)
        if not key.startswith(f"{ISO_YEAR}-"):
            excluded.append({"timestamp": t["timestamp"], "iso_week": key,
                             "pnl_pct": t["pnl_pct"]})
            continue
        r = t["pnl_pct"] / 100.0
        weekly[key] = (1.0 + weekly.get(key, 0.0)) * (1.0 + r) - 1.0
    meta = {
        "source": str(monitor_path.relative_to(REPO_ROOT)),
        "series_method": "trade pnl_pct -> semana ISO de entrada; semanas sin trade = 0",
        "n_trades_2025": len(trades),
        "n_weeks_with_trade": len(weekly),
        "trades_excluded_outside_iso_year": excluded,
    }
    return weekly, meta


def _latest_backtest_version(strategy_id: str) -> Path:
    versions = sorted(
        (STRATEGIES_DIR / strategy_id / "backtests").iterdir(),
        key=lambda p: tuple(int(x) for x in p.name.split(".")),
    )
    return versions[-1]


def equity_weekly_returns(strategy_id: str) -> tuple[dict[str, float], dict]:
    """Retornos semanales (decimal) desde la curva daily_equity del bundle publicado."""
    vdir = _latest_backtest_version(strategy_id)
    sig_path = next(vdir.glob("signals_*.json"))
    with open(sig_path, encoding="utf-8") as fh:
        sig = json.load(fh)
    assert sig["kind"] == "daily_equity", f"{sig_path}: kind={sig['kind']}"
    rows = sorted(sig["rows"], key=lambda r: r["d"])
    # ultimo equity por semana ISO
    week_last: dict[str, float] = {}
    week_order: list[str] = []
    for r in rows:
        d = date.fromisoformat(r["d"])
        key = _iso_key(d)
        if key not in week_last:
            week_order.append(key)
        week_last[key] = float(r["eq"])
    weekly: dict[str, float] = {}
    prev = None
    for key in week_order:
        eq = week_last[key]
        if prev is not None and prev > 0 and key.startswith(f"{ISO_YEAR}-"):
            weekly[key] = eq / prev - 1.0
        prev = eq
    meta = {
        "source": str(sig_path.relative_to(REPO_ROOT)),
        "series_method": ("daily_equity -> ultimo equity por semana ISO -> retorno "
                          "vs semana anterior (posiciones multi-semana: la curva de "
                          "equity es la unica serie semanal honesta)"),
        "n_weeks": len(weekly),
    }
    return weekly, meta


# ---------------------------------------------------------------------------
# ERC (equal risk contribution) — iterativo estandar, SOLO covarianza
# ---------------------------------------------------------------------------

def erc_weights(cov: np.ndarray, tol: float = 1e-12, max_iter: int = 100_000) -> np.ndarray:
    """Fixed-point multiplicativo: w_i <- w_i * (RC_target / RC_i)^0.5, normalizado.

    Converge para covarianzas definidas positivas (Ledoit-Wolf lo es).
    No usa retornos esperados: solo `cov`.
    """
    n = cov.shape[0]
    w = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        port_var = float(w @ cov @ w)
        mrc = cov @ w                      # d(var)/dw (proporcional al riesgo marginal)
        rc = w * mrc                       # contribuciones a la varianza
        if np.any(rc <= 0):                # cov no-PSD o peso degenerado: nunca silencioso
            raise SystemExit(f"[FAIL] ERC: contribucion de riesgo <= 0 (rc={rc})")
        target = port_var / n
        w_new = w * np.power(target / rc, 0.5)
        w_new = np.clip(w_new, 1e-12, None)
        w_new /= w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            return w_new
        w = w_new
    raise SystemExit(f"[FAIL] ERC no convergio en {max_iter} iteraciones (tol={tol})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evidence-date", default="2026-07-22",
                        help="Subcarpeta de .claude/evidence/book_construction/")
    parser.add_argument("--config-out", default=str(REPO_ROOT / "config/book/book_v1.yaml"))
    args = parser.parse_args()

    evidence_dir = REPO_ROOT / ".claude/evidence/book_construction" / args.evidence_date
    evidence_dir.mkdir(parents=True, exist_ok=True)

    weeks = iso_weeks_of_year(ISO_YEAR)

    series: dict[str, dict[str, float]] = {}
    sleeve_meta: dict[str, dict] = {}
    series["cop_smart_simple_v12"], sleeve_meta["cop_smart_simple_v12"] = \
        cop_weekly_returns(COP_MONITOR_JSON)
    series["xau_gold_trend_simple"], sleeve_meta["xau_gold_trend_simple"] = \
        equity_weekly_returns("gold_trend_simple")
    series["btc_hodl_b1"], sleeve_meta["btc_hodl_b1"] = \
        equity_weekly_returns("btc_hodl_b1")

    # Matriz semanal alineada; semanas sin trade = 0 SOLO es valido para COP (gate
    # cerrado = flat real). Para XAU/BTC (daily_equity continuo) un hueco seria dato
    # faltante, no flat: exigir cobertura completa de las 52 semanas ISO.
    for s in ("xau_gold_trend_simple", "btc_hodl_b1"):
        missing_wk = [wk for wk in weeks if wk not in series[s]]
        if missing_wk:
            raise SystemExit(f"[FAIL] {s}: {len(missing_wk)} semanas ISO-2025 sin dato "
                             f"de equity (primeras: {missing_wk[:4]}) — hueco, no flat")
    R = np.array([[series[s].get(wk, 0.0) for s in SLEEVES] for wk in weeks])

    # Covarianza Ledoit-Wolf (>=3 sleeves). El centrado interno de LW es parte de la
    # ESTIMACION de covarianza; ninguna media entra en la asignacion.
    lw = LedoitWolf().fit(R)
    cov = lw.covariance_

    w = erc_weights(cov)

    port_var = float(w @ cov @ w)
    port_vol_weekly = math.sqrt(port_var)
    port_vol_annual = port_vol_weekly * math.sqrt(WEEKS_PER_YEAR)

    mrc = cov @ w
    rc = w * mrc                          # contribuciones a la varianza
    rc_pct = rc / rc.sum()

    scale = TARGET_ANNUAL_VOL / port_vol_annual if port_vol_annual > 0 else None
    scaled_weights = (w * scale) if scale is not None else np.full_like(w, np.nan)

    sleeve_vol_annual = np.sqrt(np.diag(cov)) * math.sqrt(WEEKS_PER_YEAR)

    corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))

    result = {
        "task": "A1 — LIBRO multi-activo ERC (0 trials, solo covarianza)",
        "declaration": DECLARATION,
        "computed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "window": {
            "iso_year": ISO_YEAR,
            "weeks": weeks,
            "n_weeks": len(weeks),
            "annualization_weeks": WEEKS_PER_YEAR,
        },
        "sleeves": SLEEVES,
        "sleeve_meta": sleeve_meta,
        "weekly_returns_decimal": {
            s: [series[s].get(wk, 0.0) for wk in weeks] for s in SLEEVES
        },
        "covariance": {
            "estimator": "LedoitWolf (sklearn)",
            "shrinkage": float(lw.shrinkage_),
            "matrix_weekly_decimal2": cov.tolist(),
            "correlation": corr.tolist(),
            "sleeve_vol_annualized": dict(zip(SLEEVES, sleeve_vol_annual.tolist())),
        },
        "erc": {
            "weights": dict(zip(SLEEVES, w.tolist())),
            "risk_contribution_variance": dict(zip(SLEEVES, rc.tolist())),
            "risk_contribution_pct": dict(zip(SLEEVES, rc_pct.tolist())),
            "max_rc_pct_deviation_from_equal": float(np.max(np.abs(rc_pct - 1.0 / len(SLEEVES)))),
        },
        "vol_targeting": {
            "target_annual_vol": TARGET_ANNUAL_VOL,
            "unit_book_annual_vol": port_vol_annual,
            "scale_factor": scale,
            "scaled_weights": dict(zip(SLEEVES, scaled_weights.tolist())),
            "scaled_gross_exposure": float(np.sum(scaled_weights)) if scale is not None else None,
        },
    }

    out_json = evidence_dir / "book_erc_v1.json"
    safe_json_dump(result, out_json)

    # Copia del script como generator_script.py (evidencia reproducible)
    shutil.copyfile(Path(__file__).resolve(), evidence_dir / "generator_script.py")

    # Config del libro (YAML escrito a mano — sin dependencia de ruamel para orden)
    config_path = Path(args.config_out)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_lines = [
        "# book_v1.yaml — LIBRO multi-activo ERC (TAREA A1)",
        "# Generado por scripts/analysis/book_construction.py — NO editar a mano.",
        f"# Nota: {DECLARATION}",
        "",
        "book_id: book_v1",
        "method: equal_risk_contribution",
        "trials_consumed: 0  # solo covarianza; ningun retorno esperado ni grid",
        f"computed_at: '{result['computed_at']}'",
        "window:",
        f"  iso_year: {ISO_YEAR}",
        f"  weeks: {len(weeks)}",
        "  frequency: weekly_iso",
        f"  annualization_weeks: {WEEKS_PER_YEAR}",
        "covariance_estimator: ledoit_wolf",
        f"covariance_shrinkage: {float(lw.shrinkage_):.6f}",
        f"target_annual_vol: {TARGET_ANNUAL_VOL}",
        f"unit_book_annual_vol: {port_vol_annual:.6f}",
        f"vol_scale_factor: {scale:.6f}" if scale is not None else "vol_scale_factor: null",
        "sleeves:",
    ]
    for i, s in enumerate(SLEEVES):
        yaml_lines += [
            f"  - sleeve_id: {s}",
            f"    source: {sleeve_meta[s]['source'].replace(chr(92), '/')}",
            f"    erc_weight: {w[i]:.6f}",
            f"    risk_contribution_pct: {rc_pct[i]:.6f}",
            f"    scaled_weight: {scaled_weights[i]:.6f}",
            f"    annualized_vol_2025: {sleeve_vol_annual[i]:.6f}",
        ]
    yaml_lines += [
        "notes: >-",
        "  Pesos = riesgo, no retorno; juez = forward; sin claims de edge.",
        "  Series semanales 2025 (ISO). COP: pnl_pct por trade (semana de entrada,",
        "  semanas sin trade = 0). XAU/BTC: daily_equity del bundle publicado",
        "  (posiciones multi-semana). No se compara performance entre activos;",
        "  se combinan solo en riesgo.",
        "evidence: .claude/evidence/book_construction/" + args.evidence_date + "/book_erc_v1.json",
        "",
    ]
    config_path.write_text("\n".join(yaml_lines), encoding="utf-8")

    # Resumen a stdout (sin Sharpe, sin ranking)
    print(f"Semanas usadas: {len(weeks)} (ISO {ISO_YEAR})")
    print(f"Shrinkage Ledoit-Wolf: {lw.shrinkage_:.4f}")
    for i, s in enumerate(SLEEVES):
        print(f"  {s:24s} w_ERC={w[i]:.4f}  RC%={rc_pct[i]*100:6.2f}  "
              f"vol_anual={sleeve_vol_annual[i]*100:6.2f}%  w_escalado={scaled_weights[i]:.4f}")
    print(f"Vol anual libro (pesos suma-1): {port_vol_annual*100:.2f}%")
    print(f"Factor de escala a {TARGET_ANNUAL_VOL*100:.0f}%: {scale:.4f}  "
          f"(exposicion bruta escalada = {float(np.sum(scaled_weights)):.4f})")
    print(f"Evidencia: {out_json}")
    print(f"Config:    {config_path}")


if __name__ == "__main__":
    main()
