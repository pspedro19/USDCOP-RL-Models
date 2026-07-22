"""TAREA A2 — Gobernador de leverage fraccional-Kelly a nivel LIBRO (0 trials).

No prueba ninguna hipotesis: computa el f* de Kelly sobre la UNICA serie honesta
(v11 2025 purgado +7.35% / 32 trades + 2026 YTD +3.36% / 11 trades) para sellar
ex-ante una regla de apalancamiento del libro ANTES de que el forward gradue.
Los numeros pre-reparacion (+26.05 / +13.05) estan invalidados y este script
FALLA si el bundle no coincide con la serie purgada.

Salidas:
  .claude/evidence/book_kelly/2026-07-22/book_kelly.json
  .claude/evidence/book_kelly/2026-07-22/generator_script.py  (copia de este archivo)

Metodologia:
  - f* continuo = mu / sigma^2 (aproximacion log-normal / continua).
  - f* discreto = argmax_f E[log(1 + f*r)] sobre la distribucion empirica.
  - IC95 de ambos por bootstrap iid Y por block bootstrap circular b=4
    (el estandar de la casa: HYPOTHESIS-REGISTRY, juez v12).
  - Shrinkage: f* con media observada / 2, justificado por el error estandar
    del Sharpe con N~43 semanas (Lo 2002; Mertens 2002 con skew/kurtosis).
  - Simulacion de maxDD forward (52 semanas, block bootstrap) a multiplicadores
    de libro 0.5x/1.0x/1.5x/2.0x para derivar los escalones de des-apalancamiento.

Reglas respetadas: N=43 >= 20 para la serie combinada (se permiten ratios);
para 2026 solo (N=11 < 20) se reporta UNICAMENTE conteo y PnL.
JSON sin Infinity/NaN (sanitizado a null).
"""

from __future__ import annotations

import json
import math
import shutil
from datetime import date, timedelta
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

TRADES_2025 = (
    REPO
    / "usdcop-trading-dashboard/public/data/production/trades/smart_simple_v11_2025.json"
)
MONITOR_2026 = REPO / ".claude/evidence/cop_monitor_2025_2026/2026-07-21/monitor.json"
EVIDENCE_DIR = REPO / ".claude/evidence/book_kelly/2026-07-22"

SEED = 42
N_BOOT = 10_000
BLOCK = 4  # block bootstrap circular b=4 — mismo prior que el juez sellado de v12

# Guardas de datos honestos (falla si alguien apunta esto a un bundle viejo)
EXPECTED_2025 = {"total_trades": 32, "total_return_pct": 7.35}
EXPECTED_2026 = {"n_trades": 11, "ret_pct": 3.36}

INVALIDATED_RETURNS = {26.05, 13.05}  # pre-reparacion / pre-purga: PROHIBIDOS


# ----------------------------------------------------------------------------- data
def load_honest_series() -> dict:
    with open(TRADES_2025, encoding="utf-8") as f:
        b25 = json.load(f)
    s = b25["summary"]
    if round(s["total_return_pct"], 2) in INVALIDATED_RETURNS:
        raise ValueError(
            "Bundle 2025 con numeros PRE-REPARACION (invalidados). Abortando."
        )
    if (
        s["total_trades"] != EXPECTED_2025["total_trades"]
        or round(s["total_return_pct"], 2) != EXPECTED_2025["total_return_pct"]
    ):
        raise ValueError(
            f"Bundle 2025 no coincide con la serie purgada esperada "
            f"({EXPECTED_2025}); encontrado trades={s['total_trades']}, "
            f"ret={s['total_return_pct']}"
        )
    r25 = [t["pnl_pct"] / 100.0 for t in b25["trades"]]

    with open(MONITOR_2026, encoding="utf-8") as f:
        mon = json.load(f)
    y26 = mon["results"]["v11_produccion"]["2026"]
    if (
        y26["n_trades"] != EXPECTED_2026["n_trades"]
        or round(y26["ret_pct"], 2) != EXPECTED_2026["ret_pct"]
    ):
        raise ValueError(
            f"Serie 2026 no coincide con lo esperado ({EXPECTED_2026}); "
            f"encontrado n={y26['n_trades']}, ret={y26['ret_pct']}"
        )
    r26 = [t["pnl_pct"] / 100.0 for t in y26["trades"]]

    # verificacion de composicion (tolerancia por redondeo del pnl_pct publicado)
    comp25 = (np.prod([1 + r for r in r25]) - 1) * 100
    comp26 = (np.prod([1 + r for r in r26]) - 1) * 100
    assert abs(comp25 - 7.35) < 0.15, f"composicion 2025 = {comp25:.2f} != 7.35"
    assert abs(comp26 - 3.36) < 0.15, f"composicion 2026 = {comp26:.2f} != 3.36"

    return {
        "r25": np.array(r25),
        "r26": np.array(r26),
        "combined": np.array(r25 + r26),
        "comp25_pct": float(comp25),
        "comp26_pct": float(comp26),
    }


def count_mondays(start: date, end: date) -> int:
    """Semanas calendario en las que la estrategia estuvo viva (lunes)."""
    d = start
    while d.weekday() != 0:
        d += timedelta(days=1)
    n = 0
    while d <= end:
        n += 1
        d += timedelta(days=7)
    return n


# ----------------------------------------------------------------------- estimadores
def kelly_continuous(r: np.ndarray) -> float:
    mu = r.mean()
    var = r.var(ddof=1)
    return float(mu / var) if var > 0 else 0.0


def kelly_discrete(r: np.ndarray) -> float:
    """argmax_f E[log(1+f r)] sobre la empirica, f en [0, 0.99/|peor perdida|)."""
    worst = r.min()
    if worst >= 0:  # sin perdidas: Kelly no acotado -> se reporta el borde del grid
        f_max = 30.0
    else:
        f_max = 0.99 / abs(worst)
    grid = np.linspace(0.0, min(f_max, 30.0), 3001)
    growth = np.log1p(np.outer(grid, r)).mean(axis=1)
    return float(grid[int(np.argmax(growth))])


def iid_bootstrap(r: np.ndarray, stat, rng: np.random.Generator) -> np.ndarray:
    n = len(r)
    return np.array([stat(r[rng.integers(0, n, n)]) for _ in range(N_BOOT)])


def circular_block_bootstrap_sample(
    r: np.ndarray, b: int, rng: np.random.Generator
) -> np.ndarray:
    n = len(r)
    n_blocks = math.ceil(n / b)
    starts = rng.integers(0, n, n_blocks)
    idx = (starts[:, None] + np.arange(b)[None, :]) % n
    return r[idx.ravel()[:n]]


def block_bootstrap(r: np.ndarray, stat, rng: np.random.Generator) -> np.ndarray:
    return np.array(
        [stat(circular_block_bootstrap_sample(r, BLOCK, rng)) for _ in range(N_BOOT)]
    )


def ci95(v: np.ndarray) -> list[float]:
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def lo_mertens_se(r: np.ndarray) -> dict:
    """SE del Sharpe semanal: Lo (2002) iid y Mertens (2002) con skew/kurtosis."""
    n = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    sr = mu / sd
    m3 = ((r - mu) ** 3).mean() / sd**3  # skew
    m4 = ((r - mu) ** 4).mean() / sd**4  # kurtosis (no exceso)
    se_lo = math.sqrt((1 + 0.5 * sr**2) / n)
    se_mertens = math.sqrt((1 - m3 * sr + (m4 - 1) / 4 * sr**2) / n)
    return {
        "sharpe_weekly": float(sr),
        "n": n,
        "skew": float(m3),
        "kurtosis": float(m4),
        "se_lo_2002": float(se_lo),
        "se_mertens_2002": float(se_mertens),
        "sr_half_in_se_units_mertens": float((sr - sr / 2) / se_mertens),
    }


def maxdd_of_path(r: np.ndarray) -> float:
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min())


def dd_forward_sim(r: np.ndarray, mult: float, rng: np.random.Generator) -> dict:
    """MaxDD de 52 semanas forward a multiplicador de libro `mult` (block b=4).

    Semanas flat NO estan en la serie (los flat son 0 y no cambian el DD compuesto
    mas alla de estirar el calendario); 52 trades-semana es el escenario DENSO
    (conservador: mas apuestas por año que las ~28/año observadas).
    """
    dds = np.empty(N_BOOT)
    for i in range(N_BOOT):
        path = circular_block_bootstrap_sample(
            np.tile(r, 2)[: max(52, len(r))], BLOCK, rng
        )[:52]
        dds[i] = maxdd_of_path(mult * path)
    return {
        "book_multiplier": mult,
        "maxdd_p50_pct": float(np.percentile(dds, 50) * 100),
        "maxdd_p75_pct": float(np.percentile(dds, 25) * 100),  # mas profundo = peor
        "maxdd_p95_pct": float(np.percentile(dds, 5) * 100),
        "prob_dd_worse_than_7pct": float((dds < -0.07).mean()),
        "prob_dd_worse_than_10pct": float((dds < -0.10).mean()),
        "prob_dd_worse_than_12pct": float((dds < -0.12).mean()),
        "prob_dd_worse_than_15pct": float((dds < -0.15).mean()),
    }


# ------------------------------------------------------------------------ sanitizado
def sanitize(obj):
    """JSON safety: Infinity/NaN -> null (regla strategy-contract)."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.floating, np.integer)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


# ------------------------------------------------------------------------------ main
def main() -> None:
    rng = np.random.default_rng(SEED)
    data = load_honest_series()
    r = data["combined"]
    n = len(r)

    mu, sd = float(r.mean()), float(r.std(ddof=1))
    f_cont = kelly_continuous(r)
    f_disc = kelly_discrete(r)

    # IC95 de f* (continuo y discreto), iid y block b=4
    boot_cont_iid = iid_bootstrap(r, kelly_continuous, rng)
    boot_cont_blk = block_bootstrap(r, kelly_continuous, rng)
    boot_disc_blk = block_bootstrap(r, kelly_discrete, rng)

    # Haircut de POLITICA x1/2 (no estimador formal): motivado por Lo/Mertens
    # (media a la mitad = 0.46 SE, indistinguible), decision conservadora declarada
    lm = lo_mertens_se(r)
    f_cont_half = 0.5 * mu / r.var(ddof=1)
    r_half = r - mu / 2.0  # serie con la mitad de la media, misma dispersion
    f_disc_half = kelly_discrete(r_half)

    # Variante flat-as-zero (invariancia aproximada del f*)
    weeks_live = count_mondays(date(2025, 1, 6), date(2026, 7, 13))
    zeros = np.zeros(weeks_live - n)
    r_full = np.concatenate([r, zeros])
    f_cont_full = kelly_continuous(r_full)

    # Escalones: distribucion de maxDD forward por multiplicador de libro
    dd_sims = [dd_forward_sim(r, m, rng) for m in (0.5, 1.0, 1.5, 2.0)]

    frac_band = {
        "kelly_0.25x_observado": 0.25 * f_cont,
        "kelly_0.50x_observado": 0.50 * f_cont,
        "kelly_0.25x_shrunk_half": 0.25 * f_cont_half,
        "kelly_0.50x_shrunk_half": 0.50 * f_cont_half,
    }

    result = {
        "task": "A2 — gobernador de leverage fraccional-Kelly a nivel LIBRO",
        "trials_consumed": 0,
        "trials_note": (
            "0 trials: no se evalua ninguna hipotesis ni se selecciona ninguna "
            "variante; se computa un guard-rail sobre la serie ya pagada para "
            "sellarlo ANTES del veredicto forward (quant-constitution §1)."
        ),
        "inputs": {
            "trades_2025": str(TRADES_2025.relative_to(REPO)),
            "trades_2026": str(MONITOR_2026.relative_to(REPO)) + " (v11_produccion.2026)",
            "verified": {
                "2025": {"n_trades": 32, "compounded_pct": data["comp25_pct"]},
                "2026_ytd": {
                    "n_trades": 11,
                    "compounded_pct": data["comp26_pct"],
                    "nota": "N=11 < 20: solo conteo y PnL (constitucion §6)",
                },
            },
            "invalidated_forbidden": "+26.05% / +13.05% (pre-reparacion/pre-purga)",
        },
        "series": {
            "definition": (
                "retorno semanal del libro por trade-semana (pnl_pct/100, ya neto "
                "de costos y del leverage INTERNO 0.5-2.0x de la estrategia). "
                "f* es el MULTIPLICADOR DE LIBRO sobre la config congelada, "
                "no leverage absoluto de exchange."
            ),
            "n_trade_weeks": n,
            "weekly_returns": [float(x) for x in r],
            "mu_weekly": mu,
            "sigma_weekly": sd,
            "worst_week": float(r.min()),
            "weeks_live_calendar": weeks_live,
        },
        "kelly": {
            "f_star_continuous_mu_over_var": f_cont,
            "f_star_discrete_empirical": f_disc,
            "f_star_continuous_flat_as_zero_variant": f_cont_full,
            "flat_variant_note": (
                "añadir semanas flat=0 deja f* casi invariante "
                "(mu y E[r^2] escalan igual); se reporta por transparencia."
            ),
            "ci95_f_cont_bootstrap_iid": ci95(boot_cont_iid),
            "ci95_f_cont_block_b4": ci95(boot_cont_blk),
            "ci95_f_disc_block_b4": ci95(boot_disc_blk),
            "prob_f_cont_leq_0_block_b4": float((boot_cont_blk <= 0).mean()),
        },
        "shrinkage_lo_mertens": {
            **lm,
            "reading": (
                "con N=43 semanas el SE del Sharpe semanal es del orden del "
                "Sharpe mismo: una media real = mitad de la observada esta a "
                "menos de 1 SE — indistinguible. El sizing usa el f* SHRUNK."
            ),
            "f_star_cont_half_mean": f_cont_half,
            "f_star_disc_half_mean": f_disc_half,
        },
        "recommended_fraction_band": frac_band,
        "dd_forward_simulation_52w": dd_sims,
        "governor_recommendation": {
            "pre_graduation": 1.0,
            "post_graduation_rule": "min(1.5, 0.25 x f*_shrunk computado con datos FORWARD)",
            "rationale": (
                "el IC95 block-b4 del f* INCLUYE CERO — esta serie ni siquiera "
                "prueba que el Kelly optimo sea positivo; ademas el tramo 2025 "
                "esta contaminado por seleccion (DSR 0.72<0.95). Por eso "
                "pre-graduacion = 1.0x sin excepciones. Post-graduacion la banda "
                "0.25x Kelly-shrunk (~1.2 con estos datos) queda por debajo del "
                "cap 1.5x y la 0.25x sin shrink (~2.3) por encima: el cap 1.5x "
                "es el techo duro y la fraccion se recomputa SOLO con trades "
                "forward, nunca con el backtest."
            ),
        },
    }

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE_DIR / "book_kelly.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(sanitize(result), f, indent=2, ensure_ascii=False, allow_nan=False)
    shutil.copy2(__file__, EVIDENCE_DIR / "generator_script.py")

    print(f"OK -> {out}")
    print(f"N = {n} trade-semanas | mu = {mu*100:.3f}%/sem | sigma = {sd*100:.3f}%/sem")
    print(f"f* continuo = {f_cont:.2f} | f* discreto = {f_disc:.2f}")
    print(f"IC95 f* (block b=4) = {ci95(boot_cont_blk)}")
    print(f"f* shrunk (media/2) = {f_cont_half:.2f} | discreto = {f_disc_half:.2f}")
    print(f"Banda 0.25-0.5 Kelly shrunk = "
          f"[{0.25*f_cont_half:.2f}, {0.50*f_cont_half:.2f}]")
    for s in dd_sims:
        print(
            f"  mult {s['book_multiplier']:.1f}x: maxDD p50 {s['maxdd_p50_pct']:.1f}% "
            f"p95 {s['maxdd_p95_pct']:.1f}% | P(DD>12%) "
            f"{s['prob_dd_worse_than_12pct']:.3f}"
        )


if __name__ == "__main__":
    main()
