"""TAREA A3 — Stress Monte Carlo del LIBRO + presupuesto de DD del gobernador (0 trials).

Ingenieria de riesgo pura: NO se evalua ninguna hipotesis de retorno, NO se
selecciona ninguna variante, NO se reporta Sharpe ni retorno esperado. Se
cuantifica la cola de drawdown del LIBRO ERC (book_v1) bajo bootstrap y se
valida si la escalera del gobernador (BOOK-LEVERAGE-GOVERNOR.md §3) acota el
p95 del MaxDD por debajo del presupuesto de 12%.

Insumos:
  .claude/evidence/book_construction/2026-07-22/book_erc_v1.json
    (series semanales 2025 por sleeve + pesos ERC {COP 0.4205, XAU 0.3833, BTC 0.1961})
  .claude/specs/assets/usdcop/BOOK-LEVERAGE-GOVERNOR.md
    (escalones DD 7%/10%/12%, histeresis re-armado 4 semanas a umbral-2pp)

Salidas:
  .claude/evidence/book_mc_stress/2026-07-22/book_mc_stress.json
  .claude/evidence/book_mc_stress/2026-07-22/generator_script.py (copia de este archivo)

Metodologia (seed 42, 10.000 paths, horizonte 52 semanas):
  1. Bootstrap por bloques circular b=4 (el prior de la casa: juez v12, A2) de la
     serie semanal del LIBRO (pesos ERC fijos => rebalanceo semanal implicito a ERC).
     MaxDD p50/p90/p95/p99, ES97.5 del retorno anual, P(MaxDD > 7/10/12%),
     a leverage 1.0x y 1.5x.
  2. Variante iid semanal (dependencia rota) como banda de comparacion; la de
     bloques MANDA.
  3. Regimen hostil (CAUSAL, verificacion Codex #2/#3): la vol que clasifica la
     semana t usa SOLO las semanas t-4..t-1 (estrictamente previas) y el corte
     de tercil es EXPANDING causal (min 8 observaciones). El re-muestreo hostil
     es bootstrap de RUNS contiguos completos (no semanas compactadas: nada de
     vecindades falsas tipo W12-W15 ni wrap). STRESS CONSERVADOR, no pronostico.
  4. Escalera del gobernador simulada dinamicamente sobre LOS MISMOS paths
     (arranque 1.5x): DD>7% => 1.0x, DD>10% => 0.5x, DD>12% => flat (absorbente:
     con equity congelada el DD no puede cerrar bajo umbral-2pp), decision al
     cierre semanal con efecto la semana SIGUIENTE (sin look-ahead), re-armado
     de un escalon tras 4 cierres consecutivos con DD < (umbral del escalon - 2pp).
  5. Cuadro PROPUESTA del presupuesto de DD (el operador firma).

LIMITACION ESTRUCTURAL (citada tambien en el JSON): la serie madre son 52
semanas de 2025. El MC HEREDA el regimen 2025 y NO simula regimenes no vistos
(quiebre de correlaciones entre sleeves, crash conjunto, gap de liquidez).
Ademas el sleeve COP 2025 es un backtest contaminado por seleccion
(DSR 0.72 < 0.95): estos numeros son presupuesto de riesgo, jamas evidencia
de edge. CERO claims de retorno; solo colas.

JSON sin Infinity/NaN (sanitizado a null, regla strategy-contract).

Correcciones tras verificacion Codex (RECHAZADO -> re-medicion, sigue 0 trials):
  #1 MaxDD con high-water inicial NAV=1.0 (un path que arranca perdiendo ya
     esta en drawdown) — corrige todos los numeros estaticos de MaxDD.
  #2 Subset hostil causal: vol de t-4..t-1 (shift) + tercil expanding causal.
  #3 Bootstrap hostil por RUNS contiguos, no semanas compactadas.
  #4 prob_applied_* se registra cuando el multiplicador SE APLICA (t+1), no al
     disparo; prob_trigger_flat se reporta aparte.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

BOOK_JSON = REPO / ".claude/evidence/book_construction/2026-07-22/book_erc_v1.json"
EVIDENCE_DIR = REPO / ".claude/evidence/book_mc_stress/2026-07-22"

SEED = 42
N_PATHS = 10_000
HORIZON = 52
BLOCK = 4  # bootstrap por bloques circular b=4 — mismo prior que el juez v12 / A2

# Guardas: pesos ERC esperados (aborta si el JSON no coincide con book_v1)
EXPECTED_WEIGHTS = {
    "cop_smart_simple_v12": 0.420537,
    "xau_gold_trend_simple": 0.383319,
    "btc_hodl_b1": 0.196145,
}

# Escalera del gobernador (BOOK-LEVERAGE-GOVERNOR.md §3) — steps por indice:
#   0 = base (multiplicador de arranque), 1 = 1.0x, 2 = 0.5x, 3 = flat
LADDER_TRIGGER = (0.07, 0.10, 0.12)  # DD > umbral => step 1 / 2 / 3
REARM_HYST_PP = 0.02                 # re-armado: DD < umbral - 2pp
REARM_WEEKS = 4                      # ... durante 4 cierres consecutivos
LADDER_MULTS_TAIL = (1.0, 0.5, 0.0)  # multiplicadores de steps 1..3

VOL_WINDOW = 4   # vol realizada de las semanas t-4..t-1 (estrictamente previas)
MIN_VOL_OBS = 8  # observaciones minimas antes de clasificar con el tercil expanding


# ----------------------------------------------------------------------------- data
def load_book_series() -> dict:
    with open(BOOK_JSON, encoding="utf-8") as f:
        book = json.load(f)

    weights = book["erc"]["weights"]
    for sleeve, expected in EXPECTED_WEIGHTS.items():
        got = weights[sleeve]
        if abs(got - expected) > 1e-4:
            raise ValueError(
                f"Peso ERC de {sleeve} = {got:.6f} != esperado {expected:.6f}. "
                "El JSON no es book_v1. Abortando."
            )
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("Los pesos ERC no suman 1.0. Abortando.")

    sleeves = book["sleeves"]
    series = np.array(
        [book["weekly_returns_decimal"][s] for s in sleeves]
    )  # (3, 52)
    if series.shape != (3, HORIZON):
        raise ValueError(f"Series con shape {series.shape} != (3, {HORIZON}).")

    w = np.array([weights[s] for s in sleeves])
    r_book = w @ series  # retorno semanal del LIBRO a pesos ERC fijos (unit, 1.0x)

    # sanity vs la vol anualizada del libro publicada en A1 (LW-cov: 0.072670);
    # la vol muestral de la serie difiere un poco del w'Σ_LW w — tolerancia laxa
    ann_vol_sample = float(r_book.std(ddof=1) * math.sqrt(52))
    if abs(ann_vol_sample - book["vol_targeting"]["unit_book_annual_vol"]) > 0.015:
        raise ValueError(
            f"Vol anualizada muestral del libro {ann_vol_sample:.4f} demasiado "
            f"lejos de la publicada {book['vol_targeting']['unit_book_annual_vol']:.4f}."
        )

    return {
        "sleeves": sleeves,
        "weights": {s: float(weights[s]) for s in sleeves},
        "r_book": r_book,
        "ann_vol_sample": ann_vol_sample,
        "book_meta": {
            "book_id": "book_v1",
            "window": "2025-W01..2025-W52 (52 semanas ISO)",
            "source": str(BOOK_JSON.relative_to(REPO)),
        },
    }


# ------------------------------------------------------------------- path generation
def gen_block_paths(
    series: np.ndarray, n_paths: int, horizon: int, b: int, rng: np.random.Generator
) -> np.ndarray:
    """Bootstrap por bloques circular: paths (n_paths, horizon)."""
    n = len(series)
    n_blocks = math.ceil(horizon / b)
    starts = rng.integers(0, n, (n_paths, n_blocks))
    idx = (starts[..., None] + np.arange(b)) % n
    return series[idx.reshape(n_paths, -1)[:, :horizon]]


def gen_iid_paths(
    series: np.ndarray, n_paths: int, horizon: int, rng: np.random.Generator
) -> np.ndarray:
    """Bootstrap iid semanal (dependencia rota) — solo banda de comparacion."""
    n = len(series)
    return series[rng.integers(0, n, (n_paths, horizon))]


# ------------------------------------------------------------------------ static MC
def static_stats(paths: np.ndarray, mult: float) -> dict:
    """MaxDD y cola del retorno anual a multiplicador de libro constante.

    SOLO riesgo: del retorno anual se reporta exclusivamente la cola ES97.5
    (media del peor 2.5%); el centro de la distribucion NO se reporta porque
    el bootstrap hereda la media 2025 y eso seria un claim de retorno.
    """
    eq = np.cumprod(1.0 + mult * paths, axis=1)
    # high-water inicial NAV=1.0 (Codex #1): un path que arranca perdiendo ya
    # esta en drawdown respecto del capital inicial
    peak = np.maximum(np.maximum.accumulate(eq, axis=1), 1.0)
    dd = eq / peak - 1.0
    maxdd_mag = -dd.min(axis=1)  # magnitud (positiva) del peor drawdown del path

    ann = eq[:, -1] - 1.0
    var_cut = np.percentile(ann, 2.5)
    es975 = float(ann[ann <= var_cut].mean())

    p50, p90, p95, p99 = np.percentile(maxdd_mag, [50, 90, 95, 99])
    return {
        "book_multiplier": mult,
        "maxdd_pct": {
            "p50": float(p50 * 100),
            "p90": float(p90 * 100),
            "p95": float(p95 * 100),
            "p99": float(p99 * 100),
        },
        "prob_maxdd_gt_7pct": float((maxdd_mag > 0.07).mean()),
        "prob_maxdd_gt_10pct": float((maxdd_mag > 0.10).mean()),
        "prob_maxdd_gt_12pct": float((maxdd_mag > 0.12).mean()),
        "es97_5_annual_return_pct": es975 * 100,
    }


# --------------------------------------------------------------------- ladder engine
def run_ladder(paths: np.ndarray, start_mult: float) -> dict:
    """Simula la escalera del gobernador sobre paths (n, T), arrancando en start_mult.

    Reglas (BOOK-LEVERAGE-GOVERNOR.md §3, tal cual):
      - DD medido peak-to-trough sobre la equity del LIBRO, al cierre semanal.
      - DD > 7% => 1.0x ; DD > 10% => 0.5x ; DD > 12% => 0 (flat).
      - La decision del cierre t se aplica a la semana t+1 (sin look-ahead).
      - Bajadas: inmediatas (al escalon que dicte el DD). Subidas: UN escalon
        tras REARM_WEEKS cierres consecutivos con DD < (umbral del escalon - 2pp)
        (1.0x->base con DD<5%; 0.5x->1.0x con DD<8%).
      - Flat es absorbente de facto: con equity congelada el DD queda clavado
        >12% y la condicion de re-armado (DD<10%) es inalcanzable — coherente
        con '0 (flat) + revision formal' del protocolo.
    """
    n, T = paths.shape
    mults = np.array([start_mult, *LADDER_MULTS_TAIL])
    # re-armado del step k -> k-1 exige DD < LADDER_TRIGGER[k-1] - 2pp
    rearm_dd = np.array([np.inf, *[t - REARM_HYST_PP for t in LADDER_TRIGGER]])

    eq = np.ones(n)
    peak = np.ones(n)
    step = np.zeros(n, dtype=np.int64)  # multiplicador vigente ESTA semana
    counter = np.zeros(n, dtype=np.int64)
    maxdd = np.zeros(n)
    mult_sum = np.zeros(n)
    # Codex #4: 'applied' se registra cuando el multiplicador SE APLICA (la
    # semana siguiente al disparo); 'trigger' registra el disparo al cierre.
    ever_applied = {1: np.zeros(n, bool), 2: np.zeros(n, bool), 3: np.zeros(n, bool)}
    ever_trigger = {1: np.zeros(n, bool), 2: np.zeros(n, bool), 3: np.zeros(n, bool)}

    for t in range(T):
        for k in (1, 2, 3):
            ever_applied[k] |= step == k  # escalon efectivamente OPERADO esta semana
        m = mults[step]
        mult_sum += m
        eq = eq * (1.0 + m * paths[:, t])
        peak = np.maximum(peak, eq)
        dd = eq / peak - 1.0
        maxdd = np.minimum(maxdd, dd)
        ddm = -dd

        # bajada inmediata (efectiva desde la proxima semana)
        trig = np.select(
            [ddm > LADDER_TRIGGER[2], ddm > LADDER_TRIGGER[1], ddm > LADDER_TRIGGER[0]],
            [3, 2, 1],
            default=0,
        )
        down = trig > step
        step = np.where(down, trig, step)
        counter = np.where(down, 0, counter)
        for k in (1, 2, 3):
            ever_trigger[k] |= step == k

        # histeresis de re-armado (solo un escalon por evento)
        cond = (step > 0) & (ddm < rearm_dd[step])
        counter = np.where(cond, counter + 1, 0)
        up = counter >= REARM_WEEKS
        step = np.where(up, step - 1, step)
        counter = np.where(up, 0, counter)

    maxdd_mag = -maxdd
    ann = eq - 1.0
    var_cut = np.percentile(ann, 2.5)
    p50, p90, p95, p99 = np.percentile(maxdd_mag, [50, 90, 95, 99])
    return {
        "start_multiplier": start_mult,
        "maxdd_pct": {
            "p50": float(p50 * 100),
            "p90": float(p90 * 100),
            "p95": float(p95 * 100),
            "p99": float(p99 * 100),
        },
        "prob_maxdd_gt_7pct": float((maxdd_mag > 0.07).mean()),
        "prob_maxdd_gt_10pct": float((maxdd_mag > 0.10).mean()),
        "prob_maxdd_gt_12pct": float((maxdd_mag > 0.12).mean()),
        "es97_5_annual_return_pct": float(ann[ann <= var_cut].mean() * 100),
        "prob_applied_step_1p0x": float(ever_applied[1].mean()),
        "prob_applied_step_0p5x": float(ever_applied[2].mean()),
        "prob_applied_flat": float(ever_applied[3].mean()),
        "prob_trigger_flat": float(ever_trigger[3].mean()),
        "applied_vs_trigger_note": (
            "applied = el escalon llego a OPERARSE (multiplicador aplicado en "
            "t+1); trigger = el DD lo disparo al cierre t. Un disparo en la "
            "ultima semana del horizonte cuenta como trigger pero no como applied."
        ),
        "avg_multiplier_over_horizon": float((mult_sum / T).mean()),
    }


# --------------------------------------------------------------------- hostile regime
def hostile_subset(r_book: np.ndarray) -> dict:
    """Semanas hostiles CAUSALES + runs contiguos (Codex #2/#3).

    - La vol que clasifica la semana t usa SOLO r[t-4..t-1] (estrictamente
      previas: shift completo, r[t] jamas entra en su propia clasificacion).
    - El corte de tercil (percentil 66.7) es EXPANDING causal: en t se computa
      sobre las vols observadas hasta t inclusive (todas derivadas de retornos
      <= t-1), con un minimo de MIN_VOL_OBS observaciones antes de clasificar.
    - Se devuelven los RUNS contiguos de semanas hostiles, no un subset
      compactado: el bootstrap hostil re-muestrea runs completos.
    """
    n = len(r_book)
    vols = np.full(n, np.nan)
    for t in range(VOL_WINDOW, n):
        vols[t] = r_book[t - VOL_WINDOW : t].std(ddof=1)

    hostile = np.zeros(n, dtype=bool)
    cuts = np.full(n, np.nan)
    seen: list[float] = []
    for t in range(VOL_WINDOW, n):
        seen.append(float(vols[t]))
        if len(seen) >= MIN_VOL_OBS:
            cut = float(np.percentile(seen, 100 * 2 / 3))
            cuts[t] = cut
            hostile[t] = vols[t] >= cut

    runs: list[list[int]] = []
    cur: list[int] = []
    for t in range(n):
        if hostile[t]:
            cur.append(t)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    if not runs:
        raise ValueError("Sin semanas hostiles bajo la definicion causal. Abortando.")

    return {
        "hostile_iso_weeks": [f"2025-W{t + 1:02d}" for t in np.where(hostile)[0]],
        "n_weeks": int(hostile.sum()),
        "vol_window_weeks": VOL_WINDOW,
        "min_vol_obs": MIN_VOL_OBS,
        "runs_iso": [
            [f"2025-W{t + 1:02d}" for t in run] for run in runs
        ],
        "run_lengths": [len(run) for run in runs],
        "run_returns": [r_book[np.array(run)] for run in runs],
    }


def gen_run_paths(
    run_returns: list[np.ndarray], n_paths: int, horizon: int, rng: np.random.Generator
) -> np.ndarray:
    """Bootstrap de RUNS contiguos completos (uniforme con reemplazo).

    Concatena runs muestreados hasta llenar `horizon` semanas y trunca. Preserva
    la contiguidad REAL dentro de cada run; no fabrica vecindades falsas entre
    semanas que no fueron consecutivas en el calendario (Codex #3).
    """
    n_runs = len(run_returns)
    paths = np.empty((n_paths, horizon))
    for i in range(n_paths):
        buf: list[np.ndarray] = []
        total = 0
        while total < horizon:
            r = run_returns[rng.integers(0, n_runs)]
            buf.append(r)
            total += len(r)
        paths[i] = np.concatenate(buf)[:horizon]
    return paths


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
    data = load_book_series()
    r_book = data["r_book"]

    # Un solo RNG seed=42; el orden de generacion es parte de la receta reproducible
    rng = np.random.default_rng(SEED)
    paths_block = gen_block_paths(r_book, N_PATHS, HORIZON, BLOCK, rng)
    paths_iid = gen_iid_paths(r_book, N_PATHS, HORIZON, rng)

    host = hostile_subset(r_book)
    paths_host = gen_run_paths(host["run_returns"], N_PATHS, HORIZON, rng)

    # 1-2. MC estatico: bloques (manda) + iid (banda de comparacion)
    static_block = [static_stats(paths_block, m) for m in (1.0, 1.5)]
    static_iid = [static_stats(paths_iid, m) for m in (1.0, 1.5)]

    # 3. Regimen hostil (tercil alto de vol) — stress conservador, NO pronostico
    static_host = [static_stats(paths_host, m) for m in (1.0, 1.5)]

    # 4. Escalera del gobernador sobre LOS MISMOS paths
    ladder_15 = run_ladder(paths_block, 1.5)   # la pregunta central
    ladder_10 = run_ladder(paths_block, 1.0)   # secundario: pre-graduacion
    ladder_host = run_ladder(paths_host, 1.5)  # escalera bajo stress hostil

    verdict_p95_under_12 = ladder_15["maxdd_pct"]["p95"] < 12.0

    result = {
        "task": "A3 — stress Monte Carlo del LIBRO + presupuesto de DD del gobernador",
        "trials_consumed": 0,
        "trials_note": (
            "0 trials: ingenieria de riesgo pura. No se evalua ninguna hipotesis "
            "de retorno, no se selecciona ninguna variante, no se reporta Sharpe "
            "ni retorno esperado — solo colas de drawdown y ES97.5 (perdida)."
        ),
        "inputs": {
            "book": data["book_meta"],
            "erc_weights": data["weights"],
            "governor_spec": ".claude/specs/assets/usdcop/BOOK-LEVERAGE-GOVERNOR.md",
            "book_weekly_series_stats": {
                "n_weeks": HORIZON,
                "annualized_vol_sample": data["ann_vol_sample"],
                "worst_week_pct": float(r_book.min() * 100),
                "best_week_pct": float(r_book.max() * 100),
                "realized_maxdd_2025_pct_at_1x": float(
                    -(
                        (eq25 := np.concatenate([[1.0], np.cumprod(1 + r_book)]))
                        / np.maximum.accumulate(eq25)
                        - 1
                    ).min()
                    * 100
                ),
            },
        },
        "mc_config": {
            "seed": SEED,
            "n_paths": N_PATHS,
            "horizon_weeks": HORIZON,
            "block_size": BLOCK,
            "bootstrap": (
                "base: circular block (b=4) — MANDA; iid solo banda de "
                "comparacion; hostil: bootstrap de runs contiguos"
            ),
            "maxdd_definition": (
                "peak-to-trough sobre equity semanal con high-water inicial "
                "NAV=1.0 (Codex #1)"
            ),
            "weights_handling": (
                "pesos ERC fijos cada semana = rebalanceo semanal implicito a ERC; "
                "no se simula drift de pesos intra-rebalanceo"
            ),
        },
        "es97_5_disclaimer": (
            "ATENCION: bajo el regimen base el ES97.5 anual sale POSITIVO porque "
            "el bootstrap hereda la deriva 2025 del libro (Gold/BTC alcistas + COP "
            "backtest contaminado por seleccion). Eso es un ARTEFACTO de heredar "
            "2025, no un claim de que 'ni la peor cola pierde'. La cola honesta "
            "para presupuestar perdida es la del stress hostil (ES97.5 negativo)."
        ),
        "static_mc_block_b4": static_block,
        "static_mc_iid_reference_only": static_iid,
        "hostile_regime_stress": {
            "label": (
                "STRESS CONSERVADOR — NO PRONOSTICO: re-muestreo solo de semanas "
                "hostiles del libro (clasificacion CAUSAL) como proxy de regimen "
                "hostil PERSISTENTE 52 semanas (peor que cualquier año realista "
                "del regimen 2025)."
            ),
            "classification": {
                "method": (
                    "semana t es hostil si la vol realizada de r[t-4..t-1] "
                    "(estrictamente previas, shift completo) >= percentil 66.7 "
                    "EXPANDING causal de las vols observadas hasta t, con minimo "
                    f"{MIN_VOL_OBS} observaciones antes de clasificar (Codex #2)"
                ),
                "vol_window_weeks": host["vol_window_weeks"],
                "min_vol_obs": host["min_vol_obs"],
                "hostile_weeks": host["hostile_iso_weeks"],
                "n_hostile_weeks": host["n_weeks"],
            },
            "resampling": {
                "method": (
                    "bootstrap de RUNS contiguos completos (uniforme con "
                    "reemplazo), concatenados hasta llenar 52 semanas y truncado "
                    "— preserva la contiguidad real, sin vecindades falsas "
                    "(Codex #3)"
                ),
                "runs_iso": host["runs_iso"],
                "run_lengths": host["run_lengths"],
            },
            "static_mc": static_host,
            "ladder_from_1p5x": ladder_host,
        },
        "governor_ladder_simulation": {
            "rules_as_simulated": {
                "steps": "DD>7% => 1.0x ; DD>10% => 0.5x ; DD>12% => flat",
                "timing": "decision al cierre semanal, efectiva la semana siguiente (sin look-ahead)",
                "rearm": (
                    f"subida de UN escalon tras {REARM_WEEKS} cierres consecutivos "
                    "con DD < (umbral del escalon - 2pp); flat es absorbente de "
                    "facto (equity congelada => DD clavado > 12%)"
                ),
            },
            "from_1p5x_block_b4": ladder_15,
            "from_1p0x_block_b4": ladder_10,
            "comparison_same_paths": {
                "static_1p5x_maxdd_p95_pct": static_block[1]["maxdd_pct"]["p95"],
                "ladder_from_1p5x_maxdd_p95_pct": ladder_15["maxdd_pct"]["p95"],
                "static_1p5x_prob_dd_gt_12pct": static_block[1]["prob_maxdd_gt_12pct"],
                "ladder_from_1p5x_prob_dd_gt_12pct": ladder_15["prob_maxdd_gt_12pct"],
            },
            "verdict": {
                "question": "¿la escalera acota el p95 del MaxDD por debajo del 12%?",
                "ladder_p95_maxdd_pct": ladder_15["maxdd_pct"]["p95"],
                "p95_under_12pct": bool(verdict_p95_under_12),
                "hostile_regime_note": (
                    f"bajo el stress hostil causal la escalera deja el p95 en "
                    f"{ladder_host['maxdd_pct']['p95']:.2f}% y el p99 en "
                    f"{ladder_host['maxdd_pct']['p99']:.2f}% (vs "
                    f"{static_host[1]['maxdd_pct']['p95']:.2f}% / "
                    f"{static_host[1]['maxdd_pct']['p99']:.2f}% estatico 1.5x). "
                    "Overshoot posible por diseño: DD marcado al cierre + decision "
                    "efectiva en t+1 => el escalon flat congela el DD apenas "
                    "CRUZADO el umbral, no antes."
                ),
                "hostile_causal_honesty_note": (
                    "el stress hostil CAUSAL es mas benigno que la variante con "
                    "look-ahead (que seleccionaba las propias semanas de vol alta "
                    "con sus perdidas): en 2025 las semanas POSTERIORES a ventanas "
                    "de vol alta tendieron a recuperarse. Eso es herencia del "
                    "regimen 2025, no una propiedad general — otro motivo para no "
                    "leer estas colas como techo del riesgo real."
                ),
            },
        },
        "dd_budget_proposal": {
            "status": "PROPUESTA — el operador firma (BOOK-LEVERAGE-GOVERNOR.md §5)",
            "rows": [
                {
                    "threshold": "DD 7% => bajar a 1.0x",
                    "justification": (
                        f"p50 del MaxDD del libro a 1.5x = "
                        f"{static_block[1]['maxdd_pct']['p50']:.1f}% y a 1.0x = "
                        f"{static_block[0]['maxdd_pct']['p50']:.1f}%: 7% separa la "
                        "zona normal de la cola y corta la amplificacion 1.5x antes "
                        "de que el path mediano invada el presupuesto"
                    ),
                },
                {
                    "threshold": "DD 10% => bajar a 0.5x",
                    "justification": (
                        f"P(MaxDD>10%) a 1.5x = "
                        f"{static_block[1]['prob_maxdd_gt_10pct']:.1%} vs "
                        f"{static_block[0]['prob_maxdd_gt_10pct']:.1%} a 1.0x: 10% "
                        "ya es cola; a 0.5x la propagacion residual hacia 12% es "
                        "minima y frena la sangria casi con certeza"
                    ),
                },
                {
                    "threshold": "DD 12% => flat + revision formal (presupuesto total)",
                    "justification": (
                        "12% = umbral W1 del WITHDRAWAL-PROTOCOL ya firmado; un "
                        "libro apalancado no puede tener presupuesto mas laxo que "
                        "el retiro de su sleeve principal — la escalera simulada "
                        f"deja P(MaxDD>12%) = "
                        f"{ladder_15['prob_maxdd_gt_12pct']:.1%} (vs "
                        f"{static_block[1]['prob_maxdd_gt_12pct']:.1%} estatico 1.5x)"
                    ),
                },
            ],
        },
        "limitations": [
            (
                "La serie madre son 52 semanas de 2025: el MC HEREDA el regimen "
                "2025 y NO simula regimenes no vistos (quiebre de correlaciones "
                "entre sleeves, crash conjunto tipo 2020, gaps de liquidez). Es "
                "una limitacion estructural, no un parametro ajustable."
            ),
            (
                "El sleeve COP 2025 es un backtest contaminado por seleccion "
                "(DSR 0.72 < 0.95): estos numeros presupuestan riesgo, jamas "
                "evidencian edge. El juez sigue siendo el forward."
            ),
            (
                "Correlaciones 2025 entre sleeves ~0 (COP-XAU -0.02, COP-BTC 0.09, "
                "XAU-BTC 0.05): el beneficio de diversificacion del libro esta "
                "heredado tal cual; en crisis las correlaciones suelen subir y la "
                "cola real seria peor que la simulada."
            ),
            (
                "El bootstrap b=4 preserva dependencia hasta ~4 semanas; "
                "clustering de vol mas largo no esta capturado (por eso se añade "
                "el stress del tercil hostil persistente)."
            ),
            "DD marcado solo al cierre semanal: drawdowns intra-semana no se miden.",
            (
                "ES97.5 del retorno anual se reporta como métrica de COLA "
                "(perdida); el centro de la distribucion no se reporta porque el "
                "bootstrap hereda la media 2025 (seria un claim de retorno)."
            ),
        ],
    }

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE_DIR / "book_mc_stress.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(sanitize(result), f, indent=2, ensure_ascii=False, allow_nan=False)
    shutil.copy2(__file__, EVIDENCE_DIR / "generator_script.py")

    print(f"OK -> {out}")
    print(
        f"libro: ann_vol={data['ann_vol_sample']:.4f} | "
        f"maxDD 2025 realizado (1x) = "
        f"{result['inputs']['book_weekly_series_stats']['realized_maxdd_2025_pct_at_1x']:.2f}%"
    )
    for label, rows in (("BLOCK b=4", static_block), ("IID (ref)", static_iid),
                        ("HOSTIL", static_host)):
        for s in rows:
            print(
                f"  [{label}] {s['book_multiplier']:.1f}x: MaxDD p50 "
                f"{s['maxdd_pct']['p50']:.2f}% p90 {s['maxdd_pct']['p90']:.2f}% "
                f"p95 {s['maxdd_pct']['p95']:.2f}% p99 {s['maxdd_pct']['p99']:.2f}% | "
                f"P>7 {s['prob_maxdd_gt_7pct']:.3f} P>10 "
                f"{s['prob_maxdd_gt_10pct']:.3f} P>12 "
                f"{s['prob_maxdd_gt_12pct']:.3f} | ES97.5 anual "
                f"{s['es97_5_annual_return_pct']:.2f}%"
            )
    for label, lad in (("ESCALERA 1.5x", ladder_15), ("ESCALERA 1.0x", ladder_10),
                       ("ESCALERA hostil", ladder_host)):
        print(
            f"  [{label}] MaxDD p50 {lad['maxdd_pct']['p50']:.2f}% p90 "
            f"{lad['maxdd_pct']['p90']:.2f}% p95 {lad['maxdd_pct']['p95']:.2f}% "
            f"p99 {lad['maxdd_pct']['p99']:.2f}% | P>12 "
            f"{lad['prob_maxdd_gt_12pct']:.3f} | P(flat aplicado/trigger) "
            f"{lad['prob_applied_flat']:.3f}/{lad['prob_trigger_flat']:.3f} | "
            f"mult medio {lad['avg_multiplier_over_horizon']:.2f}"
        )
    print(
        f"VEREDICTO: escalera desde 1.5x => p95 MaxDD = "
        f"{ladder_15['maxdd_pct']['p95']:.2f}% "
        f"({'SI' if verdict_p95_under_12 else 'NO'} queda bajo 12%)"
    )


if __name__ == "__main__":
    main()
