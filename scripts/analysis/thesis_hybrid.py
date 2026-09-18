#!/usr/bin/env python
"""Brazo hibrido PPO + LLM, con la regla congelada en el pre-registro v3.

    w_hib(b) = w_ppo(b)   si signo(w_ppo(b)) == signo(w_llm(b)) y ninguno es 0
    w_hib(b) = 0          en cualquier otro caso

La regla se congelo el 2026-09-11 con los dos ledgers al 7 % y al 3 %, es decir **antes de que
nadie pudiera ver que produce**. Esta implementacion no la elige: la aplica.

## Por que esta regla y no un promedio

El fallo medido del PPO v2 no es falta de bruto -- la mediana del bruto es +13,50 % -- sino
exceso de operacion: 4,13 cambios por sesion y un coste mediano del 47,17 %. Un veto que exige
confirmacion independiente **solo puede reducir rotacion; no puede inventar bruto**. Eso hace la
hipotesis estrecha y falsable: si el hibrido sigue perdiendo, el problema no es el filtro de
entrada y la familia se cierra. Promediar exposiciones, ponderar por confianza del LLM o invertir
el papel de veto son hipotesis DISTINTAS y cobran trial aparte (pre-registro v3).

## Contabilidad identica por construccion

Se re-liquida con `run_session()`, el mismo motor que usan el PPO y la liquidacion del LLM. Los
costes no se recalculan aqui ni se aproximan: si el hibrido y sus componentes salieran de
motores distintos, la comparacion mediria tambien la diferencia entre los motores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.dataset import load_portable  # noqa: E402
from src.research.session_env import run_session, simple_returns  # noqa: E402

BARS = 59


def _llm_weights(ledger: Path, valid_dates: set[str]) -> tuple[dict, dict]:
    """Senda de 59 pesos por sesion, con la MISMA regla de completitud que la liquidacion.

    Una sesion incompleta se excluye; **no se rellena con ceros**. Rellenar convertiria una
    sesion no decidida en una decision de no operar, que es una afirmacion distinta.
    """
    grouped: dict[str, dict[int, float]] = {}
    with ledger.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            date, bar = str(row.get("session_date", "")), row.get("bar")
            if date in valid_dates and isinstance(bar, int) and 0 <= bar < BARS:
                grouped.setdefault(date, {})[bar] = row
    weights: dict[str, np.ndarray] = {}
    excluded: dict[str, str] = {}
    for date, bars in grouped.items():
        if set(bars) != set(range(BARS)):
            excluded[date] = "incomplete_59_decisions"
            continue
        # Misma regla que la liquidacion: si el proveedor no respondio, esa sesion no contiene
        # decisiones del modelo sino huecos de red, y no puede entrar en un veto.
        n_unavailable = sum(bool(bars[i].get("unavailable")) for i in range(BARS))
        if n_unavailable:
            excluded[date] = f"provider_unavailable_{n_unavailable}_bars"
            continue
        path = np.asarray([float(bars[i]["weight"]) for i in range(BARS)], dtype=float)
        if not np.isfinite(path).all() or (np.abs(path) > 1.0).any():
            excluded[date] = "invalid_weight"
            continue
        weights[date] = path
    return weights, excluded


def combine(w_ppo: np.ndarray, w_llm: np.ndarray) -> np.ndarray:
    """La regla congelada. Acuerdo de signo; cualquier otra cosa es no operar."""
    agree = (np.sign(w_ppo) == np.sign(w_llm)) & (w_ppo != 0.0) & (w_llm != 0.0)
    return np.where(agree, w_ppo, 0.0)


def _score(sessions: dict, weights_by_date: dict) -> list[dict]:
    out = []
    for date, spec in sessions.items():
        if date not in weights_by_date:
            continue
        scored = run_session(spec.close, weights_by_date[date], spec.spread_pips, date=spec.date)
        out.append({
            "session_date": date,
            "gross_return": float(scored.gross_return),
            "total_cost": float(scored.total_cost),
            "daily_return": float(scored.daily_return),
            "terminal_cost": float(scored.terminal_cost),
            "n_changes": int(scored.n_changes),
        })
    return out


def _summary(results: list[dict]) -> dict:
    daily = np.asarray([r["daily_return"] for r in results], dtype=float)
    if not len(daily):
        return {"compounded_return": 0.0, "sharpe_annualized_sqrt221": 0.0, "max_drawdown": 0.0}
    equity = np.cumprod(1.0 + daily)
    peak = np.maximum.accumulate(equity)
    sd = float(np.std(daily, ddof=1)) if len(daily) > 1 else 0.0
    return {
        "compounded_return": float(equity[-1] - 1.0),
        "sharpe_annualized_sqrt221": float(np.mean(daily) / sd * np.sqrt(221)) if sd > 0 else 0.0,
        "max_drawdown": float(np.min(equity / peak - 1.0)),
        "mean_changes_per_session": float(np.mean([r["n_changes"] for r in results])),
        "total_gross": float(sum(r["gross_return"] for r in results)),
        "total_cost": float(sum(r["total_cost"] for r in results)),
    }


def build(ppo_weights: Path, ledger: Path, block: str, portable: Path,
          *, allow_retrospective: bool = False,
          forward_specs: Path | None = None,
          ppo_config: str = "ppo_regime") -> dict:
    if block == "forward":
        if forward_specs is None:
            raise ValueError("forward hybrid requires --forward-specs")
        with forward_specs.open("rb") as handle:
            bundle = pickle.load(handle)
        manifest = bundle.get("manifest", {})
        if manifest.get("portable_sha256") != hashlib.sha256(portable.read_bytes()).hexdigest():
            raise ValueError("forward specs and portable hash differ")
        specs = {s.date.isoformat(): s for s in bundle.get("sessions", [])}
    else:
        data = load_portable(portable, allow_stale=allow_retrospective)
        specs = {s.date.isoformat(): s for s in data.block(block)}

    ppo_payload = json.loads(ppo_weights.read_text(encoding="utf-8"))
    payload_block = ppo_payload.get("block")
    forward_actions = ppo_payload.get("schema_version") == "confirmatory-ppo-forward-actions-v1"
    if not ((block == "forward" and forward_actions) or payload_block == block):
        raise ValueError("las exposiciones PPO son de otro bloque: "
                         f"{payload_block} en vez de {block}")
    median = ppo_payload["median"]
    if forward_actions:
        median = median.get(ppo_config, {})
        if not median:
            raise ValueError(f"forward PPO actions lack config {ppo_config}")
    ppo = {d: np.asarray(p, dtype=float) for d, p in median.items()}

    llm, llm_excluded = _llm_weights(ledger, set(specs))

    common = sorted(set(ppo) & set(llm))
    hybrid = {d: combine(ppo[d], llm[d]) for d in common}

    # Cuanto sobrevive al veto. Es la magnitud que la hipotesis predice que debe caer, asi que
    # se reporta aunque el resultado sea malo: sin ella, "el hibrido pierde" no dice por que.
    agreed = sum(int(np.count_nonzero(hybrid[d])) for d in common)
    ppo_active = sum(int(np.count_nonzero(ppo[d])) for d in common)
    llm_active = sum(int(np.count_nonzero(llm[d])) for d in common)

    # Descomposicion del bruto del PPO segun lo que el LLM avala. Es la medicion que convierte
    # "el hibrido pierde" en una afirmacion sobre el LLM: si el subconjunto avalado tuviera el
    # mismo signo que el total, el veto seria solo una muestra mas pequena. Medido el
    # 2026-09-12 sobre seleccion: total +9,90 % = avalado -12,35 % + no avalado +22,25 %. El
    # acuerdo del LLM no es ruido, selecciona las posiciones perdedoras del PPO.
    gross_all = gross_agreed = 0.0
    for d in common:
        wp, wl = ppo[d], llm[d]
        r = np.asarray(simple_returns(specs[d].close), dtype=float)
        keep = combine(wp, wl) != 0.0
        gross_all += float((wp * r).sum())
        gross_agreed += float((wp[keep] * r[keep]).sum())

    results = _score({d: specs[d] for d in common}, hybrid)
    # Los componentes se puntuan sobre EXACTAMENTE las mismas sesiones, o la comparacion
    # mediria tambien la diferencia de muestra.
    ppo_only = _score({d: specs[d] for d in common}, {d: ppo[d] for d in common})
    llm_only = _score({d: specs[d] for d in common}, {d: llm[d] for d in common})

    return {
        "contract": "CTR-RESEARCH-HYBRID-001",
        "rule": "sign_agreement_veto",
        "rule_frozen_at": "2026-09-11",
        "rule_source": ".claude/specs/planes/06-PRE-REGISTRATION-v3.md",
        "block": block,
        "confirmatory": False,
        "scope": "post_freeze_forward_partial" if block == "forward" else "retrospective_diagnostic",
        "retrospective_artifact_override": bool(allow_retrospective),
        "portable_path": str(portable),
        "portable_sha256": hashlib.sha256(portable.read_bytes()).hexdigest(),
        "ledger_path": str(ledger),
        "ppo_weights_path": str(ppo_weights),
        "ppo_aggregation": ppo_payload.get("aggregation"),
        "n_sessions_common": len(common),
        "n_sessions_block": len(specs),
        "llm_excluded": llm_excluded,
        "agreement": {
            "bars_total": len(common) * BARS,
            "bars_ppo_active": ppo_active,
            "bars_llm_active": llm_active,
            "bars_surviving_veto": agreed,
            "share_of_ppo_positions_kept": (agreed / ppo_active) if ppo_active else 0.0,
        },
        "ppo_gross_split_by_llm_agreement": {
            "all_bars": gross_all,
            "bars_the_llm_endorses": gross_agreed,
            "bars_the_llm_rejects": gross_all - gross_agreed,
            "note": ("si el tramo avalado tiene signo contrario al total, el acuerdo del LLM "
                     "no es una muestra menor: es anti-informativo"),
        },
        "sessions": results,
        "hybrid": _summary(results),
        "components": {
            "ppo_median_same_sessions": _summary(ppo_only),
            "llm_same_sessions": _summary(llm_only),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppo-weights", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--block", default="selection",
                    choices=("development", "selection", "holdout", "forward"))
    ap.add_argument("--portable", type=Path)
    ap.add_argument("--dataset-version", choices=("v1", "v2"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--allow-retrospective", action="store_true",
                    help="permit a historical portable artifact; never confirmatory")
    ap.add_argument("--forward-specs", type=Path,
                    help="post-freeze specs pickle required for --block forward")
    ap.add_argument("--ppo-config", choices=("ppo_regime", "ppo_backbone"),
                    default="ppo_regime", help="configuration when using forward PPO actions")
    args = ap.parse_args()
    if (args.portable is None) == (args.dataset_version is None):
        ap.error("da exactamente uno: --portable o --dataset-version")
    portable = args.portable or (
        ROOT / "data" / "thesis" / ("research_data_portable_v2.pkl" if args.dataset_version == "v2"
                                    else "research_data_portable.pkl"))
    report = build(args.ppo_weights, args.ledger, args.block, portable,
                   allow_retrospective=args.allow_retrospective,
                   forward_specs=args.forward_specs,
                   ppo_config=args.ppo_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "sessions": report["n_sessions_common"],
        "hybrid_return": round(report["hybrid"]["compounded_return"], 4),
        "kept_share": round(report["agreement"]["share_of_ppo_positions_kept"], 4),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
