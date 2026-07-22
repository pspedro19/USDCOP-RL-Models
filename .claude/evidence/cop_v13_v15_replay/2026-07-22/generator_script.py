"""Replay DESCRIPTIVO v13/v15 sobre 2025 y 2026 — pedido del operador (2026-07-22).

PRE-DECLARACIÓN (antes de correr): ABRE 4 celdas de monitoreo (+4 trials, N 82→86):
v13-2025, v13-2026, v15-2025, v15-2026. Se corre para RESPONDER "¿cómo les habría ido?",
NO para seleccionar — el juez sellado de v13 (y de v15 si se congela) es el forward
desde su freeze; este tramo queda MIRADO y jamás podrá reclamarse como forward limpio.
2025 está además contaminado por selección (DSR 0.72): nada de aquí prueba edge.
Techos con cutoff=None (la mediana de diseño queda congelada en 2024 por fórmula).
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pipeline"))

from train_and_export_smart_simple import (  # noqa: E402
    load_config, load_data, run_walkforward_backtest, compute_v13_leverage_ceilings,
)
from src.forecasting.enhance_v2 import enhance_features_v2  # noqa: E402

V12_CONFIG = REPO / "config" / "execution" / "smart_simple_v12_lev_cap.yaml"
OUT = REPO / ".claude" / "evidence" / "cop_v13_v15_replay" / "2026-07-22"


def stats(res):
    m = res["metrics"] or {}
    tr = res["trades"]
    return {"ret_pct": m.get("total_return_pct"), "max_dd_pct": m.get("max_dd_pct"),
            "n_trades": m.get("n_trades", 0),
            "n_hard_stops": (m.get("exit_reasons") or {}).get("hard_stop", 0),
            "lev_mean": round(float(np.mean([t["leverage"] for t in tr])), 3) if tr else None,
            "note_n": "N<20 => solo conteo y PnL" if m.get("n_trades", 99) < 20 else None}


def main():
    cfg13 = load_config(config_path=str(V12_CONFIG), version_override="13.0.0",
                        strategy_id="smart_simple_v13_qrisk")
    cfg13["v13_ceiling_enabled"] = True
    cfg15 = copy.deepcopy(cfg13)
    cfg15["strategy_id"] = "smart_simple_v15_qrisk_eme"

    df, feats = load_data()
    df, feats = enhance_features_v2(df, feats)
    c13, m13 = compute_v13_leverage_ceilings(df)                       # sin cutoff
    c15, m15 = compute_v13_leverage_ceilings(df, eme_dispersion=True)
    cfg13["_v13_ceiling_cache"], cfg13["_v13_ceiling_meta"] = c13, m13
    cfg15["_v13_ceiling_cache"], cfg15["_v13_ceiling_meta"] = c15, m15

    out = {"label": "REPLAY DESCRIPTIVO (+4 trials, N 82->86) — prohibido seleccionar; "
                    "juez = forward post-freeze; 2025 contaminado (DSR 0.72)",
           "results": {}}
    for name, cfg in (("v13", cfg13), ("v15", cfg15)):
        out["results"][name] = {}
        for year in (2025, 2026):
            r = run_walkforward_backtest(df, feats, cfg, year)
            out["results"][name][str(year)] = stats(r)
            s = out["results"][name][str(year)]
            print(f"{name} {year}: ret={s['ret_pct']}% dd={s['max_dd_pct']}% "
                  f"tr={s['n_trades']} hs={s['n_hard_stops']} lev={s['lev_mean']}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "replay_2025_2026.json").write_text(json.dumps(out, indent=2, default=str))
    shutil.copy(__file__, OUT / "generator_script.py")
    print(f"artefacto -> {OUT}")


if __name__ == "__main__":
    main()
