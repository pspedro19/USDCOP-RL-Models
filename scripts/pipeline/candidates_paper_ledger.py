"""Ledger de paper 2026 de las candidatas COP + libro — ANCLADO A ENERO 2026.

Directiva del operador (2026-07-22): "el paper debe ser siempre calculado desde enero
del 2026 y dejarlo habilitado para que corra el resto del año".

ETIQUETADO CONSTITUCIONAL (CTR-QUANT-CONSTITUTION-001):
- La serie 2026 completa de v12/v14 es HÍBRIDA: replay descriptivo Ene-2026 → freeze
  (2026-07-21, tramo ya MIRADO — trials 69→71 pagados el 2026-07-21) + forward real
  post-freeze. El JUEZ SELLADO de cada candidata consume SOLO las semanas post-freeze
  (campo `judge_window` separado en la salida) — la serie completa es monitoreo.
- v11: su 2026 entero es forward real (producción desde enero).
- v13 EXCLUIDA hasta su freeze (correrle 2026 = +1 trial nuevo; requiere operador).
- Libro book_v1: pesos ERC CONGELADOS del yaml; semana sin dato de un sleeve = NA
  (ITT), jamás 0 inventado. 0 trials: integridad y reporte, sin métricas de decisión.
- Sin Sharpe/p-values con N<20 por serie (solo conteo y PnL).

Salida (dashboard-served, tracked por política 2026-07-09):
  usdcop-trading-dashboard/public/data/production/paper/candidates_ledger_2026.json
Refresh: tarea semanal `paper_ledger_2026` en forecast_h5_l6_weekly_monitor (Vie 14:30
COT) — "habilitado parcialmente": corre el resto del año sin intervención; también
ejecutable a mano: python scripts/pipeline/candidates_paper_ledger.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pipeline"))

FREEZE_DATES = {  # inicio del juez sellado por candidata (freeze real)
    "smart_simple_v11": None,          # producción: todo 2026 es forward
    "smart_simple_v12": "2026-07-21",
    "smart_simple_v14": "2026-07-21",
}
CONFIG_OVERRIDES = {
    "smart_simple_v11": {},
    "smart_simple_v12": {"vt_max": 1.5},
    "smart_simple_v14": {"vt_max": 1.5, "ladder_enabled": True, "ladder_k2": 2.0},
}
OUT = REPO / "usdcop-trading-dashboard/public/data/production/paper/candidates_ledger_2026.json"


def _connect_lineage_db():
    """Connect without ever materializing or logging credential values."""
    import psycopg2

    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        return psycopg2.connect(db_url)
    required = ("POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "paper ledger lineage requires DATABASE_URL or complete POSTGRES_* settings; "
            "missing variable names: " + ", ".join(missing)
        )
    return psycopg2.connect(
        host=os.environ["POSTGRES_HOST"],
        port=os.environ["POSTGRES_PORT"],
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )


def iso_week(ts: str) -> str:
    d = datetime.fromisoformat(str(ts)[:10])
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def _python_values(value):
    """Remove library scalar wrappers before canonical identity sealing."""
    if isinstance(value, dict):
        return {str(key): _python_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_python_values(item) for item in value]
    if type(value).__module__.split(".", 1)[0] == "numpy":
        return value.item()
    return value


def main() -> int:
    from train_and_export_smart_simple import (
        load_config,
        run_production_backtest,
    )
    from src.forecasting.enhance_v2 import enhance_features_v2
    from src.forecasting.dataset_loader import load_data_with_provenance, rebind_dataset_provenance
    from src.contracts.strategy_schema import safe_json_dump
    from src.identity.candidate_ledger import seal_candidate_ledger
    from src.lineage.paper_writer import persist_paper_lineage

    cfg0 = load_config()
    df, feats, provenance = load_data_with_provenance(project_root=REPO)
    df, feats = enhance_features_v2(df, feats)
    provenance = rebind_dataset_provenance(provenance, df, feats)

    ledger: dict = {
        "contract": "CTR-QUANT-CONSTITUTION-001",
        "anchor": "2026-01-01 (directiva operador 2026-07-22)",
        "labels": {
            "smart_simple_v11": "forward real todo 2026 (producción)",
            "smart_simple_v12": "replay descriptivo Ene->2026-07-21 (mirado, trial pagado) + forward post-freeze",
            "smart_simple_v14": "replay descriptivo Ene->2026-07-21 (mirado, trial pagado) + forward post-freeze",
            "v13": "EXCLUIDA hasta freeze (abrir 2026 = +1 trial, decisión del operador)",
        },
        "judge_note": "el juez sellado de v12/v14 consume SOLO judge_window (post-freeze); "
                      "la serie completa es monitoreo, no evidencia confirmatoria",
        "generated_at": None,  # se sella abajo
        "strategies": {},
        "book": None,
    }

    weekly_by_strat: dict[str, dict[str, float]] = {}
    lineage_trade: dict | None = None
    for sid, over in CONFIG_OVERRIDES.items():
        c = dict(cfg0)
        c.update(over)
        r = run_production_backtest(df, feats, c, 2026)
        trades = r["trades"]
        if sid == "smart_simple_v11" and trades:
            lineage_trade = dict(trades[0])
        wk: dict[str, float] = defaultdict(float)
        for t in trades:
            w = iso_week(t["timestamp"])
            wk[w] = (1 + wk.get(w, 0.0)) * (1 + t["pnl_pct"] / 100) - 1
        weekly_by_strat[sid] = dict(wk)
        eq = 10000.0
        rows = []
        for t in trades:
            eq *= 1 + t["pnl_pct"] / 100
            rows.append({"timestamp": t["timestamp"], "side": t["side"],
                         "pnl_pct": t["pnl_pct"], "exit_reason": t["exit_reason"],
                         "leverage": t["leverage"], "equity": round(eq, 2)})
        freeze = FREEZE_DATES[sid]
        judge = None
        if freeze:
            jt = [t for t in trades if str(t["timestamp"])[:10] > freeze]
            judge = {"starts_after": freeze, "n_trades": len(jt),
                     "pnl_pct_compound": round((np.prod([1 + t["pnl_pct"] / 100 for t in jt]) - 1) * 100, 4) if jt else 0.0,
                     "note": "N<20 => solo conteo y PnL"}
        ledger["strategies"][sid] = {
            "ret_2026_ytd_pct": round((eq / 10000 - 1) * 100, 2),
            "n_trades": len(trades),
            "note_n": "N<20 => solo conteo y PnL" if len(trades) < 20 else None,
            "judge_window": judge,
            "trades": rows,
        }
        print(f"{sid}: 2026 YTD {ledger['strategies'][sid]['ret_2026_ytd_pct']}% "
              f"({len(trades)} trades)", flush=True)

    # ---- Libro book_v1: pesos ERC congelados; sleeve sin dato en la semana = NA ----
    import yaml
    book_cfg = yaml.safe_load((REPO / "config/book/book_v1.yaml").read_text(encoding="utf-8"))
    w_cop = None
    for sl in book_cfg.get("sleeves", []):
        if "cop" in str(sl.get("sleeve_id", "")).lower():
            w_cop = float(sl.get("erc_weight"))
    cop_sleeve = weekly_by_strat["smart_simple_v12"]  # sleeve COP actual de book_v1
    # XAU/BTC: el ledger del libro completo exige las curvas de equity de sus bundles;
    # si no están disponibles esta semana, la semana queda NA (ITT) — se reporta la
    # cobertura, no se inventa cero.
    book_rows = []
    for w in sorted(cop_sleeve):
        book_rows.append({"iso_week": w, "cop_component_pct": round(cop_sleeve[w] * 100, 4),
                          "xau_component_pct": None, "btc_component_pct": None,
                          "book_ret_pct": None,
                          "status": "PARTIAL — solo sleeve COP poblado por este runner; "
                                    "XAU/BTC via sus bundles (semana sin dato = NA)"})
    ledger["book"] = {
        "weights_frozen": {sl["sleeve_id"]: sl["erc_weight"]
                           for sl in book_cfg.get("sleeves", [])},
        "cop_weight_used": w_cop,
        "policy": "pesos ERC congelados de book_v1.yaml; NA=dato faltante (ITT), nunca 0",
        "weeks": book_rows,
    }

    if lineage_trade is None:
        raise RuntimeError("smart_simple_v11 produced no real trade to anchor BL-24 lineage")
    connection = _connect_lineage_db()
    staged = OUT.with_name(f".{OUT.name}.{os.getpid()}.staged")
    try:
        with connection.cursor() as cursor:
            declaration = persist_paper_lineage(
                cursor,
                strategy_id="smart_simple_v11",
                trade=lineage_trade,
                dataset=df,
                provenance=provenance,
                run_id=f"paper-ledger-2026:{date.today().isoformat()}",
                verified_at=datetime.now(timezone.utc),
            )
        ledger["strategies"]["smart_simple_v11"]["lineage"] = declaration.as_dict()
        ledger["generated_at"] = date.today().isoformat()
        producer_code_hash = "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ledger = _python_values(ledger)
        ledger = seal_candidate_ledger(ledger, producer_code_hash=producer_code_hash)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with staged.open("x", encoding="utf-8") as fh:
            safe_json_dump(ledger, fh)
            fh.flush()
            os.fsync(fh.fileno())
        connection.commit()
        os.replace(staged, OUT)
    except Exception:
        connection.rollback()
        staged.unlink(missing_ok=True)
        raise
    finally:
        connection.close()
    print(f"ledger -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
