#!/usr/bin/env python
"""Auditoría read-only de frecuencias, timestamps y uniones del dataset de tesis.

No calcula P&L ni lee hold-out para seleccionar modelos. Su salida es evidencia de ingeniería:
cada merge macro debe poder justificarse por fecha de publicación y ningún valor futuro puede
entrar al contexto de la apertura.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.dataset import MACRO_FEATURES, SEED_M5
from src.research.features import MACRO_CLEAN

ROOT = Path(__file__).resolve().parents[2]
SESSION_BARS = 60
FIVE_MINUTES = pd.Timedelta(minutes=5)
SESSION_OPEN = (8, 0)


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(m5_path: Path = SEED_M5, macro_path: Path = MACRO_CLEAN) -> dict:
    m5 = pd.read_parquet(m5_path)
    m5["time"] = pd.to_datetime(m5["time"], utc=True)
    m5 = m5.sort_values("time")
    d = m5["time"].dt.tz_convert("America/Bogota").dt.date
    counts = m5.groupby(d).size()
    diffs = m5.groupby(d)["time"].diff().dropna()
    # A source can contain multiple symbols; frequency is assessed per symbol/session.
    dup = int(m5.duplicated(subset=[c for c in ("symbol", "time") if c in m5]).sum())
    off_grid = int(((m5["time"].dt.minute % 5) != 0).sum())
    bad_counts = {str(k): int(v) for k, v in counts[counts != SESSION_BARS].items()}

    macro = pd.read_parquet(macro_path)
    if "fecha" in macro.columns:
        macro_dates = pd.to_datetime(macro["fecha"])
    else:
        macro_dates = pd.to_datetime(macro.index)
    macro_dates = pd.DatetimeIndex(macro_dates).tz_localize(None).normalize()
    macro_dup = int(macro_dates.duplicated().sum())
    macro_gap = macro_dates.to_series().sort_values().diff().dropna()
    # The clean parquet carries observation dates, not publication timestamps. Therefore a
    # future-publication count cannot be inferred honestly here; the PIT merge tests are the
    # authoritative guard. Keep this field explicit rather than pretending dates are releases.
    macro_future = None

    raw_macro = {
        "dxy": "FXRT_INDEX_DXY_USA_D_DXY",
        "brent": "COMM_OIL_BRENT_GLB_D_BRENT",
        "ibr": "FINC_RATE_IBR_OVERNIGHT_COL_D_IBR",
        "dgs2": "FINC_BOND_YIELD2Y_USA_D_DGS2",
    }
    available = [name for name, col in raw_macro.items() if col in macro.columns]
    return {
        "contract": "CTR-RESEARCH-DATA-AUDIT-001",
        "inputs": {"m5": str(m5_path), "m5_sha256": _digest(m5_path),
                   "macro": str(macro_path), "macro_sha256": _digest(macro_path)},
        "m5": {"rows": int(len(m5)), "symbols": sorted(m5.get("symbol", pd.Series()).astype(str).unique().tolist()),
               "duplicate_symbol_time": dup, "off_five_minute_grid": off_grid,
               "session_count": int(len(counts)), "sessions_not_60_bars": bad_counts,
               "complete_60_bar_sessions": int((counts == SESSION_BARS).sum()),
               "intraday_diffs_not_5m": int((diffs != FIVE_MINUTES).sum()),
               "min_time_utc": m5["time"].min().isoformat(),
               "max_time_utc": m5["time"].max().isoformat()},
        "macro": {"rows": int(len(macro)), "duplicate_dates": macro_dup,
                  "features_present": available,
                  "features_missing": [name for name in raw_macro if name not in available],
                  "date_min": str(macro_dates.min().date()),
                  "date_max": str(macro_dates.max().date()),
                  "max_calendar_gap_days": float(macro_gap.dt.days.max()) if len(macro_gap) else 0.0,
                  "same_day_values_usable": False,
                  "future_observation_count": macro_future,
                  "note": "publication timestamps are absent; PIT causality is verified by attach_macro_features tests."},
        "verdict": {"structural_m5_clean": bool(dup == 0 and off_grid == 0),
                    "complete_session_grid": bool(not bad_counts),
                    "macro_columns_complete": len(available) == len(raw_macro),
                    "requires_pit_merge": True},
    }


def require_contract(m5_path: Path = SEED_M5, macro_path: Path = MACRO_CLEAN) -> dict:
    """Fail-closed gate for research jobs; returns evidence when the contract passes."""
    report = audit(m5_path, macro_path)
    verdict = report["verdict"]
    failures = [name for name in ("structural_m5_clean", "macro_columns_complete")
                if not verdict[name]]
    if failures:
        raise RuntimeError("research data contract failed: " + ", ".join(failures))
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2))
    return 0 if report["verdict"]["structural_m5_clean"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
