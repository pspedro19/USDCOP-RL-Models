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
import yaml

from src.research.dataset import MACRO_FEATURES, SEED_M5
from src.research.features import MACRO_CLEAN

ROOT = Path(__file__).resolve().parents[2]
AVAILABILITY = ROOT / "config" / "research" / "macro_availability.yaml"
SESSION_BARS = 60
FIVE_MINUTES = pd.Timedelta(minutes=5)
SESSION_OPEN = (8, 0)


def audit_frequency_file(path: Path, *, expected_frequency: str | None = None) -> dict:
    """Inspect an OHLCV artifact without computing a strategy or return series.

    The output is deliberately about representation only: effective cadence,
    timezone, ordering, duplicate timestamps and gaps.  A file that is not
    present is reported as ``missing`` so a caller can fail closed explicitly.
    """
    if not path.is_file():
        return {"path": str(path), "status": "missing"}
    frame = pd.read_parquet(path)
    time_col = next((c for c in ("time", "datetime", "timestamp", "date") if c in frame.columns), None)
    if time_col is None:
        return {"path": str(path), "status": "invalid", "error": "no timestamp column"}
    raw = pd.to_datetime(frame[time_col], errors="coerce")
    invalid = int(raw.isna().sum())
    # Convert to UTC only after recording whether the source actually carried a
    # timezone.  A naive source is a provenance defect, not an inferred UTC.
    timezone_present = raw.dt.tz is not None
    ordered = raw.dropna().sort_values()
    delta = ordered.diff().dropna()
    mode = delta.mode().iloc[0] if not delta.empty else pd.NaT
    nonpositive = int((delta <= pd.Timedelta(0)).sum())
    result = {
        "path": str(path), "status": "ok", "rows": int(len(frame)),
        "timestamp_column": time_col, "timezone_present": bool(timezone_present),
        "invalid_timestamps": invalid, "duplicate_timestamps": int(ordered.duplicated().sum()),
        "nonpositive_deltas": nonpositive,
        "effective_delta": str(mode) if pd.notna(mode) else None,
        "min_timestamp": str(ordered.iloc[0]) if len(ordered) else None,
        "max_timestamp": str(ordered.iloc[-1]) if len(ordered) else None,
    }
    if expected_frequency:
        expected = {"5min": pd.Timedelta(minutes=5), "1h": pd.Timedelta(hours=1),
                    "daily": pd.Timedelta(days=1)}.get(expected_frequency)
        if expected is None:
            raise ValueError(f"unsupported expected frequency: {expected_frequency}")
        result["expected_frequency"] = expected_frequency
        result["expected_delta"] = str(expected)
        result["delta_mismatch_count"] = int((delta != expected).sum())
    return result


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
    ohlc_cols = [c for c in ("open", "high", "low", "close") if c in m5.columns]
    ohlc_numeric_invalid = int(
        m5[ohlc_cols].apply(pd.to_numeric, errors="coerce").isna().any(axis=1).sum()
    ) if len(ohlc_cols) == 4 else int(len(m5))

    macro = pd.read_parquet(macro_path)
    availability = yaml.safe_load(AVAILABILITY.read_text(encoding="utf-8"))
    declared = availability.get("series", {})
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
    available = [name for name, col in raw_macro.items()
                 if col in macro.columns and name in declared]
    macro_numeric_invalid = {}
    for name, col in raw_macro.items():
        if col in macro.columns:
            raw_values = macro[col]
            converted = pd.to_numeric(raw_values, errors="coerce")
            # Missing observations are a freshness/PIT concern, not a type error.
            macro_numeric_invalid[name] = int((raw_values.notna() & converted.isna()).sum())
    availability_errors = []
    for name, col in raw_macro.items():
        spec = declared.get(name)
        if spec is None:
            availability_errors.append(f"{name}:missing_declaration")
        elif spec.get("column") != col:
            availability_errors.append(f"{name}:column_mismatch")
        elif spec.get("frequency") != "daily" or spec.get("fallback") != "forbidden":
            availability_errors.append(f"{name}:frequency_or_fallback_policy")
    frequency_files = {
        "usdcop_m5": (m5_path, "5min"),
        "usdcop_1h": (ROOT / "seeds" / "latest" / "usdcop_1h_ohlcv.parquet", "1h"),
        "xauusd_m5": (ROOT / "seeds" / "latest" / "xauusd_m5_ohlcv.parquet", "5min"),
        "xauusd_daily": (ROOT / "seeds" / "latest" / "xauusd_daily_ohlcv.parquet", "daily"),
    }
    frequency_audit = {name: audit_frequency_file(path, expected_frequency=freq)
                       for name, (path, freq) in frequency_files.items()}
    return {
        "contract": "CTR-RESEARCH-DATA-AUDIT-001",
        "inputs": {"m5": str(m5_path), "m5_sha256": _digest(m5_path),
                   "macro": str(macro_path), "macro_sha256": _digest(macro_path)},
        "m5": {"rows": int(len(m5)), "symbols": sorted(m5.get("symbol", pd.Series()).astype(str).unique().tolist()),
               "duplicate_symbol_time": dup, "off_five_minute_grid": off_grid,
               "ohlc_numeric_invalid": ohlc_numeric_invalid,
               "session_count": int(len(counts)), "sessions_not_60_bars": bad_counts,
               "complete_60_bar_sessions": int((counts == SESSION_BARS).sum()),
               "intraday_diffs_not_5m": int((diffs != FIVE_MINUTES).sum()),
               "min_time_utc": m5["time"].min().isoformat(),
               "max_time_utc": m5["time"].max().isoformat()},
        "macro": {"rows": int(len(macro)), "duplicate_dates": macro_dup,
                  "features_present": available,
                  "features_missing": [name for name in raw_macro if name not in available],
                  "numeric_invalid_by_series": macro_numeric_invalid,
                  "availability_errors": availability_errors,
                  "date_min": str(macro_dates.min().date()),
                  "date_max": str(macro_dates.max().date()),
                  "max_calendar_gap_days": float(macro_gap.dt.days.max()) if len(macro_gap) else 0.0,
                  "same_day_values_usable": False,
                  "future_observation_count": macro_future,
                  "note": "publication timestamps are absent; PIT causality is verified by attach_macro_features tests."},
        "frequency_lineage": {
            "artifacts": frequency_audit,
            "merge_policy": "frequency-specific; no implicit resample or forward-fill",
            "independent_samples": False,
            "note": "M5/H1/H4/D1 variants share the same market history and are not independent trials.",
        },
        "verdict": {"structural_m5_clean": bool(dup == 0 and off_grid == 0),
                    "market_numeric_clean": ohlc_numeric_invalid == 0,
                    "complete_session_grid": bool(not bad_counts),
                    "macro_columns_complete": len(available) == len(raw_macro),
                    "macro_numeric_clean": len(macro_numeric_invalid) == len(raw_macro)
                    and all(v == 0 for v in macro_numeric_invalid.values()),
                    "macro_availability_declared": not availability_errors,
                    "requires_pit_merge": True},
    }


def require_contract(m5_path: Path = SEED_M5, macro_path: Path = MACRO_CLEAN) -> dict:
    """Fail-closed gate for research jobs; returns evidence when the contract passes."""
    report = audit(m5_path, macro_path)
    verdict = report["verdict"]
    failures = [name for name in ("structural_m5_clean", "market_numeric_clean",
                                  "macro_columns_complete", "macro_numeric_clean",
                                  "macro_availability_declared")
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
