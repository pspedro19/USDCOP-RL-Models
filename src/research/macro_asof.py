"""Point-in-time macro joins shared by features and regime construction.

The research contract has one temporal rule: a daily observation dated ``d`` is
not available at the opening of session ``d``.  This module keeps that rule in
one place and reads the freshness limit from ``macro_availability.yaml`` so the
feature and HMM paths cannot silently acquire different merge semantics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
AVAILABILITY_PATH = ROOT / "config" / "research" / "macro_availability.yaml"


def _max_staleness_business_days(path: Path = AVAILABILITY_PATH) -> int:
    """Return the contract freshness limit, failing closed on malformed SSOT."""
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        value = int(raw["max_staleness_business_days"])
    except (OSError, TypeError, ValueError, KeyError, yaml.YAMLError) as exc:
        raise RuntimeError(f"invalid macro availability SSOT: {path}") from exc
    if value < 0:
        raise ValueError("max_staleness_business_days must be non-negative")
    return value


def strict_asof(
    targets: pd.Index | pd.Series,
    series: pd.Series,
    *,
    name: str,
    availability_path: Path = AVAILABILITY_PATH,
) -> pd.Series:
    """Join ``series`` to targets using only strictly earlier observations.

    A stale or missing observation becomes NaN.  No interpolation, bfill or
    zero-fill is permitted.  The returned index is the original target index.
    """
    target_index = pd.DatetimeIndex(pd.to_datetime(targets)).sort_values()
    clean = series.dropna().copy()
    clean.index = pd.to_datetime(clean.index)
    clean = clean[~clean.index.duplicated(keep="last")].sort_index()
    left = pd.DataFrame({"d": target_index})
    right = pd.DataFrame(
        {"d": clean.index, name: clean.to_numpy(), "_source": clean.index}
    )
    merged = pd.merge_asof(
        left,
        right,
        on="d",
        direction="backward",
        allow_exact_matches=False,
    )
    max_stale = _max_staleness_business_days(availability_path)
    source_dates = pd.to_datetime(merged["_source"])
    stale = np.array(
        [
            bool(pd.isna(source))
            or np.busday_count(source.date(), target.date()) > max_stale
            for source, target in zip(source_dates, merged["d"], strict=True)
        ],
        dtype=bool,
    )
    values = merged[name].to_numpy(dtype=float)
    values[stale] = np.nan
    return pd.Series(values, index=target_index, name=name)

