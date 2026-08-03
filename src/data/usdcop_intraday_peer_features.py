"""Causal intraday LatAm peer features for daily USD/COP decisions.

The historical hourly snapshot is a reconstructed research backfill, not a
true point-in-time archive.  Twelve Data timestamps denote candle OPEN time,
so this module reconstructs availability as ``open + 1h + 5min`` and rejects
rows whose recorded ``available_at`` proves they were captured while forming.
No result from this module is promotion-eligible without prospective vintages.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


BOGOTA = ZoneInfo("America/Bogota")
UTC = ZoneInfo("UTC")
PEERS = {"USD/MXN": "mxn", "USD/BRL": "brl"}
BAR_DURATION = pd.Timedelta(hours=1)
PUBLICATION_BUFFER = pd.Timedelta(minutes=5)
OBJECTIVE_DELAY = BAR_DURATION + PUBLICATION_BUFFER

FEATURES = [
    "pit_intraday_mxn_preopen_log_return",
    "pit_intraday_brl_preopen_log_return",
    "pit_intraday_mxn_session_log_return",
    "pit_intraday_brl_session_log_return",
    "pit_intraday_latam_session_dispersion",
    "pit_intraday_latam_realized_vol_24h",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_hourly(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "time", "symbol", "tf", "open", "high", "low", "close",
        "source", "available_at",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Hourly peer snapshot missing columns: {sorted(missing)}")
    rows = frame[
        frame["symbol"].isin(PEERS) & frame["tf"].eq("1h")
    ].copy()
    rows["bar_open"] = pd.to_datetime(rows["time"], utc=True, errors="coerce")
    rows["captured_available_at"] = pd.to_datetime(
        rows["available_at"], utc=True, errors="coerce"
    )
    for column in ("open", "high", "low", "close"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    coherent = (
        rows[["open", "high", "low", "close"]].notna().all(axis=1)
        & rows[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & rows["high"].ge(rows[["open", "close"]].max(axis=1))
        & rows["low"].le(rows[["open", "close"]].min(axis=1))
    )
    rows = rows[coherent & rows["bar_open"].notna()].copy()
    rows["objective_available_at"] = rows["bar_open"] + OBJECTIVE_DELAY
    partial = (
        rows["captured_available_at"].notna()
        & rows["captured_available_at"].lt(rows["objective_available_at"])
    )
    partial_count = int(partial.sum())
    rows = rows[~partial].copy()
    duplicate_count = int(rows.duplicated(["symbol", "bar_open"]).sum())
    rows = rows.sort_values(
        ["symbol", "bar_open", "captured_available_at"]
    ).drop_duplicates(["symbol", "bar_open"], keep="last")

    pieces = []
    for symbol, group in rows.groupby("symbol", sort=False):
        group = group.sort_values("objective_available_at").copy()
        group["hourly_log_return"] = np.log(group["close"]).diff()
        squared = group.set_index("objective_available_at")[
            "hourly_log_return"
        ].pow(2)
        group["realized_vol_24h"] = np.sqrt(
            squared.rolling("24h", min_periods=4).sum().to_numpy()
        )
        pieces.append(group)
    prepared = pd.concat(pieces, ignore_index=True).sort_values(
        ["symbol", "objective_available_at"]
    )
    provenance = {
        "input_peer_rows": int(len(frame[
            frame["symbol"].isin(PEERS) & frame["tf"].eq("1h")
        ])),
        "usable_peer_rows": int(len(prepared)),
        "partial_capture_rows_excluded": partial_count,
        "duplicate_symbol_time_rows_dropped": duplicate_count,
        "objective_availability_policy": "bar_open_plus_1h_plus_5m",
        "historical_availability_is_reconstructed": True,
        "promotion_eligible": False,
    }
    return prepared, provenance


def _session_cutoffs(
    dates: pd.Series,
    *,
    hour: int,
    minute: int,
) -> pd.Series:
    values = [
        datetime.combine(value.date(), time(hour, minute), tzinfo=BOGOTA)
        .astimezone(UTC)
        for value in pd.to_datetime(dates)
    ]
    return pd.Series(pd.to_datetime(values, utc=True), index=dates.index)


def _asof_snapshot(
    cutoffs: pd.Series,
    bars: pd.DataFrame,
    symbol: str,
    *,
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
    left = pd.DataFrame({
        "row_id": cutoffs.index,
        "cutoff": pd.DatetimeIndex(cutoffs.array),
    })
    left = left.sort_values("cutoff")
    right = bars[bars["symbol"].eq(symbol)][[
        "objective_available_at", "bar_open", "close", "realized_vol_24h"
    ]].sort_values("objective_available_at")
    merged = pd.merge_asof(
        left,
        right,
        left_on="cutoff",
        right_on="objective_available_at",
        direction="backward",
        tolerance=tolerance,
        allow_exact_matches=True,
    ).set_index("row_id").reindex(cutoffs.index)
    violation = merged[
        merged["objective_available_at"].notna()
        & merged["objective_available_at"].gt(merged["cutoff"])
    ]
    if not violation.empty:
        raise AssertionError("Intraday peer as-of join admitted a future candle")
    return merged


def attach_intraday_peer_features(
    price_frame: pd.DataFrame,
    hourly_path: str | Path | None = None,
    *,
    hourly_frame: pd.DataFrame | None = None,
    preopen_hour_bogota: int = 7,
    preopen_minute_bogota: int = 30,
    decision_hour_bogota: int = 13,
    decision_minute_bogota: int = 30,
    maximum_snapshot_staleness_hours: int = 2,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Attach six fixed peer features using only information known at decision.

    ``price_frame`` must contain one row per USD/COP trading date.  The
    pre-open return compares the 07:30 snapshot with the previous USD/COP
    session's 13:30 snapshot; the session return compares 07:30 with 13:30.
    """
    if "date" not in price_frame:
        raise ValueError("price_frame requires a date column")
    if hourly_frame is None:
        if hourly_path is None:
            raise ValueError("Provide hourly_path or hourly_frame")
        resolved_path = Path(hourly_path)
        hourly_frame = pd.read_parquet(resolved_path)
    else:
        resolved_path = None
    bars, provenance = _prepare_hourly(hourly_frame)
    result = price_frame.copy().reset_index(drop=True)
    if pd.to_datetime(result["date"]).duplicated().any():
        raise ValueError("price_frame contains duplicate trading dates")
    preopen_cutoffs = _session_cutoffs(
        result["date"], hour=preopen_hour_bogota, minute=preopen_minute_bogota
    )
    decision_cutoffs = _session_cutoffs(
        result["date"], hour=decision_hour_bogota, minute=decision_minute_bogota
    )
    tolerance = pd.Timedelta(hours=maximum_snapshot_staleness_hours)

    for symbol, code in PEERS.items():
        preopen = _asof_snapshot(
            preopen_cutoffs, bars, symbol, tolerance=tolerance
        )
        decision = _asof_snapshot(
            decision_cutoffs, bars, symbol, tolerance=tolerance
        )
        result[f"audit_{code}_preopen_bar_open"] = preopen["bar_open"].to_numpy()
        result[f"audit_{code}_preopen_available_at"] = preopen[
            "objective_available_at"
        ].to_numpy()
        result[f"audit_{code}_decision_bar_open"] = decision["bar_open"].to_numpy()
        result[f"audit_{code}_decision_available_at"] = decision[
            "objective_available_at"
        ].to_numpy()
        prior_decision_close = decision["close"].shift(1)
        result[f"pit_intraday_{code}_preopen_log_return"] = np.log(
            preopen["close"].to_numpy() / prior_decision_close.to_numpy()
        )
        result[f"pit_intraday_{code}_session_log_return"] = np.log(
            decision["close"].to_numpy() / preopen["close"].to_numpy()
        )
        result[f"_pit_intraday_{code}_realized_vol_24h"] = decision[
            "realized_vol_24h"
        ].to_numpy()

    result["pit_intraday_latam_session_dispersion"] = (
        result["pit_intraday_mxn_session_log_return"]
        - result["pit_intraday_brl_session_log_return"]
    ).abs()
    result["pit_intraday_latam_realized_vol_24h"] = result[[
        "_pit_intraday_mxn_realized_vol_24h",
        "_pit_intraday_brl_realized_vol_24h",
    ]].mean(axis=1, skipna=False)
    result = result.drop(columns=[
        "_pit_intraday_mxn_realized_vol_24h",
        "_pit_intraday_brl_realized_vol_24h",
    ])
    provenance.update({
        "peer_symbols": list(PEERS),
        "preopen_time_bogota": (
            f"{preopen_hour_bogota:02d}:{preopen_minute_bogota:02d}"
        ),
        "decision_time_bogota": (
            f"{decision_hour_bogota:02d}:{decision_minute_bogota:02d}"
        ),
        "maximum_snapshot_staleness_hours": maximum_snapshot_staleness_hours,
        "first_feature_date": pd.to_datetime(
            result.loc[result[FEATURES].notna().all(axis=1), "date"]
        ).min(),
        "last_feature_date": pd.to_datetime(
            result.loc[result[FEATURES].notna().all(axis=1), "date"]
        ).max(),
    })
    if resolved_path is not None:
        provenance["hourly_snapshot_path"] = str(resolved_path).replace("\\", "/")
        provenance["hourly_snapshot_sha256"] = _sha256_file(resolved_path)
    return result, list(FEATURES), provenance
