"""Date-resolution ALFRED diagnostics; never manufacture intraday availability.

ALFRED vintage dates describe a historical information set, not our collector's
historical receipt times. Missing from yesterday's vintage does not prove a value
was unavailable at today's open through every possible source.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SERIES_COLUMNS = {
    "DGS2": "FINC_BOND_YIELD2Y_USA_D_DGS2",
    "DCOILBRENTEU": "COMM_OIL_BRENT_GLB_D_BRENT",
}
VERSION = "alfred-prior-date-v1"
MAX_BYTES = 2_000_000


def validate_daily_labels(values) -> list[str]:
    result = []
    for value in values:
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("explicit ISO calendar dates required")
        parsed = date.fromisoformat(value)
        if parsed > datetime.now(UTC).date():
            raise ValueError("future dates cannot certify historical observations")
        result.append(value)
    if len(result) != len(set(result)):
        raise ValueError("duplicate calendar dates")
    return result


def _series(series: str) -> None:
    if series not in SERIES_COLUMNS:
        raise ValueError("only declared DGS2 and DCOILBRENTEU series permitted")


def snapshot_url(series: str, vintage_date: str, start_date: str) -> str:
    _series(series)
    validate_daily_labels([vintage_date])
    validate_daily_labels([start_date])
    if start_date > vintage_date:
        raise ValueError("observation window starts after vintage")
    query = urllib.parse.urlencode({
        "id": series, "cosd": start_date, "coed": vintage_date,
        "vintage_date": vintage_date,
    })
    return "https://alfred.stlouisfed.org/graph/alfredgraph.csv?" + query


def parse_snapshot(raw: bytes, series: str, vintage_date: str) -> pd.Series:
    _series(series)
    validate_daily_labels([vintage_date])
    if len(raw) > MAX_BYTES:
        raise ValueError("oversized snapshot")
    try:
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig")), strict=True))
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("invalid CSV encoding or structure") from exc
    expected = ["observation_date", f"{series}_{vintage_date.replace('-', '')}"]
    if not rows or rows[0] != expected:
        raise ValueError("response must echo the exact series and requested vintage")
    labels, values = [], []
    for row in rows[1:]:
        if len(row) != 2:
            raise ValueError("exactly two CSV columns required")
        label, value = row
        labels.append(label)
        if value in ("", "."):
            values.append(np.nan)
        else:
            if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
                raise ValueError("ambiguous/nonfinite numeric value")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("nonfinite numeric value")
            values.append(number)
    validate_daily_labels(labels)
    if labels != sorted(labels) or any(label > vintage_date for label in labels):
        raise ValueError("unordered or post-vintage observation")
    return pd.Series(values, index=pd.to_datetime(labels), dtype=float, name=series)


def _daily_series(value: pd.Series) -> pd.Series:
    if not isinstance(value.index, pd.DatetimeIndex):
        raise ValueError("daily DatetimeIndex required")
    if (value.index.tz is not None or value.index.has_duplicates or value.index.hasnans
            or not value.index.is_monotonic_increasing
            or (value.index != value.index.normalize()).any()):
        raise ValueError("daily series index must be unique, ordered, naive calendar labels")
    result = pd.to_numeric(value, errors="raise").dropna()
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("nonfinite levels")
    return result


def _transform(value: pd.Series, count: int) -> float | None:
    if len(value) < count:
        return None
    if count == 1:
        return float(value.iloc[-1])
    if (value.iloc[-2:] <= 0).any():
        raise ValueError("log-return requires positive levels")
    return float(np.log(value.iloc[-1] / value.iloc[-2]))


def compare_snapshot(local: pd.Series, snapshot: pd.Series, *, series: str,
                     session_date: str, vintage_date: str, capture_start_date: str | None = None,
                     max_staleness_business_days: int = 5) -> dict:
    """Compare exact T-1 endpoint periods/levels; don't infer intraday release time."""
    _series(series)
    validate_daily_labels([session_date, vintage_date])
    if date.fromisoformat(vintage_date) != date.fromisoformat(session_date) - timedelta(days=1):
        raise ValueError("audit requires exactly the prior calendar date vintage")
    local, snapshot = _daily_series(local), _daily_series(snapshot)
    if type(max_staleness_business_days) is not int or max_staleness_business_days < 0:
        raise ValueError("invalid archived freshness contract")
    if capture_start_date is not None:
        validate_daily_labels([capture_start_date])
    if len(snapshot) and snapshot.index[-1] > pd.Timestamp(vintage_date):
        raise ValueError("post-vintage observation")
    local = local[local.index < pd.Timestamp(session_date)]
    count = 2 if series == "DCOILBRENTEU" else 1
    selected = local.tail(count)
    snap_selected = snapshot.tail(count)
    row = {
        "series": series, "session_date": session_date, "vintage_date": vintage_date,
        "status": "INSUFFICIENT_LOCAL_CONTEXT", "n_required_observations": count,
        "t1_selected_periods": [d.date().isoformat() for d in selected.index],
        "t1_selected_values": [float(v) for v in selected],
        "snapshot_latest_period": snapshot.index[-1].date().isoformat() if len(snapshot) else None,
        "snapshot_selected_periods": [d.date().isoformat() for d in snap_selected.index],
        "snapshot_selected_values": [float(v) for v in snap_selected],
        "t1_transform": _transform(selected, count),
        "snapshot_transform": _transform(snap_selected, count),
        "max_abs_selected_value_difference": None,
        "opening_availability_verified": False,
        "historical_collector_receipt_verified": False,
        "evidence_resolution": "ALFRED_VINTAGE_DATE_NOT_PUBLICATION_TIMESTAMP",
    }
    if len(snapshot):
        row["period_age_calendar_days"] = (pd.Timestamp(session_date) - snapshot.index[-1]).days
    if len(selected) < count:
        return row
    age_business_days = int(np.busday_count(selected.index[-1].date(), date.fromisoformat(session_date)))
    row["local_period_age_business_days"] = age_business_days
    if age_business_days > max_staleness_business_days:
        row["status"] = "LOCAL_STALE_UNDER_T1_CONTRACT"
        row["t1_transform"] = None
        return row
    if capture_start_date is not None and selected.index[0] < pd.Timestamp(capture_start_date):
        row["status"] = "NOT_AUDITED_OUTSIDE_CAPTURE_WINDOW"
        return row
    if snapshot.empty:
        row["status"] = "NO_NUMERIC_OBSERVATION_IN_VINTAGE"
        return row
    matched = snapshot.reindex(selected.index)
    if matched.isna().any():
        row["status"] = "NOT_IN_PRIOR_DATE_VINTAGE"
        row["missing_selected_periods"] = [d.date().isoformat() for d in matched[matched.isna()].index]
        return row
    delta = float(np.max(np.abs(matched.to_numpy() - selected.to_numpy())))
    row["max_abs_selected_value_difference"] = delta
    # Both downloads report decimal source levels. This tolerance addresses only
    # binary representation; it is not an economic materiality threshold.
    row["status"] = ("VALUES_MATCH_PRIOR_DATE_VINTAGE" if delta <= 1e-10
                     else "VALUE_DIFFERS_FROM_PRIOR_DATE_VINTAGE")
    return row


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("source redirect rejected; identity must be reviewed")


def capture_snapshot(directory: Path, series: str, session_date: str, *, window_days: int = 60) -> dict:
    validate_daily_labels([session_date])
    if type(window_days) is not int or not 7 <= window_days <= 366:
        raise ValueError("bounded explicit observation window required")
    vintage = date.fromisoformat(session_date) - timedelta(days=1)
    start = vintage - timedelta(days=window_days)
    url = snapshot_url(series, vintage.isoformat(), start.isoformat())
    record = {
        "version": VERSION, "series": series, "session_date": session_date,
        "vintage_date": vintage.isoformat(), "start_date": start.isoformat(), "url": url,
        "requested_at_utc": datetime.now(UTC).isoformat(),
        "parser_sha256": digest(Path(__file__)), "network_status": "PENDING",
    }
    try:
        opener = urllib.request.build_opener(_NoRedirect())
        request = urllib.request.Request(url, headers={"User-Agent": "USDCOP-Thesis-DataAudit/1.0"})
        with opener.open(request, timeout=20) as response:
            if response.geturl() != url or response.status != 200:
                raise ValueError("source URL/status mismatch")
            payload = response.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES:
            raise ValueError("response exceeds byte limit")
        raw_hash = hashlib.sha256(payload).hexdigest()
        relative = f"raw/{series}_{vintage.isoformat()}_{raw_hash}.csv"
        write_new(directory / relative, payload)
        record.update(raw_path=relative, raw_sha256=raw_hash, bytes=len(payload))
        parse_snapshot(payload, series, vintage.isoformat())
        record["network_status"] = "CAPTURED_AND_PARSED"
    except (OSError, ValueError, urllib.error.URLError) as exc:
        record["network_status"] = "FETCH_OR_PARSE_ERROR"
        record["error_type"] = type(exc).__name__
        if isinstance(exc, urllib.error.HTTPError):
            record["http_status"] = exc.code
        # Do not serialize request/headers/environment or unbounded exception text.
    record["retrieved_at_utc"] = datetime.now(UTC).isoformat()
    relative_record = f"captures/{series}_{vintage.isoformat()}.json"
    write_new(directory / relative_record, json_bytes(record))
    return record


def replay_snapshot(directory: Path, record: dict) -> pd.Series:
    if record.get("version") != VERSION or record.get("parser_sha256") != digest(Path(__file__)):
        raise ValueError("vintage parser identity mismatch")
    expected = snapshot_url(record["series"], record["vintage_date"], record["start_date"])
    validate_daily_labels([record["session_date"]])
    if date.fromisoformat(record["vintage_date"]) != date.fromisoformat(record["session_date"]) - timedelta(days=1):
        raise ValueError("session/vintage relationship mismatch")
    if record.get("url") != expected or record.get("network_status") != "CAPTURED_AND_PARSED":
        raise ValueError("uncertified capture URL/status")
    requested = datetime.fromisoformat(record["requested_at_utc"])
    retrieved = datetime.fromisoformat(record["retrieved_at_utc"])
    if (requested.tzinfo is None or retrieved.tzinfo is None
            or not requested <= retrieved <= datetime.now(UTC)):
        raise ValueError("invalid real capture timestamps")
    sha = record["raw_sha256"]
    if not re.fullmatch(r"[a-f0-9]{64}", sha):
        raise ValueError("invalid raw digest")
    expected_path = f"raw/{record['series']}_{record['vintage_date']}_{sha}.csv"
    path = directory / expected_path
    if record.get("raw_path") != expected_path or not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("raw path mismatch or escape")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha or len(payload) != record["bytes"]:
        raise ValueError("raw capture hash/length mismatch")
    return parse_snapshot(payload, record["series"], record["vintage_date"])
