"""Replayable macro source evidence; flags alone never certify an input dataset.

Payloads and manifests are content-addressed and created exclusively. Replaying the
declared source establishes numerical reproducibility, not source independence,
historical vintages, publication-time availability, or executable market prices.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PARSER_VERSION = "macro-source-v1"
SCHEMA_VERSION = 2
SOURCE_UNITS = {
    "INVESTING_DXY": "index_points", "FRED_DCOILBRENTEU": "usd_per_barrel",
    "FRED_DGS2": "percent", "BANREP_IBR": "percent",
}
SOURCE_HOSTS = {
    "INVESTING_DXY": "api.investing.com", "FRED_DCOILBRENTEU": "fred.stlouisfed.org",
    "FRED_DGS2": "fred.stlouisfed.org", "BANREP_IBR": "totoro.banrep.gov.co",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def immutable_write(path: Path, raw: bytes) -> None:
    """Never overwrite evidence, including when an existing name was tampered with."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(raw)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise ValueError(f"immutable evidence collision: {path.name}") from None


def _source_url(source: str, url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname != SOURCE_HOSTS.get(source)
            or parsed.username or parsed.password or parsed.fragment):
        raise ValueError("unexpected or credential-bearing source URL")
    allowed = {"id", "startPeriod", "endPeriod", "dimensionAtObservation", "detail",
               "start-date", "end-date", "time-frame", "add-missing-rows"}
    params = dict(urllib.parse.parse_qsl(parsed.query))
    if set(params) - allowed:
        raise ValueError("unapproved source query parameter; do not archive credentials")
    if source.startswith("FRED_") and (params.get("id") != source.removeprefix("FRED_")
                                       or parsed.path != "/graph/fredgraph.csv"):
        raise ValueError("FRED identifier differs from declared series")
    if source == "INVESTING_DXY" and parsed.path != "/api/financialdata/historical/942611":
        raise ValueError("Investing instrument must be 942611")
    if source == "BANREP_IBR" and "DF_IBR_DAILY_HIST" not in parsed.path:
        raise ValueError("BanRep flow differs from declared IBR source")
    return url


def capture_payload(raw: bytes, *, source: str, url: str, directory: Path) -> dict:
    """Archive exact response bytes; do not pass headers, credentials or request objects."""
    _source_url(source, url)
    digest = hashlib.sha256(raw).hexdigest()
    relative = f"raw/{digest}.bin"
    immutable_write(directory / relative, raw)
    record = {
        "source": source, "url": url, "unit": SOURCE_UNITS[source],
        "path": relative, "sha256": digest, "bytes": len(raw),
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "parser_version": PARSER_VERSION, "parser_sha256": sha256_file(Path(__file__)),
    }
    metadata = canonical_json(record)
    immutable_write(directory / "captures" / f"{hashlib.sha256(metadata).hexdigest()}.json", metadata)
    return record


def _clean_series(dates, values) -> pd.Series:
    index = pd.DatetimeIndex(pd.to_datetime(dates, errors="raise"))
    if index.tz is not None:
        raise ValueError("daily observation dates must be timezone-naive labels")
    if index.isna().any() or (index != index.normalize()).any() or index.has_duplicates:
        raise ValueError("invalid/duplicate daily observation date")
    result = pd.Series(values, index=index, dtype=float).sort_index()
    if result.empty or not np.isfinite(result.to_numpy()).all():
        raise ValueError("empty or nonfinite source values")
    return result


def parse_payload(raw: bytes, source: str) -> pd.Series:
    """Parse the documented numeric representation, failing instead of guessing locale."""
    if source in {"FRED_DCOILBRENTEU", "FRED_DGS2"}:
        frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
        expected = source.removeprefix("FRED_")
        if list(frame.columns) not in (["observation_date", expected], ["DATE", expected]):
            raise ValueError("unexpected FRED columns or series identity")
        dates = pd.to_datetime(frame.iloc[:, 0], errors="raise")
        if dates.duplicated().any():
            raise ValueError("duplicate FRED date")
        valid = ~frame.iloc[:, 1].isin(["", "."])
        values = pd.to_numeric(frame.loc[valid].iloc[:, 1], errors="raise")
        return _clean_series(dates[valid], values.to_numpy())
    if source == "INVESTING_DXY":
        payload = json.loads(raw)
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("unexpected Investing payload")
        dates, values = [], []
        for row in payload["data"]:
            label = str(row.get("rowDateTimestamp", row.get("rowDate", ""))).split("T")[0]
            value = row.get("last_closeRaw", row.get("last_close"))
            if value in (None, "", "-"):
                continue
            if not re.fullmatch(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", str(value)):
                raise ValueError("ambiguous Investing numeric locale")
            dates.append(label)
            values.append(float(str(value).replace(",", "")))
        return _clean_series(dates, values)
    if source == "BANREP_IBR":
        root = ET.fromstring(raw)
        ns = {"g": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic"}
        candidates = []
        for series in root.findall(".//g:Series", ns):
            keys = {n.attrib.get("id"): n.attrib.get("value")
                    for n in series.findall("./g:SeriesKey/g:Value", ns)}
            if keys.get("SUBJECT") == "IRIBRM00" and keys.get("UNIT_MEASURE") == "NR":
                candidates.append(series)
        if len(candidates) != 1:
            raise ValueError("BanRep requires exactly one IRIBRM00/NR series")
        dates, values = [], []
        for obs in candidates[0].findall("./g:Obs", ns):
            date_node, value_node = obs.find("g:ObsDimension", ns), obs.find("g:ObsValue", ns)
            if date_node is None or value_node is None:
                raise ValueError("incomplete BanRep observation")
            dates.append(pd.to_datetime(date_node.attrib["value"], format="%Y%m%d"))
            values.append(float(value_node.attrib["value"]))
        return _clean_series(dates, values)
    raise ValueError(f"unsupported source: {source}")


def replay_payloads(records: list[dict], *, source: str, evidence_root: Path) -> pd.Series:
    if not isinstance(records, list) or not records:
        raise ValueError("missing archived reference payloads")
    parts = []
    root = evidence_root.resolve()
    for record in records:
        if (record.get("source") != source or record.get("unit") != SOURCE_UNITS.get(source)
                or record.get("parser_version") != PARSER_VERSION
                or record.get("parser_sha256") != sha256_file(Path(__file__))):
            raise ValueError("source/unit/parser identity mismatch")
        _source_url(source, record["url"])
        stamp = datetime.fromisoformat(record["retrieved_at_utc"])
        if stamp.tzinfo is None or stamp > datetime.now(UTC):
            raise ValueError("invalid retrieval timestamp")
        digest = record.get("sha256", "")
        if (not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)
                or record.get("path") != f"raw/{digest}.bin"):
            raise ValueError("non-content-addressed reference path")
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root) or path.name != f"{digest}.bin":
            raise ValueError("reference payload escapes evidence root")
        raw = path.read_bytes()
        if (hashlib.sha256(raw).hexdigest() != record.get("sha256")
                or len(raw) != record.get("bytes")):
            raise ValueError("reference payload hash/size mismatch")
        parts.append(parse_payload(raw, source))
    combined = pd.concat(parts).sort_index()
    if combined.index.has_duplicates:
        raise ValueError("overlapping reference payload dates")
    return combined


def compare_series(ours: pd.Series, reference: pd.Series, *, tolerance: float) -> dict:
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("invalid tolerance")
    # Missing cells in the union parquet are permissible, never silently discard infinities.
    local = _clean_series(ours.dropna().index, ours.dropna().to_numpy())
    common = local.index.intersection(reference.index)
    if not len(common):
        raise ValueError("no common observation dates")
    diff = (local.loc[common] - reference.loc[common]).abs()
    discrepancies = diff[diff >= tolerance]
    expected_local = reference.loc[local.index.min():local.index.max()].index
    return {
        "n_local": len(local), "n_reference": len(reference), "n_common": len(common),
        "n_missing_reference": len(local.index.difference(reference.index)),
        "n_missing_local": len(expected_local.difference(local.index)),
        "n_discrepancies": len(discrepancies),
        "match_fraction": float((diff < tolerance).mean()),
        "mean_abs_diff": float(diff.mean()), "median_abs_diff": float(diff.median()),
        "p95_abs_diff": float(diff.quantile(.95)), "max_abs_diff": float(diff.max()),
        "local_first": str(local.index.min().date()), "local_last": str(local.index.max().date()),
        "reference_first": str(reference.index.min().date()),
        "reference_last": str(reference.index.max().date()),
        "discrepancies": [{"date": str(day.date()), "local": float(local.at[day]),
                            "reference": float(reference.at[day]), "abs_diff": float(value)}
                           for day, value in discrepancies.items()],
    }


def require_macro_evidence(report: dict, *, clean: Path, availability: Path,
                           evidence_root: Path | None = None) -> dict:
    """Recompute evidence from archived bytes. A positive legacy JSON must fail closed."""
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("macro evidence schema missing/legacy; raw numerical replay required")
    raw_clean, raw_availability = clean.read_bytes(), availability.read_bytes()
    inputs = report.get("inputs", {})
    if (inputs.get("clean_sha256") != hashlib.sha256(raw_clean).hexdigest()
            or inputs.get("availability_sha256") != hashlib.sha256(raw_availability).hexdigest()):
        raise ValueError("macro evidence input hash mismatch")
    timestamp = datetime.fromisoformat(report["measured_at_utc"])
    if timestamp.tzinfo is None or timestamp > datetime.now(UTC):
        raise ValueError("invalid measurement timestamp")
    frame = pd.read_parquet(io.BytesIO(raw_clean))
    declared = yaml.safe_load(raw_availability)["series"]
    if set(report.get("series", {})) != set(declared) or not declared:
        raise ValueError("macro evidence series coverage mismatch")
    tolerance = report.get("tolerance")
    if tolerance != 0.01:
        raise ValueError("unregistered numerical tolerance")
    root = evidence_root or Path(report["evidence_root"])
    recomputed = {}
    for name, spec in declared.items():
        entry = report["series"][name]
        if (entry.get("declared_source") != spec["source"]
                or entry.get("column") != spec["column"] or entry.get("unit") != spec["unit"]
                or entry.get("local_provenance_only", False)):
            raise ValueError(f"{name}: declaration/local-only mismatch")
        records = entry.get("reference_payloads")
        reference = replay_payloads(records, source=spec["source"], evidence_root=root)
        if any(datetime.fromisoformat(r["retrieved_at_utc"]) > timestamp for r in records):
            raise ValueError("report predates captured reference payloads")
        stats = compare_series(frame[spec["column"]], reference, tolerance=tolerance)
        if any(entry.get(key) != value for key, value in stats.items()):
            raise ValueError(f"{name}: numerical evidence differs from raw replay")
        if stats["n_missing_reference"] or stats["n_missing_local"] or stats["n_discrepancies"]:
            raise ValueError(f"{name}: incomplete coverage or numerical discrepancies")
        recomputed[name] = stats
    return {"schema_version": SCHEMA_VERSION, "numerical_identity_verified": True,
            "source_independence_verified": False, "historical_availability_verified": False,
            "recomputed": recomputed}
