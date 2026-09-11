"""Point-in-time ingestion for forward-looking Colombian USD/COP data.

The existing macro tables are wide and keep one row per observation date.  That
shape cannot preserve multiple publication vintages, so this module writes a
long-form PIT dataset whose primary time contract is ``available_at``.

Official sources supported here:

* Banco de la Republica (SUAMECA) daily USD/COP forward bulletins.
* Banco de la Republica monthly derivatives reports.
* Banco de la Republica EME analyst expectations workbooks.
* Superfinanciera pension-fund derivative positions (historic Socrata format
  468 and current portfolio workbook format 415).

Raw documents are cached under ``data/pipeline/01_sources`` and every output is
hashed.  Historical rows with a reconstructed release lag are retained for
research but explicitly marked as not promotion eligible.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
import unicodedata
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time as dt_time, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import yaml

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:  # pragma: no cover - exercised by the CLI dependency check
    requests = None
    HTTPAdapter = None
    Retry = None

LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "usdcop_forward_macro_sources.yaml"
DEFAULT_RAW = ROOT / "data" / "pipeline" / "01_sources" / "17_forward_looking"
DEFAULT_OUTPUT = (
    ROOT / "data" / "pipeline" / "04_cleaning" / "output"
    / "USDCOP_FORWARD_MACRO_PIT.parquet"
)
DEFAULT_MANIFEST_DIR = (
    ROOT / "data" / "pipeline" / "02_scrapers" / "storage" / "manifests"
)

PIT_COLUMNS = [
    "series_id",
    "observation_date",
    "reference_date",
    "release_date",
    "available_at",
    "value",
    "frequency",
    "unit",
    "source",
    "source_url",
    "document_sha256",
    "retrieved_at",
    "availability_policy",
    "pit_vintage",
    "promotion_eligible",
    "metadata_json",
]

SPANISH_MONTHS = {
    "ene": 1,
    "enero": 1,
    "feb": 2,
    "febrero": 2,
    "mar": 3,
    "marzo": 3,
    "abr": 4,
    "abril": 4,
    "may": 5,
    "mayo": 5,
    "jun": 6,
    "junio": 6,
    "jul": 7,
    "julio": 7,
    "ago": 8,
    "agosto": 8,
    "sep": 9,
    "sept": 9,
    "septiembre": 9,
    "oct": 10,
    "octubre": 10,
    "nov": 11,
    "noviembre": 11,
    "dic": 12,
    "diciembre": 12,
}
MONTH_ABBR = {
    1: "ene", 2: "feb", 3: "mar", 4: "abr", 5: "may", 6: "jun",
    7: "jul", 8: "ago", 9: "sep", 10: "oct", 11: "nov", 12: "dic",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ascii(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\u00a0", " ").replace("−", "-").replace("–", "-")
    return "".join(
        char for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )


def _compact(value: Any) -> str:
    return re.sub(r"\s+", " ", _ascii(value)).strip()


def _pdf_ascii(value: Any) -> str:
    """Repair common glyph splits produced by LaTeX PDFs and pypdf."""
    text = _ascii(value).replace("ı", "i")
    repairs = {
        r"\bF\s+echa\b": "Fecha",
        r"\bT\s+otal\b": "Total",
        r"\bdevaluaci\s+on\b": "devaluacion",
        r"\bnegociaci\s+on\b": "negociacion",
        r"\bparticipaci\s+on\b": "participacion",
        r"\bposici\s+on\b": "posicion",
        r"\bimpl\s+icita\b": "implicita",
        r"\bte\s+orica\b": "teorica",
        r"\bd\s+ias\b": "dias",
        r"\bd\s+ia\b": "dia",
        r"\bd\s+olar(?:es)?\b": "dolar",
        r"\bn\s+umero\b": "numero",
        r"\bGr\s+afico\b": "Grafico",
        r"\bTama\s+no\b": "Tamano",
        r"\ban\s+o\b": "ano",
    }
    for pattern, replacement in repairs.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def parse_number(value: Any) -> float:
    """Parse Spanish or English-formatted numbers without guessing silently."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = _compact(value).replace("%", "").replace("US$", "").replace("USD", "")
    text = re.sub(r"[^0-9,\.\-+]", "", text)
    if not text or text in {"-", "+"}:
        return math.nan
    if "," in text:
        # Official Colombian documents use dots for thousands and comma decimals.
        text = text.replace(".", "").replace(",", ".")
    elif text.count(".") > 1:
        text = text.replace(".", "")
    elif text.count(".") == 1:
        left, right = text.split(".")
        if len(right) == 3 and len(left.lstrip("-+")) <= 3:
            text = left + right
    try:
        return float(text)
    except ValueError:
        return math.nan


def parse_spanish_date(value: Any, default_year: int | None = None) -> pd.Timestamp | None:
    text = _compact(value).lower()
    text = re.sub(r"^el\s+", "", text)
    text = text.replace("/", " ").replace(".", " ").replace("de ", "")
    text = re.sub(r"\s+", " ", text).strip()
    direct = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if not pd.isna(direct):
        return pd.Timestamp(direct).normalize()
    match = re.search(
        r"(?P<day>\d{1,2})\s+(?P<month>[a-z]+)\s+(?P<year>\d{2,4})?", text
    )
    if not match:
        return None
    month = SPANISH_MONTHS.get(match.group("month").rstrip("."))
    if not month:
        return None
    year_text = match.group("year")
    year = int(year_text) if year_text else default_year
    if year is None:
        return None
    if year < 100:
        year += 2000 if year < 50 else 1900
    try:
        return pd.Timestamp(year=year, month=month, day=int(match.group("day")))
    except ValueError:
        # A malformed target date in one workbook must not abort the complete
        # monthly history.  The caller can retain the value with a null target.
        return None


def _bogota_to_utc(value: datetime | pd.Timestamp, end_of_day: bool = False) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if end_of_day and stamp.hour == stamp.minute == stamp.second == 0:
        stamp = stamp + pd.Timedelta(hours=23, minutes=59, seconds=59)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("America/Bogota")
    return stamp.tz_convert("UTC")


def _conservative_month_release(observation: pd.Timestamp, days: int = 31) -> pd.Timestamp:
    return _bogota_to_utc(observation.normalize() + pd.Timedelta(days=days), end_of_day=True)


def _valid_embedded_timestamp(
    candidate: datetime | pd.Timestamp | None,
    observation: pd.Timestamp,
    max_lag_days: int,
) -> pd.Timestamp | None:
    if candidate is None:
        return None
    stamp = _bogota_to_utc(candidate)
    obs_utc = _bogota_to_utc(observation)
    if obs_utc <= stamp <= obs_utc + pd.Timedelta(days=max_lag_days):
        return stamp
    return None


def _pdf_text_and_created(path: Path) -> tuple[str, pd.Timestamp | None]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency failure is explicit
        raise RuntimeError("pypdf is required to parse official BanRep PDF reports") from exc
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    created: pd.Timestamp | None = None
    metadata = reader.metadata
    if metadata is not None:
        try:
            raw = metadata.creation_date or metadata.modification_date
            if raw is not None:
                created = pd.Timestamp(raw)
        except (AttributeError, ValueError, TypeError):
            created = None
    return text, created


def _xlsx_created(path: Path) -> pd.Timestamp | None:
    if path.suffix.lower() != ".xlsx":
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("docProps/core.xml").decode("utf-8", errors="replace")
        matches = re.findall(
            r"<dcterms:(?:modified|created)[^>]*>([^<]+)</dcterms:(?:modified|created)>",
            xml,
        )
        stamps = [pd.Timestamp(item) for item in matches]
        return max(stamps) if stamps else None
    except (KeyError, ValueError, zipfile.BadZipFile):
        return None


def _xls_created(path: Path) -> pd.Timestamp | None:
    try:
        import olefile
    except ImportError:
        return None
    try:
        with olefile.OleFileIO(str(path)) as ole:
            stream = "\x05SummaryInformation"
            if not ole.exists(stream):
                return None
            props = ole.getproperties(stream)
            candidates = [props.get(12), props.get(13)]  # created, last saved
            stamps = [pd.Timestamp(item) for item in candidates if item is not None]
            return max(stamps) if stamps else None
    except (OSError, ValueError):
        return None


def parse_daily_forward_text(text: str) -> dict[str, float]:
    """Extract stable features from a two-page BanRep daily forward report."""
    normalized = _pdf_ascii(text)
    flat = re.sub(r"\s+", " ", normalized)
    result: dict[str, float] = {}

    total_match = re.search(
        r"se pacto un total de US\$\s*([\d.,]+)\s*millones.*?"
        r"devaluacion implicita.*?([\d.,]+)\s*%",
        flat,
        flags=re.IGNORECASE,
    )
    if total_match:
        result["br_forward_daily_volume_usd_m"] = parse_number(total_match.group(1))
        result["br_forward_daily_implied_devaluation_pct"] = parse_number(
            total_match.group(2)
        )

    table_one = normalized
    if "Cuadro No. 1" in normalized and "Cuadro No. 2" in normalized:
        table_one = normalized.split("Cuadro No. 1", 1)[1].split("Cuadro No. 2", 1)[0]
    rows: dict[str, list[float]] = {}
    for line in table_one.splitlines():
        clean = _compact(line)
        row_match = re.match(r"^(3-14|15-35|36-60|61-90|91-180|>180|Total)\s+(.+)$", clean)
        if not row_match:
            continue
        numbers = [parse_number(item) for item in re.findall(r"[-+]?\d[\d.,]*", row_match.group(2))]
        if len(numbers) >= 7:
            rows[row_match.group(1).lower()] = numbers[:7]
    total = rows.get("total")
    if total:
        names = [
            "br_forward_daily_financial_buy_usd_m",
            "br_forward_daily_financial_sell_usd_m",
            "br_forward_daily_offshore_buy_usd_m",
            "br_forward_daily_offshore_sell_usd_m",
            "br_forward_daily_other_buy_usd_m",
            "br_forward_daily_other_sell_usd_m",
            "br_forward_daily_intermediary_usd_m",
        ]
        result.update(dict(zip(names, total)))
        result["br_forward_daily_financial_net_usd_m"] = total[0] - total[1]
        result["br_forward_daily_offshore_net_usd_m"] = total[2] - total[3]
        result["br_forward_daily_other_net_usd_m"] = total[4] - total[5]
        result["br_forward_daily_customer_net_usd_m"] = (
            total[0] - total[1] + total[2] - total[3] + total[4] - total[5]
        )
    tenor_rows = [rows.get(key) for key in ("3-14", "15-35", "36-60", "61-90", "91-180", ">180")]
    if all(row is not None for row in tenor_rows):
        gross = [sum(row[:6]) for row in tenor_rows if row is not None]
        gross_total = sum(gross)
        if gross_total > 0:
            result["br_forward_daily_short_tenor_share"] = sum(gross[:2]) / gross_total
            result["br_forward_daily_long_tenor_share"] = sum(gross[4:]) / gross_total

    table_three = normalized.split("Cuadro No. 3", 1)[-1]
    for line in table_three.splitlines():
        clean = _compact(line)
        if not re.match(r"^Total\s+", clean):
            continue
        numbers = [parse_number(item) for item in re.findall(r"[-+]?\d[\d.,]*", clean[5:])]
        if len(numbers) >= 12:
            result["br_forward_daily_ndf_3w_net_usd_m"] = sum(
                numbers[index] - numbers[index + 1] for index in (2, 6, 10)
            )
            result["br_forward_daily_ndf_3w_gross_usd_m"] = sum(
                numbers[index] + numbers[index + 1] for index in (2, 6, 10)
            )
            break
    return {key: value for key, value in result.items() if np.isfinite(value)}


def parse_daily_publication_date(text: str) -> pd.Timestamp | None:
    normalized = _compact(_pdf_ascii(text))
    match = re.search(
        r"Fecha de Publicacion:\s*(\d{1,2}\s+de\s+[A-Za-z]+\s+de\s+\d{4})",
        normalized,
        flags=re.IGNORECASE,
    )
    return parse_spanish_date(match.group(1)) if match else None


def _parse_named_table_row(text: str, label: str) -> list[float] | None:
    for line in text.splitlines():
        clean = _compact(line)
        if not re.match(rf"^{re.escape(label)}\b", clean, flags=re.IGNORECASE):
            continue
        values = [parse_number(item) for item in re.findall(r"[-+]?\d[\d.,]*", clean[len(label):])]
        if len(values) >= 2:
            return values
    return None


def parse_monthly_derivatives_text(text: str) -> dict[str, float]:
    """Extract comparable monthly USD/COP derivative aggregates.

    BanRep changed the report layout over time.  Narrative and table fallbacks
    are deliberately combined; absent fields remain absent rather than being
    imputed in the acquisition layer.
    """
    normalized = _pdf_ascii(text)
    section = normalized
    # The table of contents repeats the same heading.  The last occurrence is
    # the actual report section, not the TOC entry.
    start_at = normalized.lower().rfind("2.1.1 tamano")
    if start_at >= 0:
        section = normalized[start_at:]
    end_at = section.lower().find("2.2 mercado")
    if end_at >= 0:
        section = section[:end_at]
    flat = re.sub(r"\s+", " ", section)
    result: dict[str, float] = {}

    patterns = {
        "br_forward_monthly_short_tenor_share": (
            r"plazos inferiores a 36 dias representaron el\s*([\d.,]+)\s*%",
            0.01,
        ),
        "br_forward_monthly_outstanding_usd_m": (
            r"contratos\s*forward\s*vigentes ascendian a USD\s*([\d.,]+)\s*m",
            1.0,
        ),
        "br_forward_monthly_weighted_devaluation_pct": (
            r"promedio de devaluacion implicita ponderado por monto.*?fue de\s*([\d.,]+)\s*%",
            1.0,
        ),
    }
    for series_id, (pattern, scale) in patterns.items():
        match = re.search(pattern, flat, flags=re.IGNORECASE)
        if match:
            result[series_id] = parse_number(match.group(1)) * scale

    amount_match = re.search(
        r"monto pactado en el mercado\s*forward\s*se.*?"
        r"(?:al pasar de USD\s*[\d.,]+\s*m.*?a USD\s*([\d.,]+)\s*m|"
        r"fue de USD\s*([\d.,]+)\s*m)",
        flat,
        flags=re.IGNORECASE,
    )
    if amount_match:
        result["br_forward_monthly_volume_usd_m"] = parse_number(
            amount_match.group(1) or amount_match.group(2)
        )
    average_match = re.search(
        r"monto promedio diario.*?(?:de USD\s*[\d.,]+\s*m\s+a USD\s*([\d.,]+)\s*m|"
        r"fue de USD\s*([\d.,]+)\s*m)",
        flat,
        flags=re.IGNORECASE,
    )
    if average_match:
        result["br_forward_monthly_avg_daily_usd_m"] = parse_number(
            average_match.group(1) or average_match.group(2)
        )

    table = section
    if "Cuadro 2:" in normalized:
        table = normalized.split("Cuadro 2:", 1)[1]
        table = table.split("2.1.2", 1)[0]
    for label, prefix in (
        ("IMC", "imc"),
        ("Extranjero", "foreign"),
        ("FPC", "pension"),
        ("Total", "total"),
    ):
        values = _parse_named_table_row(table, label)
        if not values:
            continue
        result[f"br_forward_monthly_{prefix}_buy_usd_m"] = values[0]
        result[f"br_forward_monthly_{prefix}_sell_usd_m"] = values[1]
        result[f"br_forward_monthly_{prefix}_net_usd_m"] = values[0] - values[1]
        if prefix == "total" and "br_forward_monthly_volume_usd_m" not in result:
            result["br_forward_monthly_volume_usd_m"] = values[0]
    return {key: value for key, value in result.items() if np.isfinite(value)}


def _find_row(frame: pd.DataFrame, label: str, start: int = 0) -> int | None:
    wanted = _compact(label).lower()
    for index in range(start, len(frame)):
        values = {_compact(value).lower() for value in frame.iloc[index].dropna()}
        if wanted in values:
            return index
    return None


def parse_eme_workbook(path: Path) -> tuple[dict[str, tuple[float, pd.Timestamp | None]], pd.Timestamp | None]:
    """Parse the all-participant TRM block from an official EME workbook."""
    workbook = pd.ExcelFile(path)
    sheet = next((name for name in workbook.sheet_names if "trm" in name.lower()), None)
    if sheet is None:
        raise ValueError(f"EME workbook has no TRM sheet: {path.name}")
    frame = pd.read_excel(path, sheet_name=sheet, header=None)
    if frame.empty:
        raise ValueError(f"Empty TRM sheet in {path.name}")

    survey_text = next(
        (
            _compact(value)
            for value in frame.iloc[:10].to_numpy().ravel()
            if pd.notna(value) and "fecha de realizacion" in _compact(value).lower()
        ),
        "",
    )
    date_tokens = re.findall(
        r"\d{1,2}\s+de\s+[A-Za-záéíóúñ]+(?:\s+de\s+\d{4})?", survey_text,
        flags=re.IGNORECASE,
    )
    survey_end = parse_spanish_date(date_tokens[-1]) if date_tokens else None

    header_row = next(
        (
            index
            for index in range(min(12, len(frame)))
            if any(
                "medidas estadisticas" in _compact(value).lower()
                for value in frame.iloc[index].dropna()
            )
        ),
        None,
    )
    if header_row is None:
        raise ValueError(f"Could not locate statistical header in {path.name}")
    # Depending on the vintage, the dates are on the statistical-header row or
    # one row below it.  Only columns with an actual target date are forecasts;
    # adjacent columns contain percentage changes and must not be mistaken for
    # additional horizons.
    dated_columns: dict[int, pd.Timestamp] = {}
    for row_index in range(header_row, min(header_row + 3, len(frame))):
        for column in range(frame.shape[1]):
            raw = frame.iloc[row_index, column]
            if pd.isna(raw):
                continue
            parsed = parse_spanish_date(raw)
            if parsed is not None:
                dated_columns[column] = parsed
    target_columns: list[tuple[int, pd.Timestamp | None]] = sorted(dated_columns.items())
    if not target_columns:
        # Legacy sheets often merge headings and leave every other numeric column.
        mean_row = _find_row(frame, "Media", header_row)
        if mean_row is None:
            raise ValueError(f"Could not locate EME target columns in {path.name}")
        target_columns = [
            (column, None) for column in range(1, frame.shape[1])
            if np.isfinite(parse_number(frame.iloc[mean_row, column]))
        ]

    start = _find_row(frame, "TODAS LAS ENTIDADES PARTICIPANTES", header_row) or header_row
    rows = {
        "mean": _find_row(frame, "Media", start),
        "median": _find_row(frame, "Mediana", start),
        "std": _find_row(frame, "Desviacion estandar", start),
        "participants": _find_row(frame, "Numero de participantes", start),
    }
    if rows["mean"] is None:
        raise ValueError(f"Could not locate EME all-participant mean in {path.name}")

    # Derive horizon names from target dates, not column positions. In December
    # the year-end and 12-month columns collapse onto the same sequence, while
    # in other months the workbook publishes both. Positional naming therefore
    # silently mixed year-end forecasts with true 12-month forecasts.
    horizon_columns: dict[str, tuple[int, pd.Timestamp | None]] = {}
    dated_targets = [(column, target) for column, target in target_columns if target is not None]
    if dated_targets:
        near_column, near_target = min(dated_targets, key=lambda item: item[1])
        horizon_columns["near"] = (near_column, near_target)
        for column, target in dated_targets:
            month_distance = (target.year - near_target.year) * 12 + target.month - near_target.month
            if target.month == 12 and target.year == near_target.year:
                horizon_columns.setdefault("year_end", (column, target))
            if month_distance == 12:
                horizon_columns.setdefault("12m", (column, target))
            if target.month == 12 and target.year == near_target.year + 1:
                horizon_columns.setdefault("next_year_end", (column, target))
            if month_distance == 24:
                horizon_columns.setdefault("24m", (column, target))
    else:
        fallback_names = ("near", "year_end", "12m", "next_year_end", "24m")
        horizon_columns = {
            name: target
            for name, target in zip(fallback_names, target_columns, strict=False)
        }
    result: dict[str, tuple[float, pd.Timestamp | None]] = {}
    for horizon, (column, target_date) in horizon_columns.items():
        for measure, row in rows.items():
            if row is None:
                continue
            value = parse_number(frame.iloc[row, column])
            if np.isfinite(value):
                result[f"br_eme_usdcop_{horizon}_{measure}"] = (value, target_date)
    return result, survey_end


def parse_sfc_formato_415(path: Path) -> tuple[pd.Timestamp, dict[str, float]]:
    required = {
        "Fecha Corte",
        "Código moneda derecho",
        "Nominal derecho",
        "Código moneda obligación",
        "Nominal obligación",
        "Valor del derecho (COP)",
        "Valor de la obligación (COP)",
    }
    workbook = pd.ExcelFile(path)
    sheet = next(
        (
            name
            for name in workbook.sheet_names
            if "415" in re.sub(r"[^a-z0-9]", "", _compact(name).lower())
        ),
        None,
    )
    if sheet is None:
        raise ValueError(f"Formato 415 sheet not found in {path.name}")
    frame = pd.read_excel(
        path,
        sheet_name=sheet,
        usecols=lambda column: _compact(column) in {_compact(item) for item in required},
    )
    normalized = {_compact(column).lower(): column for column in frame.columns}

    def column(name: str) -> str:
        found = normalized.get(_compact(name).lower())
        if found is None:
            raise ValueError(f"Formato 415 missing column {name!r} in {path.name}")
        return found

    cut = pd.to_datetime(frame[column("Fecha Corte")], errors="coerce").dropna()
    if cut.empty:
        raise ValueError(f"Formato 415 has no cut-off date: {path.name}")
    observation = pd.Timestamp(cut.max()).normalize()
    right_ccy = frame[column("Código moneda derecho")].astype(str).str.upper().str.strip()
    obligation_ccy = frame[column("Código moneda obligación")].astype(str).str.upper().str.strip()
    right_nominal = pd.to_numeric(frame[column("Nominal derecho")], errors="coerce").fillna(0.0)
    obligation_nominal = pd.to_numeric(
        frame[column("Nominal obligación")], errors="coerce"
    ).fillna(0.0)
    rights_cop = pd.to_numeric(
        frame[column("Valor del derecho (COP)")], errors="coerce"
    ).fillna(0.0)
    obligations_cop = pd.to_numeric(
        frame[column("Valor de la obligación (COP)")], errors="coerce"
    ).fillna(0.0)
    long_mask = (right_ccy == "USD") & (obligation_ccy == "COP")
    short_mask = (right_ccy == "COP") & (obligation_ccy == "USD")
    pair_mask = long_mask | short_mask
    long_usd = float(right_nominal[long_mask].sum() / 1_000_000)
    short_usd = float(obligation_nominal[short_mask].sum() / 1_000_000)
    values = {
        "sfc_pension_deriv_usd_long_m": long_usd,
        "sfc_pension_deriv_usd_short_m": short_usd,
        "sfc_pension_deriv_usd_net_m": long_usd - short_usd,
        "sfc_pension_deriv_usd_gross_m": long_usd + short_usd,
        "sfc_pension_deriv_fair_value_net_cop_bn": float(
            (rights_cop[pair_mask] - obligations_cop[pair_mask]).sum() / 1_000_000_000
        ),
    }
    return observation, values


FORWARD_HISTORY_COUNTERPARTIES = {
    "Extranjero": "foreign",
    "FPC": "pension",
    "Real": "real_sector",
    "Aseguradora": "insurer",
    "Fiduciaria": "trust",
    "Persona Natural": "individual",
    "Resto": "other",
    "IMC": "interbank",
}

FORWARD_HISTORY_TENORS = {
    "4 a 14": "2w",
    "15 a 35": "1m",
    "36 a 60": "2m",
    "61 a 90": "3m",
    "91 a 180": "6m",
    "mayor a 180": "long",
}


def parse_forward_history_workbook(
    path: Path,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Parse BanRep's consolidated daily USD/COP forward workbook.

    The returned frame is deliberately compact and stationary-friendly.  It
    contains daily net-position/volume ratios by counterparty, outstanding
    balances, and the market-implied devaluation curve.  The workbook is a
    mutable current snapshot; availability and vintage semantics are assigned
    by :meth:`ForwardMacroScraper.scrape_forward_history`, not by this parser.
    """
    path = Path(path)
    position = pd.read_excel(
        path,
        sheet_name="3. PosicionDiaria",
        header=8,
        usecols="A:K",
    ).dropna(how="all")
    balance = pd.read_excel(
        path,
        sheet_name="4. SaldoDiario",
        header=6,
        usecols="A:I",
    ).dropna(how="all")
    devaluation = pd.read_excel(
        path,
        sheet_name="6. DevaluacionesSectorTotal",
        header=6,
        usecols="A:M",
    ).dropna(how="all")

    for frame in (position, balance, devaluation):
        frame.columns = [_compact(column) for column in frame.columns]
        frame["Fecha"] = pd.to_datetime(frame["Fecha"], errors="coerce").dt.normalize()
        frame.dropna(subset=["Fecha"], inplace=True)
        if start is not None:
            frame.drop(frame.index[frame["Fecha"] < pd.Timestamp(start).normalize()], inplace=True)
        if end is not None:
            frame.drop(frame.index[frame["Fecha"] > pd.Timestamp(end).normalize()], inplace=True)

    numeric_position = ["PosicionNeta", "MontosNegociados"]
    for column in numeric_position:
        position[column] = pd.to_numeric(position[column], errors="coerce")
    for column in balance.columns[1:]:
        balance[column] = pd.to_numeric(balance[column], errors="coerce")
    for column in devaluation.columns[3:]:
        devaluation[column] = pd.to_numeric(devaluation[column], errors="coerce")

    dates = sorted(
        set(position["Fecha"]) | set(balance["Fecha"]) | set(devaluation["Fecha"])
    )
    daily = pd.DataFrame(index=pd.DatetimeIndex(dates, name="observation_date"))

    for official_name, slug in FORWARD_HISTORY_COUNTERPARTIES.items():
        counterpart = position[position["Contraparte"].eq(official_name)]
        aggregate = counterpart.groupby("Fecha")[numeric_position].sum(min_count=1)
        net_name = f"br_forward_history_{slug}_net_usd_m"
        volume_name = f"br_forward_history_{slug}_volume_usd_m"
        ratio_name = f"br_forward_history_{slug}_net_ratio"
        daily[net_name] = aggregate["PosicionNeta"]
        daily[volume_name] = aggregate["MontosNegociados"]
        daily[ratio_name] = aggregate["PosicionNeta"].div(
            aggregate["MontosNegociados"].abs().replace(0.0, np.nan)
        )

        ndf = counterpart[counterpart["Modalidad"].eq("NDF")]
        ndf_aggregate = ndf.groupby("Fecha")[numeric_position].sum(min_count=1)
        ndf_net = f"br_forward_history_{slug}_ndf_net_usd_m"
        ndf_volume = f"br_forward_history_{slug}_ndf_volume_usd_m"
        ndf_ratio = f"br_forward_history_{slug}_ndf_net_ratio"
        daily[ndf_net] = ndf_aggregate["PosicionNeta"]
        daily[ndf_volume] = ndf_aggregate["MontosNegociados"]
        daily[ndf_ratio] = ndf_aggregate["PosicionNeta"].div(
            ndf_aggregate["MontosNegociados"].abs().replace(0.0, np.nan)
        )

        if official_name in balance.columns:
            balance_series = balance.groupby("Fecha")[official_name].last()
            daily[f"br_forward_history_{slug}_balance_usd_m"] = balance_series

    market = devaluation[
        devaluation["Reportante"].eq("IMC")
        & devaluation["Rango"].isin(FORWARD_HISTORY_TENORS)
    ]
    for official_tenor, slug in FORWARD_HISTORY_TENORS.items():
        tenor = market[market["Rango"].eq(official_tenor)].groupby("Fecha").last()
        daily[f"br_forward_history_market_devaluation_{slug}"] = tenor["Mercado"]
        for official_name, counterparty_slug in {
            "Extranjero": "foreign",
            "FPC": "pension",
            "Real": "real_sector",
        }.items():
            daily[
                f"br_forward_history_{counterparty_slug}_devaluation_{slug}"
            ] = tenor[official_name]

    daily = daily.sort_index().replace([np.inf, -np.inf], np.nan)
    daily = daily.dropna(how="all").reset_index()
    return daily


@dataclass
class DownloadedDocument:
    url: str
    path: Path
    sha256: str
    retrieved_at: pd.Timestamp
    last_modified: pd.Timestamp | None = None


@dataclass
class IngestionResult:
    extracted: pd.DataFrame
    combined: pd.DataFrame
    manifest_path: Path
    output_path: Path
    errors: list[str] = field(default_factory=list)
    documents_seen: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "rows_extracted": int(len(self.extracted)),
            "rows_total": int(len(self.combined)),
            "series": int(self.combined["series_id"].nunique()) if not self.combined.empty else 0,
            "documents_seen": self.documents_seen,
            "errors": len(self.errors),
            "output": str(self.output_path),
            "manifest": str(self.manifest_path),
        }


class ForwardMacroScraper:
    """Incremental, idempotent official-document scraper."""

    def __init__(
        self,
        project_root: Path = ROOT,
        config_path: Path = DEFAULT_CONFIG,
        raw_dir: Path = DEFAULT_RAW,
        output_path: Path = DEFAULT_OUTPUT,
        manifest_dir: Path = DEFAULT_MANIFEST_DIR,
        force: bool = False,
        offline: bool = False,
    ) -> None:
        if requests is None:
            raise RuntimeError(
                "Missing scraper dependencies. Install the project data extra: "
                "pip install -e '.[data]'"
            )
        self.project_root = Path(project_root)
        self.config_path = Path(config_path)
        self.raw_dir = Path(raw_dir)
        self.output_path = Path(output_path)
        self.manifest_dir = Path(manifest_dir)
        self.force = force
        self.offline = offline
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        global_config = self.config.get("global", {})
        self.timeout = int(global_config.get("request_timeout_seconds", 60))
        self.delay = float(global_config.get("rate_limit_delay_seconds", 0.15))
        self.session = requests.Session()
        retry = Retry(
            total=int(global_config.get("retry_attempts", 4)),
            backoff_factor=float(global_config.get("retry_backoff_seconds", 1.0)),
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.headers.update(
            {
                "User-Agent": global_config.get(
                    "user_agent",
                    "USDCOP-Research-PIT/1.0 (+official public data ingestion)",
                ),
                "Accept-Language": "es-CO,es;q=0.9,en;q=0.7",
            }
        )
        self.errors: list[str] = []
        self.documents_seen = 0

    def _download(self, url: str, destination: Path) -> DownloadedDocument:
        destination.parent.mkdir(parents=True, exist_ok=True)
        sidecar = destination.with_suffix(destination.suffix + ".meta.json")
        if destination.exists() and destination.stat().st_size > 0 and not self.force:
            metadata: dict[str, Any] = {}
            if sidecar.exists():
                try:
                    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    metadata = {}
            retrieved = pd.Timestamp(
                metadata.get("retrieved_at", datetime.fromtimestamp(destination.stat().st_mtime, UTC))
            )
            if retrieved.tzinfo is None:
                retrieved = retrieved.tz_localize("UTC")
            last_modified = pd.to_datetime(
                metadata.get("last_modified"), errors="coerce", utc=True
            )
            return DownloadedDocument(
                url=url,
                path=destination,
                sha256=sha256_file(destination),
                retrieved_at=retrieved,
                last_modified=None if pd.isna(last_modified) else pd.Timestamp(last_modified),
            )

        if self.offline:
            raise FileNotFoundError(f"offline cache miss: {destination}")

        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        if not response.content:
            raise RuntimeError(f"Empty response from {url}")
        partial = destination.with_suffix(destination.suffix + ".part")
        partial.write_bytes(response.content)
        partial.replace(destination)
        retrieved = pd.Timestamp.now(tz="UTC")
        header = response.headers.get("Last-Modified")
        last_modified = None
        if header:
            try:
                last_modified = pd.Timestamp(parsedate_to_datetime(header)).tz_convert("UTC")
            except (TypeError, ValueError):
                last_modified = None
        metadata = {
            "url": url,
            "retrieved_at": retrieved.isoformat(),
            "last_modified": last_modified.isoformat() if last_modified is not None else None,
            "content_type": response.headers.get("Content-Type"),
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }
        sidecar.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        time.sleep(self.delay)
        return DownloadedDocument(
            url=url,
            path=destination,
            sha256=metadata["sha256"],
            retrieved_at=retrieved,
            last_modified=last_modified,
        )

    def _record_rows(
        self,
        values: Mapping[str, float | tuple[float, pd.Timestamp | None]],
        observation: pd.Timestamp,
        available_at: pd.Timestamp,
        document: DownloadedDocument,
        frequency: str,
        source: str,
        policy: str,
        pit_vintage: bool,
        units: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        release_date = available_at.tz_convert("America/Bogota").date()
        rows = []
        for series_id, raw in values.items():
            if isinstance(raw, tuple):
                value, reference_date = raw
            else:
                value, reference_date = raw, None
            if not np.isfinite(value):
                continue
            rows.append(
                {
                    "series_id": series_id,
                    "observation_date": observation.date(),
                    "reference_date": (
                        pd.Timestamp(reference_date).date() if reference_date is not None else None
                    ),
                    "release_date": release_date,
                    "available_at": available_at,
                    "value": float(value),
                    "frequency": frequency,
                    "unit": (units or {}).get(series_id, _infer_unit(series_id)),
                    "source": source,
                    "source_url": document.url,
                    "document_sha256": document.sha256,
                    "retrieved_at": document.retrieved_at,
                    "availability_policy": policy,
                    "pit_vintage": bool(pit_vintage),
                    "promotion_eligible": bool(pit_vintage),
                    "metadata_json": json.dumps(metadata or {}, sort_keys=True, default=str),
                }
            )
        return rows

    def _bulletins(self) -> list[dict[str, Any]]:
        source = self.config["sources"]["banrep_bulletins"]
        url = f"{source['base_url'].rstrip('/')}/{source['endpoint'].lstrip('/')}"
        document = self._download(url, self.raw_dir / "banrep" / "boletines.json")
        payload = json.loads(document.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Unexpected SUAMECA bulletin response")
        return payload

    def scrape_daily_forwards(
        self, bulletins: Sequence[Mapping[str, Any]], start: pd.Timestamp, end: pd.Timestamp
    ) -> list[dict[str, Any]]:
        configured = self.config["sources"]["banrep_daily_forward"]
        bulletin_type = configured["bulletin_type"]
        documents = [
            item for item in bulletins
            if item.get("tipoBoletin") == bulletin_type
            and start <= pd.Timestamp(item["fechaBoletin"]) <= end
        ]
        rows: list[dict[str, Any]] = []
        for index, item in enumerate(sorted(documents, key=lambda item: item["fechaBoletin"])):
            observation = pd.Timestamp(item["fechaBoletin"]).normalize()
            url = str(item["rutaReporte"])
            path = self.raw_dir / "banrep" / "daily_forward" / f"{observation:%Y}" / f"{observation:%Y-%m-%d}.pdf"
            try:
                document = self._download(url, path)
                self.documents_seen += 1
                text, embedded = _pdf_text_and_created(path)
                values = parse_daily_forward_text(text)
                if not values:
                    raise ValueError("no daily-forward fields parsed")
                available = _valid_embedded_timestamp(embedded, observation, 14)
                policy = "official_pdf_creation_timestamp"
                pit = available is not None
                if available is None:
                    publication = parse_daily_publication_date(text)
                    if publication is None:
                        publication = observation + pd.Timedelta(days=3)
                        policy = "observation_plus_3d_conservative"
                    else:
                        policy = "official_publication_date_eod_conservative"
                    available = _bogota_to_utc(publication, end_of_day=True)
                rows.extend(
                    self._record_rows(
                        values, observation, available, document, "daily", "banrep",
                        policy, pit, metadata={"bulletin_type": bulletin_type}
                    )
                )
            except Exception as exc:  # individual documents must not abort the backfill
                self.errors.append(f"daily_forward {observation.date()}: {exc}")
            if (index + 1) % 50 == 0:
                LOGGER.info("Daily forward documents processed: %d/%d", index + 1, len(documents))
        return rows

    def scrape_forward_history(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> list[dict[str, Any]]:
        """Snapshot the consolidated BanRep forward history with honest vintages.

        The first local snapshot is a mutable-history bootstrap and is therefore
        never promotion eligible.  On later runs, only new observations or
        changed values are emitted, stamped at their actual local first-seen
        retrieval time.  This turns future workbook updates into a real vintage
        ledger without pretending that today's revised workbook existed in the
        past.
        """
        configured = self.config["sources"]["banrep_forward_history"]
        url = str(configured["url"])
        snapshot_day = pd.Timestamp.now(tz="America/Bogota").strftime("%Y-%m-%d")
        path = (
            self.raw_dir / "banrep" / "forward_history" / "snapshots"
            / f"{snapshot_day}.xlsx"
        )
        document = self._download(url, path)
        self.documents_seen += 1
        parsed = parse_forward_history_workbook(document.path, start, end)
        if parsed.empty:
            raise ValueError("consolidated forward workbook produced no daily rows")

        existing = pd.DataFrame(columns=PIT_COLUMNS)
        if self.output_path.exists():
            existing = _coerce_pit_frame(pd.read_parquet(self.output_path))
            existing = existing[existing["source"].eq("banrep_forward_history")]
        bootstrap = existing.empty
        previous: dict[tuple[str, date], float] = {}
        if not existing.empty:
            latest = (
                existing.sort_values(["available_at", "retrieved_at"])
                .drop_duplicates(["series_id", "observation_date"], keep="last")
            )
            previous = {
                (str(row.series_id), row.observation_date): float(row.value)
                for row in latest.itertuples(index=False)
            }

        lag_days = int(configured.get("reconstructed_availability_lag_calendar_days", 5))
        units = {
            column: (
                "ratio"
                if column.endswith("_ratio") or "_devaluation_" in column
                else "USD_million"
            )
            for column in parsed.columns
            if column != "observation_date"
        }
        rows: list[dict[str, Any]] = []
        for record in parsed.itertuples(index=False):
            observation = pd.Timestamp(record.observation_date).normalize()
            values = {
                series_id: float(value)
                for series_id, value in record._asdict().items()
                if series_id != "observation_date" and pd.notna(value)
            }
            if bootstrap:
                changed = values
                available = _bogota_to_utc(
                    observation + pd.Timedelta(days=lag_days), end_of_day=True
                )
                policy = f"observation_plus_{lag_days}d_reconstructed_research_only"
                pit_vintage = False
            else:
                changed = {
                    series_id: value
                    for series_id, value in values.items()
                    if (
                        (series_id, observation.date()) not in previous
                        or not np.isclose(
                            previous[(series_id, observation.date())],
                            value,
                            rtol=1e-10,
                            atol=1e-12,
                            equal_nan=True,
                        )
                    )
                }
                if not changed:
                    continue
                available = document.retrieved_at
                policy = "local_first_seen_snapshot_timestamp"
                pit_vintage = True
            rows.extend(
                self._record_rows(
                    changed,
                    observation,
                    available,
                    document,
                    "daily",
                    "banrep_forward_history",
                    policy,
                    pit_vintage,
                    units=units,
                    metadata={
                        "workbook": "series-historico-forward-desde-2016",
                        "snapshot_day": snapshot_day,
                        "initial_mutable_history_bootstrap": bootstrap,
                        "revision_risk": "official workbook is revised in place",
                    },
                )
            )
        return rows

    def scrape_monthly_derivatives(
        self, bulletins: Sequence[Mapping[str, Any]], start: pd.Timestamp, end: pd.Timestamp
    ) -> list[dict[str, Any]]:
        configured = self.config["sources"]["banrep_monthly_derivatives"]
        bulletin_type = configured["bulletin_type"]
        documents = [
            item for item in bulletins
            if item.get("tipoBoletin") == bulletin_type
            and start <= pd.Timestamp(item["fechaBoletin"]) <= end
        ]
        rows: list[dict[str, Any]] = []
        for item in sorted(documents, key=lambda item: item["fechaBoletin"]):
            observation = pd.Timestamp(item["fechaBoletin"]).normalize()
            url = str(item["rutaReporte"])
            path = self.raw_dir / "banrep" / "monthly_derivatives" / f"{observation:%Y}" / f"{observation:%Y-%m}.pdf"
            try:
                document = self._download(url, path)
                self.documents_seen += 1
                text, embedded = _pdf_text_and_created(path)
                values = parse_monthly_derivatives_text(text)
                if not values:
                    raise ValueError("no monthly-derivative fields parsed")
                available = _valid_embedded_timestamp(embedded, observation, 90)
                pit = available is not None
                policy = "official_pdf_creation_timestamp"
                if available is None:
                    available = _conservative_month_release(observation, days=45)
                    policy = "observation_plus_45d_conservative"
                rows.extend(
                    self._record_rows(
                        values, observation, available, document, "monthly", "banrep",
                        policy, pit, metadata={"bulletin_type": bulletin_type}
                    )
                )
            except Exception as exc:
                self.errors.append(f"monthly_derivatives {observation.date()}: {exc}")
        return rows

    def _eme_candidates(self, observation: pd.Timestamp) -> list[str]:
        configured = self.config["sources"]["banrep_eme"]
        abbr = MONTH_ABBR[observation.month]
        year = observation.year
        relatives = (
            f"res_inf_{abbr}{year}",
            f"res_inf_{abbr}{str(year)[-2:]}",
            f"res_inf_{abbr}_{year}",
        )
        bases = configured["candidate_base_urls"]
        urls = []
        for base in bases:
            for relative in relatives:
                for extension in ("xlsx", "xls"):
                    urls.append(f"{base.rstrip('/')}/{relative}.{extension}")
        return list(dict.fromkeys(urls))

    def scrape_eme(
        self, bulletins: Sequence[Mapping[str, Any]], start: pd.Timestamp, end: pd.Timestamp
    ) -> list[dict[str, Any]]:
        configured = self.config["sources"]["banrep_eme"]
        ranking_type = configured["month_index_bulletin_type"]
        months = sorted(
            {
                pd.Timestamp(item["fechaBoletin"]).to_period("M").to_timestamp("M")
                for item in bulletins
                if item.get("tipoBoletin") == ranking_type
                and start.to_period("M") <= pd.Timestamp(item["fechaBoletin"]).to_period("M")
                <= end.to_period("M")
            }
        )
        rows: list[dict[str, Any]] = []
        for observation in months:
            document = None
            last_error: Exception | None = None
            month_dir = self.raw_dir / "banrep" / "eme" / f"{observation:%Y}"
            cached_paths = [
                candidate
                for extension in (".xlsx", ".xls")
                for candidate in month_dir.glob(f"{observation:%Y-%m}{extension}")
                if candidate.is_file() and candidate.stat().st_size > 0
            ]
            # Prefer an already archived official workbook before probing URL
            # variants. This matters because BanRep changed filename conventions
            # over time and avoids a failed .xlsx request when the .xls vintage
            # is already present locally with its provenance sidecar.
            if cached_paths and not self.force:
                cached = cached_paths[0]
                sidecar = cached.with_suffix(cached.suffix + ".meta.json")
                metadata: dict[str, Any] = {}
                if sidecar.exists():
                    try:
                        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        metadata = {}
                cached_url = str(metadata.get("url") or self._eme_candidates(observation)[0])
                document = self._download(cached_url, cached)
            for url in self._eme_candidates(observation):
                if document is not None:
                    break
                extension = Path(url).suffix.lower()
                path = self.raw_dir / "banrep" / "eme" / f"{observation:%Y}" / f"{observation:%Y-%m}{extension}"
                try:
                    document = self._download(url, path)
                    break
                except Exception as exc:
                    last_error = exc
            if document is None:
                self.errors.append(f"eme {observation:%Y-%m}: {last_error}")
                continue
            try:
                self.documents_seen += 1
                values, survey_end = parse_eme_workbook(document.path)
                record_observation = survey_end if survey_end is not None else observation
                embedded = _xlsx_created(document.path) or _xls_created(document.path)
                available = _valid_embedded_timestamp(embedded, record_observation, 60)
                pit = available is not None
                policy = "official_workbook_modified_timestamp"
                if available is None and document.last_modified is not None:
                    available = _valid_embedded_timestamp(document.last_modified, record_observation, 60)
                    pit = available is not None
                    policy = "official_http_last_modified"
                if available is None:
                    available = _bogota_to_utc(observation, end_of_day=True)
                    policy = "observation_month_end_conservative"
                metadata = {
                    "survey_end": survey_end.date().isoformat() if survey_end is not None else None,
                    "survey_period": observation.strftime("%Y-%m"),
                }
                rows.extend(
                    self._record_rows(
                        values, record_observation, available, document, "monthly", "banrep_eme",
                        policy, pit, metadata=metadata
                    )
                )
            except Exception as exc:
                self.errors.append(f"eme parse {observation:%Y-%m}: {exc}")
        return rows

    def _socrata_aggregate(self, nominal: str, where: str) -> pd.DataFrame:
        configured = self.config["sources"]["sfc_socrata_468"]
        url = configured["endpoint"]
        params = {
            "$select": f"fecha_corte,sum({nominal}) as total",
            "$where": where,
            "$group": "fecha_corte",
            "$order": "fecha_corte",
            "$limit": "50000",
        }
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        frame = pd.DataFrame(payload)
        if frame.empty:
            return pd.DataFrame(columns=["observation_date", "total"])
        frame["observation_date"] = pd.to_datetime(frame["fecha_corte"], errors="coerce").dt.normalize()
        frame["total"] = pd.to_numeric(frame["total"], errors="coerce") / 1_000_000
        return frame[["observation_date", "total"]].dropna()

    def scrape_sfc_socrata(self, start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, Any]]:
        configured = self.config["sources"]["sfc_socrata_468"]
        try:
            long_frame = self._socrata_aggregate(
                "nominal_moneda_comprada",
                "codigo_moneda_comprada='USD' AND cod_moneda_vendida='PESO'",
            ).rename(columns={"total": "long"})
            short_frame = self._socrata_aggregate(
                "nominal_moneda_vendida",
                "codigo_moneda_comprada='PESO' AND cod_moneda_vendida='USD'",
            ).rename(columns={"total": "short"})
            combined = long_frame.merge(short_frame, on="observation_date", how="outer").fillna(0.0)
            combined = combined[
                (combined["observation_date"] >= start) & (combined["observation_date"] <= end)
            ]
            raw_path = self.raw_dir / "sfc" / "socrata_468_aggregated.csv"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(raw_path, index=False)
            document = DownloadedDocument(
                url=configured["endpoint"], path=raw_path, sha256=sha256_file(raw_path),
                retrieved_at=pd.Timestamp.now(tz="UTC"), last_modified=None,
            )
            self.documents_seen += 1
            rows: list[dict[str, Any]] = []
            for item in combined.itertuples(index=False):
                observation = pd.Timestamp(item.observation_date)
                values = {
                    "sfc_pension_deriv_usd_long_m": item.long,
                    "sfc_pension_deriv_usd_short_m": item.short,
                    "sfc_pension_deriv_usd_net_m": item.long - item.short,
                    "sfc_pension_deriv_usd_gross_m": item.long + item.short,
                }
                available = _conservative_month_release(observation, days=int(configured["conservative_release_lag_days"]))
                rows.extend(
                    self._record_rows(
                        values, observation, available, document, "monthly", "sfc_open_data",
                        "observation_plus_45d_reconstructed", False,
                        metadata={"dataset_id": configured["dataset_id"]},
                    )
                )
            return rows
        except Exception as exc:
            self.errors.append(f"sfc_socrata_468: {exc}")
            return []

    def _discover_sfc_workbooks(self, start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, str]]:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("beautifulsoup4 is required for SFC workbook discovery") from exc
        configured = self.config["sources"]["sfc_formato_415"]
        pages: Mapping[Any, str] = configured["year_pages"]
        found: dict[pd.Timestamp, str] = {}
        for year in range(max(start.year, int(configured["start_year"])), end.year + 1):
            page_url = pages.get(year) or pages.get(str(year))
            if not page_url:
                continue
            response = self.session.get(page_url, timeout=self.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for anchor in soup.find_all("a", href=True):
                month = SPANISH_MONTHS.get(_compact(anchor.get_text(" ")).lower())
                href = str(anchor.get("href"))
                if month is None or "loader.php" not in href:
                    continue
                observation = pd.Timestamp(year=year, month=month, day=1).to_period("M").to_timestamp("M")
                if start.to_period("M") <= observation.to_period("M") <= end.to_period("M"):
                    found[observation] = urljoin(page_url, href)
        return sorted(found.items())

    def _cached_sfc_workbooks(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> list[tuple[pd.Timestamp, str]]:
        root = self.raw_dir / "sfc" / "formato_415"
        found: dict[pd.Timestamp, str] = {}
        for path in sorted(root.glob("*/*.*")):
            if path.suffix.lower() not in {".xls", ".xlsx"}:
                continue
            try:
                observation = pd.Timestamp(path.stem + "-01").to_period("M").to_timestamp("M")
            except ValueError:
                continue
            if not (start.to_period("M") <= observation.to_period("M") <= end.to_period("M")):
                continue
            metadata: dict[str, Any] = {}
            sidecar = path.with_suffix(path.suffix + ".meta.json")
            if sidecar.exists():
                try:
                    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    metadata = {}
            found[observation] = str(metadata.get("url") or path.resolve())
        return sorted(found.items())

    def scrape_sfc_formato_415(self, start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, Any]]:
        configured = self.config["sources"]["sfc_formato_415"]
        cached = dict(self._cached_sfc_workbooks(start, end))
        discovered: dict[pd.Timestamp, str] = {}
        if not self.offline:
            try:
                discovered = dict(self._discover_sfc_workbooks(start, end))
            except Exception as exc:
                self.errors.append(f"sfc_formato_415 discovery: {exc}")
        documents = sorted({**cached, **discovered}.items())
        rows: list[dict[str, Any]] = []
        for expected_observation, url in documents:
            base = self.raw_dir / "sfc" / "formato_415" / f"{expected_observation:%Y}" / f"{expected_observation:%Y-%m}"
            path = next(
                (candidate for candidate in (base.with_suffix(".xls"), base.with_suffix(".xlsx")) if candidate.exists()),
                base.with_suffix(".xls"),
            )
            try:
                document = self._download(url, path)
                self.documents_seen += 1
                observation, values = parse_sfc_formato_415(path)
                embedded = _xls_created(path) or _xlsx_created(path)
                available = _valid_embedded_timestamp(embedded, observation, 90)
                pit = available is not None
                policy = "official_workbook_modified_timestamp"
                if available is None and document.last_modified is not None:
                    available = _valid_embedded_timestamp(document.last_modified, observation, 90)
                    pit = available is not None
                    policy = "official_http_last_modified"
                if available is None:
                    available = _conservative_month_release(
                        observation, int(configured["conservative_release_lag_days"])
                    )
                    policy = "observation_plus_45d_conservative"
                rows.extend(
                    self._record_rows(
                        values, observation, available, document, "monthly", "sfc",
                        policy, pit, metadata={"format": 415}
                    )
                )
            except Exception as exc:
                self.errors.append(f"sfc_formato_415 {expected_observation:%Y-%m}: {exc}")
        return rows

    def run(
        self,
        start: str | date | pd.Timestamp,
        end: str | date | pd.Timestamp,
        sources: Iterable[str] = (
            "daily_forward",
            "forward_history",
            "monthly_derivatives",
            "eme",
            "sfc",
        ),
    ) -> IngestionResult:
        requested_at = pd.Timestamp.now(tz="UTC")
        start_stamp = pd.Timestamp(start).normalize()
        end_stamp = pd.Timestamp(end).normalize()
        selected = set(sources)
        rows: list[dict[str, Any]] = []
        needs_bulletins = bool(selected & {"daily_forward", "monthly_derivatives", "eme"})
        bulletins = self._bulletins() if needs_bulletins else []
        if "daily_forward" in selected:
            rows.extend(self.scrape_daily_forwards(bulletins, start_stamp, end_stamp))
        if "forward_history" in selected:
            history_start = pd.Timestamp(
                self.config["sources"]["banrep_forward_history"].get(
                    "first_observation", start_stamp
                )
            ).normalize()
            rows.extend(self.scrape_forward_history(history_start, end_stamp))
        if "monthly_derivatives" in selected:
            rows.extend(self.scrape_monthly_derivatives(bulletins, start_stamp, end_stamp))
        if "eme" in selected:
            rows.extend(self.scrape_eme(bulletins, start_stamp, end_stamp))
        if selected & {"sfc", "sfc_socrata"}:
            if self.offline:
                self.errors.append("sfc_socrata_468: skipped in offline mode")
            else:
                rows.extend(self.scrape_sfc_socrata(start_stamp, end_stamp))
        if selected & {"sfc", "sfc_formato_415"}:
            rows.extend(self.scrape_sfc_formato_415(start_stamp, end_stamp))

        extracted = _coerce_pit_frame(pd.DataFrame(rows, columns=PIT_COLUMNS))
        existing = (
            _coerce_pit_frame(pd.read_parquet(self.output_path))
            if self.output_path.exists() else pd.DataFrame(columns=PIT_COLUMNS)
        )
        # Reprocessing a document replaces its former parsed representation.
        # This makes parser corrections idempotent even when observation-date
        # semantics improve (for example EME month-end -> survey-end).
        if not existing.empty and not extracted.empty:
            # An EME workbook is an atomic document. Clear all of its former
            # series before inserting a corrected horizon schema. API endpoints
            # such as Socrata are intentionally excluded because an incremental
            # query only contains a slice of that endpoint's history.
            eme_urls = set(
                extracted.loc[extracted["source"].eq("banrep_eme"), "source_url"]
                .dropna()
                .astype(str)
            )
            if eme_urls:
                existing = existing[~existing["source_url"].astype(str).isin(eme_urls)]
            # Consolidated-history snapshots are vintages.  Replacing by URL
            # would erase earlier first-seen values because every snapshot has
            # the same official URL.
            replaceable = extracted[
                ~extracted["source"].eq("banrep_forward_history")
            ]
            replacements = replaceable[["series_id", "source_url"]].drop_duplicates().assign(
                _replace=True
            )
            existing = existing.merge(
                replacements, on=["series_id", "source_url"], how="left"
            )
            existing = existing[existing["_replace"].isna()].drop(columns="_replace")
        combined = (
            extracted.copy() if existing.empty
            else existing.copy() if extracted.empty
            else pd.concat([existing, extracted], ignore_index=True)
        )
        if not combined.empty:
            combined = _coerce_pit_frame(combined)
            combined = combined.sort_values(["series_id", "observation_date", "available_at", "retrieved_at"])
            combined = combined.drop_duplicates(
                ["series_id", "observation_date", "available_at"], keep="last"
            ).reset_index(drop=True)
            invalid = combined["available_at"] < pd.to_datetime(combined["observation_date"], utc=True)
            if invalid.any():
                bad = combined.loc[invalid, ["series_id", "observation_date", "available_at"]]
                raise ValueError(f"PIT contract violation: available_at before observation: {bad.head().to_dict('records')}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.output_path.with_suffix(".tmp.parquet")
        combined.to_parquet(temp, index=False)
        temp.replace(self.output_path)
        csv_path = self.output_path.with_suffix(".csv")
        csv_temp = csv_path.with_suffix(".tmp.csv")
        combined.to_csv(csv_temp, index=False)
        csv_temp.replace(csv_path)

        completed_at = pd.Timestamp.now(tz="UTC")
        manifest = {
            "schema_version": 1,
            "run_id": f"usdcop-forward-macro-{requested_at:%Y%m%dT%H%M%SZ}",
            "provider": "public",
            "asset": "usdcop_forward_macro",
            "requested_at": requested_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "rows_received": int(len(rows)),
            "rows_accepted": int(len(extracted)),
            "rows_rejected": int(max(0, len(rows) - len(extracted))),
            "sha256": sha256_file(self.output_path),
            "frequency": "native_daily_monthly",
            "timezone": "America/Bogota",
            "available_at_policy": "official_document_timestamp_else_explicit_conservative_lag",
            "status": "success" if not self.errors else ("partial" if len(extracted) else "failed"),
            "error_count": len(self.errors),
            "retry_count": 0,
            "sources": sorted(selected),
            "start": start_stamp.date().isoformat(),
            "end": end_stamp.date().isoformat(),
            "documents_seen": self.documents_seen,
            "series_count": int(combined["series_id"].nunique()) if not combined.empty else 0,
            "pit_vintage_rows": int(combined["pit_vintage"].sum()) if not combined.empty else 0,
            "promotion_eligible": bool(
                not combined.empty and combined["promotion_eligible"].all()
            ),
            "errors": self.errors[:200],
            "output_path": str(self.output_path.relative_to(self.project_root)),
        }
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.manifest_dir / f"{manifest['run_id']}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        latest = self.manifest_dir / "usdcop_forward_macro_latest.json"
        latest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return IngestionResult(
            extracted=extracted,
            combined=combined,
            manifest_path=manifest_path,
            output_path=self.output_path,
            errors=list(self.errors),
            documents_seen=self.documents_seen,
        )


def _infer_unit(series_id: str) -> str:
    if series_id.endswith("_share") or series_id.endswith("_ratio"):
        return "ratio"
    if "_devaluation_" in series_id:
        return "ratio"
    if series_id.endswith("_pct"):
        return "percent"
    if series_id.endswith("_cop_bn"):
        return "COP_billion"
    if series_id.endswith("_m") or series_id.endswith("_usd_m"):
        return "USD_million"
    if series_id.endswith("participants"):
        return "count"
    if series_id.startswith("br_eme_usdcop"):
        return "COP_per_USD"
    return "value"


def _coerce_pit_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=PIT_COLUMNS)
    result = frame.copy()
    for column in PIT_COLUMNS:
        if column not in result:
            result[column] = None
    for column in ("observation_date", "reference_date", "release_date"):
        result[column] = pd.to_datetime(result[column], errors="coerce").dt.date
    for column in ("available_at", "retrieved_at"):
        result[column] = pd.to_datetime(result[column], errors="coerce", utc=True)
    result["value"] = pd.to_numeric(result["value"], errors="coerce")
    result["pit_vintage"] = result["pit_vintage"].astype("boolean").fillna(False).astype(bool)
    result["promotion_eligible"] = (
        result["promotion_eligible"].astype("boolean").fillna(False).astype(bool)
    )
    result = result.dropna(subset=["series_id", "observation_date", "available_at", "value"])
    return result[PIT_COLUMNS]


def upsert_pit_rows(connection: Any, frame: pd.DataFrame) -> int:
    """Idempotently upsert extracted rows into ``macro_indicators_pit``."""
    if frame.empty:
        return 0
    try:
        from psycopg2.extras import execute_values
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("psycopg2 is required for PostgreSQL PIT upsert") from exc
    columns = PIT_COLUMNS
    values = [tuple(None if pd.isna(row[column]) else row[column] for column in columns) for _, row in frame.iterrows()]
    sql = f"""
        INSERT INTO macro_indicators_pit ({', '.join(columns)})
        VALUES %s
        ON CONFLICT (series_id, observation_date, available_at)
        DO UPDATE SET
            reference_date = EXCLUDED.reference_date,
            release_date = EXCLUDED.release_date,
            value = EXCLUDED.value,
            frequency = EXCLUDED.frequency,
            unit = EXCLUDED.unit,
            source = EXCLUDED.source,
            source_url = EXCLUDED.source_url,
            document_sha256 = EXCLUDED.document_sha256,
            retrieved_at = EXCLUDED.retrieved_at,
            availability_policy = EXCLUDED.availability_policy,
            pit_vintage = EXCLUDED.pit_vintage,
            promotion_eligible = EXCLUDED.promotion_eligible,
            metadata_json = EXCLUDED.metadata_json::jsonb,
            updated_at = NOW()
    """
    with connection.cursor() as cursor:
        execute_values(cursor, sql, values, page_size=500)
    connection.commit()
    return len(values)


def _causal_rolling_z(
    values: pd.Series,
    *,
    window: int = 252,
    min_periods: int = 60,
    clip: float = 5.0,
) -> pd.Series:
    """Normalize a signal against past-only moments and cap data errors.

    The one-row shift is intentional: today's observation may enter the
    signal, but it cannot alter the mean or scale used to normalize itself.
    """
    history = pd.to_numeric(values, errors="coerce")
    mean = history.rolling(window, min_periods=min_periods).mean().shift(1)
    scale = history.rolling(window, min_periods=min_periods).std().shift(1)
    z_score = (history - mean) / scale.replace(0.0, np.nan)
    return z_score.clip(-clip, clip)


def attach_forward_macro_features(
    market_frame: pd.DataFrame,
    pit_path: Path = DEFAULT_OUTPUT,
    *,
    decision_hour_bogota: int = 16,
    promotion_only: bool = False,
    max_age_days: Mapping[str, int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Attach strictly as-of forward-looking features to a daily market frame."""
    frame = market_frame.copy()
    if not Path(pit_path).exists():
        return frame, []
    pit = _coerce_pit_frame(pd.read_parquet(pit_path))
    if promotion_only:
        pit = pit[pit["promotion_eligible"]]
    if pit.empty:
        return frame, []
    local = pd.to_datetime(frame["date"]).dt.normalize() + pd.Timedelta(hours=decision_hour_bogota)
    decision = local.dt.tz_localize("America/Bogota", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    left = pd.DataFrame({"decision_at": decision, "_row": np.arange(len(frame))}).sort_values("decision_at")
    age_limits = {"daily": 10, "monthly": 75}
    if max_age_days:
        age_limits.update({str(key): int(value) for key, value in max_age_days.items()})
    raw_columns: dict[str, str] = {}
    raw_values: dict[str, np.ndarray] = {}
    feature_names: list[str] = []
    for series_id, group in pit.groupby("series_id"):
        slug = re.sub(r"[^a-z0-9]+", "_", series_id.lower()).strip("_")
        frequencies = group["frequency"].dropna().astype(str).str.lower()
        frequency = frequencies.iloc[-1] if not frequencies.empty else "monthly"
        # A single release timestamp may contain several reference periods.  Do
        # not collapse those observations merely because they arrived together;
        # retain the period identity and only deduplicate an identical
        # (series, period, availability) vintage.
        group = group.sort_values(
            ["observation_date", "available_at", "retrieved_at"]
        ).drop_duplicates(
            ["series_id", "observation_date", "available_at"], keep="last"
        )
        group = group[["observation_date", "available_at", "value"]].copy()
        group["observation_date"] = pd.to_datetime(group["observation_date"])
        group[f"{slug}__release_delta"] = group["value"].diff()
        denominator = group["value"].shift(1).abs().replace(0, np.nan)
        group[f"{slug}__release_pct"] = group["value"].diff() / denominator
        group = group.rename(columns={"available_at": "published_at", "value": slug})
        merged = pd.merge_asof(
            left,
            group.sort_values("published_at"),
            left_on="decision_at",
            right_on="published_at",
            direction="backward",
        ).sort_values("_row")
        age_days = (
            merged["decision_at"] - merged["published_at"]
        ).dt.total_seconds() / 86_400
        stale = age_days > age_limits.get(frequency, 75)
        value_columns = [slug, f"{slug}__release_delta", f"{slug}__release_pct"]
        merged.loc[stale, value_columns] = np.nan
        raw_values[slug] = merged[slug].to_numpy()
        raw_values[f"{slug}__observation_date"] = merged["observation_date"].to_numpy()
        raw_values[f"{slug}__available_at"] = merged["published_at"].to_numpy()
        raw_values[f"{slug}__release_delta"] = merged[f"{slug}__release_delta"].to_numpy()
        raw_values[f"{slug}__release_pct"] = merged[f"{slug}__release_pct"].to_numpy()
        raw_columns[series_id] = slug

    frame = pd.concat([frame, pd.DataFrame(raw_values, index=frame.index)], axis=1)
    engineered: dict[str, pd.Series] = {}

    def has(*series: str) -> bool:
        return all(item in raw_columns for item in series)

    def add(name: str, values: pd.Series) -> None:
        engineered[name] = values.replace([np.inf, -np.inf], np.nan)
        feature_names.append(name)

    near = raw_columns.get("br_eme_usdcop_near_mean")
    twelve = raw_columns.get("br_eme_usdcop_12m_mean")
    std = raw_columns.get("br_eme_usdcop_near_std")
    if near:
        add("pit_eme_near_gap_pct", frame[near] / frame["close"] - 1.0)
        add("pit_eme_near_revision_pct", frame[f"{near}__release_pct"])
    if near and twelve:
        add("pit_eme_curve_12m_pct", frame[twelve] / frame[near] - 1.0)
        add("pit_eme_12m_gap_pct", frame[twelve] / frame["close"] - 1.0)
        add("pit_eme_12m_revision_pct", frame[f"{twelve}__release_pct"])
    if near and std:
        add("pit_eme_near_dispersion", frame[std] / frame[near].abs().replace(0, np.nan))

    daily_volume = raw_columns.get("br_forward_daily_volume_usd_m")
    daily_dev = raw_columns.get("br_forward_daily_implied_devaluation_pct")
    if daily_volume:
        add("pit_forward_daily_volume_change", frame[f"{daily_volume}__release_pct"])
    if daily_dev:
        add("pit_forward_daily_devaluation", frame[daily_dev])
        add("pit_forward_daily_devaluation_change", frame[f"{daily_dev}__release_delta"])
        if "carry_policy_us2y" in frame:
            add("pit_forward_daily_devaluation_carry_gap", frame[daily_dev] - frame["carry_policy_us2y"])
    for source_id, name in (
        ("br_forward_daily_financial_net_usd_m", "financial"),
        ("br_forward_daily_offshore_net_usd_m", "offshore"),
        ("br_forward_daily_other_net_usd_m", "other"),
        ("br_forward_daily_ndf_3w_net_usd_m", "ndf_3w"),
    ):
        column = raw_columns.get(source_id)
        if column and daily_volume:
            add(f"pit_forward_daily_{name}_net_ratio", frame[column] / frame[daily_volume].abs().replace(0, np.nan))
    for source_id, name in (
        ("br_forward_daily_short_tenor_share", "short_share"),
        ("br_forward_daily_long_tenor_share", "long_share"),
    ):
        column = raw_columns.get(source_id)
        if column:
            add(f"pit_forward_daily_{name}", frame[column])

    # Consolidated official history: expose a small, economically motivated
    # feature family.  Raw balances and volumes remain available for audit but
    # never enter selection directly.
    for counterparty in ("foreign", "pension", "real_sector"):
        net_ratio = raw_columns.get(
            f"br_forward_history_{counterparty}_net_ratio"
        )
        ndf_ratio = raw_columns.get(
            f"br_forward_history_{counterparty}_ndf_net_ratio"
        )
        balance = raw_columns.get(
            f"br_forward_history_{counterparty}_balance_usd_m"
        )
        if net_ratio:
            net_signal = np.arcsinh(frame[net_ratio])
            add(
                f"pit_forward_history_{counterparty}_net_ratio_asinh",
                net_signal,
            )
            add(
                f"pit_forward_history_{counterparty}_net_ratio_mean5_z252",
                _causal_rolling_z(net_signal.rolling(5, min_periods=3).mean()),
            )
        if ndf_ratio:
            ndf_signal = np.arcsinh(frame[ndf_ratio])
            add(
                f"pit_forward_history_{counterparty}_ndf_net_ratio_asinh",
                ndf_signal,
            )
            add(
                f"pit_forward_history_{counterparty}_ndf_net_ratio_mean5_z252",
                _causal_rolling_z(ndf_signal.rolling(5, min_periods=3).mean()),
            )
        if balance:
            balance_level = frame[balance]
            rolling_scale = balance_level.rolling(252, min_periods=60).std()
            balance_change = (
                balance_level.diff(5) / rolling_scale.replace(0.0, np.nan)
            )
            add(
                f"pit_forward_history_{counterparty}_balance_change_5_scaled",
                balance_change,
            )
            add(
                f"pit_forward_history_{counterparty}_balance_change_5_z252",
                _causal_rolling_z(balance_change),
            )

    market_1m = raw_columns.get("br_forward_history_market_devaluation_1m")
    market_3m = raw_columns.get("br_forward_history_market_devaluation_3m")
    market_6m = raw_columns.get("br_forward_history_market_devaluation_6m")
    if market_1m:
        market_signal = frame[market_1m]
        market_change = market_signal.diff(5)
        add("pit_forward_history_market_devaluation_1m", market_signal)
        add(
            "pit_forward_history_market_devaluation_1m_z252",
            _causal_rolling_z(market_signal),
        )
        add(
            "pit_forward_history_market_devaluation_1m_change_5",
            market_change,
        )
        add(
            "pit_forward_history_market_devaluation_1m_change_5_z252",
            _causal_rolling_z(market_change),
        )
    if market_1m and market_3m:
        curve_3m_1m = frame[market_3m] - frame[market_1m]
        add(
            "pit_forward_history_curve_3m_1m",
            curve_3m_1m,
        )
        add(
            "pit_forward_history_curve_3m_1m_z252",
            _causal_rolling_z(curve_3m_1m),
        )
    if market_1m and market_6m:
        curve_6m_1m = frame[market_6m] - frame[market_1m]
        add(
            "pit_forward_history_curve_6m_1m",
            curve_6m_1m,
        )
        add(
            "pit_forward_history_curve_6m_1m_z252",
            _causal_rolling_z(curve_6m_1m),
        )
    if market_1m:
        for counterparty in ("foreign", "pension", "real_sector"):
            counterparty_1m = raw_columns.get(
                f"br_forward_history_{counterparty}_devaluation_1m"
            )
            if counterparty_1m:
                spread = frame[counterparty_1m] - frame[market_1m]
                add(
                    f"pit_forward_history_{counterparty}_devaluation_1m_spread",
                    spread,
                )
                add(
                    f"pit_forward_history_{counterparty}_devaluation_1m_spread_z252",
                    _causal_rolling_z(spread),
                )

    monthly_volume = raw_columns.get("br_forward_monthly_volume_usd_m")
    if monthly_volume:
        add("pit_forward_monthly_volume_change", frame[f"{monthly_volume}__release_pct"])
    for source_id, name in (
        ("br_forward_monthly_imc_net_usd_m", "imc"),
        ("br_forward_monthly_foreign_net_usd_m", "foreign"),
        ("br_forward_monthly_pension_net_usd_m", "pension"),
    ):
        column = raw_columns.get(source_id)
        if column and monthly_volume:
            add(f"pit_forward_monthly_{name}_net_ratio", frame[column] / frame[monthly_volume].abs().replace(0, np.nan))
    for source_id, name in (
        ("br_forward_monthly_short_tenor_share", "short_share"),
        ("br_forward_monthly_weighted_devaluation_pct", "devaluation"),
        ("br_forward_monthly_outstanding_usd_m", "outstanding"),
    ):
        column = raw_columns.get(source_id)
        if column:
            add(f"pit_forward_monthly_{name}", frame[column])
            add(f"pit_forward_monthly_{name}_change", frame[f"{column}__release_pct"])

    sfc_net = raw_columns.get("sfc_pension_deriv_usd_net_m")
    sfc_gross = raw_columns.get("sfc_pension_deriv_usd_gross_m")
    if sfc_net and sfc_gross:
        add("pit_sfc_pension_usd_net_gross", frame[sfc_net] / frame[sfc_gross].abs().replace(0, np.nan))
        add("pit_sfc_pension_usd_net_change", frame[f"{sfc_net}__release_pct"])

    # Only expose engineered, stationary/relative fields to feature selection.
    if engineered:
        frame = pd.concat([frame, pd.DataFrame(engineered, index=frame.index)], axis=1)
    return frame, list(dict.fromkeys(feature_names))


__all__ = [
    "ForwardMacroScraper",
    "IngestionResult",
    "PIT_COLUMNS",
    "attach_forward_macro_features",
    "parse_daily_forward_text",
    "parse_forward_history_workbook",
    "parse_monthly_derivatives_text",
    "parse_eme_workbook",
    "parse_sfc_formato_415",
    "parse_number",
    "upsert_pit_rows",
]
