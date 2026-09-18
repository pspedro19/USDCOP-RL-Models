#!/usr/bin/env python
"""Construye el macro del carril de investigacion DESDE LA FUENTE QUE DECLARA SU SSOT.

`config/research/macro_availability.yaml` nombra una fuente por serie con
`fallback: forbidden`, y el pre-registro v3 congela esa identidad. El 2026-09-11 se midio que
**no se cumple**: el Brent de `MACRO_DAILY_CLEAN.parquet` coincide con `FRED_DCOILBRENTEU` solo
el 1,8 % de las veces porque son futuros con un parche spot de 59 filas, y el DGS2 coincide el
94 % porque viene de Investing. Entrenar la version reparada sobre datos cuya identidad
contradice su propio pre-registro reproduciria el defecto que el programa existe para corregir.

Este script no parchea el fichero existente -- meter una tercera fuente encima de las dos que ya
conviven amplia el problema. Escribe un artefacto **separado** donde cada serie viene entera de
su fuente declarada y cada fila lleva procedencia:

    data/pipeline/04_cleaning/output/MACRO_RESEARCH_v2.parquet
    data/pipeline/04_cleaning/output/MACRO_RESEARCH_v2.provenance.json

Las series se descargan desde las fuentes declaradas: DXY desde Investing (instrumento
942611), Brent/DGS2 desde FRED e IBR desde BanRep. Si una fuente falla, el proceso aborta
en modo estricto; `--allow-unverified` solo produce un artefacto diagnóstico marcado como
no verificable.

Uso:
    python scripts/data/build_research_macro.py [--out-dir DIR]
    python scripts/data/build_research_macro.py --allow-unverified  # solo diagnóstico
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.macro_evidence import (
    canonical_json, capture_payload, immutable_write, parse_payload,
)

AVAILABILITY = ROOT / "config" / "research" / "macro_availability.yaml"
CLEAN = ROOT / "data" / "pipeline" / "04_cleaning" / "output" / "MACRO_DAILY_CLEAN.parquet"
MARKET_M5 = ROOT / "seeds" / "latest" / "usdcop_m5_ohlcv.parquet"
DEFAULT_OUT = ROOT / "data" / "pipeline" / "04_cleaning" / "output"
FRED_SOURCES = {"FRED_DCOILBRENTEU": "DCOILBRENTEU", "FRED_DGS2": "DGS2"}
BANREP_IBR_URL = (
    "https://totoro.banrep.gov.co/nsi-jax-ws/rest/data/"
    "ESTAT,DF_IBR_DAILY_HIST,1.0/all/ALL/"
    "?startPeriod=2008&endPeriod=2027&dimensionAtObservation=TIME_PERIOD&detail=full"
)
INVESTING_DXY_URL = "https://api.investing.com/api/financialdata/historical/942611"


def _fred(series_id: str, *, evidence_dir=None, records=None):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    with urllib.request.urlopen(url, timeout=90) as resp:
        raw = resp.read()
    if evidence_dir is not None:
        evidence = capture_payload(raw, source=f"FRED_{series_id}", url=url,
                                   directory=evidence_dir)
        if records is not None:
            records.append(evidence)
    series = parse_payload(raw, f"FRED_{series_id}")
    return series, hashlib.sha256(raw).hexdigest()


def _banrep_ibr(*, evidence_dir=None, records=None):
    try:
        import requests
        response = requests.get(
            BANREP_IBR_URL,
            headers={"User-Agent": "USDCOP-Research-PIT/1.0", "Accept": "application/xml"},
            timeout=90,
        )
        response.raise_for_status()
        raw = response.content
    except ImportError:
        request = urllib.request.Request(
            BANREP_IBR_URL,
            headers={"User-Agent": "USDCOP-Research-PIT/1.0", "Accept": "application/xml"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read()
    if evidence_dir is not None:
        evidence = capture_payload(raw, source="BANREP_IBR", url=BANREP_IBR_URL,
                                   directory=evidence_dir)
        if records is not None:
            records.append(evidence)
    return parse_payload(raw, "BANREP_IBR"), hashlib.sha256(raw).hexdigest()


def _investing_dxy(start_date, end_date, *, evidence_dir=None, records=None):
    """Fetch Investing DXY instrument 942611 with immutable payload evidence."""
    import pandas as pd
    chunks = []
    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(cursor + timedelta(days=6570), end_date)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    parts = []
    raw_payloads = []
    for chunk_start, chunk_end in chunks:
        query = urllib.parse.urlencode({
            "start-date": chunk_start.strftime("%Y-%m-%d"),
            "end-date": chunk_end.strftime("%Y-%m-%d"),
            "time-frame": "Daily",
            "add-missing-rows": "false",
        })
        request = urllib.request.Request(
            f"{INVESTING_DXY_URL}?{query}",
            headers={
                "User-Agent": "USDCOP-Research/1.0",
                "Accept": "application/json",
                "Referer": "https://www.investing.com/indices/usdollar-historical-data",
                "domain-id": "www",
            },
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read()
        if evidence_dir is not None:
            evidence = capture_payload(raw, source="INVESTING_DXY", url=request.full_url,
                                       directory=evidence_dir)
            if records is not None:
                records.append(evidence)
        raw_payloads.append(raw)
        parts.append(parse_payload(raw, "INVESTING_DXY"))
    if not parts:
        raise ValueError("Investing DXY returned no usable daily rows")
    values = pd.concat(parts).sort_index()
    if values.index.has_duplicates:
        raise ValueError("Investing DXY overlapping chunk dates")
    index = values.index
    if index.min().date() > start_date + timedelta(days=31) or index.max().date() < end_date - timedelta(days=31):
        raise ValueError(
            "Investing DXY coverage incomplete: "
            f"requested {start_date}..{end_date}, received {index.min().date()}..{index.max().date()}"
        )
    digest = hashlib.sha256(b"".join(raw_payloads)).hexdigest()
    return values, digest


def main() -> int:
    import pandas as pd
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--evidence-dir", type=Path,
                        help="content-addressed raw responses; default <out-dir>/macro_evidence")
    parser.add_argument(
        "--allow-unverified", action="store_true",
        help="escribe un artefacto parcial para diagnóstico; nunca usarlo para entrenar",
    )
    args = parser.parse_args()
    # Convert relative output paths before provenance calls `relative_to(ROOT)`.
    args.out_dir = args.out_dir.resolve()
    evidence_dir = (args.evidence_dir or args.out_dir / "macro_evidence").resolve()
    if (args.out_dir / "MACRO_RESEARCH_v2.parquet").exists():
        print("ABORTA: output already exists; select a new --out-dir", file=sys.stderr)
        return 2

    if not (AVAILABILITY.is_file() and CLEAN.is_file()):
        print("faltan el SSOT de disponibilidad o el macro limpio", file=sys.stderr)
        return 2

    declared = yaml.safe_load(AVAILABILITY.read_text(encoding="utf-8"))["series"]
    clean = pd.read_parquet(CLEAN)
    columns, provenance = {}, {}

    for name, spec in declared.items():
        column, source = spec["column"], spec["source"]
        records = []
        if source in FRED_SOURCES:
            try:
                series, payload_sha = _fred(FRED_SOURCES[source], evidence_dir=evidence_dir,
                                           records=records)
            except Exception as exc:
                provenance[name] = {"column": column, "declared_source": source,
                                    "source_verified": False,
                                    "status": f"FETCH_ERROR:{type(exc).__name__}"}
                continue
            series.index = pd.to_datetime(series.index).normalize()
            columns[column] = series
            provenance[name] = {
                "column": column, "declared_source": source, "source_verified": True,
                "fetched_from": f"fredgraph.csv?id={FRED_SOURCES[source]}",
                "payload_sha256": payload_sha, "rows": len(series),
                "reference_payloads": records,
                "first": str(series.index.min().date()), "last": str(series.index.max().date()),
            }
        elif source == "BANREP_IBR":
            try:
                series, payload_sha = _banrep_ibr(evidence_dir=evidence_dir, records=records)
            except Exception as exc:
                provenance[name] = {"column": column, "declared_source": source,
                                    "source_verified": False, "status": f"FETCH_ERROR:{type(exc).__name__}"}
                continue
            series.index = pd.to_datetime(series.index).normalize()
            columns[column] = series
            provenance[name] = {
                "column": column, "declared_source": source, "source_verified": True,
                "fetched_from": BANREP_IBR_URL, "selection": "SUBJECT=IRIBRM00, UNIT_MEASURE=NR",
                "payload_sha256": payload_sha, "rows": len(series),
                "reference_payloads": records,
                "first": str(series.index.min().date()), "last": str(series.index.max().date()),
            }
        elif source == "INVESTING_DXY":
            try:
                market = pd.read_parquet(MARKET_M5, columns=["time"])
                market_dates = pd.to_datetime(market["time"], utc=True).dt.tz_convert(
                    "America/Bogota"
                ).dt.date
                start = market_dates.min()
                # The legacy macro parquet can be stale.  Bound the declared source by
                # the freshest primary market artifact, not by the stale union used to
                # bootstrap the new parquet; otherwise DXY silently stops at 2026-08-24.
                end = market_dates.max()
                series, payload_sha = _investing_dxy(start, end, evidence_dir=evidence_dir,
                                                    records=records)
            except Exception as exc:
                provenance[name] = {
                    "column": column, "declared_source": source,
                    "source_verified": False,
                    "status": f"FETCH_ERROR:{type(exc).__name__}",
                    "instrument_id": 942611,
                    "url": INVESTING_DXY_URL,
                }
                continue
            columns[column] = series
            provenance[name] = {
                "column": column, "declared_source": source,
                "source_verified": True, "instrument_id": 942611,
                "fetched_from": INVESTING_DXY_URL,
                "payload_sha256": payload_sha, "rows": len(series),
                "reference_payloads": records,
                "first": str(series.index.min().date()), "last": str(series.index.max().date()),
            }
        else:
            if column not in clean.columns:
                provenance[name] = {"column": column, "declared_source": source,
                                    "source_verified": False, "status": "COLUMNA AUSENTE"}
                continue
            series = clean[column].dropna()
            series.index = pd.to_datetime(series.index).normalize()
            columns[column] = series
            provenance[name] = {
                "column": column, "declared_source": source, "source_verified": False,
                "copied_from": "MACRO_DAILY_CLEAN.parquet",
                "warning": ("la fuente declarada no es alcanzable desde este script; la columna "
                            "se copia TAL CUAL y NO se certifica que sea el instrumento "
                            "declarado"),
                "rows": len(series),
                "first": str(series.index.min().date()), "last": str(series.index.max().date()),
            }

    unverified = [name for name, prov in provenance.items()
                  if not prov.get("source_verified", False)]
    if unverified and not args.allow_unverified:
        print(
            "ABORTA: fuentes no verificadas " + ", ".join(sorted(unverified))
            + ". Use --allow-unverified solo para diagnóstico; no se escribe ningún artefacto derivado; "
              "las respuestas crudas capturadas se conservan.",
            file=sys.stderr,
        )
        return 2

    frame = pd.DataFrame(columns).sort_index()
    frame.index.name = "fecha"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "MACRO_RESEARCH_v2.parquet"
    immutable_write(out, frame.to_parquet())

    verified = [n for n, p in provenance.items() if p.get("source_verified")]
    meta = {
        "schema_version": 2,
        "evidence_root": str(evidence_dir),
        "availability_sha256": hashlib.sha256(AVAILABILITY.read_bytes()).hexdigest(),
        "contract": "CTR-RESEARCH-MACRO-AVAILABILITY-001",
        "built_at_utc": datetime.now(UTC).isoformat(),
        "artifact": str(out),
        "artifact_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "rows": len(frame),
        "first": str(frame.index.min().date()), "last": str(frame.index.max().date()),
        "series": provenance,
        "series_verified_against_declared_source": sorted(verified),
        "all_series_verified": len(verified) == len(provenance),
        "write_mode": "diagnostic_partial" if unverified else "strict_verified",
        "note": ("Artefacto SEPARADO a proposito: MACRO_DAILY_CLEAN no se parchea. Las series "
                 "no verificadas se copian y se marcan, nunca se presentan como correctas."),
    }
    metadata = canonical_json(meta)
    immutable_write(args.out_dir / "MACRO_RESEARCH_v2.provenance.json", metadata)
    immutable_write(evidence_dir / "manifests" / f"{hashlib.sha256(metadata).hexdigest()}.json",
                    metadata)

    for name, prov in provenance.items():
        mark = "VERIFICADA" if prov.get("source_verified") else "COPIADA sin verificar"
        print(f"  {name:6s} {prov['declared_source']:22s} {mark:22s} "
              f"{prov.get('first', '?')} -> {prov.get('last', '?')}")
    print(f"\n  escrito {out} "
          f"({len(frame)} filas, {len(verified)}/{len(provenance)} series verificadas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
