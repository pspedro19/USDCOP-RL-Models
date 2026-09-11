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

Las series cuya fuente declarada no es alcanzable desde aqui (ICE DXY, BanRep IBR) se copian
del fichero actual y se marcan `source_verified: false`. **No se presentan como correctas**: la
diferencia entre verificado y no mirado es lo que esta auditoria vino a instaurar, y un
artefacto que la borre es peor que no tenerlo.

Uso:
    python scripts/data/build_research_macro.py [--out-dir DIR]
    python scripts/data/build_research_macro.py --allow-unverified  # solo diagnóstico
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AVAILABILITY = ROOT / "config" / "research" / "macro_availability.yaml"
CLEAN = ROOT / "data" / "pipeline" / "04_cleaning" / "output" / "MACRO_DAILY_CLEAN.parquet"
DEFAULT_OUT = ROOT / "data" / "pipeline" / "04_cleaning" / "output"
FRED_SOURCES = {"FRED_DCOILBRENTEU": "DCOILBRENTEU", "FRED_DGS2": "DGS2"}


def _fred(series_id: str):
    import pandas as pd

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    with urllib.request.urlopen(url, timeout=90) as resp:
        raw = resp.read()
    frame = pd.read_csv(io.BytesIO(raw))
    frame.columns = ["date", "value"]
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    series = frame.dropna().set_index("date")["value"]
    return series, hashlib.sha256(raw).hexdigest()


def main() -> int:
    import pandas as pd
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--allow-unverified", action="store_true",
        help="escribe un artefacto parcial para diagnóstico; nunca usarlo para entrenar",
    )
    args = parser.parse_args()

    if not (AVAILABILITY.is_file() and CLEAN.is_file()):
        print("faltan el SSOT de disponibilidad o el macro limpio", file=sys.stderr)
        return 2

    declared = yaml.safe_load(AVAILABILITY.read_text(encoding="utf-8"))["series"]
    clean = pd.read_parquet(CLEAN)
    columns, provenance = {}, {}

    for name, spec in declared.items():
        column, source = spec["column"], spec["source"]
        if source in FRED_SOURCES:
            series, payload_sha = _fred(FRED_SOURCES[source])
            series.index = pd.to_datetime(series.index).normalize()
            columns[column] = series
            provenance[name] = {
                "column": column, "declared_source": source, "source_verified": True,
                "fetched_from": f"fredgraph.csv?id={FRED_SOURCES[source]}",
                "payload_sha256": payload_sha, "rows": int(len(series)),
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
                "rows": int(len(series)),
                "first": str(series.index.min().date()), "last": str(series.index.max().date()),
            }

    unverified = [name for name, prov in provenance.items()
                  if not prov.get("source_verified", False)]
    if unverified and not args.allow_unverified:
        print(
            "ABORTA: fuentes no verificadas " + ", ".join(sorted(unverified))
            + ". Use --allow-unverified solo para diagnóstico; no se escribe ningún artefacto.",
            file=sys.stderr,
        )
        return 2

    frame = pd.DataFrame(columns).sort_index()
    frame.index.name = "fecha"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "MACRO_RESEARCH_v2.parquet"
    frame.to_parquet(out)

    verified = [n for n, p in provenance.items() if p.get("source_verified")]
    meta = {
        "contract": "CTR-RESEARCH-MACRO-AVAILABILITY-001",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": out.relative_to(ROOT).as_posix(),
        "artifact_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "rows": int(len(frame)),
        "first": str(frame.index.min().date()), "last": str(frame.index.max().date()),
        "series": provenance,
        "series_verified_against_declared_source": sorted(verified),
        "all_series_verified": len(verified) == len(provenance),
        "write_mode": "diagnostic_partial" if unverified else "strict_verified",
        "note": ("Artefacto SEPARADO a proposito: MACRO_DAILY_CLEAN no se parchea. Las series "
                 "no verificadas se copian y se marcan, nunca se presentan como correctas."),
    }
    (args.out_dir / "MACRO_RESEARCH_v2.provenance.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for name, prov in provenance.items():
        mark = "VERIFICADA" if prov.get("source_verified") else "COPIADA sin verificar"
        print(f"  {name:6s} {prov['declared_source']:22s} {mark:22s} "
              f"{prov.get('first', '?')} -> {prov.get('last', '?')}")
    print(f"\n  escrito {out.relative_to(ROOT).as_posix()} "
          f"({len(frame)} filas, {len(verified)}/{len(provenance)} series verificadas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
