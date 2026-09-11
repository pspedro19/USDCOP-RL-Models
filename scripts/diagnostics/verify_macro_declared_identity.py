#!/usr/bin/env python
"""Comprueba que las series macro que lee la investigacion SON las que declara su SSOT.

`config/research/macro_availability.yaml` nombra una fuente por serie y pone
`fallback: forbidden`. Eso es una afirmacion sobre el instrumento, no una preferencia: decir
"brent = FRED_DCOILBRENTEU" y leer un futuro es exactamente el defecto que la auditoria del
2026-09-10 encontro, y que la declaracion se escribio para cerrar.

Medido el 2026-09-11 sobre `MACRO_DAILY_CLEAN.parquet`:

    COMM_OIL_BRENT_GLB_D_BRENT  vs FRED DCOILBRENTEU : coincide el  1,8 %  (dif. media 0,87 USD)
    FINC_BOND_YIELD2Y_USA_D_DGS2 vs FRED DGS2        : coincide el 94,1 %  (dif. media 0,0017)

O sea que la declaracion **no se cumple** para Brent y solo aproximadamente para DGS2. Una
declaracion que el dato no honra es peor que ninguna: crea confianza donde no la hay, y el
pre-registro v3 congela esa identidad como si fuera cierta.

Este script no repara nada: mide y reporta. Repararlo es re-obtener la serie de su fuente
declarada, y es requisito de la Etapa 4 de BL-50.

Uso:
    python scripts/diagnostics/verify_macro_declared_identity.py [--output ruta.json]

Necesita red (FRED). Las series cuya fuente declarada no es FRED (ICE DXY, BanRep IBR) se
reportan como NO COMPROBABLES aqui, no como correctas: la diferencia importa.
"""
from __future__ import annotations

import argparse
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

# Solo las fuentes que este script sabe comprobar. El resto se reporta como no comprobable.
FRED_SOURCES = {"FRED_DCOILBRENTEU": "DCOILBRENTEU", "FRED_DGS2": "DGS2"}
TOLERANCE = 0.01


def _fred(series_id: str):
    import pandas as pd

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        frame = pd.read_csv(io.BytesIO(resp.read()))
    frame.columns = ["date", "value"]
    frame["date"] = pd.to_datetime(frame["date"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna().set_index("date")["value"]


def main() -> int:
    import pandas as pd
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not (AVAILABILITY.is_file() and CLEAN.is_file()):
        print("faltan el SSOT de disponibilidad o el macro limpio", file=sys.stderr)
        return 2

    declared = yaml.safe_load(AVAILABILITY.read_text(encoding="utf-8"))["series"]
    clean = pd.read_parquet(CLEAN)
    report, honoured = {}, True

    for name, spec in declared.items():
        column, source = spec["column"], spec["source"]
        entry = {"column": column, "declared_source": source,
                 "fallback": spec.get("fallback")}
        if column not in clean.columns:
            entry.update(status="COLUMNA AUSENTE", honoured=False)
            honoured = False
            report[name] = entry
            continue
        ours = clean[column].dropna()
        entry["our_last"] = str(ours.index.max().date())
        if source not in FRED_SOURCES:
            entry.update(status="NO COMPROBABLE AQUI",
                         note="la fuente declarada no es FRED; comprobarla exige su proveedor")
            report[name] = entry
            continue
        try:
            reference = _fred(FRED_SOURCES[source])
        except Exception as exc:                     # red caida: no se finge un verde
            entry.update(status=f"NO VERIFICADO ({type(exc).__name__})")
            report[name] = entry
            continue
        common = ours.index.intersection(reference.index)
        if len(common) == 0:
            entry.update(status="SIN FECHAS COMUNES", honoured=False)
            honoured = False
            report[name] = entry
            continue
        diff = (ours.loc[common] - reference.loc[common]).abs()
        match = float((diff < TOLERANCE).mean())
        entry.update(status="COINCIDE" if match > 0.99 else "NO COINCIDE",
                     honoured=match > 0.99, n_common=int(len(common)),
                     match_fraction=round(match, 4),
                     mean_abs_diff=round(float(diff.mean()), 6),
                     max_abs_diff=round(float(diff.max()), 6),
                     reference_last=str(reference.index.max().date()))
        honoured &= entry["honoured"]
        report[name] = entry

    out = {"contract": "CTR-RESEARCH-MACRO-AVAILABILITY-001",
           "measured_at_utc": datetime.now(timezone.utc).isoformat(),
           "tolerance": TOLERANCE, "series": report,
           "all_declared_identities_honoured": honoured}
    text = json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    for name, entry in report.items():
        extra = (f" coincide {100 * entry['match_fraction']:.1f}%"
                 if "match_fraction" in entry else "")
        print(f"  {name:6s} {entry['declared_source']:22s} {entry['status']}{extra}")
    print(f"\n  identidades declaradas honradas: {out['all_declared_identities_honoured']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
