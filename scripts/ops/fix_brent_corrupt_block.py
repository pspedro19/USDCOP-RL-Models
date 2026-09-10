#!/usr/bin/env python
"""Repara el bloque de Brent corrupto de 2025 usando FRED como fuente autoritativa.

Contract: CTR-OPS-FIX-BRENT-001 · Date: 2026-08-25

## Qué pasó

Entre **2025-09-25 y 2025-12-19** la serie `comm_oil_brent_glb_d_brent` cae de 69,31 a 23,50,
se queda ~59 días en 21-23 y vuelve a 61,58. Brent real en ese período estuvo en 60-70.

Dos comprobaciones independientes señalan **exactamente las mismas 59 filas** y ninguna más
desde 2021:

1. **Nivel**: `brent < 30`. Desde 2021 Brent nunca bajó de 40 en la realidad.
2. **Ratio contra WTI**: Brent cotiza siempre algo por ENCIMA de WTI (media 1,062 en el resto
   de la serie). En el bloque malo el ratio es 0,366 — imposible: implicaría que el crudo del
   Mar del Norte vale un tercio que el de Texas.

Que dos criterios sin relación entre sí converjan en el mismo conjunto es lo que convierte la
sospecha en diagnóstico.

## Por qué existía un guard y no sirvió

`config/l0_macro_sources.yaml:509` declara `comm_oil_brent_glb_d_brent: [30, 150]` bajo
`validation: {enabled: true}`, y los 59 valores fuera de rango entraron igual.

**CORRECCION (2026-08-25).** La primera version de esta cabecera decia que `RangeValidator`
"solo se instancia en tests". **Es falso**: `data_validators.py:604` lo incluye en la lista por
defecto de `ValidationPipeline` y `l0_macro_backfill.py:835` construye esa pipeline. El
validador SI corre en el backfill.

Los huecos reales son cuatro, y ninguno es "falta cablear el validador":

- **A** — `l0_macro_update.py`, el DAG de ingesta DIARIA, no tiene tarea de validacion:
  el grafo va `extract >> upsert`. **Es por aqui por donde entro este dato.**
- **B** — `l0_macro_backfill.py::validate_data` nunca lanza: acumula errores, hace
  `logger.warning` y el upsert corre igual (`fail_fast=False`).
- **C** — la rama `restore_from_seeds` va directa a `merge`, saltandose la validacion.
- **D** — `RangeValidator._load_ranges_from_config` devuelve `{}` **en silencio** si no
  encuentra el YAML, y entonces valida cero variables sin que nadie se entere.

## La regla que este script NO viola

`fix_macro_fx_scale.py` ya enseñó la lección: el **umbral de detección no es el rango de
plausibilidad**. Aquel confundió los dos y marcó como corrupto el pico real de MXN del COVID.
Aquí se exige que los dos criterios coincidan Y se restringe a `>= 2021-01-01`, porque antes
de esa fecha Brent SÍ estuvo por debajo de 30 legítimamente (2001-2004, y el desplome de abril
de 2020). Las filas pre-2021 bajo 30 son datos buenos y no se tocan.

## Fuente de reemplazo

FRED `DCOILBRENTEU` (Europe Brent Spot Price FOB, diario), vía el endpoint CSV público que no
requiere clave. Es spot y la serie original era el futuro de Investing.com: en la frontera del
bloque difieren un 0,5% (2025-09-24: FRED 69,64 vs BD 69,31), que es la base spot-futuro
normal y no un problema de escala.

**Verificación previa a escribir**: cada valor de reemplazo debe caer dentro de
`[0.85, 1.25] x WTI` del mismo día. Si alguno no lo cumple, el script aborta sin escribir nada.

Uso:
    python scripts/ops/fix_brent_corrupt_block.py            # dry-run (por defecto)
    python scripts/ops/fix_brent_corrupt_block.py --apply
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

COLUMN = "comm_oil_brent_glb_d_brent"
WTI_COLUMN = "comm_oil_wti_glb_d_wti"
PARQUET = REPO / "data" / "pipeline" / "04_cleaning" / "output" / "MACRO_DAILY_CLEAN.parquet"
PARQUET_COL = "COMM_OIL_BRENT_GLB_D_BRENT"
BACKUP_DIR = REPO / "data" / "backups" / "macro_fixes"

# Solo desde aqui: antes, Brent < 30 es un dato REAL (2001-2004, abril 2020).
SAFE_FLOOR_SINCE = date(2021, 1, 1)
LEVEL_FLOOR = 30.0                 # criterio 1
RATIO_BAND = (0.85, 1.25)          # criterio 2, contra WTI
FRED_SERIES = "DCOILBRENTEU"


def fetch_fred(start: date, end: date) -> pd.Series:
    """Brent diario desde FRED. `curl_cffi` porque el `urllib` plano hace timeout."""
    from curl_cffi import requests

    url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={FRED_SERIES}"
           f"&cosd={start:%Y-%m-%d}&coed={end:%Y-%m-%d}")
    resp = requests.get(url, impersonate="chrome", timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["fecha", "valor"]
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.date
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    return df.dropna().set_index("fecha")["valor"]


def connect():
    import psycopg2
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        user=os.getenv("POSTGRES_USER", "admin"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        dbname=os.getenv("POSTGRES_DB", "usdcop_trading"),
    )


def find_corrupt(conn) -> pd.DataFrame:
    """Filas donde AMBOS criterios coinciden. Uno solo no basta."""
    q = f"""
        SELECT fecha, {COLUMN} AS brent, {WTI_COLUMN} AS wti
        FROM macro_indicators_daily
        WHERE fecha >= %s
          AND {COLUMN} IS NOT NULL AND {WTI_COLUMN} IS NOT NULL
          AND {COLUMN} < %s
          AND ({COLUMN} / {WTI_COLUMN}) NOT BETWEEN %s AND %s
        ORDER BY fecha
    """
    return pd.read_sql(q, conn, params=(SAFE_FLOOR_SINCE, LEVEL_FLOOR, *RATIO_BAND))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="escribe; por defecto dry-run")
    args = ap.parse_args()

    conn = connect()
    bad = find_corrupt(conn)
    if bad.empty:
        print("Sin filas corruptas: nada que hacer (script idempotente).")
        return 0

    lo, hi = bad["fecha"].min(), bad["fecha"].max()
    print(f"Filas corruptas: {len(bad)}  |  {lo} -> {hi}")
    print(f"  ratio Brent/WTI observado: {(bad.brent / bad.wti).mean():.3f} (normal ~1.06)")

    fred = fetch_fred(lo, hi)
    print(f"FRED {FRED_SERIES}: {len(fred)} observaciones en la ventana")

    bad["fred"] = bad["fecha"].map(fred)
    missing = bad[bad["fred"].isna()]
    if not missing.empty:
        print(f"ABORTA: FRED no cubre {len(missing)} fechas: "
              f"{[str(d) for d in missing.fecha.head(5)]}")
        return 2

    bad["ratio_nuevo"] = bad["fred"] / bad["wti"]
    off = bad[~bad["ratio_nuevo"].between(*RATIO_BAND)]
    if not off.empty:
        print(f"ABORTA: {len(off)} reemplazos siguen fuera de la banda; no se escribe nada.")
        print(off.head(10).to_string(index=False))
        return 2

    print(f"  ratio Brent/WTI tras reemplazo: {bad.ratio_nuevo.mean():.3f}  OK")
    print(bad.head(3).to_string(index=False))

    if not args.apply:
        print("\nDRY-RUN. Nada escrito. Relanza con --apply.")
        return 0

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"{lo:%Y%m%d}_{hi:%Y%m%d}"
    backup = BACKUP_DIR / f"brent_corrupt_{stamp}.csv"
    bad.to_csv(backup, index=False)
    print(f"\nBackup de las filas originales -> {backup}")

    with conn, conn.cursor() as cur:
        for _, row in bad.iterrows():
            cur.execute(
                f"UPDATE macro_indicators_daily SET {COLUMN} = %s WHERE fecha = %s",
                (float(row["fred"]), row["fecha"]),
            )
        cur.execute(
            f"""SELECT count(*) FROM macro_indicators_daily
                WHERE fecha >= %s AND {COLUMN} IS NOT NULL AND {WTI_COLUMN} IS NOT NULL
                  AND ({COLUMN} / {WTI_COLUMN}) NOT BETWEEN %s AND %s""",
            (SAFE_FLOOR_SINCE, *RATIO_BAND),
        )
        left = cur.fetchone()[0]
        if left:
            raise RuntimeError(f"tras el UPDATE quedan {left} filas fuera de banda; rollback")
    print(f"BD actualizada: {len(bad)} filas. Quedan 0 fuera de banda desde {SAFE_FLOOR_SINCE}.")

    if PARQUET.is_file():
        pq = pd.read_parquet(PARQUET)
        repl = bad.set_index(pd.to_datetime(bad["fecha"]))["fred"]
        hit = pq.index.isin(repl.index)
        pq.loc[hit, PARQUET_COL] = repl.reindex(pq.index[hit]).to_numpy()
        pq.to_parquet(PARQUET)
        print(f"Parquet sincronizado ({int(hit.sum())} filas) -> {PARQUET.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
