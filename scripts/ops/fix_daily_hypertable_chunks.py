#!/usr/bin/env python
"""Consolida los chunks de `asset_daily_ohlcv`. Dry-run por defecto.

Contract: CTR-OPS-FIX-CHUNKS-001 · Date: 2026-08-25

## El problema, medido

    SELECT (range_end - range_start) AS ancho, count(*)
    FROM timescaledb_information.chunks WHERE hypertable_name='asset_daily_ohlcv'
    GROUP BY 1;
    ->  7 days   | 2430
        943 days |    1

**2.430 chunks de 7 días para 60.542 filas**: unas 25 filas por chunk. El hypertable se creó
con el `chunk_time_interval` por defecto (7 días) y se cambió después a 3.600, pero TimescaleDB
**no reescribe los chunks existentes**: el intervalo nuevo solo rige para los que se creen a
partir de entonces. El backfill de historia completa (1979-2026,
`scripts/ops/backfill_max_history.py`) escribió décadas de barras diarias con el intervalo viejo.

Consecuencia: una consulta que recorra el hypertable entero toca 2.430 chunks, agota la memoria
del contenedor (límite 2 GB, `work_mem` 4 MB) y **el OOM killer se lleva el backend**:

    LOG: server process (PID ...) was terminated by signal 9: Killed

Eso tumba 4 tests de `tests/regression/test_data_quality_floor.py` con un modo de fallo
desconcertante —`server closed the connection unexpectedly` y tres `connection already
closed`, porque la fixture `db` es de módulo—. **No es un fallo de datos**: los datos están
bien.

## Alcance real del CASCADE: 3 vistas, no 17

Una versión anterior de esta cabecera decía "15 objetos anidados". Ese número contaba filas de
`pg_depend` —una por columna referenciada—, no objetos. El árbol recursivo real es:

    nivel 1: asset_daily_coverage, market_ohlcv_daily
    nivel 2: market_ohlcv_daily_wide

Tres vistas. Se capturan sus definiciones con `pg_get_viewdef` **antes** de tocar nada y se
recrean en orden de nivel al terminar. Se capturan de la BD viva en vez de replayar las
migraciones 051/060/061 a propósito: si las vistas hubieran derivado de su migración, replayar
la migración las cambiaría en silencio; restaurar lo capturado devuelve exactamente lo que
había.

Renombrar en vez de dropear no vale: las vistas apuntan a la tabla por OID, no por nombre.

## Las redes, en orden

1. **Volcado CSV de las 60.542 filas**, con `COPY`, y **se cuentan**. Es la red real.
   *No* con `pg_dump -t`: sobre un hypertable eso produce 584 bytes y CERO filas —los datos
   viven en los chunks de `_timescaledb_internal`, no en la tabla padre— y la primera versión
   de este script generaba ese respaldo vacío reportándolo como "0.0 MB".
2. Definiciones de las 3 vistas capturadas y volcadas a fichero.
3. Verificación **dentro de la transacción**: filas, `min`/`max`, símbolos y `sum(close)`.
   La suma se compara sobre `numeric`, no sobre `double precision`: `close` es float y **la
   suma de floats depende del orden**, así que la tabla nueva —llenada en otro orden— difiere
   en los últimos bits y cualquier comparación exacta sobre float falla siempre. En `numeric`
   la suma es decimal exacta e independiente del orden. Si falla algo, `RuntimeError`
   ⇒ ROLLBACK y la tabla original intacta (verificado: el primer intento abortó así).
4. Tras recrear las vistas, se comprueba que **las tres existen** y que sus conteos coinciden
   con los de antes.

Uso:
    python scripts/ops/fix_daily_hypertable_chunks.py                 # diagnóstico
    python scripts/ops/fix_daily_hypertable_chunks.py --apply --i-understand-this-recreates-the-table
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TABLE = "asset_daily_ohlcv"
TARGET_INTERVAL = "3650 days"          # 10 anos por chunk: ~5 chunks para 1979-2026
BACKUP_DIR = REPO / "data" / "backups" / "hypertable_rebuild"

DEPS_SQL = """
WITH RECURSIVE deps AS (
  SELECT v.oid, v.relname::text AS relname, 1 AS lvl
  FROM pg_depend d JOIN pg_rewrite r ON r.oid = d.objid
  JOIN pg_class v ON v.oid = r.ev_class JOIN pg_class s ON s.oid = d.refobjid
  WHERE s.relname = %s AND v.relname <> %s
  UNION
  SELECT v.oid, v.relname::text, deps.lvl + 1
  FROM deps JOIN pg_depend d ON d.refobjid = deps.oid
  JOIN pg_rewrite r ON r.oid = d.objid JOIN pg_class v ON v.oid = r.ev_class
  WHERE v.oid <> deps.oid AND deps.lvl < 6
)
SELECT lvl, relname, pg_get_viewdef(oid, true)
FROM (SELECT DISTINCT ON (relname) lvl, relname, oid FROM deps ORDER BY relname, lvl) x
ORDER BY lvl, relname
"""


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


def snapshot(cur) -> dict:
    """Fotografía completa: chunks, contenido y vistas dependientes con sus conteos."""
    cur.execute(
        "SELECT (range_end - range_start)::text, count(*) "
        "FROM timescaledb_information.chunks WHERE hypertable_name = %s "
        "GROUP BY 1 ORDER BY 2 DESC", (TABLE,))
    widths = cur.fetchall()

    # `sum(close::numeric)` y NO `sum(close)`: `close` es `double precision` y la suma de
    # floats DEPENDE DEL ORDEN. La tabla nueva se llena en otro orden, asi que los ultimos
    # bits difieren y una comparacion exacta sobre float falla siempre. Convertir a numeric
    # suma en decimal exacto: independiente del orden y comparable con `==`.
    cur.execute(f"SELECT count(*), min(time), max(time), count(DISTINCT symbol), "
                f"sum(close::numeric) FROM {TABLE}")
    rows, lo, hi, syms, total = cur.fetchone()

    cur.execute(DEPS_SQL, (TABLE, TABLE))
    views = [{"level": lvl, "name": name, "definition": ddl} for lvl, name, ddl in cur.fetchall()]
    for v in views:
        cur.execute(f"SELECT count(*) FROM {v['name']}")
        v["row_count"] = cur.fetchone()[0]

    return {"widths": widths, "rows": rows, "min": lo, "max": hi, "symbols": syms,
            "sum_close": total, "views": views,
            "total_chunks": sum(n for _, n in widths)}


def pg_dump_table(container: str = "usdcop-postgres-timescale",
                  expected_rows: int = 0) -> Path:
    """Vuelca las filas con `COPY`, NO con `pg_dump -t`.

    `pg_dump -t public.asset_daily_ohlcv` sobre un hypertable produce un fichero de 584
    bytes con **cero** filas: los datos no viven en la tabla padre sino en los chunks de
    `_timescaledb_internal`, y `-t` no los sigue. La primera version de este script generaba
    ese respaldo vacio y lo reportaba como "0.0 MB" — una red que no sujeta nada.

    `COPY (SELECT * FROM tabla) TO STDOUT` lee A TRAVES del hypertable y saca las 60.542
    filas. Se cuenta el resultado y se aborta si no cuadra con lo esperado: un respaldo que
    no se verifica es una suposicion.
    """
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = BACKUP_DIR / f"{TABLE}_{stamp}.csv"
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    cmd = ["docker", "exec", container, "psql", "-U",
           os.getenv("POSTGRES_USER", "admin"), "-d",
           os.getenv("POSTGRES_DB", "usdcop_trading"), "--no-align", "-c",
           f"COPY (SELECT * FROM {TABLE} ORDER BY time, symbol) TO STDOUT WITH CSV HEADER"]
    with out.open("w", encoding="utf-8", newline="") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE, env=env, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"COPY fallo: {proc.stderr[-500:]}")

    with out.open("r", encoding="utf-8") as fh:
        n = sum(1 for _ in fh) - 1          # menos la cabecera
    if expected_rows and n != expected_rows:
        raise RuntimeError(f"el respaldo tiene {n} filas y la tabla {expected_rows}; "
                           "NO se continua sin una red verificada")
    print(f"  respaldo verificado: {n:,} filas")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--i-understand-this-recreates-the-table", action="store_true",
                    dest="confirmed")
    ap.add_argument("--container", default="usdcop-postgres-timescale")
    args = ap.parse_args()

    conn = connect()
    before = snapshot(conn.cursor())

    print(f"{TABLE}: {before['rows']:,} filas en {before['total_chunks']} chunks "
          f"({before['rows'] / max(before['total_chunks'], 1):.0f} filas por chunk)")
    for ancho, n in before["widths"]:
        print(f"  {n:>5} chunks de {ancho}")
    print(f"  rango {before['min']} -> {before['max']}, {before['symbols']} simbolos")
    print(f"\nVistas dependientes ({len(before['views'])}):")
    for v in before["views"]:
        print(f"  nivel {v['level']}  {v['name']:<28} {v['row_count']:>8,} filas")

    if before["total_chunks"] < 50:
        print("\nNada que consolidar.")
        return 0

    span = (before["max"] - before["min"]).days
    print(f"\nPropuesta: recrear con chunk_time_interval = {TARGET_INTERVAL} "
          f"(~{span // 3650 + 1} chunks en vez de {before['total_chunks']}).")

    if not (args.apply and args.confirmed):
        print("\nDRY-RUN. Nada escrito. Para aplicar hacen falta LAS DOS banderas:\n"
              "  --apply --i-understand-this-recreates-the-table")
        return 0

    # --- red 1 y 2: dump + definiciones ---------------------------------
    dump = pg_dump_table(args.container, expected_rows=before["rows"])
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    defs = BACKUP_DIR / f"{dump.stem}_views.json"
    defs.write_text(json.dumps(before["views"], indent=2, default=str), encoding="utf-8")
    print(f"\nRespaldo   -> {dump.relative_to(REPO)} ({dump.stat().st_size / 1e6:.1f} MB)")
    print(f"Vistas     -> {defs.relative_to(REPO)}")

    # --- rebuild dentro de UNA transaccion ------------------------------
    with conn, conn.cursor() as cur:
        cur.execute(f"CREATE TABLE {TABLE}_new (LIKE {TABLE} INCLUDING ALL)")
        cur.execute("SELECT create_hypertable(%s, 'time', "
                    "chunk_time_interval => INTERVAL %s, migrate_data => false)",
                    (f"{TABLE}_new", TARGET_INTERVAL))
        cur.execute(f"INSERT INTO {TABLE}_new SELECT * FROM {TABLE}")

        cur.execute(f"SELECT count(*), min(time), max(time), count(DISTINCT symbol), "
                    f"sum(close::numeric) FROM {TABLE}_new")
        rows, lo, hi, syms, total = cur.fetchone()
        checks = {
            "filas": rows == before["rows"],
            "min": lo == before["min"],
            "max": hi == before["max"],
            "simbolos": syms == before["symbols"],
            "sum_close": total == before["sum_close"],   # Decimal: igualdad EXACTA
        }
        if not all(checks.values()):
            raise RuntimeError(f"verificacion fallida {checks}; ROLLBACK, la tabla "
                               "original NO se toca")
        print(f"  copia verificada: {checks}")

        cur.execute(f"DROP TABLE {TABLE} CASCADE")
        cur.execute(f"ALTER TABLE {TABLE}_new RENAME TO {TABLE}")

        # Vistas, en el mismo orden de nivel en que se capturaron.
        for v in before["views"]:
            cur.execute(f"CREATE OR REPLACE VIEW {v['name']} AS {v['definition']}")
            print(f"  recreada {v['name']}")

    # --- verificacion posterior -----------------------------------------
    after = snapshot(conn.cursor())
    print(f"\nHecho: {after['total_chunks']} chunks (antes {before['total_chunks']}), "
          f"{after['rows']:,} filas.")

    problems = []
    names_after = {v["name"]: v["row_count"] for v in after["views"]}
    for v in before["views"]:
        if v["name"] not in names_after:
            problems.append(f"vista {v['name']} NO se recreo")
        elif names_after[v["name"]] != v["row_count"]:
            problems.append(f"{v['name']}: {names_after[v['name']]:,} filas, "
                            f"antes {v['row_count']:,}")
    if problems:
        print("\nPROBLEMAS (la tabla esta bien; restaura las vistas desde "
              f"{defs.name}):")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"Las {len(after['views'])} vistas dependientes existen y sirven las mismas filas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
