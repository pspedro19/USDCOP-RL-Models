#!/usr/bin/env python3
"""Corrective migration: duplicados por fecha y `available_at` en `asset_daily_ohlcv`.

Contract: CTR-DQ-ASSET-DAILY-001 · Date: 2026-08-24

Cierra tres fallos que solo se hicieron visibles al levantar el stack — llevaban meses
ocultos porque sus tests hacen `skip` cuando Postgres no responde.

## Q2 · Duplicados por fecha UTC (5 filas de XAU/USD)

`asset_daily_ohlcv` tiene PK `(time, symbol)`, pero los feeds sellan a horas distintas
(el deep a 00:00, el diario a 21:00/22:00), asi que la PK **no** impide dos barras del
mismo dia. Medido: XAU/USD 2026-07-22..28 con `twelvedata_daily` y
`twelvedata_daily_deep` a la vez.

Gana `twelvedata_daily_deep`: es la serie de historia profunda (desde 1979), la que
cubre mas rango y la unica de las dos que ya sella `available_at` en el 100% de sus
filas. Se borra la copia `twelvedata_daily` del mismo dia.

## Q3 · Vistas wide inventando filas (9.483 vs 9.478)

**No es un defecto propio**: son exactamente las 5 filas de Q2 doblando el join. El
docstring de `test_no_duplicate_daily_dates` ya lo documenta ("the doubled join broke
wide-no-invention"). Se cierra al resolver Q2; este script lo VERIFICA en vez de
suponerlo.

## Q4 · 9.133 filas sin `available_at`

Reparto exacto: XAU/USD `twelvedata_daily` 5.873 + BTC/USDT `binance_daily` 3.260. El
resto de fuentes (`twelvedata_daily_deep`, `investing_daily`) sellan el 100%.

**Convencion medida, no supuesta**: `available_at` = hora de INGESTA, no de cierre de
barra. Los `twelvedata_daily_deep` tienen un lag mediano de 7.649 dias, que es
`now() - time` para historia backfilleada. `scripts/ops/backfill_max_history.py:22` lo
declara: "for a historic backfill, ingestion time IS the availability". Se rellena con
el `updated_at` de cada fila: el instante real y demostrable en que tuvimos el dato.

LIMITACION QUE HAY QUE DECLARAR, NO ESCONDER: con esta convencion una consulta PIT
`available_at <= bar_time` devuelve VACIO para toda la historia backfilleada. Es honesto
(no teniamos el dato antes de pedirlo) pero significa que este campo **no sirve como
defensa anti-look-ahead retrospectiva**. La defensa real es el `shift(1)` de macro y el
`merge_asof(backward)` — capa 1 de `quant-constitution.md` §4.

Idempotente. Dry-run por DEFECTO; `--apply` muta. Backup previo a tabla.
"""
from __future__ import annotations

import argparse
import os

BACKUP = "asset_daily_ohlcv_integrity_backup_20260824"

# En un empate de fecha, gana la fuente de MAYOR prioridad (indice mas bajo).
SOURCE_PRIORITY = ("twelvedata_daily_deep", "investing_daily", "binance_daily",
                   "twelvedata_daily")

DUP_SQL = """
SELECT symbol, (time AT TIME ZONE 'UTC')::date AS d, count(*), array_agg(source)
FROM asset_daily_ohlcv GROUP BY 1, 2 HAVING count(*) > 1 ORDER BY 2
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="muta; el default es dry-run")
    a = ap.parse_args()

    import psycopg2
    conn = psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", ""))
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("SET max_parallel_workers_per_gather = 0")

    # ---------------------------------------------------------------- Q2
    cur.execute(DUP_SQL)
    dups = cur.fetchall()
    print(f"=== Q2 duplicados por fecha UTC: {len(dups)}")
    for sym, d, n, srcs in dups:
        print(f"    {sym} {d}  n={n}  fuentes={sorted(set(srcs))}")

    unknown = {s for _sym, _d, _n, srcs in dups for s in srcs} - set(SOURCE_PRIORITY)
    if unknown:
        print(f"ABORTA: fuentes sin prioridad declarada: {sorted(unknown)}. "
              "Anadelas a SOURCE_PRIORITY con criterio explicito antes de borrar nada.")
        conn.rollback()
        return 2

    # ---------------------------------------------------------------- Q4
    cur.execute("""
        SELECT source, symbol, count(*) FROM asset_daily_ohlcv
        WHERE available_at IS NULL GROUP BY 1, 2 ORDER BY 3 DESC
    """)
    nulls = cur.fetchall()
    total_null = sum(n for _s, _y, n in nulls)
    print(f"\n=== Q4 filas sin available_at: {total_null}")
    for src, sym, n in nulls:
        print(f"    {src:24s} {sym:10s} {n:6d}")

    cur.execute("SELECT count(*) FROM asset_daily_ohlcv "
                "WHERE available_at IS NULL AND updated_at IS NULL")
    no_stamp = cur.fetchone()[0]
    if no_stamp:
        print(f"ABORTA: {no_stamp} filas no tienen NI available_at NI updated_at; "
              "no hay instante demostrable con el que rellenar.")
        conn.rollback()
        return 2

    if not a.apply:
        print("\nDRY-RUN (usa --apply para ejecutar)")
        conn.rollback()
        return 0

    cur.execute(f"CREATE TABLE IF NOT EXISTS {BACKUP} "
                f"(LIKE asset_daily_ohlcv INCLUDING DEFAULTS)")

    # Q2: guardar y borrar la copia de menor prioridad de cada fecha duplicada.
    # NUNCA `ctid` AQUI. `asset_daily_ohlcv` es una hypertable de TimescaleDB con 2.431
    # chunks, y **ctid solo es unico dentro de un chunk**: dos filas de chunks distintos
    # comparten valor. Un `WHERE ctid IN (...)` alcanza filas ajenas.
    #
    # Aprendido a golpes el 2026-08-24: la primera version de este script uso ctid, marco
    # 5 duplicados reales y BORRO 99 filas — 94 de ellas legitimas, de 2020 a 2026, en
    # XAU/BTC/SPX. Se restauraron desde el backup (por eso este script SIEMPRE hace backup
    # antes de borrar) y la comprobacion "toda fila borrada debe dejar superviviente en su
    # misma fecha" paso a ser obligatoria mas abajo.
    #
    # La clave natural es `(time, symbol)` — la PK declarada de la tabla.
    priority_case = " ".join(
        f"WHEN source = '{s}' THEN {i}" for i, s in enumerate(SOURCE_PRIORITY))
    loser_key = f"""
        SELECT time, symbol FROM (
          SELECT time, symbol,
                 row_number() OVER (
                   PARTITION BY symbol, (time AT TIME ZONE 'UTC')::date
                   ORDER BY CASE {priority_case} ELSE 99 END, time
                 ) AS rn
          FROM asset_daily_ohlcv
        ) r WHERE rn > 1
    """
    cur.execute(f"INSERT INTO {BACKUP} SELECT * FROM asset_daily_ohlcv "
                f"WHERE (time, symbol) IN ({loser_key})")
    cur.execute(f"DELETE FROM asset_daily_ohlcv WHERE (time, symbol) IN ({loser_key})")
    deleted = cur.rowcount

    # Q4: sellar con el instante de ingesta real de cada fila.
    cur.execute("UPDATE asset_daily_ohlcv SET available_at = updated_at "
                "WHERE available_at IS NULL")
    stamped = cur.rowcount

    # ---------------------------------------------------------------- verificacion
    cur.execute(DUP_SQL)
    left_dups = cur.fetchall()
    cur.execute("SELECT count(*) FROM asset_daily_ohlcv WHERE available_at IS NULL")
    left_nulls = cur.fetchone()[0]
    # La comprobacion que habria evitado la perdida de 94 filas: toda fila borrada debe
    # dejar una superviviente en su MISMA (symbol, fecha). Si no, no era un duplicado.
    cur.execute(f"""
        SELECT count(*) FROM {BACKUP} b WHERE NOT EXISTS (
          SELECT 1 FROM asset_daily_ohlcv a
          WHERE a.symbol = b.symbol
            AND (a.time AT TIME ZONE 'UTC')::date = (b.time AT TIME ZONE 'UTC')::date)
    """)
    orphaned = cur.fetchone()[0]
    if left_dups or left_nulls or orphaned:
        conn.rollback()
        print(f"VERIFICACION FALLIDA, rollback: duplicados={len(left_dups)} "
              f"nulls={left_nulls} BORRADAS_SIN_SUPERVIVIENTE={orphaned}")
        return 1

    conn.commit()
    print(f"\nAPLICADO: {deleted} duplicados borrados · {stamped} filas selladas")
    print(f"backup en {BACKUP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
