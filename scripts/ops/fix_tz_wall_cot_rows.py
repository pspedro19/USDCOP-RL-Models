#!/usr/bin/env python3
"""One-off corrective migration: wall-COT timestamps mislabeled as UTC -> true instants.

Contract: CTR-DQ-TZ-001 (Fase TZ del plan 2026-07-21, aprobado por el operador)

## What was found (empirical, 2026-07-21)

`usdcop_m5_ohlcv` mixes THREE timestamp conventions in one TIMESTAMPTZ column:

    source                              stored-hour(UTC)   meaning
    twelvedata (realtime, ->2026-01-08)      13-17         TRUE instants (8:00-12:55 COT)
    twelvedata_multi (2026-03->)             13-17         TRUE instants
    twelvedata_backfill  (COP, MXN)           8-12         COT WALL CLOCK mislabeled as UTC
    twelvedata_gap_fill / _manual_test        8-12         same wall-COT defect
    twelvedata_backfill  (BRL)               13-17         TRUE instants -- saved by the
                                                           documented "request BRL in UTC" quirk

A backfill bar displaying "12:55+00" claims an instant of 07:55 COT (pre-session); the bar it
represents actually closed at 12:55 COT = 17:55 UTC. Consequences: cross-source intraday
ordering is inverted (a backfill 12:55 sorts BEFORE a realtime 13:00 of the same day while
occurring 5h AFTER it), and any hour-based logic answers differently per source.

Plus one stray day: 2026-01-08 (the old realtime source's LAST day) fetched 00:00-12:55 in the
wall convention -- h8-12 is that day's session (fix like the others), h0-7 is off-session under
BOTH readings (delete; the OHLCV validator would hard-reject those bars today).

## What this does

1. BACKUP every affected row to `usdcop_m5_ohlcv_tz_backup_20260721` (created if absent).
2. DELETE wall rows whose corrected slot (+5h) is already occupied by a true-instant row
   (~815): the instant-true row wins; the backfill was filling gaps it perceived under the
   wrong clock.
3. UPDATE the remaining wall rows: time := time + interval '5 hours'.
4. DELETE the 2026-01-08 h0-7 off-session strays.
5. VERIFY: post-migration hour histogram per (symbol, source) must be mono-convention 13-17
   for session pairs; exits non-zero otherwise.

Idempotent: re-running finds zero wall rows (hours 8-12 no longer exist for those sources)
and changes nothing. Dry-run is the DEFAULT; --apply mutates.

Usage:
    python scripts/ops/fix_tz_wall_cot_rows.py            # dry-run: counts only
    python scripts/ops/fix_tz_wall_cot_rows.py --apply
"""
from __future__ import annotations

import argparse
import os
import sys

WALL_SOURCES = ("twelvedata_backfill", "twelvedata_gap_fill", "twelvedata_manual_test")
# USD/MXN salio de aqui el 2026-08-24: dejo de ser una serie acotada por sesion (hoy son
# 462.193 filas cubriendo las 24 horas), asi que la premisa "hora 8-12 UTC == reloj-pared" ya
# no le aplica y el paso de colisiones habria borrado 95.562 barras legitimas. El pre-flight de
# `main()` lo vuelve a comprobar en cada corrida por si el dato cambia otra vez.
# BRL nunca estuvo: su rama de fetch pedia UTC y localizaba bien (el quirk documentado).
WALL_SYMBOLS = ("USD/COP",)
BACKUP = "usdcop_m5_ohlcv_tz_backup_20260721"

WALL_PRED = f"""
    source IN {WALL_SOURCES!r} AND symbol IN {WALL_SYMBOLS!r}
    AND EXTRACT(hour FROM time AT TIME ZONE 'UTC') BETWEEN 8 AND 12
""".replace("'", "'")  # psycopg2 will interpolate tuples properly below

STRAY_PRED = """
    symbol = 'USD/COP' AND source = 'twelvedata'
    AND time::date = DATE '2026-01-08'
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="mutate; default is dry-run")
    a = ap.parse_args()

    import psycopg2
    conn = psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
    )
    conn.autocommit = False
    cur = conn.cursor()

    # ------------------------------------------------------------------------
    # PRE-FLIGHT (anadido 2026-08-24). Este script asume que WALL_SYMBOLS son pares
    # ACOTADOS POR SESION (08:00-12:55 COT = 13-17 UTC), donde "hora 8-12 UTC" solo puede
    # significar reloj-COT mal etiquetado.
    #
    # Esa premisa CADUCO para USD/MXN: alguien backfilleo su historia completa despues de
    # escribirse este script, y hoy es una serie de 24 HORAS (462.193 filas, horas 0-23).
    # En una serie 24h la hora 8-12 UTC es legitima, y ademas el hueco +5h siempre esta
    # ocupado, asi que el paso de colisiones las clasificaria todas como borrables:
    #
    #     medido el 2026-08-24: 96.544 filas de MXN clasificadas como "pared",
    #     de las cuales 95.562 SE BORRARIAN. Todas legitimas.
    #
    # La verificacion final habria hecho rollback (MXN seguiria teniendo filas fuera de
    # 13-17), asi que el script falla en seguro — pero es una mina: basta con que alguien
    # relaje esa verificacion para creer que "ya pasa" y perder 95k barras.
    #
    # USD/COP si sigue siendo mono-convencion 13-17 y con 0 filas pared: su migracion ya
    # se completo el 2026-07-21. Este guard deja constancia y aborta antes de tocar nada.
    cur.execute("""
        SELECT symbol, COUNT(DISTINCT EXTRACT(hour FROM time AT TIME ZONE 'UTC'))
        FROM usdcop_m5_ohlcv WHERE symbol = ANY(%s) GROUP BY 1
    """, (list(WALL_SYMBOLS),))
    not_session_bound = [(sym, int(h)) for sym, h in cur.fetchall() if h > 5]
    if not_session_bound:
        print("ABORTA: estos WALL_SYMBOLS ya NO son series acotadas por sesion:")
        for sym, h in not_session_bound:
            print(f"  {sym}: {h} horas distintas (una sesion COT tiene 5). "
                  f"Sus barras en 8-12 UTC son LEGITIMAS, no reloj-pared.")
        print("Quita esos simbolos de WALL_SYMBOLS antes de volver a correr esto.")
        conn.rollback()
        return 2

    wall_where = """
        source IN %s AND symbol IN %s
        AND EXTRACT(hour FROM time AT TIME ZONE 'UTC') BETWEEN 8 AND 12
    """
    args = (WALL_SOURCES, WALL_SYMBOLS)

    cur.execute(f"SELECT COUNT(*) FROM usdcop_m5_ohlcv WHERE {wall_where}", args)
    n_wall = cur.fetchone()[0]
    cur.execute(f"""
        SELECT COUNT(*) FROM usdcop_m5_ohlcv w
        WHERE {wall_where.replace('source', 'w.source').replace('symbol IN', 'w.symbol IN')
               .replace('FROM time', 'FROM w.time')}
        AND EXISTS (SELECT 1 FROM usdcop_m5_ohlcv e
                    WHERE e.symbol = w.symbol AND e.time = w.time + interval '5 hours')
    """, args)
    n_coll = cur.fetchone()[0]
    cur.execute(f"""SELECT COUNT(*) FROM usdcop_m5_ohlcv WHERE {STRAY_PRED}
                    AND EXTRACT(hour FROM time AT TIME ZONE 'UTC') < 8""")
    n_stray_off = cur.fetchone()[0]
    cur.execute(f"""SELECT COUNT(*) FROM usdcop_m5_ohlcv WHERE {STRAY_PRED}
                    AND EXTRACT(hour FROM time AT TIME ZONE 'UTC') BETWEEN 8 AND 12""")
    n_stray_sess = cur.fetchone()[0]

    print(f"pared-COT a corregir : {n_wall}")
    print(f"  de ellas, colisiones (se BORRAN, gana la fila instante-real): {n_coll}")
    print(f"  a desplazar +5h    : {n_wall - n_coll}")
    print(f"dia extraviado 2026-01-08: sesion(h8-12)={n_stray_sess} -> +5h | "
          f"fuera-de-sesion(h0-7)={n_stray_off} -> BORRAR")

    if not a.apply:
        print("\nDRY-RUN (usa --apply para ejecutar)")
        conn.rollback()
        return 0

    # 1) backup
    cur.execute(f"""CREATE TABLE IF NOT EXISTS {BACKUP} (LIKE usdcop_m5_ohlcv INCLUDING ALL)""")
    cur.execute(f"""INSERT INTO {BACKUP}
                    SELECT * FROM usdcop_m5_ohlcv WHERE {wall_where} ON CONFLICT DO NOTHING""",
                args)
    cur.execute(f"""INSERT INTO {BACKUP}
                    SELECT * FROM usdcop_m5_ohlcv WHERE {STRAY_PRED} ON CONFLICT DO NOTHING""")
    # 2) collisions: delete the wall copy
    cur.execute(f"""
        DELETE FROM usdcop_m5_ohlcv w
        WHERE {wall_where.replace('source', 'w.source').replace('symbol IN', 'w.symbol IN')
               .replace('FROM time', 'FROM w.time')}
        AND EXISTS (SELECT 1 FROM usdcop_m5_ohlcv e
                    WHERE e.symbol = w.symbol AND e.time = w.time + interval '5 hours')
    """, args)
    deleted_coll = cur.rowcount
    # 3) shift the rest
    cur.execute(f"""UPDATE usdcop_m5_ohlcv SET time = time + interval '5 hours'
                    WHERE {wall_where}""", args)
    shifted = cur.rowcount
    # 3b) stray day: session block +5h (guard collisions the same way), off-session delete
    cur.execute(f"""
        DELETE FROM usdcop_m5_ohlcv w WHERE {STRAY_PRED.replace('symbol','w.symbol')
            .replace('source','w.source').replace('time::date','w.time::date')}
        AND EXTRACT(hour FROM w.time AT TIME ZONE 'UTC') BETWEEN 8 AND 12
        AND EXISTS (SELECT 1 FROM usdcop_m5_ohlcv e
                    WHERE e.symbol = w.symbol AND e.time = w.time + interval '5 hours')""")
    deleted_coll += cur.rowcount
    cur.execute(f"""UPDATE usdcop_m5_ohlcv SET time = time + interval '5 hours'
                    WHERE {STRAY_PRED}
                    AND EXTRACT(hour FROM time AT TIME ZONE 'UTC') BETWEEN 8 AND 12""")
    shifted += cur.rowcount
    # 4) off-session strays
    cur.execute(f"""DELETE FROM usdcop_m5_ohlcv WHERE {STRAY_PRED}
                    AND EXTRACT(hour FROM time AT TIME ZONE 'UTC') < 8""")
    deleted_stray = cur.rowcount

    # 5) verify mono-convention: session symbols must now live in 13-17 UTC only
    cur.execute("""
        -- Solo los simbolos realmente acotados por sesion. MXN/BRL son series 24h desde su
        -- backfill de historia completa (ver nota en WALL_SYMBOLS): exigirles 13-17 haria
        -- fallar la verificacion siempre y tentaria a relajarla.
        SELECT symbol, source, COUNT(*) FROM usdcop_m5_ohlcv
        WHERE symbol = ANY(%s)
        AND EXTRACT(hour FROM time AT TIME ZONE 'UTC') NOT BETWEEN 13 AND 17
        GROUP BY 1,2
    """, (list(WALL_SYMBOLS),))
    leftovers = cur.fetchall()
    if leftovers:
        conn.rollback()
        print(f"VERIFICACION FALLIDA, rollback: filas fuera de 13-17 UTC: {leftovers}")
        return 1

    conn.commit()
    print(f"\nAPLICADO: colisiones borradas={deleted_coll}  desplazadas +5h={shifted}  "
          f"fuera-de-sesion borradas={deleted_stray}")
    print(f"backup en {BACKUP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
