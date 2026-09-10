#!/usr/bin/env python3
"""Corrective migration: barras diarias en dias que el mercado estuvo CERRADO.

Contract: CTR-DQ-ASSET-DAILY-002 · Date: 2026-08-24

## El defecto

`market_ohlcv_daily_wide` clasifica cada dia asi:

    WHEN is_trading_day IS NOT TRUE THEN
        CASE WHEN close IS NULL THEN 'closed' ELSE 'off_session' END

`test_weekend_is_closed_and_today_is_pending_never_missing` fallaba con
`cop=off_session` en 10 fines de semana. La vista **no** estaba equivocada: decia
literalmente "el mercado estaba cerrado pero hay una barra". Lo que sobra es la barra.

Medido el 2026-08-24: 10 filas de `usdcop` en sabados y domingos entre 2026-07-25 y
2026-08-23, todas de `twelvedata_daily_deep`. USD/COP no cotiza en fin de semana; son
fabricacion del endpoint de historia profunda del proveedor.

## Criterio (derivado, no hardcodeado)

Una barra sobra si su activo NO opera 24/7 (`dim_asset.calendar_kind`) y
`market_session_calendar` dice `is_trading_day = false` para esa fecha.

BTC queda fuera por `calendar_kind = 'utc_24_7'`: para el, un sabado es un dia de mercado
normal y borrar sus barras seria el error contrario.

## Dos clases, y solo una se borra por defecto

Aplicar el criterio completo encuentra **152** barras, no 10. Se separan porque no pesan
lo mismo (medido 2026-08-24):

    FIN DE SEMANA   57   usdcop 10 (deep, 2026-07-25..08-23)
                         xauusd 47 (2025-04-26..2026-07-04)
    FESTIVO         95   usdcop 92 (`daily_native`, 2020-01-06..2026-01-01) + 3 deep

Los de FIN DE SEMANA son fabricacion inequivoca: ni USD/COP ni el oro spot cotizan sabado
o domingo bajo estos calendarios. Se borran por defecto.

Los de FESTIVO llevan SEIS ANOS en la serie diaria que usa produccion (H5) y borrarlos
reescribe historia del track en produccion. Son igual de sinteticos —el mercado colombiano
estuvo cerrado— pero es decision del operador, no fontaneria: exigen `--include-holidays`
y recalcular el forward publicado.

## Aviso

Toca la serie DIARIA que usa produccion (H5). El cambio se registra en el backup y en
`CLAUDE.md`; si algun bundle publicado dependiera de esas fechas, el recalculo debe
declararse en el `HYPOTHESIS-REGISTRY` en vez de aparecer como deriva silenciosa.

Idempotente. Dry-run por DEFECTO; `--apply` muta. Backup previo a tabla.
"""
from __future__ import annotations

import argparse
import os

BACKUP = "asset_daily_ohlcv_nonsession_backup_20260824"

# Calendarios que SI operan todos los dias: sus barras en sabado son legitimas.
ALWAYS_OPEN_CALENDARS = ("utc_24_7",)
# `repr` de una tupla de UN elemento deja la coma final — `('utc_24_7',)` es sintaxis
# invalida en SQL. Se construye la lista explicitamente.
_ALWAYS_OPEN_SQL = ", ".join(f"'{c}'" for c in ALWAYS_OPEN_CALENDARS)

OFFENDERS_SQL = f"""
SELECT d.symbol, da.asset_id, da.calendar_kind,
       (d.time AT TIME ZONE 'UTC')::date AS session_date, d.source
FROM asset_daily_ohlcv d
JOIN dim_asset da ON da.symbol = d.symbol
JOIN market_session_calendar c
  ON c.asset_id = da.asset_id
 AND c.session_date = (d.time AT TIME ZONE 'UTC')::date
WHERE da.calendar_kind NOT IN ({_ALWAYS_OPEN_SQL})
  AND c.is_trading_day IS NOT TRUE
  {{scope}}
ORDER BY session_date
"""

WEEKEND_ONLY = "AND EXTRACT(dow FROM (d.time AT TIME ZONE 'UTC')::date) IN (0, 6)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="muta; el default es dry-run")
    ap.add_argument("--include-holidays", action="store_true",
                    help="incluye festivos (reescribe historia de produccion)")
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

    scope = "" if a.include_holidays else WEEKEND_ONLY
    sql = OFFENDERS_SQL.format(scope=scope)
    cur.execute(sql)
    rows = cur.fetchall()
    label = "fin de semana + festivos" if a.include_holidays else "fin de semana"
    print(f"=== barras en dia NO habil ({label}, activos no-24/7): {len(rows)}")
    if not a.include_holidays:
        cur.execute(OFFENDERS_SQL.format(scope=""))
        print(f"    (con --include-holidays serian {len(cur.fetchall())}: los "
              "festivos reescriben historia de produccion, decision del operador)")
    by_asset: dict[str, int] = {}
    for sym, aid, cal, d, src in rows:
        by_asset[aid] = by_asset.get(aid, 0) + 1
    for aid, n in sorted(by_asset.items()):
        print(f"    {aid}: {n}")
    for sym, aid, cal, d, src in rows[:20]:
        print(f"      {aid:9s} {d}  cal={cal:12s} src={src}")
    if len(rows) > 20:
        print(f"      ... y {len(rows) - 20} mas")

    if not rows:
        print("nada que hacer")
        conn.rollback()
        return 0
    if not a.apply:
        print("\nDRY-RUN (usa --apply para ejecutar)")
        conn.rollback()
        return 0

    cur.execute(f"CREATE TABLE IF NOT EXISTS {BACKUP} "
                f"(LIKE asset_daily_ohlcv INCLUDING DEFAULTS)")
    # Clave natural `(time, symbol)` — NUNCA ctid: `asset_daily_ohlcv` es una hypertable
    # de TimescaleDB con miles de chunks y ctid solo es unico dentro de un chunk.
    key_sql = f"SELECT d.time, d.symbol FROM ({sql}) x " \
              f"JOIN asset_daily_ohlcv d ON d.symbol = x.symbol " \
              f"AND (d.time AT TIME ZONE 'UTC')::date = x.session_date"
    cur.execute(f"INSERT INTO {BACKUP} SELECT * FROM asset_daily_ohlcv "
                f"WHERE (time, symbol) IN ({key_sql})")
    cur.execute(f"DELETE FROM asset_daily_ohlcv WHERE (time, symbol) IN ({key_sql})")
    deleted = cur.rowcount

    cur.execute(sql)
    left = cur.fetchall()
    if left:
        conn.rollback()
        print(f"VERIFICACION FALLIDA, rollback: quedan {len(left)} barras en dia no habil")
        return 1

    conn.commit()
    print(f"\nAPLICADO: {deleted} barras borradas. Backup en {BACKUP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
