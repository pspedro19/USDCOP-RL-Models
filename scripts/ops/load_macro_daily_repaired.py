#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Carga `macro_indicators_daily` desde el backup, reparando la escala FX antes de insertar.

POR QUÉ EXISTE ESTE SCRIPT Y NO SE REUSA OTRO
---------------------------------------------
* `scripts/data/seed_database.py` hace `DELETE FROM usdcop_m5_ohlcv WHERE TRUE` antes de
  cargar. Sobre la base viva eso destruiría **2.206.214 filas** de OHLCV para sustituirlas
  por un seed de 98.160 de un solo activo. Queda PROHIBIDO para restore en vivo.
* `scripts/ops/backup/restore_master.py` carga `.env` y trunca en su fallback CSV.
* `init-scripts/04-data-seeding.py::seed_macro_data` sí es *empty-only* + `ON CONFLICT DO
  NOTHING`, pero lee el backup **tal cual**, y ese fichero contiene las 15 celdas con el
  empalme de escala. Insertaría el daño en la base.

De ahí este cargador acotado: hace una sola cosa, sobre una sola tabla, y **pasa por
`validate_and_repair_macro_scale` antes del INSERT**.

GARANTÍAS
---------
1. **Empty-table-only**: si la tabla tiene filas, aborta. No es un sincronizador ni un
   reparador de historia existente; es un arranque.
2. **La reparación va antes**, y es fail-closed: cualquier empalme no declarado detiene la
   carga en vez de escribir datos sucios (ver `src/data_quality/macro_scale.py`).
3. **Transacción única**: o entran todas las filas o ninguna.
4. `ON CONFLICT (fecha) DO NOTHING`, así que reejecutarlo no duplica.
5. **Provenance por stdout**: fichero de origen, sha256, filas leídas/insertadas y las
   celdas reparadas con su factor.

Uso:
    python scripts/ops/load_macro_daily_repaired.py [--dry-run]

`--dry-run` hace todo salvo el INSERT: sirve para ver el informe de reparación antes de
tocar la base.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
# El paquete `src.data_quality` usa imports ABSOLUTOS (`from src.data_quality.rules ...`),
# así que lo que va al path es el raíz del repo, no `src/`. Con `src/` el import del
# `__init__` revienta con `No module named 'src'`. Los tests no lo notaban porque pytest
# ya deja el raíz en `sys.path`; un script suelto sí.
sys.path.insert(0, str(REPO))

from src.data_quality.macro_scale import (  # noqa: E402
    MacroScaleError,
    manifiesto_backup_2026_06,
    validate_and_repair_macro_scale,
)

ORIGEN = REPO / "data/backups/seeds/macro_indicators_daily_backup.parquet"
TABLA = "macro_indicators_daily"


def _conexion():
    """Conexión desde el entorno. NO lee `.env` ni imprime credenciales."""
    import psycopg2

    faltan = [v for v in ("PGHOST", "PGUSER", "PGDATABASE") if not os.environ.get(v)]
    if faltan:
        raise SystemExit(
            f"faltan variables de entorno {faltan}. Este script no lee `.env` a propósito: "
            f"exportarlas en la sesión que lo invoca"
        )
    return psycopg2.connect(
        host=os.environ["PGHOST"],
        port=int(os.environ.get("PGPORT", 5432)),
        user=os.environ["PGUSER"],
        password=os.environ.get("PGPASSWORD"),
        dbname=os.environ["PGDATABASE"],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="no inserta; sólo informa")
    ap.add_argument(
        "--export-csv",
        metavar="RUTA",
        help=(
            "escribe el frame YA REPARADO a CSV y termina, sin conectar a la base. Es la "
            "via que evita manejar credenciales: el CSV se carga con COPY desde "
            "dentro del contenedor, usando el entorno del propio Postgres"
        ),
    )
    args = ap.parse_args()

    if not ORIGEN.is_file():
        print(f"[FAIL] no existe el origen {ORIGEN}")
        return 1
    sha = hashlib.sha256(ORIGEN.read_bytes()).hexdigest()
    print(f"[origen] {ORIGEN.relative_to(REPO)}  sha256={sha[:16]}")

    frame = pd.read_parquet(ORIGEN)
    print(f"[origen] {len(frame):,} filas, {len(frame.columns)} columnas")

    try:
        reparado, reporte = validate_and_repair_macro_scale(
            frame, manifiesto_backup_2026_06()
        )
    except MacroScaleError as exc:
        print(f"[FAIL] la reparación se negó a continuar: {exc}")
        print("       NO se ha tocado la base de datos.")
        return 1

    print(f"[reparacion] {reporte['n_celdas_reparadas']} celdas · "
          f"filas {reporte['n_filas_entrada']:,} -> {reporte['n_filas_salida']:,}")
    for celda in reporte["celdas_reparadas"]:
        print(f"    {celda['fecha']}  {celda['columna']}  "
              f"{celda['antes']:,.4f} / {celda['factor']:g} = {celda['despues']:,.4f}")

    if args.export_csv:
        destino = Path(args.export_csv)
        reparado.to_csv(destino, index=False)
        print(f"[export] {len(reparado):,} filas -> {destino}")
        print("[export] la base NO se ha tocado; cargar con COPY desde el contenedor")
        return 0

    conexion = _conexion()
    try:
        with conexion, conexion.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {TABLA}")
            existentes = cur.fetchone()[0]
            if existentes:
                print(f"[ABORTA] {TABLA} ya tiene {existentes:,} filas. Este cargador es "
                      f"sólo para arranque: no sincroniza ni repara historia existente.")
                return 1

            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = %s",
                (TABLA,),
            )
            columnas_tabla = {r[0] for r in cur.fetchall()}
            columnas = [c for c in reparado.columns if c in columnas_tabla]
            ausentes = sorted(set(reparado.columns) - columnas_tabla)
            if ausentes:
                print(f"[aviso] {len(ausentes)} columnas del backup no existen en la tabla "
                      f"y NO se cargan: {ausentes[:6]}{' …' if len(ausentes) > 6 else ''}")

            datos = reparado[columnas].where(pd.notna(reparado[columnas]), None)
            filas = [tuple(r) for r in datos.itertuples(index=False, name=None)]
            if args.dry_run:
                print(f"[dry-run] se insertarían {len(filas):,} filas en {TABLA} "
                      f"({len(columnas)} columnas). Base NO tocada.")
                return 0

            marcas = ", ".join(["%s"] * len(columnas))
            nombres = ", ".join(f'"{c}"' for c in columnas)
            cur.executemany(
                f"INSERT INTO {TABLA} ({nombres}) VALUES ({marcas}) "
                f"ON CONFLICT (fecha) DO NOTHING",
                filas,
            )
            cur.execute(f"SELECT count(*) FROM {TABLA}")
            final = cur.fetchone()[0]
        print(f"[OK] {TABLA}: {final:,} filas tras la carga ({len(columnas)} columnas)")
        return 0
    finally:
        conexion.close()


if __name__ == "__main__":
    raise SystemExit(main())
