"""Inventario de LLENADO de la base: tabla por tabla, columna por columna.

QUE RESPONDE
------------
"Estan todas las tablas consolidadas y con todos los campos llenos?" La respuesta honesta no
es un si/no: es, por cada tabla, cuantas filas tiene y cuantas de sus columnas estan
COMPLETAMENTE vacias (NULL en el 100% de las filas) o parcialmente vacias.

POR QUE NO BASTA CON `count(*)`
-------------------------------
Una tabla con un millon de filas y la mitad de sus columnas a NULL no esta "llena": esta
poblada por un writer que solo escribe una parte del contrato. Ese es el fallo que un conteo
de filas no ve, y es el que rompe los consumidores aguas abajo.

Uso:
    python scripts/diagnostics/inventory_db_fill.py                # todo menos infra
    python scripts/diagnostics/inventory_db_fill.py --incluir-infra
    python scripts/diagnostics/inventory_db_fill.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg2

# Esquemas y prefijos que NO son datos del negocio: metadatos de Airflow y de Flask-AppBuilder.
# Se excluyen por defecto porque "estan llenos" o no es una propiedad de los pipelines.
INFRA_PREFIJOS = (
    "public.ab_", "public.dag", "public.log", "public.job", "public.task",
    "public.xcom", "public.sla_", "public.import_error", "public.connection",
    "public.variable", "public.slot_pool", "public.serialized_dag", "public.rendered_",
    "public.session", "public.alembic_version", "public.callback_request",
    "public.dataset", "public.trigger", "public.dagrun_", "public.dag_",
)


def conectar():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", "admin123"),
        connect_timeout=10,
    )


def es_infra(nombre: str) -> bool:
    return any(nombre.startswith(p) for p in INFRA_PREFIJOS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incluir-infra", action="store_true")
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--solo-vacias", action="store_true", help="listar solo tablas con 0 filas")
    args = ap.parse_args()

    conn = conectar()
    conn.set_session(readonly=True, autocommit=True)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_schema, table_name FROM information_schema.tables
            WHERE table_type='BASE TABLE'
              AND table_schema NOT IN ('pg_catalog','information_schema',
                  '_timescaledb_internal','_timescaledb_catalog','_timescaledb_config',
                  '_timescaledb_cache')
            ORDER BY 1,2;""")
        tablas = [(s, t) for s, t in cur.fetchall()]

        cur.execute("""
            SELECT table_schema, table_name, column_name, is_nullable, data_type
            FROM information_schema.columns
            WHERE table_schema NOT IN ('pg_catalog','information_schema')
            ORDER BY ordinal_position;""")
        cols: dict[tuple, list] = {}
        for s, t, c, nul, dt in cur.fetchall():
            cols.setdefault((s, t), []).append((c, nul == "YES", dt))

        filas = []
        for s, t in tablas:
            nombre = f"{s}.{t}"
            if not args.incluir_infra and es_infra(nombre):
                continue
            try:
                cur.execute(f'SELECT count(*) FROM "{s}"."{t}";')
                n = int(cur.fetchone()[0])
            except Exception as exc:  # noqa: BLE001
                filas.append({"tabla": nombre, "error": str(exc).strip()[:120]})
                continue

            columnas = cols.get((s, t), [])
            detalle = []
            if n > 0 and columnas:
                # Un solo barrido: count(col) por columna. count() ignora NULL, asi que
                # n - count(col) son los NULL de esa columna.
                expr = ", ".join(f'count("{c}")' for c, _, _ in columnas)
                cur.execute(f'SELECT {expr} FROM "{s}"."{t}";')
                llenos = cur.fetchone()
                for (c, nullable, dt), lleno in zip(columnas, llenos):
                    detalle.append({
                        "columna": c, "tipo": dt, "nullable": nullable,
                        "no_nulos": int(lleno), "pct_nulo": round(100 * (n - int(lleno)) / n, 2),
                    })
            filas.append({
                "tabla": nombre,
                "filas": n,
                "columnas": len(columnas),
                "cols_100pct_nulas": sum(1 for d in detalle if d["pct_nulo"] == 100.0),
                "cols_parcialmente_nulas": sum(1 for d in detalle if 0 < d["pct_nulo"] < 100),
                "detalle": detalle,
            })
    conn.close()

    con_datos = [f for f in filas if f.get("filas", 0) > 0]
    vacias = [f for f in filas if f.get("filas") == 0]
    errores = [f for f in filas if "error" in f]

    print(f"\n{'='*104}")
    print(f"INVENTARIO DE LLENADO — {len(filas)} tablas de negocio "
          f"({len(con_datos)} con datos · {len(vacias)} VACIAS · {len(errores)} con error)")
    print(f"{'='*104}")

    if not args.solo_vacias:
        print(f"\n{'tabla':<44}{'filas':>12}{'cols':>6}{'100% NULL':>11}{'parc. NULL':>12}")
        for f in sorted(con_datos, key=lambda x: -x["filas"]):
            alerta = "  <--" if f["cols_100pct_nulas"] else ""
            print(f"{f['tabla']:<44}{f['filas']:>12,}{f['columnas']:>6}"
                  f"{f['cols_100pct_nulas']:>11}{f['cols_parcialmente_nulas']:>12}{alerta}")

    print(f"\nTABLAS VACIAS ({len(vacias)}):")
    for f in vacias:
        print(f"  {f['tabla']:<44} {f['columnas']} columnas, 0 filas")

    if errores:
        print(f"\nERRORES ({len(errores)}):")
        for f in errores:
            print(f"  {f['tabla']:<44} {f['error']}")

    huecos = [(f["tabla"], d["columna"], d["tipo"])
              for f in con_datos for d in f["detalle"] if d["pct_nulo"] == 100.0]
    if huecos:
        print(f"\nCOLUMNAS 100% NULAS EN TABLAS CON DATOS ({len(huecos)}) — "
              f"el writer puebla la tabla pero no ese campo:")
        actual = None
        for tabla, col, tipo in huecos:
            if tabla != actual:
                print(f"  {tabla}")
                actual = tabla
            print(f"      {col} ({tipo})")

    if args.json:
        Path(args.json).write_text(json.dumps(filas, indent=2, default=str), encoding="utf-8")
        print(f"\nJSON -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
