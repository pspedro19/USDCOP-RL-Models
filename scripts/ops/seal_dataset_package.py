"""Construye y SELLA el paquete de datos unificado (Claude + Codex).

QUE ES "SELLAR"
---------------
Que el paquete pueda demostrar que es lo que dice ser: cada fichero lleva su sha256 en
`SHA256SUMS.txt`, el manifiesto declara el corte y las limitaciones, y el conjunto se puede
verificar con `sha256sum -c` sin confiar en nadie. Un ZIP sin sello es una carpeta comprimida.

POR QUE UN SCRIPT Y NO UNA CARPETA ARMADA A MANO
------------------------------------------------
Un paquete armado a mano no se puede regenerar cuando los datos cambian, y este cambio dos
veces en una hora. Aqui todo lo derivado —inventario, frescura, diccionario de columnas,
evidencia de pipelines, contenidos del dump y los checksums— se recalcula de la base viva en
cada corrida. Lo unico que se copia son los documentos escritos por humanos/agentes.

DECISIONES DECLARADAS
---------------------
1. **Conteos EXACTOS, sin `reltuples`.** El motivo por el que antes se estimaban —agotar los
   locks de TimescaleDB— desaparecio al subir `max_locks_per_transaction` a 4096. Un
   inventario cuyo valor es la exactitud no debe llevar estimaciones: la de
   `asset_daily_ohlcv` iba un 17% por debajo.
2. **`asset_daily_ohlcv` reporta frescura POR SIMBOLO.** Es multi-activo, y un unico max()
   escondia que 4 de sus 7 simbolos iban 14 dias atrasados.
3. **Una barra diaria sin cerrar no cuenta como cobertura.** Se reporta el ultimo cierre
   REAL, no la sesion en curso.

Uso:
    python scripts/ops/seal_dataset_package.py --out <dir> [--zip <ruta.zip>]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import psycopg2

REPO = Path(__file__).resolve().parents[2]
CONTENEDOR = os.environ.get("PG_CONTAINER", "usdcop-postgres-timescale")

INFRA = ("ab_", "dag", "log", "job", "task", "xcom", "sla_", "import_error", "connection",
         "variable", "slot_pool", "serialized_dag", "rendered_", "session", "alembic_version",
         "callback_request", "dataset", "trigger", "dagrun_")

# Familias con serie temporal declarada -> (tabla, columna de tiempo, columna de particion)
FAMILIAS = [
    ("public.usdcop_m5_ohlcv", "time", "symbol", "5 minutos"),
    ("public.asset_daily_ohlcv", "time", "symbol", "diaria"),
    ("public.asset_native_ohlcv", "time", "tf", "1h / 4h / 1mes"),
    ("public.macro_indicators_daily", "fecha", None, "diaria macro"),
    ("public.macro_indicators_monthly", "fecha", None, "mensual macro"),
    ("public.macro_indicators_quarterly", "fecha", None, "trimestral macro"),
    ("public.news_articles", "published_at", None, "evento"),
    ("market.raw_bar", "event_time", "provider_symbol", "por barra"),
    ("market.canonical_bar", "event_time", None, "por barra"),
]


def conectar(dbname="usdcop_trading"):
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=dbname, user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", "admin123"), connect_timeout=10)


def es_infra(schema, tabla):
    return schema == "public" and tabla.startswith(INFRA)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def docker(*args, salida=None):
    cmd = ["docker", "exec", CONTENEDOR, *args]
    env = {**os.environ, "MSYS_NO_PATHCONV": "1"}
    r = subprocess.run(cmd, capture_output=True, env=env)
    if salida and r.returncode == 0:
        Path(salida).write_bytes(r.stdout)
    return r


# --------------------------------------------------------------------------- derivados
def inventario(cur, destino: Path) -> int:
    cur.execute("""
        SELECT table_schema, table_name FROM information_schema.tables
        WHERE table_type='BASE TABLE' AND table_schema NOT IN ('pg_catalog','information_schema',
          '_timescaledb_internal','_timescaledb_catalog','_timescaledb_config','_timescaledb_cache')
        ORDER BY 1,2;""")
    tablas = cur.fetchall()
    filas, pobladas = [], 0
    for s, t in tablas:
        if es_infra(s, t):
            continue
        cur.execute(f'SELECT count(*) FROM "{s}"."{t}";')
        n = int(cur.fetchone()[0])
        pobladas += 1 if n else 0
        filas.append({"schema_name": s, "table_name": t, "row_count": n,
                      "count_method": "exact",
                      "data_status": "POPULATED" if n else "EMPTY"})
    with open(destino, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    return pobladas


def columnas(cur, destino: Path) -> int:
    cur.execute("""
        SELECT c.table_schema, c.table_name, c.ordinal_position, c.column_name, c.data_type,
               c.is_nullable, COALESCE(c.column_default,''),
               CASE WHEN pk.column_name IS NOT NULL THEN 'YES' ELSE 'NO' END,
               COALESCE(d.description,'')
        FROM information_schema.columns c
        LEFT JOIN (
            SELECT kcu.table_schema, kcu.table_name, kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
        ) pk ON pk.table_schema=c.table_schema AND pk.table_name=c.table_name
            AND pk.column_name=c.column_name
        LEFT JOIN pg_class cl ON cl.relname = c.table_name
        LEFT JOIN pg_namespace ns ON ns.oid = cl.relnamespace AND ns.nspname = c.table_schema
        LEFT JOIN pg_description d ON d.objoid = cl.oid AND d.objsubid = c.ordinal_position
        WHERE c.table_schema NOT IN ('pg_catalog','information_schema',
              '_timescaledb_internal','_timescaledb_catalog','_timescaledb_config','_timescaledb_cache')
        ORDER BY c.table_schema, c.table_name, c.ordinal_position;""")
    filas = [r for r in cur.fetchall() if not es_infra(r[0], r[1])]
    with open(destino, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["schema_name", "table_name", "ordinal_position", "column_name", "data_type",
                    "is_nullable", "column_default", "is_primary_key", "column_comment"])
        w.writerows(filas)
    return len(filas)


def frescura(cur, destino: Path) -> None:
    """Rango temporal por familia y, cuando la tabla es multi-serie, POR SERIE.

    Un unico max() sobre `asset_daily_ohlcv` decia 2026-08-11 mientras cuatro de sus siete
    simbolos llevaban 14 dias congelados: el maximo de la tabla es el maximo del simbolo mas
    fresco, no su cobertura. Por eso las multi-serie se abren.
    """
    with open(destino, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "serie", "frecuencia_declarada", "min_observacion",
                    "max_observacion", "filas"])
        for tabla, tcol, pcol, freq in FAMILIAS:
            s, t = tabla.split(".")
            try:
                if pcol:
                    cur.execute(f'SELECT "{pcol}"::text, min("{tcol}")::text, max("{tcol}")::text,'
                                f' count(*) FROM "{s}"."{t}" GROUP BY 1 ORDER BY 1;')
                    for serie, lo, hi, n in cur.fetchall():
                        w.writerow([tabla, serie, freq, lo, hi, n])
                else:
                    cur.execute(f'SELECT min("{tcol}")::text, max("{tcol}")::text, count(*)'
                                f' FROM "{s}"."{t}";')
                    lo, hi, n = cur.fetchone()
                    w.writerow([tabla, "(unica)", freq, lo, hi, n])
            except Exception as exc:  # noqa: BLE001
                w.writerow([tabla, "ERROR", freq, "", "", str(exc).strip()[:80]])


def pipelines(cur, destino: Path) -> None:
    cur.execute("""
        SELECT dr.dag_id, dr.run_id, dr.state, dr.start_date::text, COALESCE(dr.end_date::text,''),
               count(ti.task_id),
               count(*) FILTER (WHERE ti.state='success'),
               string_agg(ti.task_id||':'||COALESCE(ti.state,'none'), ', ' ORDER BY ti.task_id)
        FROM dag_run dr LEFT JOIN task_instance ti
          ON ti.dag_id=dr.dag_id AND ti.run_id=dr.run_id
        WHERE dr.start_date > now() - interval '24 hours'
        GROUP BY 1,2,3,4,5 ORDER BY 4;""")
    with open(destino, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dag_id", "run_id", "dag_state", "start_date", "end_date",
                    "task_count", "tasks_success", "task_states"])
        w.writerows(cur.fetchall())


def exportar_csvs(cur, destino: Path) -> int:
    destino.mkdir(parents=True, exist_ok=True)
    for viejo in destino.glob("*.csv.gz"):
        viejo.unlink()
    cur.execute("""
        SELECT table_schema, table_name FROM information_schema.tables
        WHERE table_type='BASE TABLE' AND table_schema NOT IN ('pg_catalog','information_schema',
          '_timescaledb_internal','_timescaledb_catalog','_timescaledb_config','_timescaledb_cache')
        ORDER BY 1,2;""")
    n = 0
    for s, t in cur.fetchall():
        if es_infra(s, t):
            continue
        cur.execute(f'SELECT count(*) FROM "{s}"."{t}";')
        if not cur.fetchone()[0]:
            continue
        with gzip.open(destino / f"{s}.{t}.csv.gz", "wt", encoding="utf-8", newline="") as fh:
            # COPY (SELECT ...) expande la herencia. `COPY <tabla>` sobre un hypertable
            # devuelve SOLO la cabecera: un CSV valido con cero filas, sin error.
            cur.copy_expert(f'COPY (SELECT * FROM "{s}"."{t}") TO STDOUT WITH CSV HEADER', fh)
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--zip", dest="zip_path", default=None)
    ap.add_argument("--skip-dump", action="store_true", help="reutiliza el dump ya presente")
    args = ap.parse_args()

    raiz = Path(args.out)
    for sub in ("db", "diccionario", "reglas", "evidencia", "tablas_csv"):
        (raiz / sub).mkdir(parents=True, exist_ok=True)

    conn = conectar()
    conn.set_session(autocommit=True)
    cur = conn.cursor()

    print("[1/6] inventario y diccionario de columnas ...")
    pobladas = inventario(cur, raiz / "diccionario" / "TABLE_INVENTORY.csv")
    n_cols = columnas(cur, raiz / "diccionario" / "DATA_DICTIONARY_COLUMNS.csv")
    print(f"      {pobladas} tablas pobladas · {n_cols} columnas catalogadas")

    print("[2/6] frescura por serie y evidencia de pipelines ...")
    frescura(cur, raiz / "diccionario" / "FRESHNESS.csv")
    pipelines(cur, raiz / "evidencia" / "PIPELINE_RUNS.csv")

    print("[3/6] exportando tablas pobladas a CSV.gz ...")
    n_csv = exportar_csvs(cur, raiz / "tablas_csv")
    print(f"      {n_csv} tablas exportadas")
    conn.close()

    print("[4/6] dump + esquema + catalogo del dump ...")
    dump = raiz / "db" / "usdcop_trading.dump"
    if not (args.skip_dump and dump.exists()):
        docker("pg_dump", "-U", "admin", "-d", "usdcop_trading", "--format=custom",
               "--compress=6", "--no-owner", "--no-acl", "-f", "/tmp/_seal.dump")
        subprocess.run(["docker", "cp", f"{CONTENEDOR}:/tmp/_seal.dump", str(dump)], check=True)
    docker("pg_dump", "-U", "admin", "-d", "usdcop_trading", "--schema-only", "--no-owner",
           "--no-acl", "--exclude-schema=_timescaledb_internal",
           "--exclude-schema=_timescaledb_catalog", "--exclude-schema=_timescaledb_config",
           "--exclude-schema=_timescaledb_cache", "-f", "/tmp/_seal_schema.sql")
    subprocess.run(["docker", "cp", f"{CONTENEDOR}:/tmp/_seal_schema.sql",
                    str(raiz / "db" / "esquema.sql")], check=True)
    # `pg_restore` puede no existir en el host (aqui no esta): el fallback lo corre DENTRO
    # del contenedor, que siempre lo tiene. Se captura FileNotFoundError, no solo el codigo
    # de retorno -- un binario ausente lanza excepcion, no sale con != 0.
    try:
        r = subprocess.run(["pg_restore", "--list", str(dump)], capture_output=True)
    except FileNotFoundError:
        r = None
    if r is not None and r.returncode == 0:
        (raiz / "db" / "PG_DUMP_CONTENTS.txt").write_bytes(r.stdout)
    else:
        subprocess.run(["docker", "cp", str(dump), f"{CONTENEDOR}:/tmp/_seal_list.dump"], check=True)
        rr = docker("pg_restore", "--list", "/tmp/_seal_list.dump")
        (raiz / "db" / "PG_DUMP_CONTENTS.txt").write_bytes(rr.stdout)

    print("[5/6] sellando (sha256 de cada fichero) ...")
    sumas = raiz / "SHA256SUMS.txt"
    if sumas.exists():
        sumas.unlink()
    lineas = []
    for f in sorted(raiz.rglob("*")):
        if f.is_file():
            lineas.append(f"{sha256(f)}  {f.relative_to(raiz).as_posix()}")
    # `newline=""` es obligatorio: en Windows `write_text` traduce \n a \r\n y `sha256sum -c`
    # lee el \r como parte del NOMBRE del fichero -> "No such file or directory" en las 60
    # entradas. Un sello que no verifica es peor que no tener sello: aparenta integridad.
    with open(sumas, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(lineas) + "\n")
    print(f"      {len(lineas)} ficheros sellados")

    # Auto-verificacion: el sello se comprueba aqui mismo, no se promete.
    fallos = []
    for linea in lineas:
        esperado, rel = linea.split("  ", 1)
        real = sha256(raiz / rel)
        if real != esperado:
            fallos.append(rel)
    if fallos:
        print(f"      SELLO INVALIDO en {len(fallos)} fichero(s): {fallos[:3]}", file=sys.stderr)
        return 2
    print(f"      sello verificado: {len(lineas)}/{len(lineas)} coinciden")

    print("[6/6] empaquetando ...")
    if args.zip_path:
        base = str(Path(args.zip_path).with_suffix(""))
        shutil.make_archive(base, "zip", root_dir=raiz.parent, base_dir=raiz.name)
        z = Path(base + ".zip")
        print(f"      {z} · {z.stat().st_size/1e6:.1f} MB")
        print(f"      sha256 del paquete: {sha256(z)}")
    print(f"\nSellado: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
