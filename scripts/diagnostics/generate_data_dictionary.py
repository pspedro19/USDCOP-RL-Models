"""Genera el DICCIONARIO DE DATOS de la base, en Markdown.

QUE PRODUCE
-----------
Un documento por tabla con: dominio, frecuencia MEDIDA (no declarada), rango temporal,
numero de filas, y la lista completa de columnas con tipo, nulabilidad declarada y
**porcentaje real de NULL**.

POR QUE LA FRECUENCIA SE MIDE Y NO SE DECLARA
---------------------------------------------
Decir "macro_indicators_daily es diaria" es una etiqueta; medir la separacion modal entre
sellos consecutivos dice lo que la tabla CONTIENE. Cuando las dos difieren, la que importa
para un consumidor es la medida. Las tablas sin columna temporal se marcan `sin serie`.

POR QUE EL % DE NULL ES PARTE DEL DICCIONARIO
---------------------------------------------
Una columna declarada NULLABLE y 100% nula no esta "permitida vacia": esta SIN POBLAR, y su
consumidor aguas abajo va a romper o a rellenar con un supuesto. El diccionario lo dice en
vez de dejar que se descubra en produccion.

Uso:
    python scripts/diagnostics/generate_data_dictionary.py --out DICCIONARIO-DATOS.md
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import psycopg2

# ---------------------------------------------------------------------------
# Clasificacion por DOMINIO. El orden importa: gana la primera regla que casa.
# ---------------------------------------------------------------------------
DOMINIOS = [
    ("Mercado · OHLCV",        lambda s, t: "ohlcv" in t or s == "market" or t.endswith("_bar")),
    ("Macro",                  lambda s, t: t.startswith("macro_")),
    ("Forecasting · H5",       lambda s, t: t.startswith("forecast_h5_")),
    ("Forecasting · fabric",   lambda s, t: s == "forecast"),
    ("Noticias & Analisis",    lambda s, t: t.startswith("news_") or "analysis" in t),
    ("Features",               lambda s, t: "features" in t or s == "quality"),
    ("Ejecucion & ordenes",    lambda s, t: s == "exec" or t in ("executions", "trades_history", "signals") or t.startswith("sb_") or "execution" in t),
    ("Cartera & riesgo",       lambda s, t: s in ("portfolio", "fact") or "risk" in t or "kill_switch" in t),
    ("Experimentos & modelos", lambda s, t: "experiment" in t or "model" in t or s in ("bi", "metrics", "dw")),
    ("Gobierno & linaje",      lambda s, t: s in ("control", "lineage", "audit", "config", "reference")),
    ("Usuarios & RBAC",        lambda s, t: t.startswith(("users", "user_", "rbac_", "ab_")) or "credential" in t or t == "audit_log"),
    ("Trading state",          lambda s, t: t.startswith("trading_") or t == "equity_snapshots"),
]

INFRA_PREFIJOS = (
    "public.ab_", "public.dag", "public.log", "public.job", "public.task",
    "public.xcom", "public.sla_", "public.import_error", "public.connection",
    "public.variable", "public.slot_pool", "public.serialized_dag", "public.rendered_",
    "public.session", "public.alembic_version", "public.callback_request",
    "public.dataset", "public.trigger", "public.dagrun_",
)

# Frecuencia: (etiqueta, segundos minimos, segundos maximos) sobre la separacion MODAL.
ESCALA = [
    ("1 minuto",      45,        90),
    ("5 minutos",     240,       360),
    ("15 minutos",    800,       1000),
    ("1 hora",        3400,      3800),
    ("diaria",        80000,     95000),
    ("diaria (habil)", 95001,    280000),
    ("semanal",       500000,    700000),
    ("mensual",       2300000,   2800000),
    ("trimestral",    7000000,   8500000),
    ("anual",         30000000,  33000000),
]


def dominio(schema: str, tabla: str) -> str:
    for nombre, test in DOMINIOS:
        if test(schema, tabla):
            return nombre
    return "Otros"


def es_infra(nombre: str) -> bool:
    return any(nombre.startswith(p) for p in INFRA_PREFIJOS)


def clasifica_gap(segundos: float | None) -> str:
    if segundos is None:
        return "—"
    for etiqueta, lo, hi in ESCALA:
        if lo <= segundos <= hi:
            return etiqueta
    if segundos < 45:
        return f"sub-minuto (~{segundos:.0f}s)"
    return f"irregular (~{segundos/86400:.1f} d)"


def conectar():
    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", "admin123"),
        connect_timeout=10,
    )


def columna_temporal(columnas) -> str | None:
    """La columna que define la SERIE, por prioridad de nombre y tipo."""
    tipos_t = ("timestamp with time zone", "timestamp without time zone", "date")
    candidatas = [c for c, _, dt in columnas if dt in tipos_t]
    # `fecha` va en la lista porque `macro_indicators_daily` nombra asi su columna de negocio;
    # sin ella el script caia en `created_at` y reportaba "sub-minuto" para una serie DIARIA
    # -- medir el sello de INGESTA en vez del de observacion invierte el significado.
    for preferida in ("time", "bar_time", "fecha", "signal_date", "inference_date", "trade_date",
                      "date", "observation_date", "published_at", "target_date",
                      "occurred_at", "event_time", "created_at", "timestamp"):
        for c in candidatas:
            if c == preferida:
                return c
    return candidatas[0] if candidatas else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--incluir-infra", action="store_true")
    args = ap.parse_args()

    conn = conectar()
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()

    cur.execute("""
        SELECT table_schema, table_name FROM information_schema.tables
        WHERE table_type='BASE TABLE'
          AND table_schema NOT IN ('pg_catalog','information_schema',
              '_timescaledb_internal','_timescaledb_catalog','_timescaledb_config',
              '_timescaledb_cache')
        ORDER BY 1,2;""")
    tablas = cur.fetchall()

    cur.execute("""
        SELECT table_schema, table_name, column_name, is_nullable, data_type,
               COALESCE(character_maximum_length::text, numeric_precision::text, '')
        FROM information_schema.columns
        WHERE table_schema NOT IN ('pg_catalog','information_schema')
        ORDER BY table_schema, table_name, ordinal_position;""")
    cols: dict[tuple, list] = {}
    for s, t, c, nul, dt, largo in cur.fetchall():
        cols.setdefault((s, t), []).append((c, nul == "YES", dt + (f"({largo})" if largo and dt in ("character varying", "numeric") else "")))

    cur.execute("""
        SELECT c.relname, obj_description(c.oid) FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace WHERE c.relkind='r';""")
    comentarios = {r[0]: r[1] for r in cur.fetchall() if r[1]}

    try:
        cur.execute("SELECT hypertable_name FROM timescaledb_information.hypertables;")
        hypertables = {r[0] for r in cur.fetchall()}
    except Exception:  # noqa: BLE001
        hypertables = set()

    fichas = []
    for s, t in tablas:
        nombre = f"{s}.{t}"
        if not args.incluir_infra and es_infra(nombre):
            continue
        columnas = cols.get((s, t), [])
        try:
            cur.execute(f'SELECT count(*) FROM "{s}"."{t}";')
            n = int(cur.fetchone()[0])
        except Exception as exc:  # noqa: BLE001
            fichas.append({"tabla": nombre, "error": str(exc).strip()[:150], "columnas": columnas})
            continue

        detalle, gap, rango = [], None, None
        if n > 0 and columnas:
            expr = ", ".join(f'count("{c}")' for c, _, _ in columnas)
            cur.execute(f'SELECT {expr} FROM "{s}"."{t}";')
            llenos = cur.fetchone()
            for (c, nullable, dt), lleno in zip(columnas, llenos):
                detalle.append({"columna": c, "tipo": dt, "nullable": nullable,
                                "pct_nulo": round(100 * (n - int(lleno)) / n, 1)})

            tcol = columna_temporal(columnas)
            if tcol:
                cur.execute(f'SELECT min("{tcol}"), max("{tcol}") FROM "{s}"."{t}";')
                lo, hi = cur.fetchone()
                if lo is not None:
                    rango = f"{lo} .. {hi}"
                # separacion MODAL sobre una muestra de sellos distintos
                # Una tabla multi-serie (varios `symbol`) intercala los sellos de todas sus
                # series, y la separacion entre sellos DISTINTOS deja de ser la frecuencia de
                # ninguna de ellas: `asset_daily_ohlcv` daba "1 hora" siendo diaria, porque
                # distintos activos sellan el dia a horas distintas. Se mide UNA serie.
                partic = next((c for c, _, _ in columnas
                               if c in ("symbol", "asset_id", "instrument_id", "model_id",
                                        "strategy_id", "variable", "ticker")), None)
                filtro = ""
                if partic:
                    cur.execute(f'SELECT "{partic}" FROM "{s}"."{t}" GROUP BY 1 '
                                f'ORDER BY count(*) DESC LIMIT 1;')
                    fila = cur.fetchone()
                    if fila and fila[0] is not None:
                        cur.execute("SELECT %s", (fila[0],))
                        filtro = cur.mogrify(f'AND "{partic}" = %s', (fila[0],)).decode()

                # `v::timestamp` normaliza: restar dos DATE da integer (dias), no interval,
                # y EXTRACT(EPOCH FROM integer) no existe. El cast unifica ambos casos.
                cur.execute(f'''
                    SELECT EXTRACT(EPOCH FROM (v::timestamp - lag(v::timestamp) OVER (ORDER BY v)))
                    FROM (SELECT DISTINCT "{tcol}" AS v FROM "{s}"."{t}"
                          WHERE "{tcol}" IS NOT NULL {filtro} ORDER BY 1 DESC LIMIT 4000) q;''')
                gaps = [float(r[0]) for r in cur.fetchall() if r[0] is not None and float(r[0]) > 0]
                if gaps:
                    # modal redondeado a la escala mas cercana, robusto a huecos de fin de semana
                    cuenta = Counter(round(g / 60) for g in gaps)
                    gap = cuenta.most_common(1)[0][0] * 60

        fichas.append({
            "tabla": nombre, "schema": s, "nombre": t, "filas": n,
            "dominio": dominio(s, t), "frecuencia": clasifica_gap(gap),
            "rango": rango, "hypertable": t in hypertables,
            "comentario": comentarios.get(t), "detalle": detalle, "columnas": columnas,
        })
    conn.close()

    # ---------------------------------------------------------------- Markdown
    con_datos = [f for f in fichas if f.get("filas", 0) > 0]
    vacias = [f for f in fichas if f.get("filas") == 0]
    errores = [f for f in fichas if "error" in f]
    generado = datetime.now().astimezone().isoformat(timespec="seconds")

    L = []
    L.append("# Diccionario de datos — `usdcop_trading`\n")
    L.append(f"> Generado por `scripts/diagnostics/generate_data_dictionary.py` el {generado}.\n")
    L.append("> **La frecuencia es MEDIDA, no declarada**: es la separación modal entre sellos\n"
             "> consecutivos de la columna temporal de la tabla. El `% NULL` es real, contado sobre\n"
             "> todas las filas. Una columna declarada nullable y 100% nula no está *permitida\n"
             "> vacía*: está **sin poblar**, y eso es lo que rompe a un consumidor aguas abajo.\n")
    L.append(f"\n**Resumen**: {len(fichas)} tablas de negocio · "
             f"**{len(con_datos)} con datos** · **{len(vacias)} vacías** · {len(errores)} con error.\n")

    # ---- indice por dominio
    L.append("\n## 1. Índice por dominio\n")
    por_dom: dict[str, list] = {}
    for f in fichas:
        por_dom.setdefault(f.get("dominio", "Otros"), []).append(f)
    for dom in sorted(por_dom):
        grupo = sorted(por_dom[dom], key=lambda x: -x.get("filas", 0))
        vivas = sum(1 for g in grupo if g.get("filas", 0) > 0)
        L.append(f"\n### {dom}  ·  {vivas}/{len(grupo)} pobladas\n")
        L.append("| tabla | filas | frecuencia medida | rango temporal | cols | % cols 100% NULL |")
        L.append("|---|---:|---|---|---:|---:|")
        for g in grupo:
            if "error" in g:
                L.append(f"| `{g['tabla']}` | ERROR | — | — | {len(g['columnas'])} | — |")
                continue
            nc = len(g["columnas"])
            vac = sum(1 for d in g["detalle"] if d["pct_nulo"] == 100.0)
            pct = f"{100*vac/nc:.0f}%" if nc and g["filas"] else "—"
            L.append(f"| `{g['tabla']}` | {g['filas']:,} | {g['frecuencia']} | "
                     f"{g['rango'] or '—'} | {nc} | {pct} |")

    # ---- fichas de tablas pobladas
    L.append("\n\n## 2. Fichas — tablas con datos\n")
    for f in sorted(con_datos, key=lambda x: (x["dominio"], -x["filas"])):
        L.append(f"\n### `{f['tabla']}`\n")
        if f["comentario"]:
            L.append(f"> {f['comentario']}\n")
        meta = [f"**Dominio**: {f['dominio']}", f"**Filas**: {f['filas']:,}",
                f"**Frecuencia medida**: {f['frecuencia']}"]
        if f["rango"]:
            meta.append(f"**Rango**: `{f['rango']}`")
        if f["hypertable"]:
            meta.append("**TimescaleDB hypertable**")
        L.append(" · ".join(meta) + "\n")
        L.append("| columna | tipo | nullable | % NULL real | estado |")
        L.append("|---|---|---|---:|---|")
        for d in f["detalle"]:
            if d["pct_nulo"] == 100.0:
                estado = "**SIN POBLAR**"
            elif d["pct_nulo"] == 0.0:
                estado = "completa"
            else:
                estado = "parcial"
            L.append(f"| `{d['columna']}` | {d['tipo']} | {'sí' if d['nullable'] else 'NO'} | "
                     f"{d['pct_nulo']:.1f} | {estado} |")

    # ---- tablas vacias con su razon
    L.append("\n\n## 3. Tablas vacías — y por qué\n")
    L.append("Una tabla vacía no es necesariamente un fallo. Se separan por causa, porque la\n"
             "acción correctiva es distinta en cada caso.\n")
    L.append("\n| tabla | dominio | cols | causa probable |")
    L.append("|---|---|---:|---|")
    for f in sorted(vacias, key=lambda x: (x["dominio"], x["tabla"])):
        dom = f["dominio"]
        if dom in ("Ejecucion & ordenes", "Cartera & riesgo"):
            causa = "el sistema **nunca ha operado en vivo**; poblarla exigiría fabricar trades"
        elif dom == "Usuarios & RBAC":
            causa = "requiere alta de usuarios/credenciales por el operador"
        elif dom == "Forecasting · fabric" or f["schema"] in ("bi", "dw", "control", "lineage"):
            causa = "capa fabric: su **writer no corre** (DAG pausado o no cableado)"
        elif dom == "Noticias & Analisis":
            causa = "requiere correr el pipeline de noticias/análisis (scrapers + LLM)"
        else:
            causa = "pipeline correspondiente no ejecutado en esta base"
        L.append(f"| `{f['tabla']}` | {dom} | {len(f['columnas'])} | {causa} |")

    if errores:
        L.append("\n\n## 4. Tablas con error de lectura\n")
        for f in errores:
            L.append(f"- `{f['tabla']}`: `{f['error']}`")

    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Diccionario -> {args.out}")
    print(f"  {len(fichas)} tablas · {len(con_datos)} con datos · {len(vacias)} vacias · {len(errores)} error")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
