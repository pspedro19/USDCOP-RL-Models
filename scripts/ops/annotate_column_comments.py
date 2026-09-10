"""Escribe las descripciones de columna en el catalogo de PostgreSQL (`COMMENT ON COLUMN`).

POR QUE EN EL CATALOGO Y NO EN UN CSV APARTE
--------------------------------------------
Un CSV de descripciones se desincroniza del esquema en cuanto alguien anade una columna.
`pg_description` viaja DENTRO del dump, aparece en `\\d+` de psql y lo lee cualquier
herramienta BI. La descripcion deja de ser un anexo y pasa a ser parte de la base.

REGLA QUE NO SE VIOLA
---------------------
**Cada descripcion sale de una fuente declarada o de una convencion documentada. Donde no
hay ninguna, la columna se queda SIN describir.** Rellenar 2.340 columnas con parrafos
plausibles generados a ojo produciria un diccionario que parece completo y miente, que es
exactamente el fallo que este paquete lleva todo el dia corrigiendo. Cada comentario lleva su
procedencia entre parentesis para que el lector pueda ir a comprobarla.

FUENTES, en orden de prioridad
------------------------------
1. `config/macro_variables_ssot.yaml` -> 51 variables macro con display_name, categoria,
   pais, frecuencia, fuente de extraccion, rezago de publicacion y rango esperado.
2. `config.feature_definitions` (tabla) -> descripcion declarada por feature.
3. Convencion estructural: columnas cuyo significado es inequivoco en todo el esquema
   (`time`, `symbol`, OHLCV, `created_at`, ...). Se declaran aqui, no se adivinan.
4. Convencion de nombres macro `dominio_concepto_pais_frecuencia_ticker`, usada SOLO para
   descomponer el nombre; nunca para inferir semantica que el SSOT no declare.

Uso:
    python scripts/ops/annotate_column_comments.py            # aplica
    python scripts/ops/annotate_column_comments.py --dry-run  # solo reporta cobertura
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg2
import yaml

REPO = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------------------
# 3. Convencion estructural. Solo entran columnas cuyo significado es el MISMO en todo el
#    esquema. Una columna ambigua no se describe: se queda vacia.
# --------------------------------------------------------------------------------------
ESTRUCTURALES = {
    "time":            "Sello temporal de la barra u observacion (TIMESTAMPTZ, UTC).",
    "event_time":      "Instante al que se refiere el dato (no cuando se ingirio).",
    "available_at":    "Instante desde el que el dato estuvo DISPONIBLE. Es la columna que "
                       "gobierna la causalidad: nada posterior a `available_at` pudo usarse "
                       "en una decision tomada antes.",
    "provider_published_at": "Instante en que el proveedor publico el dato.",
    "retrieved_at":    "Instante en que este sistema descargo el dato de su fuente.",
    "ingested_at":     "Instante en que el dato se escribio en esta base.",
    "created_at":      "Instante de creacion de la fila en esta base.",
    "updated_at":      "Instante de la ultima modificacion de la fila.",
    "fecha":           "Fecha de OBSERVACION del dato (el periodo que describe), no la de "
                       "publicacion ni la de ingesta.",
    "publication_date": "Fecha de publicacion efectiva. El dato solo puede usarse a partir "
                        "de aqui, no desde `fecha`.",
    "source_date":     "Fecha declarada por la fuente original.",
    "symbol":          "Simbolo del instrumento (p.ej. `USD/COP`, `XAU/USD`, `BTC/USDT`).",
    "provider_symbol": "Simbolo tal como lo nombra el proveedor, antes de normalizar.",
    "open":            "Precio de apertura de la barra.",
    "high":            "Precio maximo de la barra.",
    "low":             "Precio minimo de la barra.",
    "close":           "Precio de cierre de la barra.",
    "volume":          "Volumen negociado en la barra.",
    "source":          "Identificador de la fuente de datos que produjo la fila.",
    "tf":              "Marco temporal de la barra (`1h`, `4h`, `1month`).",
    "interval_id":     "Intervalo de la barra en notacion ISO-8601 de duracion (p.ej. `P1D`).",
    "is_complete":     "Marca si la fila tiene todas sus series pobladas para esa fecha.",
    "ffill_count":     "Numero de valores rellenados hacia adelante en esta fila. Un valor "
                       "alto indica que la fila es mayoritariamente arrastre, no observacion.",
    "metadata":        "Carga util opaca especifica del productor. Ningun consumidor debe "
                       "ramificar por su contenido.",
    "source_payload_hash": "Hash del payload original recibido, para trazabilidad.",
    "source_uri":      "URI que identifica de donde salio la fila.",
    "strategy_id":     "Clave universal de la estrategia. Determina bundle, ficheros de "
                       "trades y badge en el dashboard.",
    "exit_reason":     "Motivo de cierre de la operacion. Vocabulario CERRADO: take_profit, "
                       "trailing_stop, hard_stop, week_end, session_close, circuit_breaker, "
                       "no_bars.",
    "leverage":        "Apalancamiento aplicado a la posicion, ya final (con vol-target).",
    "pnl_pct":         "Resultado de la operacion en porcentaje.",
    "pnl_usd":         "Resultado de la operacion en dolares.",
}

# Categorias tal como las escribe el SSOT -> etiqueta legible. La lista se comprobo contra
# los valores REALES del fichero (`Counter` sobre identity.category), no contra los que yo
# suponia: once categorias que existen en el SSOT no estaban en la primera version y salian
# en ingles crudo dentro de una descripcion en castellano.
CATEGORIAS = {
    "fixed_income": "renta fija", "fx": "divisas", "exchange_rates": "tipos de cambio",
    "commodity": "materias primas", "commodities": "materias primas",
    "equity": "renta variable", "equities": "renta variable",
    "volatility": "volatilidad", "credit_risk": "riesgo de credito",
    "country_risk": "riesgo pais", "inflation": "inflacion",
    "labor": "empleo", "labor_market": "mercado laboral",
    "monetary": "politica monetaria", "policy_rate": "tasa de politica",
    "policy_rates": "tasas de politica", "activity": "actividad",
    "production": "produccion", "money_supply": "agregados monetarios",
    "sentiment": "confianza", "external": "sector externo",
    "trade": "comercio exterior", "foreign_trade": "comercio exterior",
    "growth": "crecimiento", "economic_growth": "crecimiento economico",
    "reserves": "reservas", "reserves_bop": "reservas y balanza de pagos",
    "balance_of_payments": "balanza de pagos",
}

FRECUENCIAS = {"daily": "diaria", "monthly": "mensual", "quarterly": "trimestral",
               "weekly": "semanal", "annual": "anual"}


def desde_ssot() -> dict[str, str]:
    """Descripcion de cada variable macro, compuesta de campos DECLARADOS en el SSOT."""
    ruta = REPO / "config" / "macro_variables_ssot.yaml"
    if not ruta.exists():
        return {}
    doc = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    fuera = {}
    for nombre, v in (doc.get("variables") or {}).items():
        if not isinstance(v, dict):
            continue
        ident = v.get("identity") or {}
        extr = v.get("extraction") or {}
        sched = (v.get("schedule") or {}).get("publication") or {}
        val = v.get("validation") or {}
        ff = v.get("ffill") or {}

        partes = [ident.get("display_name") or nombre]
        rasgos = []
        if ident.get("category"):
            rasgos.append(CATEGORIAS.get(ident["category"], ident["category"]))
        if ident.get("country"):
            rasgos.append(ident["country"])
        if ident.get("frequency"):
            rasgos.append(FRECUENCIAS.get(ident["frequency"], ident["frequency"]))
        if rasgos:
            partes.append(" — " + ", ".join(rasgos) + ".")
        else:
            partes.append(".")

        if extr.get("primary_source"):
            f = f" Fuente: {extr['primary_source']}"
            if extr.get("fallback_source"):
                f += f" (respaldo: {extr['fallback_source']})"
            serie = (extr.get("fred") or {}).get("series_id")
            if serie:
                f += f", serie FRED {serie}"
            partes.append(f + ".")

        if sched.get("delay_days") is not None:
            d = sched["delay_days"]
            partes.append(f" Rezago de publicacion: {d} dia(s)"
                          + (f" ({sched['timezone']})" if sched.get("timezone") else "") + ".")
        if val.get("expected_range"):
            lo, hi = val["expected_range"][0], val["expected_range"][-1]
            partes.append(f" Rango esperado [{lo}, {hi}].")
        if val.get("leakage_risk"):
            partes.append(f" Riesgo de fuga: {val['leakage_risk']}.")
        if ff.get("max_days") is not None:
            partes.append(f" ffill maximo {ff['max_days']} dia(s).")
        partes.append(" (fuente: config/macro_variables_ssot.yaml)")
        fuera[nombre] = "".join(partes)
    return fuera


def desde_features(cur) -> dict[str, str]:
    try:
        cur.execute("SELECT feature_name, description, feature_group, transformation "
                    "FROM config.feature_definitions WHERE description IS NOT NULL;")
    except Exception:  # noqa: BLE001
        return {}
    fuera = {}
    for nombre, desc, grupo, transf in cur.fetchall():
        txt = desc.strip()
        extra = [x for x in (grupo, transf) if x]
        if extra:
            txt += f" [{' · '.join(extra)}]"
        fuera[nombre] = txt + " (fuente: config.feature_definitions)"
    return fuera


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", "admin123"))
    conn.autocommit = True
    cur = conn.cursor()

    ssot = desde_ssot()
    feats = desde_features(cur)
    print(f"fuentes: {len(ssot)} variables macro (SSOT) · {len(feats)} features (DB) · "
          f"{len(ESTRUCTURALES)} columnas estructurales declaradas")

    cur.execute("""
        SELECT c.table_schema, c.table_name, c.column_name
        FROM information_schema.columns c
        JOIN information_schema.tables t
          ON t.table_schema=c.table_schema AND t.table_name=c.table_name
         AND t.table_type='BASE TABLE'
        WHERE c.table_schema NOT IN ('pg_catalog','information_schema','_timescaledb_internal',
              '_timescaledb_catalog','_timescaledb_config','_timescaledb_cache')
        ORDER BY 1,2,3;""")
    columnas = [r for r in cur.fetchall()
                if not (r[0] == "public" and r[1].startswith(
                    ("ab_", "dag", "log", "job", "task", "xcom", "sla_", "import_error",
                     "connection", "variable", "slot_pool", "serialized_dag", "rendered_",
                     "session", "alembic_version", "callback_request", "dataset", "trigger",
                     "dagrun_")))]

    por_fuente = {"ssot_macro": 0, "features": 0, "estructural": 0, "sin_fuente": 0}
    aplicados = 0
    for s, t, c in columnas:
        if c in ssot:
            texto, origen = ssot[c], "ssot_macro"
        elif c in feats:
            texto, origen = feats[c], "features"
        elif c in ESTRUCTURALES:
            texto, origen = ESTRUCTURALES[c] + " (convencion estructural declarada)", "estructural"
        else:
            por_fuente["sin_fuente"] += 1
            continue
        por_fuente[origen] += 1
        if not args.dry_run:
            cur.execute(f'COMMENT ON COLUMN "{s}"."{t}"."{c}" IS %s;', (texto,))
            aplicados += 1

    total = len(columnas)
    descritas = total - por_fuente["sin_fuente"]
    print(f"\ncolumnas de negocio: {total}")
    print(f"  descritas: {descritas} ({100*descritas/total:.1f}%)")
    for k in ("ssot_macro", "features", "estructural"):
        print(f"      {k:<12} {por_fuente[k]}")
    print(f"  SIN FUENTE, se dejan vacias: {por_fuente['sin_fuente']} "
          f"({100*por_fuente['sin_fuente']/total:.1f}%)")
    if not args.dry_run:
        print(f"\n{aplicados} COMMENT ON COLUMN aplicados al catalogo.")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
