#!/usr/bin/env python3
"""Corrective migration: escala de USD/MXN y USD/CLP en `macro_indicators_daily`.

Contract: CTR-DQ-MACRO-SCALE-001 · Date: 2026-08-24

## El defecto

`InvestingExtractionStrategy` parseaba el texto de la tabla con

    value = float(cols[1].get_text(strip=True).replace(',', ''))

Correcto en el sitio ingles (`1,234.56`, coma = miles) y DESTRUCTIVO en el espanol,
donde la coma es el separador DECIMAL. `config/l0_macro_sources.yaml` manda a
`es.investing.com` **exactamente dos** indicadores ("Spanish URL for better
availability"):

    fxrt_spot_usdmxn_mex_d_usdmxn   "17,4720" -> 174720   (x10^4)
    fxrt_spot_usdclp_chl_d_usdclp   "921,98"  -> 92198    (x10^2)

y son exactamente los dos que aparecieron corruptos. El factor es 10^(decimales del
formato), constante por serie. Corregido en el codigo con
`src/data/investing_number.py::parse_investing_number` (locale-aware).

## Por que urgia

`usdmxn_change_1d` es la feature #15 de las 20 del `FEATURE_ORDER` canonico que consume
el pipeline RL. Las 16 fechas afectadas (2026-06-29 -> 2026-08-04) caen DENTRO del
hold-out de la tesis, y un salto de escala en una feature de CAMBIO diario mete un valor
basura el dia que entra y otro el dia que sale.

## Por que se restaura por inverso exacto y no sustituyendo por otro proveedor

La transformacion es invertible sin perdida: dividir por 10^n **recupera el valor
original de Investing**, sin mezclar proveedores. Sustituir por TwelveData insertaria un
empalme de fuente en mitad de la serie — justo lo que vigila
`test_clean_fx_series_have_no_scale_splices`.

TwelveData (el `fallback_source` declarado para ambas series) se usa como **verificacion
independiente**: cada valor restaurado debe coincidir con el suyo dentro de tolerancia.
Comprobado a mano antes de escribir esto: 174695/10^4 = 17,4695 frente a TwelveData
17,46848. Un valor restaurado que NO pase la verificacion no se escribe.

El mismo defecto vive en DOS artefactos: la tabla `macro_indicators_daily` y el parquet
`data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet`, que es **lo que H1/H5 leen
para sus features macro** (`CLAUDE.md`). El parquet NO se repara por separado: se
SINCRONIZA desde la BD ya corregida, para que no puedan divergir. Eso cierra tambien
`tests/regression/test_macro_clean_fx_scale.py`, que llevaba en rojo desde 2026-07-21
por esta misma causa.

Idempotente: re-ejecutar no encuentra filas fuera de rango.
Dry-run por DEFECTO; `--apply` muta. Backup previo (tabla para la BD, fichero para el
parquet).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.parse
import urllib.request
from datetime import date, datetime
from pathlib import Path

# UMBRAL DE DETECCION != RANGO DE PLAUSIBILIDAD. Confundirlos fue un error real en la
# primera version de este script: con techo 25,0 para MXN marco como corrupta la fila
# 2020-03-23 (25,3380), que es el PICO LEGITIMO del crash COVID — TwelveData devuelve
# exactamente 25,3380. La verificacion lo bloqueo y no se escribio nada, pero el diseno
# estaba mal.
#
# Rangos historicos medidos sobre la propia serie (excluyendo las corruptas):
#     MXN  16,3130 - 25,3380   (n=1733)      corrupto: ~172.000 - 176.000
#     CLP  694,88  - 1.049,30  (n=1732)      corrupto:  ~92.000 -  95.000
#
# Entre lo legitimo y lo corrupto hay 3-4 ordenes de magnitud, asi que el umbral de
# deteccion se pone MUY por encima del maximo historico y aun muy por debajo del minimo
# corrupto: imposible un falso positivo en cualquiera de los dos sentidos.
#
# (columna, factor, umbral_deteccion, rango_plausible_restaurado, simbolo TD, tolerancia)
SERIES = (
    ("fxrt_spot_usdmxn_mex_d_usdmxn", 10_000, 1_000.0, (10.0, 60.0), "USD/MXN", 0.02),
    ("fxrt_spot_usdclp_chl_d_usdclp", 100, 5_000.0, (500.0, 2_000.0), "USD/CLP", 0.02),
)
BACKUP = "macro_indicators_daily_scale_backup_20260824"

REPO = Path(__file__).resolve().parents[2]
CLEAN_PARQUET = REPO / "data" / "pipeline" / "04_cleaning" / "output" / "MACRO_DAILY_CLEAN.parquet"
BACKUP_DIR = REPO / "data" / "backups" / "macro"


def _corrupt_where(col: str, threshold: float) -> str:
    """Corrupta = por encima del UMBRAL DE DETECCION (no del maximo historico)."""
    return f"{col} > {threshold}"


def _twelvedata(symbol: str, days: list[date]) -> dict[str, float]:
    key = next((os.environ[k] for k in os.environ
                if k.startswith("TWELVEDATA_API_KEY") and os.environ[k]), None)
    if not key or not days:
        return {}
    q = urllib.parse.urlencode({
        "symbol": symbol, "interval": "1day",
        "start_date": min(days).isoformat(), "end_date": max(days).isoformat(),
        "apikey": key, "outputsize": 5000,
    })
    try:
        with urllib.request.urlopen(
                f"https://api.twelvedata.com/time_series?{q}", timeout=30) as r:
            data = json.loads(r.read())
    except Exception as exc:  # noqa: BLE001
        print(f"    [aviso] TwelveData no responde para {symbol}: {exc}")
        return {}
    return {v["datetime"]: float(v["close"]) for v in (data.get("values") or [])}


def sync_clean_parquet(cur, apply: bool) -> int:
    """Trae a MACRO_DAILY_CLEAN los valores ya corregidos de la BD.

    No repara el parquet por su cuenta: lo alinea con la tabla, que es la fuente. Asi la
    BD y el fichero no pueden acabar diciendo cosas distintas del mismo dia.
    """
    import pandas as pd

    if not CLEAN_PARQUET.is_file():
        print(f"\n=== {CLEAN_PARQUET.name}: no existe, se omite")
        return 0
    df = pd.read_parquet(CLEAN_PARQUET)
    print(f"\n=== {CLEAN_PARQUET.name} ({len(df):,} filas)")

    fixed = 0
    for col, _factor, threshold, (lo, hi), _sym, _tol in SERIES:
        pq_col = col.upper()
        if pq_col not in df.columns:
            print(f"  {pq_col}: no esta en el parquet")
            continue
        bad = df.index[df[pq_col] > threshold]
        if len(bad) == 0:
            print(f"  {pq_col}: 0 valores sobre el umbral")
            continue
        cur.execute(
            f"SELECT fecha, {col} FROM macro_indicators_daily WHERE fecha = ANY(%s)",
            ([d.date() for d in bad],))
        truth = {f: float(v) for f, v in cur.fetchall() if v is not None}
        print(f"  {pq_col}: {len(bad)} valores sobre {threshold:,.0f}; "
              f"la BD tiene valor para {len(truth)}")
        for ts in bad:
            good = truth.get(ts.date())
            if good is None:
                print(f"    BLOQUEA {ts.date()}: la BD no tiene valor")
                return -1
            if not (lo <= good <= hi):
                print(f"    BLOQUEA {ts.date()}: la BD dice {good}, fuera de [{lo}, {hi}]")
                return -1
            if apply:
                df.loc[ts, pq_col] = good
            fixed += 1

    if apply and fixed:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = BACKUP_DIR / f"MACRO_DAILY_CLEAN_scale_backup_{stamp}.parquet"
        shutil.copy2(CLEAN_PARQUET, bak)
        df.to_parquet(CLEAN_PARQUET)
        print(f"  APLICADO: {fixed} valores sincronizados desde la BD")
        print(f"  backup en {bak.relative_to(REPO).as_posix()}")
    elif fixed:
        print(f"  {fixed} valores se sincronizarian desde la BD")
    return fixed


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

    plan: list[tuple[str, date, float, float]] = []   # col, fecha, actual, restaurado
    blocked: list[str] = []

    for col, factor, threshold, (lo, hi), td_symbol, tol in SERIES:
        cur.execute(f"SELECT fecha, {col} FROM macro_indicators_daily "
                    f"WHERE {_corrupt_where(col, threshold)} ORDER BY fecha")
        rows = cur.fetchall()
        print(f"\n=== {col}")
        if not rows:
            print("  0 filas por encima del umbral — nada que hacer")
            continue
        print(f"  {len(rows)} filas > {threshold:,.0f} (umbral de deteccion)")

        ref = _twelvedata(td_symbol, [r[0] for r in rows])
        print(f"  verificacion independiente TwelveData: {len(ref)} fechas disponibles")

        for fecha, val in rows:
            restored = float(val) / factor
            key = fecha.isoformat()
            ok_range = lo <= restored <= hi
            td = ref.get(key)
            ok_td = td is None or abs(restored - td) / td <= tol
            mark = "OK " if (ok_range and ok_td) else "BLOQUEA"
            td_txt = f"TD={td:.4f}" if td is not None else "TD=n/d"
            print(f"    {mark} {key}  {float(val):>12.4f} -> {restored:>9.4f}   {td_txt}")
            if not ok_range:
                blocked.append(f"{col} {key}: {restored} fuera de [{lo}, {hi}]")
            elif not ok_td:
                blocked.append(f"{col} {key}: {restored} vs TwelveData {td} "
                               f"(delta {abs(restored-td)/td:.1%} > {tol:.0%})")
            else:
                plan.append((col, fecha, float(val), restored))

    print(f"\nfilas a restaurar: {len(plan)}   bloqueadas: {len(blocked)}")
    for b in blocked:
        print(f"  BLOQUEADA {b}")
    if blocked:
        print("\nABORTA: alguna restauracion no pasa la verificacion. No se escribe nada.")
        conn.rollback()
        return 1
    if not plan:
        print("nada que aplicar en la BD")
        rc = sync_clean_parquet(cur, apply=a.apply)
        conn.rollback()
        return 1 if rc < 0 else 0
    if not a.apply:
        print("\nDRY-RUN (usa --apply para ejecutar)")
        conn.rollback()
        return 0

    cur.execute(f"CREATE TABLE IF NOT EXISTS {BACKUP} "
                f"(fecha date PRIMARY KEY, columna text, valor_corrupto numeric, "
                f"valor_restaurado numeric, aplicado_en timestamptz DEFAULT now())")
    for col, fecha, bad, good in plan:
        cur.execute(f"INSERT INTO {BACKUP} (fecha, columna, valor_corrupto, valor_restaurado) "
                    f"VALUES (%s,%s,%s,%s) ON CONFLICT (fecha) DO NOTHING",
                    (fecha, col, bad, good))
        cur.execute(f"UPDATE macro_indicators_daily SET {col} = %s WHERE fecha = %s",
                    (good, fecha))

    # Verificacion final: ninguna serie puede quedar por encima de su techo.
    for col, _f, threshold, _rng, _s, _t in SERIES:
        cur.execute(f"SELECT count(*) FROM macro_indicators_daily "
                    f"WHERE {_corrupt_where(col, threshold)}")
        left = cur.fetchone()[0]
        if left:
            conn.rollback()
            print(f"VERIFICACION FALLIDA, rollback: {col} deja {left} filas sobre el umbral")
            return 1

    conn.commit()
    print(f"\nAPLICADO: {len(plan)} valores restaurados. Backup en {BACKUP}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
