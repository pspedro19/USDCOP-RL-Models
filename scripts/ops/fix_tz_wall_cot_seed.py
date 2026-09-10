#!/usr/bin/env python3
"""Corrective migration para las SEEDS: wall-COT mal etiquetado -> instantes verdaderos.

Contract: CTR-DQ-TZ-001 (hermano de `fix_tz_wall_cot_rows.py`, que hace lo mismo en la BD)

## Por que existe

`fix_tz_wall_cot_rows.py` reparo la BD el 2026-07-21 (15.865 filas). **Las seeds nunca se
regeneraron**, y son lo que leen la tesis, cualquier corrida local y un clon limpio. Medido el
2026-08-24 sobre `seeds/latest/usdcop_m5_ohlcv.parquet`:

    225 sesiones (13,4%) con las barras 5 horas antes de la ventana real.

## El defecto (causa raiz ya aislada)

`airflow/dags/l0_ohlcv_backfill.py` insertaba datetimes **naive COT** en una columna
`TIMESTAMPTZ` con `PGTZ: UTC`. Postgres los leia como UTC, asi que cada barra quedo 5 horas
antes: un cierre de 12:55 COT se guardo como el instante 12:55 UTC = 07:55 COT, ANTES de la
apertura. El codigo ya esta corregido (comentario `ROOT CAUSE of the tri-convention table` en
ese fichero); lo que queda es el dato.

## Por que es reparable sin perdida

Solo la ETIQUETA esta mal. Verificado contra `seeds/latest/usdcop_daily_ohlcv.parquet`, que es
fuente independiente: el open/close de las sesiones rotas coincide con diferencia **0,00**. Y el
salto overnight de las rotas (0,148% medio) es MENOR que el de las sanas (0,286%) — la serie es
continua. La correccion es `+5h`, determinista.

## Discriminador

La BD usa la columna `source` (`twelvedata_backfill` / `_gap_fill` / `_manual_test`). **La seed
no tiene esa columna**, asi que se usa la HORA, que separa igual de limpio porque las dos
convenciones no se solapan:

    03-07 COT  -> convencion PARED (rota)      = la ventana 08-12 desplazada -5h
    08-12 COT  -> convencion INSTANTE (buena)  = la sesion real 08:00-12:55

Alcance (medido): USD/COP 11.304 barras pared, USD/MXN 4.620 (98,7% del par). USD/BRL tiene
CERO porque su rama de fetch pedia UTC y localizaba bien — el mismo quirk que documenta el
script de BD. `XAU/USD` y `BTC/USDT` NO se tocan: por contrato (`data-governance.md`) guardan
instantes 24h, no COT localizado, y su histograma horario es plano por diseno.

## Que hace (calcado del script de BD)

1. BACKUP del parquet completo a `data/backups/seeds/`.
2. COLISIONES: si el hueco `+5h` de una barra pared ya lo ocupa una barra buena, gana la buena
   y se descarta la pared (el backfill rellenaba huecos que percibia bajo el reloj erroneo).
   Medido: 815 colisiones — el mismo numero que reporta el script de BD.
3. `+5h` al resto de barras pared.
4. BORRAR las barras fuera de ambas convenciones (86 en COP, todas de 2026-01-07/08 — el
   "stray day" que el script de BD tambien elimina).
5. VERIFICAR: histograma mono-convencion 08-12 COT por par de sesion, sin duplicados, y cotejo
   del open/close contra el seed diario en las sesiones COMPLETAS (60 barras) — a una sesion
   con huecos no se le puede exigir que cuadre. Sale != 0 si algo no cuadra.

Efecto medido del fix: las sesiones que no cuadran con el seed diario bajan de **55 a 13**, y
esas 13 son todas incompletas (44-59 barras) — huecos de la serie, no timezone.

Idempotente: re-ejecutar no encuentra barras pared y no cambia nada.
Dry-run es el DEFAULT; `--apply` muta.

Uso:
    python scripts/ops/fix_tz_wall_cot_seed.py            # dry-run
    python scripts/ops/fix_tz_wall_cot_seed.py --apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SEEDS = REPO / "seeds" / "latest"
BACKUP_DIR = REPO / "data" / "backups" / "seeds"

# Seeds a reparar. Comparten las mismas filas de COP y deben quedar coherentes.
TARGETS = (SEEDS / "usdcop_m5_ohlcv.parquet", SEEDS / "fx_multi_m5_ohlcv.parquet")

# Pares con sesion 08:00-12:55 COT. XAU/BTC quedan fuera A PROPOSITO (instantes 24h).
SESSION_SYMBOLS = ("USD/COP", "USD/MXN", "USD/BRL")

WALL_LO, WALL_HI = 3, 7        # convencion rota (= sesion desplazada -5h)
SESSION_LO, SESSION_HI = 8, 12  # convencion buena
SHIFT = pd.Timedelta(hours=5)
BARS_PER_SESSION = 60   # 08:00-12:55 COT en pasos de 5 min

DAILY_SEED = SEEDS / "usdcop_daily_ohlcv.parquet"


def _norm(sym: pd.Series) -> pd.Series:
    return sym.astype(str).str.upper().str.replace("/", "", regex=False)


def _classify(df: pd.DataFrame) -> pd.DataFrame:
    """Etiqueta cada fila: wall / session / off (solo para pares de sesion)."""
    out = df.copy()
    out["_t"] = pd.to_datetime(out["time"])
    out["_h"] = out["_t"].dt.hour
    is_session_pair = _norm(out["symbol"]).isin([s.replace("/", "") for s in SESSION_SYMBOLS])
    kind = pd.Series("exempt", index=out.index)   # XAU/BTC: intocables
    kind[is_session_pair & out["_h"].between(WALL_LO, WALL_HI)] = "wall"
    kind[is_session_pair & out["_h"].between(SESSION_LO, SESSION_HI)] = "session"
    kind[is_session_pair & ~out["_h"].between(WALL_LO, SESSION_HI)] = "off"
    out["_kind"] = kind
    return out


def repair(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Devuelve (df_reparado, stats). Pura: no toca disco."""
    w = _classify(df)
    stats = {k: int((w["_kind"] == k).sum()) for k in ("wall", "session", "off", "exempt")}

    # Huecos ya ocupados por una barra buena -> la pared se descarta.
    occupied = set(zip(_norm(w.loc[w["_kind"] != "wall", "symbol"]),
                       w.loc[w["_kind"] != "wall", "_t"]))
    wall_mask = w["_kind"] == "wall"
    target = w.loc[wall_mask, "_t"] + SHIFT
    # `dtype=bool` explicito: con CERO barras pared (re-ejecucion idempotente) una lista
    # vacia produce dtype=object, y `w.index[mask][serie.values]` revienta con
    # "arrays used as indices must be of integer (or boolean) type".
    collides = pd.Series(
        [(s, t) in occupied for s, t in zip(_norm(w.loc[wall_mask, "symbol"]), target)],
        index=w.index[wall_mask],
        dtype=bool,
    )
    stats["collisions_dropped"] = int(collides.sum())
    stats["shifted"] = int(wall_mask.sum() - collides.sum())

    drop_idx = w.index[wall_mask][collides.values]
    shift_idx = w.index[wall_mask][~collides.values]
    off_idx = w.index[w["_kind"] == "off"]

    out = w.drop(index=drop_idx.union(off_idx)).copy()
    out.loc[out.index.intersection(shift_idx), "_t"] += SHIFT
    out["time"] = out["_t"]
    out = out.drop(columns=[c for c in ("_t", "_h", "_kind") if c in out.columns])
    out = out.sort_values(["symbol", "time"]).reset_index(drop=True)
    return out, stats


def verify(df: pd.DataFrame, label: str) -> list[str]:
    """Verificacion dura. Devuelve lista de problemas (vacia == correcto)."""
    problems: list[str] = []
    w = _classify(df)
    sess = w[w["_kind"] != "exempt"]
    bad = sess[~sess["_h"].between(SESSION_LO, SESSION_HI)]
    if len(bad):
        hist = bad.groupby([bad["symbol"], bad["_h"]]).size().to_dict()
        problems.append(f"{label}: {len(bad)} filas fuera de 08-12 COT -> {hist}")

    dup = w.duplicated(subset=["symbol", "time"]).sum()
    if dup:
        problems.append(f"{label}: {dup} timestamps duplicados por (symbol, time)")

    # Cotejo contra el seed diario (fuente INDEPENDIENTE: proveedor distinto).
    #
    # Solo se exige a las sesiones COMPLETAS (60 barras). Una sesion a la que le faltan
    # barras no PUEDE cuadrar: si le faltan las primeras, su `open` no es el open del dia;
    # si le falta la ultima, su `close` no es el close del dia. Medido el 2026-08-24: las 13
    # sesiones que no cuadran tras reparar tienen 44-59 barras, y su desviacion va de 0,58 a
    # 6,00 COP salvo 2023-12-26 (44 barras, le faltan las primeras). Son huecos de la serie de
    # 5 minutos, un problema DISTINTO del de timezone y anterior a esta reparacion — el
    # conteo de sesiones discordantes baja de 55 a 13 precisamente porque el fix funciona.
    # Los huecos los caza `test_seed_ohlcv_integrity.py`; aqui solo se verifica el fix de tz.
    if DAILY_SEED.is_file():
        cop = w[_norm(w["symbol"]) == "USDCOP"].copy()
        if len(cop):
            cop["_d"] = cop["_t"].dt.date
            agg = (cop.sort_values("_t").groupby("_d")
                   .agg(o=("open", "first"), c=("close", "last"), n=("_t", "size")))
            agg = agg[agg["n"] == BARS_PER_SESSION]      # solo sesiones completas
            d = pd.read_parquet(DAILY_SEED)
            d["_d"] = pd.to_datetime(d["time"]).dt.date
            d = d.set_index("_d")
            common = agg.index.intersection(d.index)
            if len(common):
                a = agg.loc[common, ["o", "c"]].astype(float)
                b = d.loc[common, ["open", "close"]].astype(float)
                mismatch = ((a["o"] - b["open"]).abs() > 0.01) | ((a["c"] - b["close"]).abs() > 0.01)
                if mismatch.any():
                    ex = [str(x) for x in a.index[mismatch][:5]]
                    problems.append(
                        f"{label}: {int(mismatch.sum())}/{len(common)} sesiones COMPLETAS no "
                        f"cuadran con el seed diario (ej. {ex}) — el fix de tz no es correcto"
                    )
            else:
                problems.append(f"{label}: ninguna sesion completa que cotejar")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="muta; el default es dry-run")
    a = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rc = 0
    for path in TARGETS:
        if not path.is_file():
            print(f"[SKIP] {path.name}: no existe")
            continue
        df = pd.read_parquet(path)
        fixed, st = repair(df)

        print(f"\n=== {path.name}  ({len(df):,} filas)")
        print(f"  pared-COT (03-07)      : {st['wall']:,}")
        print(f"    colisiones -> BORRAR : {st['collisions_dropped']:,}")
        print(f"    a desplazar +5h      : {st['shifted']:,}")
        print(f"  sesion  (08-12) intactas: {st['session']:,}")
        print(f"  fuera de convencion -> BORRAR: {st['off']:,}")
        print(f"  exentas (XAU/BTC, 24h)  : {st['exempt']:,}")
        print(f"  resultado               : {len(fixed):,} filas")

        problems = verify(fixed, path.name)
        if problems:
            print("  VERIFICACION FALLIDA:")
            for p in problems:
                print(f"    - {p}")
            rc = 1
            continue
        print("  verificacion: OK (mono-convencion 08-12, sin duplicados, cuadra con seed diario)")

        if not a.apply:
            continue
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        bak = BACKUP_DIR / f"{path.stem}_tz_backup_{stamp}.parquet"
        shutil.copy2(path, bak)
        fixed.to_parquet(path, index=False)
        print(f"  APLICADO. backup en {bak.relative_to(REPO).as_posix()}")

    if not a.apply:
        print("\nDRY-RUN (usa --apply para ejecutar)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
