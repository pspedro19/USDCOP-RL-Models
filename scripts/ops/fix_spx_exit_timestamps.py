"""CXD-828 — corrige el `exit_timestamp` del bundle SPX. SOLO metadata temporal.

QUE ESTABA MAL
--------------
Los bundles de SPX publican `exit_timestamp = D` mientras su `exit_price` es el **cierre de
D+1**. Auditado por Codex sobre 916 trades y confirmado por mi sobre la serie de precios:
10 de 12 trades de la version 2.0.0 casan con el cierre de D+1 con **0.0 bp** de error.

QUE SE CORRIGE Y POR QUE ESE LADO
---------------------------------
Se mueve el **SELLO**, no el precio. Lo decide la propia serie de equity del bundle
(`signals_YYYY.json`), que ninguno de los dos habia mirado: en la salida declarada el
2025-03-10, la equity **sigue moviendose el 2025-03-11** (9492.47 -> 9489.62) y se aplana
solo desde el 12. La posicion vive hasta el cierre de D+1, luego `exit_price` es correcto y
el sello va una barra por delante. Precios y equity concuerdan entre si; el sello discrepa
de ambos.

CONSECUENCIAS DECLARADAS
------------------------
1. **Ningun PnL cambia.** `pnl_pct`, `pnl_usd`, `equity_*` y el `summary` quedan intactos:
   esto es una correccion de METADATOS. Por eso **cuesta 0 trials** -- no se elige ningun
   parametro y el criterio sale de un fichero del propio bundle.
2. Lo que si se mueve es la colocacion temporal en cualquier consumidor que componga la
   serie DIARIA: deja de acreditarse el retorno de D+1 en el dia D. Medido en el harness de
   cartera, la cota superior del efecto era -0.21 pp en 2025.
3. **No se toca la version 1.0.0**: publica precios en dolares de SPY, otra escala, y sus
   sellos no son comparables contra la serie SPX/500. Queda declarado, no corregido.
4. **No se toca `HYPOTHESIS-REGISTRY`** (instruccion de Codex en CXD-833). Si el refreeze lo
   exige, se propone aparte y append-only.

REGLA DURA DEL CORRECTOR
------------------------
Un sello solo se mueve si el `exit_price` casa **exactamente** (<=1 bp) con el cierre de otra
barra. Si no casa con ninguna, **no se toca y se reporta**: el trade de salida 2026-07-27 no
tiene barra D+1 en la serie versionada y se queda como esta.

Uso:
    python scripts/ops/fix_spx_exit_timestamps.py --dry-run
    python scripts/ops/fix_spx_exit_timestamps.py --apply
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
BUNDLES = REPO / "usdcop-trading-dashboard" / "public" / "data" / "strategies"
PRECIOS = REPO / "data" / "backups" / "features" / "asset_daily_ohlcv.parquet"
TOLERANCIA = 1e-4          # 1 bp
VERSION_OBJETIVO = "2.0.0"  # la que el manifest declara PRODUCTION y la que usa la serie SPX


def _fecha(v) -> "pd.Timestamp":
    ts = pd.Timestamp(v)
    return (ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")).date()


def sello_corregido(exit_timestamp: str, exit_price: float,
                    cierres: dict, dias: list) -> tuple[str | None, str]:
    """FUNCION PURA: devuelve (sello_nuevo_o_None, motivo). No lee ni escribe ficheros.

    Extraida para que el corrector sea comprobable pieza a pieza, a peticion de Codex
    (CXD-836). La logica completa vivia dentro del bucle de E/S y no se podia ejercitar sin
    tocar bundles reales, asi que el bug de abajo no lo cazaba ningun test.

    BUG CORREGIDO (CXD-835, encontrado por Codex): la primera version usaba
    `bisect_left(dias, ed)` y luego `dias[i + 1]`. Eso solo acierta cuando `ed` **pertenece**
    al calendario: si el sello cae fuera —un sabado, un festivo— `bisect_left` ya apunta a la
    primera barra POSTERIOR, y sumarle 1 **salta esa barra y coge la siguiente**. Reproducido:
    con `dias = [03-07, 03-10, 03-11, 03-12]` y `ed = 2025-03-08` (sabado), `bisect_left+1`
    devuelve **03-11** saltandose **03-10**. `bisect_right` da la primera barra estrictamente
    posterior en AMBOS casos, que es lo que este corrector necesita siempre.

    Reglas:
      * si el precio ya casa con el cierre del dia que el sello nombra -> None ("ya casa").
        Esto hace la funcion IDEMPOTENTE: aplicarla dos veces no mueve nada.
      * si casa con la primera barra estrictamente posterior -> se devuelve ese sello,
        conservando la hora original y cambiando solo la fecha.
      * si no casa con ninguna -> None ("sin barra que case"). Nunca se inventa un sello.
    """
    ed = _fecha(exit_timestamp)
    cierre_d = cierres.get(ed)
    if cierre_d is not None and abs(exit_price / cierre_d - 1) <= TOLERANCIA:
        return None, "ya casa con su propio dia"

    # `bisect_right` = primera barra ESTRICTAMENTE posterior, pertenezca `ed` o no al calendario.
    i = bisect.bisect_right(dias, ed)
    siguiente = dias[i] if i < len(dias) else None
    cierre_s = cierres.get(siguiente) if siguiente else None
    if cierre_s is not None and abs(exit_price / cierre_s - 1) <= TOLERANCIA:
        original = str(exit_timestamp)
        hora = original[10:] if len(original) > 10 else ""
        return f"{siguiente}{hora}", "-> primera barra posterior, casa a 0.0 bp"

    if cierre_d is not None:
        return None, f"sin barra que case (err_D={abs(exit_price / cierre_d - 1) * 1e4:.1f}bp)"
    return None, "sin barra que case (el sello no esta en el calendario)"


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(PRECIOS, columns=["time", "symbol", "close"])
    df = df[df["symbol"] == "SPX/500"].copy()
    df["d"] = pd.to_datetime(df["time"], utc=True).dt.date
    df = df.sort_values("d").drop_duplicates("d", keep="last")
    cierres = {r.d: float(r.close) for r in df.itertuples()}
    dias = sorted(cierres)
    print(f"serie SPX/500: {dias[0]} .. {dias[-1]} ({len(dias)} barras)\n")

    movidos = intactos = 0
    for f in sorted(BUNDLES.glob(f"spx500_*/backtests/{VERSION_OBJETIVO}/trades_*.json")):
        original_texto = f.read_text(encoding="utf-8")
        raw = json.loads(original_texto)
        trades = raw.get("trades", raw) if isinstance(raw, dict) else raw
        cambios = []
        for t in trades:
            nuevo, motivo = sello_corregido(t["exit_timestamp"], float(t["exit_price"]),
                                            cierres, dias)
            if nuevo is None:
                if motivo != "ya casa con su propio dia":
                    intactos += 1
                    cambios.append((t["exit_timestamp"], None, motivo))
                continue
            cambios.append((str(t["exit_timestamp"]), nuevo, motivo))
            t["exit_timestamp"] = nuevo
            movidos += 1

        rel = f.relative_to(BUNDLES)
        print(f"{rel}")
        for viejo, nuevo, nota in cambios:
            flecha = f"{viejo}  ->  {nuevo}" if nuevo else f"{viejo}  (INTACTO)"
            print(f"    {flecha}   {nota}")
        if not cambios:
            print("    (sin cambios)")
        if args.apply and any(c[1] for c in cambios):
            # DOS detalles de serializacion que no son cosmeticos, porque de ellos depende que
            # el diff sea auditable:
            #
            # `ensure_ascii=True` (por defecto) a proposito: el fichero original escapa los no
            # ASCII (`"S&P 500 \\u00b7 MA200"`). Con `ensure_ascii=False` el valor no cambia
            # pero el BYTE si, y el diff ensuciaba 8 lineas de `strategy_name` -- un campo que
            # esta correccion no tiene permiso para tocar.
            #
            # El salto final se CONSERVA tal como estaba (CXD-835 punto 4): anadirlo a ciegas
            # metia 8 lineas mas de ruido en 4 ficheros y convertia en falso el claim "el diff
            # solo contiene sellos". Un diff que solo contiene lo que dice contener es lo que
            # hace VERIFICABLE la afirmacion "solo metadata"; con ruido es solo una afirmacion.
            salto = "\n" if original_texto.endswith("\n") else ""
            f.write_text(json.dumps(raw, indent=2) + salto, encoding="utf-8")

    print(f"\nsellos movidos: {movidos} · dejados intactos por no casar: {intactos}")
    print("PnL, equity y summary NO se tocan: es correccion de metadatos (0 trials).")
    if args.dry_run:
        print("\n(dry-run: no se ha escrito nada)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
