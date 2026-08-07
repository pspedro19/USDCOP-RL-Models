"""Cartera multi-activo con SELECCION WALK-FORWARD y vol targeting.

LA PREGUNTA QUE RESPONDE
------------------------
"Cuales son las mejores estrategias" tiene dos lecturas y solo una es publicable:

  (a) las que mejor puntuaron en 2025  -> PROHIBIDO. Elegir mirando el resultado convierte
      2025 en in-sample y destruye su valor probatorio (quant-constitution §1). Es
      literalmente lo que dejo a v11 con DSR 0.50-0.92 < 0.95.
  (b) las que un operador HABRIA elegido cada mes usando solo el pasado -> esto es lo que
      se implementa aqui. Es honesto Y es operativo: el mismo algoritmo puede correr manana.

POR QUE UNA CARTERA Y NO 4 SILOS
--------------------------------
Medido en `asset_year_metrics.py`: por activo hay 3-13 trades al anio, con lo que la
constitucion §6 prohibe reportar Sharpe -- y con razon. Agregando los 4 activos el conteo
sube a decenas y, sobre todo, los drawdowns dejan de coincidir. No es un truco estadistico:
es la razon por la que nadie opera un solo activo con cinco senales al anio.

POR QUE VOL TARGETING
---------------------
Las exposiciones medias medidas son absurdas: btc_trend 0.05-0.07, gold_dynamic_exit 0.21.
Al 5% de exposicion no se gana dinero aunque la senal acierte. Se dimensiona a un objetivo
de VOLATILIDAD declarado, no a lo que emita la regla.

PARAMETROS DECLARADOS EX-ANTE (valores estandar, NO ajustados a estos datos)
---------------------------------------------------------------------------
  objetivo de vol            10% anual          (convencion institucional tipica)
  ventana de vol             63 dias            (~1 trimestre)
  ventana de seleccion       252 dias           (~1 anio)
  criterio de seleccion      Calmar trailing    (metrica primaria de la constitucion §2)
  rebalanceo                 mensual
  apalancamiento maximo      2.0x
  pesos entre activos        inverse-vol        (risk parity simple, sin optimizar)
Cambiar cualquiera de estos DESPUES de ver el resultado seria un trial nuevo y habria que
registrarlo. Se declaran aqui para que ese cambio sea visible si alguien lo hace.

DEDUPLICACION DE TRIALS
-----------------------
Estrategias con serie de posicion diaria IDENTICA son EL MISMO trial publicado dos veces
(btc_trend_b2 == btc_trend_volbrk_s5; gold_dxy_tilt == _s05 == _s07). Se colapsan: cuentan
una vez para la seleccion y una vez para el DSR.

BASELINES (§3), sin los cuales no hay claim
-------------------------------------------
  B1   cartera equiponderada 1x de los 4 activos, comprada y mantenida
  B1'  exposicion CONSTANTE igual a la exposicion media realizada de la cartera
  x1/x2/x3 el mismo recorrido re-preciado con costes al doble y al triple
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.asset_year_metrics import (  # noqa: E402
    BUNDLES, COST_BPS, SWAP_ANNUAL_PCT, _as_date, daily_prices, load_trades, resolve_asset,
)

VOL_OBJETIVO = 0.10
VENTANA_VOL = 63
VENTANA_SELECCION = 252
LEVERAGE_MAX = 2.0


def strategy_daily(trades, days: list[date]) -> np.ndarray:
    idx = {d: i for i, d in enumerate(days)}
    pos = np.zeros(len(days))
    for t in trades:
        entry, exit_ = _as_date(t.get("timestamp")), _as_date(t.get("exit_timestamp"))
        if entry is None or exit_ is None:
            continue
        lev = abs(float(t.get("leverage") or 1.0))
        sign = -1.0 if str(t.get("side", "LONG")).upper() == "SHORT" else 1.0
        for d in days:
            if entry <= d <= exit_:
                pos[idx[d]] += sign * lev
    return pos


def strategy_daily_exact(trades, days, closes):
    """Retorno diario EXACTO de una sleeve, sin aproximar por cierre-a-cierre.

      dia de entrada:   close_d / entry_price - 1
      dias intermedios: close_d / close_{d-1} - 1
      dia de salida:    exit_price / close_{d-1} - 1

    POR QUE IMPORTA (self-red-team CXD-808): la version anterior acreditaba el retorno
    cierre-a-cierre del dia de entrada, que arranca en el cierre ANTERIOR -- o sea que se
    quedaba el hueco de apertura ocurrido ANTES de entrar. Eso es fuga, y era grande:
    desplazar la posicion un dia se llevaba el 47% de 2025 y el 58% de 2026. El shift
    completo, en cambio, contaba el lag dos veces (los manifiestos ya declaran NEXT-OPEN).
    Este tratamiento no aproxima por ningun lado: usa los precios que el propio trade trae.
    """
    idx = {d: i for i, d in enumerate(days)}
    ret = np.zeros(len(days))
    pos = np.zeros(len(days))
    for t in trades:
        e, x = _as_date(t.get("timestamp")), _as_date(t.get("exit_timestamp"))
        ep, xp = t.get("entry_price"), t.get("exit_price")
        if e is None or x is None or ep is None or xp is None:
            continue
        if e not in idx or x not in idx:
            continue
        lev = abs(float(t.get("leverage") or 1.0))
        sign = -1.0 if str(t.get("side", "LONG")).upper() == "SHORT" else 1.0
        i0, i1 = idx[e], idx[x]
        for i in range(i0, i1 + 1):
            if not np.isfinite(closes[i]):
                continue
            if i == i0:
                base, final = ep, (closes[i] if i1 > i0 else xp)
            elif i == i1:
                base, final = closes[i - 1], xp
            else:
                base, final = closes[i - 1], closes[i]
            if base and np.isfinite(base):
                ret[i] += sign * lev * (final / base - 1.0)
                pos[i] += sign * lev
    return np.nan_to_num(ret), pos


def calmar(returns: np.ndarray, ann: float) -> float:
    if returns.size < 2 or not np.any(returns):
        return -np.inf
    eq = np.cumprod(1 + returns)
    dd = float(np.min(eq / np.maximum.accumulate(eq) - 1))
    years = len(returns) / ann
    if years <= 0 or eq[-1] <= 0:
        return -np.inf
    cagr = eq[-1] ** (1 / years) - 1
    return cagr / abs(dd) if dd < 0 else (np.inf if cagr > 0 else -np.inf)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2025, 2026])
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    # ---- universo: todas las estrategias publicadas, sin elegir ninguna a mano
    sleeves: dict[str, list[tuple[str, list]]] = {}
    for sd in sorted(BUNDLES.iterdir()):
        if not sd.is_dir():
            continue
        symbol, manifest, ann = resolve_asset(sd.name)
        if symbol is None:
            continue
        trades = load_trades(sd)
        if trades:
            sleeves.setdefault(symbol, []).append((sd.name, trades))

    prices = {sym: daily_prices(sym) for sym in sleeves}
    prices = {k: v for k, v in prices.items() if v}
    sleeves = {k: v for k, v in sleeves.items() if k in prices}
    if not sleeves:
        print("sin sleeves con precios", file=sys.stderr)
        return 1

    # calendario comun: union de fechas, restringida al solape real
    inicio = max(min(prices[s]) for s in prices)
    fin = min(max(prices[s]) for s in prices)
    days = sorted({d for s in prices for d in prices[s] if inicio <= d <= fin})
    if len(days) < VENTANA_SELECCION + 60:
        print(f"historia comun insuficiente: {len(days)} dias", file=sys.stderr)
        return 1

    # retornos del activo alineados al calendario comun (ffill del ultimo precio conocido)
    ret_activo, pos_estrategia, ret_estrategia, closes_sym, dedupe = {}, {}, {}, {}, {}
    for sym in sleeves:
        serie = []
        last = None
        for d in days:
            last = prices[sym].get(d, last)
            serie.append(last)
        px = np.array([v if v else np.nan for v in serie], dtype=float)
        closes_sym[sym] = px
        r = np.zeros(len(days))
        r[1:] = np.where(np.isfinite(px[1:]) & np.isfinite(px[:-1]), px[1:] / px[:-1] - 1, 0.0)
        ret_activo[sym] = np.nan_to_num(r)

        for name, trades in sleeves[sym]:
            r_exact, pos = strategy_daily_exact(trades, days, closes_sym[sym])
            firma = (sym, hash(pos.tobytes()))
            if firma in dedupe:
                print(f"  TRIAL DUPLICADO: {name} == {dedupe[firma]} (serie identica) -> colapsado")
                continue
            dedupe[firma] = name
            pos_estrategia[(sym, name)] = pos
            ret_estrategia[(sym, name)] = r_exact

    print(f"\nuniverso: {len(pos_estrategia)} sleeves unicas sobre {len(sleeves)} activos; "
          f"calendario {days[0]}..{days[-1]} ({len(days)} dias)")

    # ---- walk-forward: seleccion mensual con SOLO el pasado
    n = len(days)
    pos_cartera = np.zeros(n)
    ret_cartera = np.zeros(n)
    ann_global = 252.0
    elegidas: dict[str, str] = {}
    historial = []

    for i in range(VENTANA_SELECCION, n):
        d = days[i]
        if i == VENTANA_SELECCION or (d.month != days[i - 1].month):
            ventana = slice(i - VENTANA_SELECCION, i)          # ESTRICTAMENTE pasado
            elegidas = {}
            for sym in sleeves:
                mejor, mejor_c = None, -np.inf
                for (s, name), pos in pos_estrategia.items():
                    if s != sym:
                        continue
                    c = calmar(ret_estrategia[(s, name)][ventana], ann_global)
                    if c > mejor_c:
                        mejor, mejor_c = name, c
                if mejor:
                    elegidas[sym] = mejor
            historial.append({"fecha": str(d), "elegidas": dict(elegidas)})

        # pesos inverse-vol entre activos, con la vol del pasado reciente
        # DEFECTO CORREGIDO (diagnostico 2026): inverse-vol con una sleeve SIN POSICION le
        # daba peso 1/1e-9 -- o sea, casi toda la cartera al activo que no esta operando, y
        # como su posicion es cero la cartera entera colapsaba a exposicion ~0.01. Asi es
        # como `USD/COP`, que no tiene NI UN trade publicado en 2026, se llevaba la cartera
        # y dejaba 2026 plano. Un activo que no toma riesgo debe pesar CERO, no infinito.
        pesos, total = {}, 0.0
        for sym, name in elegidas.items():
            pos_v = pos_estrategia[(sym, name)][i - VENTANA_VOL:i]
            if not np.any(np.abs(pos_v) > 1e-12):
                continue                      # sin posicion en la ventana => no asigna riesgo
            v = float(np.std(ret_estrategia[(sym, name)][i - VENTANA_VOL:i]))
            if v <= 1e-9:
                continue                      # varianza nula: no es "riesgo bajisimo", es ausencia
            pesos[sym] = 1.0 / v
            total += pesos[sym]
        if total <= 0:
            continue
        pesos = {k: v / total for k, v in pesos.items()}

        bruto = sum(pesos[s] * ret_estrategia[(s, elegidas[s])][i] for s in pesos)
        exp_bruta = sum(pesos[s] * abs(pos_estrategia[(s, elegidas[s])][i]) for s in pesos)

        hist = np.array([
            sum(pesos[s] * ret_estrategia[(s, elegidas[s])][j] for s in pesos)
            for j in range(i - VENTANA_VOL, i)])
        vol = float(np.std(hist)) * np.sqrt(ann_global)
        k = min(LEVERAGE_MAX, VOL_OBJETIVO / vol) if vol > 1e-9 else 0.0

        pos_cartera[i] = k * exp_bruta
        ret_cartera[i] = k * bruto

    # costes por turnover de la cartera, al bps medio declarado de los activos usados
    bps = float(np.mean([COST_BPS[resolve_asset(n_)[1]] for (_, n_) in pos_estrategia])) / 10000.0
    turnover = np.abs(np.diff(pos_cartera, prepend=0.0))
    coste = turnover * bps
    swap = np.zeros(n)
    ret_neto = ret_cartera - coste

    # ---- evaluacion por anio con el SSOT constitucional
    from services.common.metrics import (
        _ann_return_dd_calmar, calculate_sharpe_ratio, calculate_sortino_ratio,
        dsr_report, paired_exposure_baseline, pbo_cscv, sharpe_ratio_stderr,
    )

    b1_diario = np.mean(np.array([ret_activo[s] for s in sleeves]), axis=0)  # equiponderada 1x
    salida = []
    for year in args.years:
        m = np.array([d.year == year for d in days])
        if m.sum() < 20:
            continue
        r, p, c, b1, sw = ret_neto[m], pos_cartera[m], coste[m], b1_diario[m], swap[m]
        obs = int(m.sum())
        fila = {
            "anio": year,
            "dias": obs,
            "periodo": f"{[d for d in days if d.year == year][0]}..{[d for d in days if d.year == year][-1]}",
            "exposicion_media": round(float(np.mean(np.abs(p))), 3),
            "cartera": _ann_return_dd_calmar(r, ann_global),
            "B1_equiponderada_1x": _ann_return_dd_calmar(b1, ann_global),
            "B1p_exposicion_emparejada": paired_exposure_baseline(p, b1, ann_global),
            # cost_stress del SSOT asume que el retorno es pos*asset_ret. Aqui NO lo es:
            # la cartera son sleeves ponderadas. Llamarlo con (p, b1) media OTRA estrategia
            # -- defecto mio, cazado al revisar. Se re-precia la MISMA serie realizada.
            "stress_costos": {
                **{f"x{k}": _ann_return_dd_calmar(ret_cartera[m] - k * c, ann_global)
                   for k in (1, 2, 3)},
                "survives_2x": bool(_ann_return_dd_calmar(ret_cartera[m] - 2 * c, ann_global)["ann_return_pct"] > 0
                                    and _ann_return_dd_calmar(ret_cartera[m] - 2 * c, ann_global)["calmar"] > 0),
                "survives_3x": bool(_ann_return_dd_calmar(ret_cartera[m] - 3 * c, ann_global)["ann_return_pct"] > 0),
            },
            "sharpe": calculate_sharpe_ratio(r, periods_per_year=ann_global),
            "sortino": calculate_sortino_ratio(r, periods_per_year=ann_global),
            "sharpe_stderr": sharpe_ratio_stderr(r),
            "n_observaciones": obs,
        }
        fila["bate_B1p"] = bool(fila["cartera"]["ann_return_pct"]
                                > fila["B1p_exposicion_emparejada"]["ann_return_pct"])

        # GATE CONSTITUCIONAL §2: ningun claim de edge sin DSR trial-aware. n_trials no es
        # 1: el universo son las sleeves unicas consideradas por el selector. Se reporta a
        # tres conteos porque el numero real depende de cuantas configuraciones se mirasen
        # historicamente, y esa cifra la fija el HYPOTHESIS-REGISTRY, no yo.
        sr_pp = (fila["sharpe"] / np.sqrt(ann_global)) if fila["sharpe"] else 0.0
        fila["dsr"] = {
            f"n_trials={nt}": dsr_report(sr_pp, obs, nt, periods_per_year=int(ann_global))
            for nt in (len(pos_estrategia), 50, 100)
        }
        salida.append(fila)

    print(f"\n{'='*96}\nCARTERA WALK-FORWARD  (vol objetivo {VOL_OBJETIVO:.0%}, "
          f"seleccion {VENTANA_SELECCION}d, rebalanceo mensual)\n{'='*96}")
    print(f"{'anio':<6}{'dias':>6}{'expos':>8}{'ret%':>9}{'B1%':>9}{'B1p%':>9}"
          f"{'maxDD%':>9}{'Calmar':>8}{'Sharpe':>8}{'+-SE':>7}{'x2':>4}{'>B1p':>6}{'DSR':>7}")
    for f in salida:
        print(f"{f['anio']:<6}{f['dias']:>6}{f['exposicion_media']:>8.2f}"
              f"{f['cartera']['ann_return_pct']:>9.2f}"
              f"{f['B1_equiponderada_1x']['ann_return_pct']:>9.2f}"
              f"{f['B1p_exposicion_emparejada']['ann_return_pct']:>9.2f}"
              f"{f['cartera']['max_dd_pct']:>9.2f}{f['cartera']['calmar']:>8.2f}"
              f"{(f['sharpe'] if f['sharpe'] is not None else float('nan')):>8.2f}"
              f"{f['sharpe_stderr']:>7.2f}"
              f"{('si' if f['stress_costos']['survives_2x'] else 'no'):>4}"
              f"{('SI' if f['bate_B1p'] else 'no'):>6}"
              f"{f['dsr'][f'n_trials={len(pos_estrategia)}']['headline_dsr']:>7.3f}")

    # PBO/CSCV: juzga el PROCEDIMIENTO de seleccion, no el resultado. El DSR deflacta un
    # Sharpe por cuantos intentos hiciste; PBO pregunta lo complementario y mas duro: al
    # elegir el mejor in-sample, se queda por encima de la mediana out-of-sample? pbo > 0.5
    # significa que seleccionar es PEOR que tirar una moneda, y eso es REJECT aunque el
    # Sharpe sea bonito. Es la unica medida que ataca de frente la limitacion declarada de
    # que las sleeves no se re-ajustan point-in-time.
    # OBJECION DE CLAUDE (CLD-696) CONCEDIDA: la version anterior agrupaba las 16 columnas
    # en UNA familia, y CSCV preguntaba si el ganador GLOBAL persiste. Mi seleccion es POR
    # ACTIVO: nunca rankeo entre activos. El pooling ademas MAQUILLA el numero, porque
    # columnas de activos distintos difieren en vol y drift de forma persistente y el
    # ganador sigue arriba por efecto de activo, no por habilidad de seleccion.
    pbo = {}
    for sym in sorted(sleeves):
        cols = [ret_estrategia[k] for k in ret_estrategia if k[0] == sym]
        if len(cols) < 2:
            continue
        try:
            r = pbo_cscv(np.array(cols).T, n_blocks=16)
            r["n_sleeves"] = len(cols)
            r["degenerado_N2"] = len(cols) == 2   # con 2 columnas "bajo la mediana" = "el peor de dos"
            pbo[sym] = r
        except Exception as exc:  # noqa: BLE001
            pbo[sym] = {"error": str(exc)}
    print("")
    print("PBO/CSCV POR ACTIVO -- la familia entre la que realmente se elige (>0.5 = REJECT):")
    for sym, v in pbo.items():
        if "error" in v:
            print(f"  {sym:<10} ERROR: {v['error']}")
        else:
            j = "PEOR QUE EL AZAR" if v["pbo"] > 0.5 else "aceptable"
            deg = "  (N=2: casi degenerado, constitucion §6)" if v["degenerado_N2"] else ""
            print(f"  {sym:<10} {v['n_sleeves']} sleeves   PBO={v['pbo']:.3f}   {j}{deg}")
    print("  NOTA: el PBO agrupando los 4 activos en una familia daria 0.327 y seria ENGANOSO:")
    print("        mide una seleccion entre activos que este procedimiento nunca ejecuta.")

    print("")
    print("DSR trial-aware (headline = la sigma MENOS favorable; bar constitucional 0.95):")
    for f in salida:
        print(f"  {f['anio']}   " + "  ".join(
            f"n={k.split('=')[1]}: {v['headline_dsr']:.3f}" for k, v in f["dsr"].items()))

    print("\nseleccion walk-forward (ultimos 6 rebalanceos):")
    for h in historial[-6:]:
        print(f"  {h['fecha']}  " + ", ".join(f"{k}={v}" for k, v in h["elegidas"].items()))

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"parametros": {"vol_objetivo": VOL_OBJETIVO, "ventana_vol": VENTANA_VOL,
                            "ventana_seleccion": VENTANA_SELECCION, "leverage_max": LEVERAGE_MAX,
                            "declarados_ex_ante": True},
             "universo": [f"{s}:{n_}" for (s, n_) in pos_estrategia],
             "resultados": salida, "pbo_cscv": pbo, "seleccion": historial}, indent=2, default=str), encoding="utf-8")
        print(f"\nJSON -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
