"""Metricas por ACTIVO x ANIO con la disciplina completa de `quant-constitution.md`.

QUE RESPONDE
------------
"Si esta estrategia hubiera estado operando en 2025 (y en 2026 hasta donde hay datos),
que habria pasado, y es distinguible de no haber hecho nada?"

POR QUE NO BASTA EL RETORNO
---------------------------
El retorno solo no separa timing de beta. Una estrategia que esta invertida el 30% del
tiempo en un activo que subio 40% "gana" sin haber decidido nada util. Por eso cada celda
lleva, ademas del retorno:

  B1   buy&hold 1x del activo en el MISMO periodo
  B1'  **exposicion emparejada**: exposicion CONSTANTE igual a la exposicion media realizada
       de la estrategia. Es la prueba dura (constitucion §3.2) y es costless por
       construccion, o sea que el liston queda a proposito mas alto
  x1/x2/x3  el MISMO recorrido de posiciones re-preciado con costes al doble y al triple.
       Si muere al doble, no hay edge (§3.4)

RECONSTRUCCION DIARIA, NO POR TRADE
-----------------------------------
Los trades publicados se convierten en una serie DIARIA de posicion (leverage mientras el
trade esta abierto, 0 fuera). Eso permite marcar a mercado dia a dia y alimentar el SSOT
constitucional `services/common/metrics.py`, que es el que la constitucion nombra como gate
de release. No se reimplementa ni un estadistico.

DECISIONES DECLARADAS (para que nadie tenga que adivinarlas)
------------------------------------------------------------
1. Un trade pertenece al anio de su **salida**: es cuando el PnL se realiza.
2. Retorno de estrategia = producto de (1 + pnl_pct/100) de los trades de ese anio, y por
   separado el compuesto de la serie diaria; se publican AMBOS y su discrepancia, porque
   diferir significa que la reconstruccion diaria no reproduce el bundle y eso hay que verlo.
3. Costes: los DECLARADOS en `config/strategy_manifests/<asset>.yaml`, no inventados.
4. Anualizacion: la de `reference.asset.annualization` (BTC 365, COP 261, Gold/SPX 250).
   Nunca se comparan activos entre si en una misma tabla de ranking (strategy-contract §5).
5. **Con N < 20 trades NO se emiten Sharpe, Sortino, p-value ni DSR** (constitucion §6).
   Se emite conteo, PnL, exposicion y los baselines, que son descriptivos y validos a
   cualquier N. Esta es la razon por la que la mayoria de celdas de este informe no llevan
   Sharpe: no es que falte, es que reportarlo seria falso.
6. Este script NO decide nada ni elige estrategia: LEE bundles ya publicados. No gasta
   trials (§1) porque no busca sobre el resultado.

Uso:
    python scripts/analysis/asset_year_metrics.py --years 2025 2026
    python scripts/analysis/asset_year_metrics.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

BUNDLES = ROOT / "usdcop-trading-dashboard" / "public" / "data" / "strategies"

# Prefijo de strategy_id -> (simbolo en asset_daily_ohlcv, manifiesto de costes, anualizacion)
ASSET_MAP = [
    ("btc_",          "BTC/USDT", "btcusdt", 365),
    ("gold_",         "XAU/USD",  "xauusd",  250),
    ("xau",           "XAU/USD",  "xauusd",  250),
    ("spx500_",       "SPX/500",  "spx500",  250),
    ("smart_simple_", "USD/COP",  "usdcop",  261),
    ("usdcop",        "USD/COP",  "usdcop",  261),
]

# Coste por unidad de turnover, en bps, LEIDO de los manifiestos (ver docstring §3).
COST_BPS = {"btcusdt": 13.0, "xauusd": 2.0, "spx500": 3.0, "usdcop": 1.0}
SWAP_ANNUAL_PCT = {"xauusd": 2.5}


def resolve_asset(strategy_id: str):
    for prefix, symbol, manifest, ann in ASSET_MAP:
        if strategy_id.startswith(prefix):
            return symbol, manifest, ann
    return None, None, None


def latest_backtest_dir(strategy_dir: Path) -> Path | None:
    """El bundle a evaluar: el que el manifest declara PRODUCTION, si lo declara.

    POR QUE NO BASTA CON "LA VERSION MAS ALTA" (defecto medido el 2026-08-06)
    ------------------------------------------------------------------------
    Esta funcion ordenaba por los digitos del nombre del directorio. Para
    `smart_simple_v11` eso da `3.0.0-A` y `3.0.0-B` -> [3,0,0], por encima de `2.0.0`, y
    se elegia una de las 3.0.0. Pero 3.0.0-A/B son variantes de INVESTIGACION: aparecen
    en el manifest solo como `backtests`, no tienen rol de produccion y **no contienen
    2026**. El manifest declara `production.model_version = 2.0.0`, que si trae los dos
    anios.

    Consecuencia observada aguas abajo: la cartera walk-forward veia `USD/COP` con 0.0%
    de dias en mercado durante 2026 (CXD-807) y le daba peso casi infinito por
    inverse-vol a una sleeve sin posicion. El sintoma parecia "COP no opera en 2026"; la
    causa era que se estaba leyendo un bundle que nunca se desplego.

    Evaluar una cartera "como si hubiera estado operativa" obliga a leer lo que HABRIA
    estado operativo. El puntero `production` del manifest es esa declaracion, y es
    anterior a cualquiera de estos analisis (`generated_at` 2026-07-21).

    DIRECCION DEL EFECTO, declarada a proposito: seguir el manifest SUBE COP 2025 de
    +18.73% a +25.63% y crea un 2026 de +1.77% donde no habia nada. Que mueva el numero
    a favor obliga a comprobar que la regla no es "elegir el mejor": `3.0.0-A` da +26.58%
    en 2025 y tampoco se elige, porque no es la declarada. Se sigue el manifest, no el
    resultado.

    Alcance medido: 1 de 18 sleeves. Las otras 17 no declaran `production` y caen al
    orden por digitos de siempre, sin cambio.
    """
    bt = strategy_dir / "backtests"
    if not bt.is_dir():
        return None
    versions = [d for d in bt.iterdir() if d.is_dir()]
    if not versions:
        return None

    declarada = _production_version(strategy_dir)
    if declarada:
        for d in versions:
            if d.name == declarada:
                return d
        # Declarada pero ausente en disco: se avisa y se cae al orden por digitos. Callarlo
        # dejaria el mismo defecto que este arreglo cierra, solo que mas dificil de ver.
        print(
            f"[aviso] {strategy_dir.name}: el manifest declara production={declarada} "
            f"pero ese directorio no existe; caigo al orden por version"
        )

    def key(d: Path):
        return [int(x) for x in re.findall(r"\d+", d.name)] or [0]

    return sorted(versions, key=key)[-1]


def _production_version(strategy_dir: Path) -> str | None:
    """`production.model_version` del manifest, o None si no lo declara o no se puede leer."""
    manifiesto = strategy_dir / "manifest.json"
    if not manifiesto.is_file():
        return None
    try:
        datos = json.loads(manifiesto.read_text(encoding="utf-8"))
    except Exception:
        return None
    produccion = datos.get("production")
    if not isinstance(produccion, dict):
        return None
    version = produccion.get("model_version")
    return version if isinstance(version, str) and version else None


def load_trades(strategy_dir: Path) -> list[dict]:
    """Todos los trades del bundle mas reciente, deduplicados por (entrada, salida, precio).

    Los ficheros `trades_2025.json` / `trades_2026.json` de BTC y Gold son la MISMA corrida
    de historia completa publicada en dos fechas (hallazgo CLD-692): el sufijo es la fecha
    de PUBLICACION, no el periodo. Por eso se unen y se deduplican, y el corte por anio lo
    hace este script con la fecha de salida real, nunca el nombre del fichero.
    """
    d = latest_backtest_dir(strategy_dir)
    if d is None:
        return []
    seen, out = set(), []
    for path in sorted(d.glob("trades_*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        trades = raw.get("trades", raw) if isinstance(raw, dict) else raw
        if not isinstance(trades, list):
            continue
        for t in trades:
            k = (t.get("timestamp"), t.get("exit_timestamp"), t.get("entry_price"))
            if k in seen:
                continue
            seen.add(k)
            out.append(t)
    return out


def _as_date(value) -> date | None:
    """Fecha del sello, NORMALIZADA A UTC antes de truncar.

    Defecto que corrijo tras la objecion de Claude (CLD-695): antes devolvia la fecha en la
    zona en que venia escrito el sello (`-05:00` para COP) mientras el indice de precios se
    construye con `to_datetime(..., utc=True).dt.date`. Eran DOS convenciones distintas a
    los dos lados de la misma comparacion. Hoy no muerde porque la sesion COP es matinal y
    un trade de las 09:00 COT cae en el mismo dia UTC -- pero uno de las 20:00 COT no, y en
    un activo 24/7 como BTC eso desplaza la barra un dia entero.

    Se normaliza a UTC en vez de a Bogota porque el indice de barras ya esta en UTC: la
    regla no es "que zona es la correcta" sino que AMBOS lados usen la misma.
    """
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc)
    return ts.date()


def daily_prices(symbol: str) -> dict[date, float]:
    """Cierres diarios del activo.

    Fuente PRIMARIA el parquet versionado `data/backups/features/asset_daily_ohlcv.parquet`,
    no la DB: (a) es reproducible por cualquiera que clone el repo, sin stack levantado; y
    (b) consultar el hypertable tumbaba la corrida con `out of shared memory` en cuanto otra
    sesion trabajaba en paralelo -- un informe que depende de que nadie mas use la base no
    es un informe. La DB queda como respaldo si el parquet falta.
    """
    parquet = ROOT / "data" / "backups" / "features" / "asset_daily_ohlcv.parquet"
    if parquet.exists():
        import pandas as pd

        df = pd.read_parquet(parquet, columns=["time", "symbol", "close"])
        df = df[(df["symbol"] == symbol) & (df["close"] > 0)]
        if not df.empty:
            fechas = pd.to_datetime(df["time"], utc=True).dt.date
            return dict(zip(fechas, df["close"].astype(float)))

    import psycopg2

    conn = psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
        connect_timeout=5,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT time::date, close FROM asset_daily_ohlcv "
                "WHERE symbol = %s AND close > 0 ORDER BY 1;", (symbol,))
            return {r[0]: float(r[1]) for r in cur.fetchall()}
    finally:
        conn.close()


def build_series(trades, prices: dict[date, float], year: int):
    """Serie DIARIA (fecha, retorno del activo, posicion, turnover) para el anio."""
    days = sorted(d for d in prices if d.year == year)
    if len(days) < 2:
        return None
    px = np.array([prices[d] for d in days], dtype=float)
    asset_ret = np.zeros(len(days))
    asset_ret[1:] = px[1:] / px[:-1] - 1.0

    pos = np.zeros(len(days))
    idx = {d: i for i, d in enumerate(days)}
    for t in trades:
        entry, exit_ = _as_date(t.get("timestamp")), _as_date(t.get("exit_timestamp"))
        if entry is None or exit_ is None:
            continue
        lev = abs(float(t.get("leverage") or 1.0))
        sign = -1.0 if str(t.get("side", "LONG")).upper() == "SHORT" else 1.0
        for d in days:
            if entry <= d <= exit_:
                pos[idx[d]] += sign * lev

    turnover = np.abs(np.diff(pos, prepend=0.0))
    return days, asset_ret, pos, turnover


def compute(strategy_id: str, trades, prices, year: int) -> dict | None:
    from services.common.metrics import (
        _ann_return_dd_calmar, calculate_sharpe_ratio, calculate_sortino_ratio,
        cost_stress, paired_exposure_baseline,
    )

    symbol, manifest, ann = resolve_asset(strategy_id)
    built = build_series(trades, prices, year)
    if built is None:
        return None
    days, asset_ret, pos, turnover = built

    # DOS conteos, porque uno solo engana: una posicion abierta todo 2025 que cierra en 2026
    # da N=0 cerrados y sin embargo tuvo exposicion y PnL el anio entero. Se publican ambos.
    year_trades = [t for t in trades if (_as_date(t.get("exit_timestamp")) or date(1900, 1, 1)).year == year]
    n = len(year_trades)
    n_activos = sum(
        1 for t in trades
        if (e := _as_date(t.get("timestamp"))) and (x := _as_date(t.get("exit_timestamp")))
        and e <= days[-1] and x >= days[0]
    )
    ret_bundle = float(np.prod([1 + float(t.get("pnl_pct", 0.0)) / 100.0 for t in year_trades]) - 1) * 100 if n else 0.0

    bps = COST_BPS.get(manifest, 0.0) / 10000.0
    cost = turnover * bps
    swap = np.full(len(days), SWAP_ANNUAL_PCT.get(manifest, 0.0) / 100.0 / ann) * (np.abs(pos) > 0)

    strat_ret = pos * asset_ret - cost - swap
    ret_daily = float(np.prod(1 + strat_ret) - 1) * 100

    out = {
        "strategy_id": strategy_id, "asset": symbol, "year": year,
        "n_trades": n,
        "periodo": f"{days[0]}..{days[-1]}",
        "dias_de_mercado": len(days),
        "exposicion_media": round(float(np.mean(np.abs(pos))), 4),
        "dias_en_mercado_pct": round(float(np.mean(np.abs(pos) > 0)) * 100, 1),
        "ret_bundle_pct": round(ret_bundle, 2),
        "ret_diario_pct": round(ret_daily, 2),
        "discrepancia_pp": round(ret_daily - ret_bundle, 2),
        "estrategia": _ann_return_dd_calmar(strat_ret, ann),
        "B1_buy_hold": _ann_return_dd_calmar(asset_ret, ann),
        "B1p_exposicion_emparejada": paired_exposure_baseline(pos, asset_ret, ann),
        "coste_bps_declarado": COST_BPS.get(manifest, 0.0),
        "stress_costos": cost_stress(pos, asset_ret, cost, swap, ann),
    }

    if n >= 20:
        sharpe = calculate_sharpe_ratio(strat_ret, periods_per_year=ann)
        out["inferencia"] = {
            "sharpe": sharpe,
            "sortino": calculate_sortino_ratio(strat_ret, periods_per_year=ann),
            "nota": "N>=20 trades: la constitucion permite estadistica inferencial",
        }
    else:
        out["inferencia"] = {
            "sharpe": None, "sortino": None,
            "nota": f"N={n} < 20 trades: constitucion §6 PROHIBE Sharpe/p-value/DSR aqui. "
                    f"No es que falte el dato, es que reportarlo seria falso.",
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2025, 2026])
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    if not BUNDLES.is_dir():
        print(f"sin bundles en {BUNDLES}", file=sys.stderr)
        return 1

    price_cache: dict[str, dict] = {}
    results = []
    for sd in sorted(BUNDLES.iterdir()):
        if not sd.is_dir():
            continue
        symbol, manifest, ann = resolve_asset(sd.name)
        if symbol is None:
            print(f"  SIN MAPEO DE ACTIVO: {sd.name} (se omite, declarado)", file=sys.stderr)
            continue
        trades = load_trades(sd)
        if not trades:
            continue
        if symbol not in price_cache:
            price_cache[symbol] = daily_prices(symbol)
        if not price_cache[symbol]:
            print(f"  SIN PRECIOS para {symbol}: {sd.name} omitido", file=sys.stderr)
            continue
        for year in args.years:
            row = compute(sd.name, trades, price_cache[symbol], year)
            if row:
                results.append(row)

    for asset in sorted({r["asset"] for r in results}):
        print(f"\n{'='*104}\n{asset}\n{'='*104}")
        print(f"{'estrategia':<26}{'anio':<6}{'N':>4}{'expos':>8}{'ret%':>9}"
              f"{'B1%':>9}{'B1p%':>9}{'maxDD%':>9}{'Calmar':>8}{'x2 vive':>9}{'Sharpe':>9}")
        for r in sorted([x for x in results if x["asset"] == asset],
                        key=lambda x: (x["strategy_id"], x["year"])):
            sh = r["inferencia"]["sharpe"]
            print(f"{r['strategy_id']:<26}{r['year']:<6}{r['n_trades']:>4}"
                  f"{r['exposicion_media']:>8.2f}"
                  f"{r['estrategia']['ann_return_pct']:>9.2f}"
                  f"{r['B1_buy_hold']['ann_return_pct']:>9.2f}"
                  f"{r['B1p_exposicion_emparejada']['ann_return_pct']:>9.2f}"
                  f"{r['estrategia']['max_dd_pct']:>9.2f}"
                  f"{r['estrategia']['calmar']:>8.2f}"
                  f"{('SI' if r['stress_costos']['survives_2x'] else 'NO'):>9}"
                  f"{(f'{sh:.2f}' if sh is not None else 'N<20'):>9}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"\nJSON -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
