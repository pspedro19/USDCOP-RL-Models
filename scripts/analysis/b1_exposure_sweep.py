"""B1 por ACTIVO: alguna sleeve bate a comprar y esperar?

El hallazgo de CLD-711 fue que ninguna sleeve de ORO bate a B1 en 2025. La pregunta obvia que
se deriva -y que no habia hecho nadie- es si eso pasa tambien en los otros tres activos. Si
pasa, el numero de la cartera no descansa en seleccion ni en timing: descansa en tomar MENOS
riesgo que el activo, que es una propiedad, no una habilidad.

Se compara, por activo y anio:
  B1 1x        retorno de comprar el activo el primer dia y venderlo el ultimo
  cada sleeve  serie diaria exacta (strategy_daily_exact), su exposicion media y sus dias en
               mercado, para poder distinguir "gana menos porque acierta menos" de "gana menos
               porque esta menos dentro"
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\USUARIO\Documents\USDCOP-RL-Models")
sys.path.insert(0, str(ROOT))
from scripts.analysis.asset_year_metrics import BUNDLES, daily_prices, load_trades, resolve_asset  # noqa
from scripts.analysis.portfolio_walkforward import strategy_daily_exact  # noqa

por_activo = {}
for sd in sorted(BUNDLES.iterdir()):
    if not sd.is_dir():
        continue
    sym = resolve_asset(sd.name)[0]
    if sym is None:
        continue
    t = load_trades(sd)
    if t:
        por_activo.setdefault(sym, []).append((sd.name, t))

for sym in sorted(por_activo):
    precios = daily_prices(sym)
    if not precios:
        continue
    days = sorted(precios)
    px = np.array([precios[d] for d in days], dtype=float)
    print(f"\n{'='*92}\n{sym}\n{'='*92}")
    print(f"{'sleeve':<26}{'anio':>6}{'ret%':>9}{'B1 1x%':>9}{'vs B1':>9}"
          f"{'expos':>8}{'dias_mkt%':>11}{'ret/expos':>11}")
    for anio in (2025, 2026):
        m = np.array([d.year == anio for d in days])
        if m.sum() < 20:
            continue
        p = px[m]
        b1 = 100 * (p[-1] / p[0] - 1)
        filas = []
        for nombre, trades in por_activo[sym]:
            r, pos = strategy_daily_exact(trades, days, px)
            ret = 100 * (np.prod(1 + r[m]) - 1)
            expos = float(np.mean(np.abs(pos[m])))
            dias = 100 * float(np.mean(np.abs(pos[m]) > 0))
            # retorno por unidad de exposicion: separa "acierta" de "esta mas dentro"
            norm = ret / expos if expos > 1e-9 else float("nan")
            filas.append((ret, nombre, b1, expos, dias, norm))
        for ret, nombre, b1v, expos, dias, norm in sorted(filas, reverse=True):
            marca = "  BATE B1" if ret > b1v else ""
            print(f"{nombre:<26}{anio:>6}{ret:>9.2f}{b1v:>9.2f}{ret-b1v:>9.2f}"
                  f"{expos:>8.2f}{dias:>11.1f}{norm:>11.2f}{marca}")
        gana = sum(1 for f in filas if f[0] > b1)
        print(f"{'-> baten a B1:':<26}{anio:>6}   {gana} de {len(filas)}")
