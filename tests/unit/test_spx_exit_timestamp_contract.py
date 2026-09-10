"""Candado causal de CXD-828: el sello de salida de SPX debe casar con su propio precio.

QUE FIJA
--------
Que en el bundle SPX que el manifest declara PRODUCTION, el `exit_price` de cada trade sea el
cierre de la barra que su `exit_timestamp` senala. Antes de la correccion el sello iba una
barra POR DELANTE del precio: 10 de 12 trades casaban con el cierre de D+1 a 0.0 bp mientras
declaraban D.

POR QUE ES UN CANDADO Y NO UNA COMPROBACION DECORATIVA
------------------------------------------------------
El defecto no producia error ni numero absurdo: el total del trade se conserva (el producto
telescopa), asi que solo se desplaza la colocacion temporal. Un consumidor que componga la
serie DIARIA acredita el retorno de D+1 en el dia D -- una fuga de un dia, invisible en el
PnL. Si alguien regenera el bundle con la convencion vieja, este test lo detiene; ningun gate
de retorno lo haria.

EXCEPCION DECLARADA, no silenciosa
----------------------------------
La salida del 2026-07-27 no tiene barra D+1 en la serie versionada (termina ese dia) y su
precio queda a ~1.3 bp del cierre de D. No se corrigio -sin barra que case no se inventa un
sello- y por eso se tolera aqui con nombre y razon, en vez de relajar la tolerancia global.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

BUNDLES = REPO / "usdcop-trading-dashboard" / "public" / "data" / "strategies"
PRECIOS = REPO / "data" / "backups" / "features" / "asset_daily_ohlcv.parquet"
TOLERANCIA_BP = 2.0            # 2 bp: holgura de redondeo del publicador, no de convencion
VERSION = "2.0.0"              # la que el manifest declara PRODUCTION

# Salidas sin barra D+1 en la serie versionada. Se listan con su razon.
EXCEPCIONES: set[date] = {date(2026, 7, 27)}


def _cierres() -> dict[date, float]:
    df = pd.read_parquet(PRECIOS, columns=["time", "symbol", "close"])
    df = df[df["symbol"] == "SPX/500"].copy()
    df["d"] = pd.to_datetime(df["time"], utc=True).dt.date
    df = df.sort_values("d").drop_duplicates("d", keep="last")
    return {r.d: float(r.close) for r in df.itertuples()}


def _fecha(v) -> date:
    ts = pd.Timestamp(v)
    return (ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")).date()


def _ficheros() -> list[Path]:
    return sorted(BUNDLES.glob(f"spx500_*/backtests/{VERSION}/trades_*.json"))


@pytest.mark.skipif(not PRECIOS.exists(), reason="serie de precios versionada ausente")
def test_every_spx_exit_price_belongs_to_the_bar_its_timestamp_names() -> None:
    """El sello y el precio tienen que hablar del mismo dia."""
    cierres = _cierres()
    ficheros = _ficheros()
    assert ficheros, f"no hay bundles SPX {VERSION} que verificar"

    desalineados: list[str] = []
    revisados = 0
    for f in ficheros:
        raw = json.loads(f.read_text(encoding="utf-8"))
        for t in raw.get("trades", raw):
            d = _fecha(t["exit_timestamp"])
            if d in EXCEPCIONES:
                continue
            cierre = cierres.get(d)
            if cierre is None:
                desalineados.append(f"{f.name}: salida {d} sin barra en la serie")
                continue
            revisados += 1
            err_bp = abs(float(t["exit_price"]) / cierre - 1) * 1e4
            if err_bp > TOLERANCIA_BP:
                desalineados.append(
                    f"{f.parent.parent.parent.name}/{f.name}: salida {d} "
                    f"precio={t['exit_price']} vs cierre_D={cierre:.2f} ({err_bp:.1f} bp)")

    assert revisados >= 10, f"solo se revisaron {revisados} salidas; el candado no muerde"
    assert not desalineados, (
        "el sello de salida no corresponde a la barra de su precio:\n  "
        + "\n  ".join(desalineados))


@pytest.mark.skipif(not PRECIOS.exists(), reason="serie de precios versionada ausente")
def test_the_correction_did_not_touch_the_money() -> None:
    """La correccion es de METADATOS: el dinero publicado debe seguir cuadrando consigo mismo.

    Complementa al test anterior: sin esto, "alinear el sello" podria lograrse moviendo el
    PRECIO, lo que cambiaria los retornos publicados.

    EL INVARIANTE CORRECTO NO ES EL RATIO DE PRECIOS, y averiguarlo costo un test fallido.
    La primera version exigia `pnl_pct == leverage * (exit/entry - 1)` y fallaba en AMBAS
    versiones del bundle, antes y despues de la correccion -- o sea que no delataba una
    regresion mia sino una suposicion mia equivocada. Medido: en `daily_ma200` 2025 el trade 3
    va de 5886.55 a 6858.47 (**+16.51% de precio**) y declara **pnl_pct +17.4562**, que es
    exactamente su cambio de equity. La posicion se escala por EXPOSICION dentro del trade;
    `leverage: 1.0` es nominal, no la exposicion realizada. El contrato interno del artefacto
    es equity <-> pnl, y eso es lo que se fija aqui.
    """
    problemas: list[str] = []
    for f in _ficheros():
        raw = json.loads(f.read_text(encoding="utf-8"))
        for t in raw.get("trades", raw):
            e0, e1 = float(t["equity_at_entry"]), float(t["equity_at_exit"])
            esperado_pct = (e1 / e0 - 1) * 100
            if abs(esperado_pct - float(t["pnl_pct"])) > 0.01:
                problemas.append(
                    f"{f.name} trade {t.get('trade_id')}: pnl_pct={t['pnl_pct']} "
                    f"pero la equity implica {esperado_pct:.4f}")
            if abs((e1 - e0) - float(t["pnl_usd"])) > 0.01:
                problemas.append(
                    f"{f.name} trade {t.get('trade_id')}: pnl_usd={t['pnl_usd']} "
                    f"pero la equity implica {e1 - e0:.2f}")
    assert not problemas, "el PnL dejo de cuadrar con la equity:\n  " + "\n  ".join(problemas)


# ---------------------------------------------------------------------------
# CXD-836/837 — los cuatro casos del CORRECTOR, no del bundle final. Los de
# arriba fijan el resultado publicado; estos ejercitan la herramienta, que es
# donde vivia el bug `bisect_left + 1` que encontro Codex.
# ---------------------------------------------------------------------------
from scripts.ops.fix_spx_exit_timestamps import sello_corregido  # noqa: E402

# Calendario bursatil de juguete: el 2025-03-08 y 09 son fin de semana y NO existen.
_DIAS = [date(2025, 3, 6), date(2025, 3, 7), date(2025, 3, 10), date(2025, 3, 11)]
_CIERRES = {date(2025, 3, 6): 100.0, date(2025, 3, 7): 101.0,
            date(2025, 3, 10): 102.0, date(2025, 3, 11): 103.0}


def test_a_stamp_on_a_trading_day_moves_to_the_next_bar() -> None:
    """(a) El caso normal: sello el 10, precio del 11 -> el sello pasa al 11."""
    nuevo, _ = sello_corregido("2025-03-10 20:00:00", 103.0, _CIERRES, _DIAS)
    assert nuevo == "2025-03-11 20:00:00"


def test_a_weekend_stamp_takes_the_first_bar_after_it_not_the_second() -> None:
    """(b) EL BUG QUE ENCONTRO CODEX, fijado.

    Sello el sabado 2025-03-08 con el precio del lunes 2025-03-10. La primera barra
    estrictamente posterior es el **10**. `bisect_left(dias, ed) + 1` devolvia el **11**,
    porque con `ed` fuera del calendario `bisect_left` ya apunta al 10 y sumarle uno lo salta.
    """
    nuevo, _ = sello_corregido("2025-03-08 20:00:00", 102.0, _CIERRES, _DIAS)
    assert nuevo == "2025-03-10 20:00:00", "salto la primera barra posterior"


def test_a_price_matching_nothing_is_left_untouched() -> None:
    """(c) Sin barra que case no se inventa un sello."""
    nuevo, motivo = sello_corregido("2025-03-10 20:00:00", 999.0, _CIERRES, _DIAS)
    assert nuevo is None and "sin barra que case" in motivo


def test_applying_the_correction_twice_changes_nothing() -> None:
    """(d) Idempotencia: la segunda pasada no mueve nada, porque el sello ya casa."""
    primero, _ = sello_corregido("2025-03-10 20:00:00", 103.0, _CIERRES, _DIAS)
    assert primero == "2025-03-11 20:00:00"
    segundo, motivo = sello_corregido(primero, 103.0, _CIERRES, _DIAS)
    assert segundo is None and motivo == "ya casa con su propio dia"


def test_a_stamp_already_matching_its_own_day_is_not_moved() -> None:
    """Guardarrail del anterior: un sello correcto nunca se desplaza 'por si acaso'."""
    nuevo, motivo = sello_corregido("2025-03-10 20:00:00", 102.0, _CIERRES, _DIAS)
    assert nuevo is None and motivo == "ya casa con su propio dia"
