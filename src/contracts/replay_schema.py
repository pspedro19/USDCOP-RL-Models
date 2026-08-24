"""Contrato de REPLAY DIARIO — C039 (CTR-REPLAY-001).

QUE PROBLEMA CIERRA
-------------------
Los bundles publicaban su serie diaria reducida a `{d, eq}` mientras sus productores
calculaban el stream completo: exposicion ejecutada, retorno bruto, coste y neto. Un consumidor
que quisiera reconstruir la serie tenia que **inferir** lo que faltaba, y la unica pista
disponible —`leverage` en el trade— es un PROMEDIO del segmento, no la exposicion diaria.

Medido en SPX: reconstruir desde `precio x leverage` difiere de la equity publicada **hasta
6.60 pp por trade**. La causa no era un bug del bundle sino un contrato desconocido: el PnL del
motor viene de `open_to_open_return` mientras `entry_price`/`exit_price` son niveles de CIERRE
usados solo como referencia. Ninguna cantidad de cuidado en el consumidor cierra esa brecha:
hay que publicar lo que el productor ya tiene.

Auditoria cruzada (Codex, CXD-842): **5 bundles BTC y 9 de Gold comparten el mismo `{d,eq}`
reducido**. El defecto es de cartera, no de un activo.

POR QUE `return_convention` ES OBLIGATORIO
------------------------------------------
Este tipo es GENERICO: lo usaran BTC (24/7), Gold (metals) y SPX (exchange hours), y **nada
obliga a que compartan convencion de retorno**. Un consumidor que calcule B1' con cierres de
`asset_daily_ohlcv` contra un stream open-to-open estaria mezclando series -- exactamente el
error de 6.60 pp que este contrato existe para cerrar, y saldria un numero plausible. Declararla
en el documento convierte una suposicion en un dato verificable.

POR QUE `exposure_exec` Y NO `target_exposure`
-----------------------------------------------
La fuente es `weights_exec`: la exposicion EJECUTADA. Un campo llamado `target_exposure`
prometeria un objetivo. En este repositorio ya nos costo caro tres veces que un nombre
prometiera algo distinto de su contenido: `leverage: 1.0` que era nominal, `exit_timestamp` que
iba una barra por delante de su precio, y `date_range` que el consumidor no leia.

Contract: CTR-REPLAY-001 · Espejo TS: usdcop-trading-dashboard/lib/contracts/replay.contract.ts
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

# Tolerancia de la invariante 1. `eq` se publica redondeada a 2 decimales sobre un capital de
# 10.000, o sea ~1e-6 relativo; los retornos decimales no se redondean. 1e-9 absorbe el error
# de coma flotante sin tolerar un error de contrato.
TOLERANCIA_NETO = 1e-9

# Tolerancia de la invariante 2 (recurrencia de la equity). Mas holgada que la anterior a
# proposito: aqui SI muerde el redondeo publicado de `eq`, acumulado a lo largo del anio.
TOLERANCIA_EQUITY_REL = 1e-4

CONVENCIONES_VALIDAS = ("open_to_open", "close_to_close")


@dataclass
class DailyReplayRow:
    """Una fila = un dia de la serie reconstruible de una estrategia.

    Campos DECIMALES (0.01 = 1%), nunca porcentajes: mezclar ambas escalas en el mismo
    documento es un error que ningun tipo detecta y que produce numeros 100x.
    """

    d: str                        # "YYYY-MM-DD"
    eq: float                     # equity al cierre del dia, en moneda
    exposure_exec: float          # exposicion EJECUTADA (weights_exec), no un objetivo
    gross_return_decimal: float   # retorno del dia ANTES de costes
    cost_return_decimal: float    # coste del dia, POSITIVO (se resta del bruto)
    net_return_decimal: float     # gross - cost

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DailyReplayRow":
        conocidos = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in conocidos})


def validar_fila(fila: DailyReplayRow) -> list[str]:
    """Invariantes 1 y 4 de C039. Devuelve la lista de fallos (vacia = conforme)."""
    fallos: list[str] = []
    numericos = {
        "eq": fila.eq, "exposure_exec": fila.exposure_exec,
        "gross_return_decimal": fila.gross_return_decimal,
        "cost_return_decimal": fila.cost_return_decimal,
        "net_return_decimal": fila.net_return_decimal,
    }
    for nombre, v in numericos.items():
        if v is None or not math.isfinite(float(v)):
            fallos.append(f"{fila.d}: {nombre} no es finito ({v!r})")
    if fallos:
        return fallos

    # Invariante 1: net == gross - cost.
    esperado = fila.gross_return_decimal - fila.cost_return_decimal
    if abs(esperado - fila.net_return_decimal) > TOLERANCIA_NETO:
        fallos.append(
            f"{fila.d}: net={fila.net_return_decimal!r} pero gross-cost={esperado!r}")

    # El coste es POSITIVO por convencion declarada. Un coste negativo significaria que el
    # espejo lo invirtio, y la invariante 1 seguiria cuadrando: por eso se comprueba aparte.
    if fila.cost_return_decimal < 0:
        fallos.append(
            f"{fila.d}: cost_return_decimal={fila.cost_return_decimal!r} es NEGATIVO; "
            f"la convencion es positivo-se-resta")
    return fallos


@dataclass
class DailyReplayDocument:
    """El documento `signals_YYYY.json` completo.

    `return_convention` es OBLIGATORIA: sin ella el consumidor no puede saber contra que serie
    del activo comparar, y compararia contra la equivocada sin enterarse.
    """

    kind: str                     # "daily_replay"
    strategy_id: str
    year: int
    initial_capital: float
    return_convention: str        # "open_to_open" | "close_to_close"
    rows: list                    # list[DailyReplayRow]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rows"] = [r.to_dict() if isinstance(r, DailyReplayRow) else r for r in self.rows]
        return d


def validar_documento(doc: DailyReplayDocument) -> list[str]:
    """Invariantes 1-4 de C039 sobre el documento entero."""
    fallos: list[str] = []
    if doc.return_convention not in CONVENCIONES_VALIDAS:
        fallos.append(
            f"return_convention={doc.return_convention!r} no esta declarada; "
            f"validas: {CONVENCIONES_VALIDAS}. Sin ella el consumidor compara contra la "
            f"serie equivocada sin enterarse.")
    filas = [r if isinstance(r, DailyReplayRow) else DailyReplayRow.from_dict(r)
             for r in doc.rows]
    for f in filas:
        fallos.extend(validar_fila(f))
    if fallos:
        return fallos

    # Invariante 2: `eq` recurre desde `initial_capital` componiendo `net_return_decimal`.
    equity = float(doc.initial_capital)
    for f in filas:
        equity *= (1.0 + f.net_return_decimal)
        if abs(equity - f.eq) / max(abs(f.eq), 1e-9) > TOLERANCIA_EQUITY_REL:
            fallos.append(
                f"{f.d}: eq publicada {f.eq} pero la recurrencia da {equity:.6f}")
            break     # una sola vez: a partir del primer desajuste el resto es consecuencia
    return fallos
