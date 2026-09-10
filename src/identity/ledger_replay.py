"""BL-17 — reconstrucción independiente del `semantic_hash` de un paper ledger.

`src/identity/canonical.py` ya sabe convertir un valor en bytes canónicos y hashearlos.
Lo que faltaba —y lo que BL-17 pide literalmente— es un **gate que reconstruya desde
cero** el hash de un ledger anclado, de forma que *mutar una fila rompa la reproducción
y nombre ambos hashes*.

La distinción que hace útil a este módulo:

* **`id` y `created_at` NO entran en el hash.** Son metadatos de escritura, no hechos
  económicos. Si entraran, restaurar el ledger desde un backup —que reasigna `id` y
  `created_at`— produciría un hash distinto para exactamente los mismos trades, y el
  gate gritaría "corrupción" ante una operación legítima. Un candado que llora con
  cualquier restauración acaba desactivado, y entonces no protege nada.
* **`week_pnl_pct` y compañía SÍ entran.** Son el resultado económico: si alguien los
  edita, eso *es* la corrupción que este gate existe para detectar.

El orden de las filas se impone al leer (por semana), no se hereda del `SELECT`: dos
lecturas de la misma tabla deben producir el mismo hash aunque el planificador devuelva
las filas en otro orden.

Contract: CTR-QLAB-FABRIC-004 (BL-17) · Date: 2026-08-04
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from src.identity.canonical import semantic_hash

#: **Todas** las columnas persistidas del ledger salvo el surrogate técnico.
#:
#: La primera versión de esta tupla listaba once campos "económicos" y dejaba fuera las
#: métricas acumuladas. Codex lo refutó con una mutación (CXD-449): cambiar
#: `running_da_pct` de 55.0 a 99.0 producía **hashes idénticos**. Esas columnas no son
#: derivadas decorativas — `running_sharpe` lo lee `control_system_health`, y el DA y el
#: drawdown gobiernan gates y circuit breaker. Un gate de integridad que ignora las
#: métricas por las que se decide protege lo que menos importa.
#:
#: La lección detrás: la frontera correcta no es "lo económico" —un juicio mío sobre qué
#: importa— sino "lo persistido", que es un hecho comprobable contra el esquema.
LEDGER_SEMANTIC_FIELDS: tuple[str, ...] = (
    "strategy_id",
    "signal_date",
    "inference_year",
    "inference_week",
    "direction",
    "leverage",
    "week_pnl_pct",
    "n_subtrades",
    "cumulative_pnl_pct",
    "running_da_pct",
    "running_da_short_pct",
    "running_da_long_pct",
    "running_sharpe",
    "running_max_dd_pct",
    "n_weeks",
    "n_long",
    "n_short",
    "long_pct_8w",
    "consecutive_losses",
    "circuit_breaker",
    "gate_status",
    # `notes` es evidencia auditada: excluirla exigiría un contrato explícito que hoy
    # no existe, así que entra (CXD-449 §1).
    "notes",
)

#: Surrogate técnico, lo ÚNICO fuera del hash: `id` lo asigna la secuencia y `created_at`
#: la hora de escritura. Restaurar un backup reasigna ambos sin que cambie un solo hecho,
#: y un gate que gritara "corrupción" ante una restauración legítima acabaría apagado.
LEDGER_EXCLUDED_FIELDS: tuple[str, ...] = ("id", "created_at")

#: Clave de ordenación: la identidad de una fila de paper trading es su semana.
_ORDER_KEY = ("strategy_id", "inference_year", "inference_week", "signal_date")

LEDGER_TABLE = "forecast_h5_paper_trading"


class LedgerReproductionError(AssertionError):
    """El ledger no reprodujo su hash anclado. Nombra AMBOS hashes, siempre.

    Un error que dijera sólo "no coincide" obligaría a reconstruir a mano qué se
    esperaba; nombrar los dos permite pegarlos en un diff y seguir el rastro.
    """

    def __init__(self, esperado: str, obtenido: str, detalle: str = "") -> None:
        self.esperado = esperado
        self.obtenido = obtenido
        mensaje = (
            f"el paper ledger no reproduce su hash anclado\n"
            f"  esperado (anclado)     : {esperado}\n"
            f"  obtenido (reconstruido): {obtenido}"
        )
        if detalle:
            mensaje += f"\n  {detalle}"
        super().__init__(mensaje)


def _semantic_row(fila: Mapping[str, Any]) -> dict[str, Any]:
    """Proyecta una fila a sus campos semánticos, normalizando a texto estable."""
    faltantes = [c for c in LEDGER_SEMANTIC_FIELDS if c not in fila]
    if faltantes:
        raise LedgerReproductionError(
            "n/a",
            "n/a",
            f"la fila no trae los campos semánticos {faltantes}: no se puede hashear "
            "un ledger incompleto sin inventar valores",
        )
    return {campo: _stable(fila[campo]) for campo in LEDGER_SEMANTIC_FIELDS}


def _stable(valor: Any) -> Any:
    """Representación estable: fechas a ISO, decimales/floats a texto normalizado.

    Los `float` pasan por `repr` para que `1.0` y `1` no colisionen ni divergan según
    de dónde venga la fila (psycopg2 devuelve `float`; un JSON restaurado puede traer
    `int`). La canonicalización numérica de `canonical.py` opera sobre el árbol ya
    construido; aquí sólo se garantiza que dos lecturas equivalentes lleguen iguales.
    """
    if valor is None:
        return None
    if isinstance(valor, bool):
        return valor
    if isinstance(valor, (int,)):
        return str(valor)
    if isinstance(valor, float):
        return repr(float(valor))
    if hasattr(valor, "isoformat"):
        return valor.isoformat()
    return str(valor)


def normalize_ledger(filas: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Ordena y proyecta el ledger. El orden lo impone esta función, no el `SELECT`."""
    proyectadas = [_semantic_row(f) for f in filas]
    return sorted(proyectadas, key=lambda r: tuple(str(r[c]) for c in _ORDER_KEY))


def ledger_semantic_hash(filas: Iterable[Mapping[str, Any]]) -> str:
    """`semantic_hash` del ledger completo, reconstruido desde las filas crudas."""
    return semantic_hash({"table": LEDGER_TABLE, "rows": normalize_ledger(filas)})


def assert_ledger_reproduces(
    esperado: str, filas: Iterable[Mapping[str, Any]], *, detalle: str = ""
) -> str:
    """Reconstruye el hash y **aborta** si no coincide con el anclado."""
    obtenido = ledger_semantic_hash(filas)
    if obtenido != esperado:
        raise LedgerReproductionError(esperado, obtenido, detalle)
    return obtenido


def ledger_prefix(
    filas: Iterable[Mapping[str, Any]], *, until_year: int, until_week: int
) -> list[dict[str, Any]]:
    """Filas hasta la semana de corte, inclusive.

    El ancla se calcula sobre un PREFIJO, no sobre el ledger entero, porque el ledger
    es append-only: cada lunes de paper trading añade una fila. Un ancla sobre el total
    se invalidaría sola cada semana, y un gate que se pone rojo por funcionamiento
    normal es un gate que alguien acaba desactivando — con lo que dejaría de detectar
    la mutación que sí importa. Anclar el pasado cerrado detecta reescrituras de la
    historia, que es exactamente la amenaza.
    """
    return [
        dict(f)
        for f in filas
        if (int(f["inference_year"]), int(f["inference_week"])) <= (until_year, until_week)
    ]


def anchor_payload(
    filas: Iterable[Mapping[str, Any]], *, until_year: int, until_week: int
) -> dict[str, Any]:
    """Ancla verificable: hash del prefijo + su corte + cuántas filas lo componen.

    `n_rows` va dentro a propósito: sin él, borrar una fila y ajustar otra podría
    recomponer un hash por casualidad de la agregación; con él, el conteo tiene que
    cuadrar además del contenido.
    """
    prefijo = ledger_prefix(filas, until_year=until_year, until_week=until_week)
    return {
        "table": LEDGER_TABLE,
        "until_year": until_year,
        "until_week": until_week,
        "n_rows": len(prefijo),
        "semantic_hash": ledger_semantic_hash(prefijo),
        "semantic_fields": list(LEDGER_SEMANTIC_FIELDS),
    }


def assert_anchor_holds(ancla: Mapping[str, Any], filas: Iterable[Mapping[str, Any]]) -> str:
    """Verifica un ancla contra el ledger vivo. Falla nombrando ambos hashes."""
    prefijo = ledger_prefix(
        filas, until_year=int(ancla["until_year"]), until_week=int(ancla["until_week"])
    )
    if len(prefijo) != int(ancla["n_rows"]):
        raise LedgerReproductionError(
            str(ancla["semantic_hash"]),
            ledger_semantic_hash(prefijo),
            f"el prefijo anclado tenía {ancla['n_rows']} filas y ahora tiene "
            f"{len(prefijo)}: se añadieron o borraron filas del pasado cerrado",
        )
    if list(ancla.get("semantic_fields", LEDGER_SEMANTIC_FIELDS)) != list(
        LEDGER_SEMANTIC_FIELDS
    ):
        raise LedgerReproductionError(
            str(ancla["semantic_hash"]),
            "n/a",
            "los campos semánticos cambiaron desde que se ancló: el hash viejo ya no "
            "es comparable, hay que re-anclar de forma explícita y auditada",
        )
    return assert_ledger_reproduces(
        str(ancla["semantic_hash"]),
        prefijo,
        detalle=f"corte {ancla['until_year']}-W{int(ancla['until_week']):02d}",
    )


def read_ledger(conn, strategy_id: str | None = None) -> list[dict[str, Any]]:
    """Lee el paper ledger de la base como `dict`s, sin depender del orden del motor."""
    columnas = ", ".join(LEDGER_SEMANTIC_FIELDS)
    sql = f"SELECT {columnas} FROM {LEDGER_TABLE}"  # noqa: S608 - columnas de constante
    parametros: Sequence[Any] = ()
    if strategy_id is not None:
        sql += " WHERE strategy_id = %s"
        parametros = (strategy_id,)
    with conn.cursor() as cur:
        cur.execute(sql, parametros)
        return [dict(zip(LEDGER_SEMANTIC_FIELDS, fila)) for fila in cur.fetchall()]
