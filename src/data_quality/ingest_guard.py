"""BL-40 — el consumidor productivo que le faltaba a la cuarentena de calidad.

`QualityRuleSet.evaluate_provider_bar` existía, era correcto y **nadie lo invocaba**
fuera de su propio módulo (medido por Codex en CXD-448). Un evaluador sin llamador no
protege ninguna barra: es el mismo estado del que salió BL-16.

No podía tener llamador antes por una razón concreta, no por olvido: `evaluate_provider_bar`
exige un `ProviderSymbolRegistry` que resuelva `(provider_id, provider_symbol)` a un
`instrument_id` canónico, y `reference.provider_symbol` estaba **vacía**. Poblar la
espina (BL-37) es lo que hace construible este guard; de ahí que la cadena de la ficha
sea *producer → quarantine → canonical*.

Diseño:

* **La cuarentena no corrige nada.** Una barra rechazada no se recorta ni se ajusta: se
  registra en `quality.quarantine_event` con su regla, su valor observado y el registro
  original, y **no se escribe** en la tabla de mercado. Corregir en silencio es peor que
  rechazar, porque produce una serie que nadie sabe que fue tocada.
* **Fail-closed de verdad.** Si el registry no resuelve el alias, la barra no entra. Un
  proveedor desconocido escribiendo bajo un símbolo desconocido es exactamente el caso
  que la identidad canónica existe para impedir.
* **El evento se escribe aunque la barra se rechace** — de hecho *porque* se rechaza. Un
  guard que bloquea sin dejar rastro convierte un dato malo en un dato ausente, que es
  indistinguible de "el proveedor no publicó".

**BLOQUEADO — no cablear todavía.** Al probar el guard contra la espina viva apareció
un choque de namespaces que hace que hoy rechazaría *todas* las barras, incluidas las
buenas:

* `reference.instrument.instrument_id` es un **UUID** (migración 072:
  `instrument_id UUID PRIMARY KEY DEFAULT gen_random_uuid()`), y
  `quality.quarantine_event.instrument_id` es UUID con FK a esa tabla.
* pero `QualityRuleSet` usa el `instrument_id` devuelto por el registry como **clave de
  `price_ranges`**, y ese YAML está indexado por *slugs* (`usdmxn`, `usdclp`).

Medido con una barra USD/COP perfectamente válida:

    evaluate_provider_bar('twelvedata', 'USD/COP', barra_buena)
    -> accepted=False  rule='bar.unknown_instrument'
       "no versioned price range declared for canonical instrument"

Es decir: la migración y el evaluador **no están de acuerdo sobre qué es un
`instrument_id`**. Conectar el guard ahora convertiría la cuarentena en un apagón de
ingesta, y traducir el UUID a slug dentro de este módulo sería fabricar una equivalencia
que ninguna de las dos capas declara. La decisión —clavar `price_ranges` al símbolo
canónico, o hacer determinista el `instrument_id`— toca `rules.py` y el esquema, así que
es bilateral y no la tomo por mi cuenta.

Contract: CTR-QLAB-FABRIC-004 (BL-40) · Date: 2026-08-04
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.data_quality.rules import QualityDecision, QualityRuleSet
from src.market.identity import ProviderSymbol, ProviderSymbolRegistry

REPO = Path(__file__).resolve().parents[2]
QUALITY_RANGES = REPO / "config" / "quality" / "market_price_ranges.yaml"


@dataclass(frozen=True, slots=True)
class ScreeningResult:
    """Qué entró, qué quedó en cuarentena, y por qué."""

    accepted: list[Mapping[str, Any]]
    quarantined: list[tuple[Mapping[str, Any], QualityDecision]]

    @property
    def n_quarantined(self) -> int:
        return len(self.quarantined)


class IngestGuardError(RuntimeError):
    """El guard no puede operar. Nunca degrada a 'dejar pasar todo'."""


def registry_from_spine(conn) -> ProviderSymbolRegistry:
    """Construye el registry de alias desde `reference.provider_symbol`.

    La espina es la fuente: si un alias no está ahí, no existe canónicamente. Un
    registry vacío es un error duro y no un registry permisivo — con cero alias, todas
    las barras se rechazarían y alguien apagaría el guard por "roto".
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT provider_id, provider_symbol, instrument_id::text "
            "FROM reference.provider_symbol"
        )
        filas = cur.fetchall()

    if not filas:
        raise IngestGuardError(
            "reference.provider_symbol está vacía: sin espina de identidad no hay alias "
            "canónico que resolver, y este guard rechazaría toda barra. Puebla la espina "
            "(scripts/data/seed_reference_spine.py) antes de activar la cuarentena"
        )
    return ProviderSymbolRegistry(
        ProviderSymbol(provider_id=p, provider_symbol=s, instrument_id=i)
        for p, s, i in filas
    )


def ruleset_from_spine(conn, ranges_path: Path = QUALITY_RANGES) -> QualityRuleSet:
    """`QualityRuleSet` con su registry ya enganchado a la identidad canónica."""
    return QualityRuleSet.from_yaml(ranges_path, identity_registry=registry_from_spine(conn))


def _instrument_id_of(decision: QualityDecision) -> str | None:
    valor = decision.observed_value
    if isinstance(valor, Mapping):
        candidato = valor.get("instrument_id")
        if isinstance(candidato, str):
            return candidato
    return None


def record_quarantine(
    conn,
    *,
    provider_id: str,
    provider_symbol: str,
    row: Mapping[str, Any],
    decision: QualityDecision,
    rule_version: str,
) -> None:
    """Escribe el evento. Es la mitad que convierte un rechazo en evidencia."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO quality.quarantine_event "
            "(entity_type, entity_id, instrument_id, rule_id, rule_version, "
            " observed_value, source_record) "
            "VALUES ('ohlcv_bar', %s, %s::uuid, %s, %s, %s::jsonb, %s::jsonb)",
            (
                f"{provider_id}:{provider_symbol}",
                _instrument_id_of(decision),
                decision.rule_id or "bar.unknown",
                rule_version,
                json.dumps(_jsonable(decision.observed_value)),
                json.dumps(_jsonable(dict(row))),
            ),
        )


def _jsonable(valor: Any) -> Any:
    if isinstance(valor, Mapping):
        return {str(k): _jsonable(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_jsonable(v) for v in valor]
    if isinstance(valor, datetime):
        return valor.isoformat()
    if isinstance(valor, (str, int, float, bool)) or valor is None:
        return valor
    return str(valor)


def screen_bars(
    conn,
    ruleset: QualityRuleSet,
    *,
    provider_id: str,
    provider_symbol: str,
    rows: Iterable[Mapping[str, Any]],
    rule_version: str = "unversioned",
    observed_at: datetime | None = None,
) -> ScreeningResult:
    """Evalúa cada barra; las rechazadas van a cuarentena y **no** se devuelven.

    El llamador escribe únicamente `result.accepted`. No hay modo "avisar y escribir
    igual": eso sería un log, no una cuarentena.
    """
    aceptadas: list[Mapping[str, Any]] = []
    rechazadas: list[tuple[Mapping[str, Any], QualityDecision]] = []

    for row in rows:
        decision = ruleset.evaluate_provider_bar(
            provider_id, provider_symbol, row, observed_at=observed_at
        )
        if decision.accepted:
            aceptadas.append(row)
            continue
        rechazadas.append((row, decision))
        record_quarantine(
            conn,
            provider_id=provider_id,
            provider_symbol=provider_symbol,
            row=row,
            decision=decision,
            rule_version=rule_version,
        )

    return ScreeningResult(accepted=aceptadas, quarantined=rechazadas)


def rows_from_frame(df, columnas: Sequence[str]) -> list[dict[str, Any]]:
    """Convierte un DataFrame a filas planas para el evaluador, sin tocar valores."""
    return [
        {col: fila[col] for col in columnas if col in fila}
        for fila in df.to_dict(orient="records")
    ]
