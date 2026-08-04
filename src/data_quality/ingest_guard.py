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

**El choque de namespaces, y cómo se resolvió sin tocar `rules.py`.** La primera versión
de este guard rechazaba *todas* las barras, incluidas las buenas:

    evaluate_provider_bar('twelvedata', 'USD/COP', barra_buena)
    -> accepted=False  rule='bar.unknown_instrument'

Causa: `QualityRuleSet` indexa `price_ranges` por el `instrument_id` que devuelve el
registry —desde la espina, un **UUID** (072)—, mientras que
`config/quality/market_price_ranges.yaml` los indexa por slugs (`usdmxn`, `usdclp`), el
vocabulario anterior a que existiera identidad canónica.

Resuelto según **CXD-454, opción (a)**: los rangos se resuelven por `canonical_symbol` y
el UUID sigue siendo la identidad y la FK persistida — nada se hace determinista ni se
reescribe en una tabla poblada. La traducción `canonical_symbol → instrument_id` sale del
**registro** (`reference.instrument`), no de una equivalencia inferida en código; y los
rangos por símbolo salen de dos declaraciones existentes (el YAML normativo y el
`price_range` que cada activo declara sobre sí mismo). Sin heurística y sin tocar el
evaluador. Medido:

    USD/COP dentro de rango   -> accepted=True
    USD/COP a 99999           -> QUARANTINED  rule='bar.range.<uuid>'
    alias no registrado       -> QUARANTINED  rule='bar.unknown_alias'

Queda un límite real y declarado: **`usdmxn` y `usdclp` —los dos únicos instrumentos con
rango escalonado por proveedor y fecha— no tienen `AssetProfile`**, así que no están en la
espina y no pueden identificarse canónicamente. El instrumento con la regla de calidad
más cuidada del repositorio (corte Banxico CF373 de 1993) es precisamente el que la
identidad no alcanza. Darles perfil es la vía; inventarles instrumento, no.

Contract: CTR-QLAB-FABRIC-004 (BL-40) · Date: 2026-08-04
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime

import yaml
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.data_quality.rules import QualityDecision, QualityRuleSet, ScopedPriceRange
from src.market.identity import ProviderSymbol, ProviderSymbolRegistry

REPO = Path(__file__).resolve().parents[2]
QUALITY_RANGES = REPO / "config" / "quality" / "market_price_ranges.yaml"
ASSETS_DIR = REPO / "config" / "assets"


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


def canonical_price_ranges(
    conn, assets_dir: Path = ASSETS_DIR, ranges_path: Path = QUALITY_RANGES
) -> dict[str, tuple[Decimal, Decimal]]:
    """Rangos de precio **re-clavados a la identidad canónica** (el `instrument_id` UUID).

    Forma acordada en CXD-454, opción (a): **los rangos se resuelven por
    `canonical_symbol`; el UUID sigue siendo la identidad y la FK persistida.** No se
    hace determinista ni se reescribe `instrument_id` en una tabla ya poblada.

    La traducción `canonical_symbol → instrument_id` sale del **registro**
    (`reference.instrument`), no de una equivalencia inferida en código. Esa distinción
    es el fondo del asunto: dos identificadores que "se parecen" no son lo mismo salvo
    que alguna capa lo declare, y aquí la declara la espina.

    Un símbolo sin rango declarado **no recibe rango**, y sin rango sus barras se van a
    cuarentena. Lo no declarado no se rellena.
    """
    por_simbolo = declared_ranges_by_canonical_symbol(assets_dir, ranges_path)

    # La traducción `canonical_symbol -> instrument_id` sale del REGISTRO
    # (`reference.instrument`), no de una equivalencia inferida en código (CXD-454).
    with conn.cursor() as cur:
        cur.execute("SELECT canonical_symbol, instrument_id::text FROM reference.instrument")
        uuid_por_simbolo = dict(cur.fetchall())

    return {
        uuid_por_simbolo[simbolo]: bounds
        for simbolo, bounds in sorted(por_simbolo.items())
        if simbolo in uuid_por_simbolo
    }


def declared_scoped_ranges_by_canonical_symbol(
    ranges_path: Path = QUALITY_RANGES,
) -> dict[str, tuple[ScopedPriceRange, ...]]:
    """Rangos **escalonados** por proveedor y fecha, indexados por símbolo canónico.

    Son las entradas del YAML normativo con forma de lista de dicts
    (`provider_id` + `valid_from` + `bounds`). El caso que las justifica es `usdmxn`:
    el corte de unidad monetaria de Banxico (SIE CF373, 1993-01-01) hace que un mismo
    símbolo tenga rangos económicos distintos antes y después, y que sólo valgan para
    el proveedor que los publicó.

    Aplanarlas a un `(low, high)` único —que es lo que hacía la versión anterior de
    este módulo al no mirarlas— **borra el alcance**: el corte deja de significar nada
    y una barra de 1990 se validaría contra el rango de hoy.
    """
    config = yaml.safe_load(ranges_path.read_text(encoding="utf-8")) or {}
    escalonados: dict[str, tuple[ScopedPriceRange, ...]] = {}

    for clave, valor in (config.get("price_ranges") or {}).items():
        if not isinstance(valor, list) or not valor or not isinstance(valor[0], Mapping):
            continue  # entrada plana: la maneja `declared_ranges_by_canonical_symbol`
        reglas = []
        for entrada in valor:
            limites = entrada.get("bounds")
            if not (isinstance(limites, (list, tuple)) and len(limites) == 2):
                raise IngestGuardError(
                    f"{clave}: rango escalonado sin 'bounds' [low, high] — un alcance "
                    "sin límites no puede gobernar nada"
                )
            momento = entrada.get("valid_from")
            if not momento:
                raise IngestGuardError(
                    f"{clave}: rango escalonado sin 'valid_from'. Es su razón de ser: "
                    "sin fecha de corte el escalón no existe"
                )
            desde = datetime.fromisoformat(str(momento).replace("Z", "+00:00"))
            if desde.tzinfo is None:
                raise IngestGuardError(f"{clave}: 'valid_from' debe llevar zona horaria")
            reglas.append(
                ScopedPriceRange(
                    provider_id=str(entrada["provider_id"]).strip().lower(),
                    valid_from=desde,
                    low=Decimal(str(limites[0])),
                    high=Decimal(str(limites[1])),
                )
            )
        escalonados[str(clave)] = tuple(reglas)

    return escalonados


def _symbol_of(asset_id: str, assets_dir: Path) -> str | None:
    ruta = assets_dir / f"{asset_id}.yaml"
    if not ruta.is_file():
        return None
    declarado = yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}
    simbolo = declarado.get("symbol")
    return str(simbolo) if simbolo else None


def scoped_symbols(assets_dir: Path = ASSETS_DIR, ranges_path: Path = QUALITY_RANGES) -> set[str]:
    """Símbolos canónicos que tienen regla escalonada, bajo cualquiera de sus nombres.

    El YAML normativo indexa por `asset_id` (`usdmxn`) y la espina por `canonical_symbol`
    (`USD/MXN`); ambos apuntan al mismo instrumento. Resolver los dos evita el fallo
    silencioso de que la regla escalonada exista y no se aplique por no reconocer la
    clave.
    """
    escalonados = set(declared_scoped_ranges_by_canonical_symbol(ranges_path))
    resueltos = set(escalonados)
    for clave in escalonados:
        simbolo = _symbol_of(clave, assets_dir)
        if simbolo:
            resueltos.add(simbolo)
    return resueltos


def declared_ranges_by_canonical_symbol(
    assets_dir: Path = ASSETS_DIR, ranges_path: Path = QUALITY_RANGES
) -> dict[str, tuple[Decimal, Decimal]]:
    """Rangos declarados, indexados por **símbolo canónico** (CXD-454, opción (a)).

    Dos fuentes, ambas declarativas y ninguna inventada:

    1. `config/quality/market_price_ranges.yaml` — los rangos normativos. Hoy sólo
       declara `usdmxn`/`usdclp`, que no tienen símbolo canónico porque carecen de
       `AssetProfile`; se conservan aquí para que el día que lo tengan entren solos.
    2. `config/assets/<id>.yaml::price_range` — el prior económico que cada activo
       declara sobre sí mismo, bajo su `symbol` canónico.

    Un activo sin ninguna de las dos **no recibe rango**: sin rango sus barras caen a
    cuarentena. Lo no declarado no se rellena.
    """
    rangos: dict[str, tuple[Decimal, Decimal]] = {}

    normativos = (yaml.safe_load(ranges_path.read_text(encoding="utf-8")) or {}).get(
        "price_ranges", {}
    )
    for clave, valor in normativos.items():
        if isinstance(valor, (list, tuple)) and len(valor) == 2 and not isinstance(valor[0], Mapping):
            rangos[str(clave)] = (Decimal(str(valor[0])), Decimal(str(valor[1])))
        # Las entradas escalonadas (lista de dicts con provider_id/valid_from) NO se
        # aplanan aquí: perderían su alcance por proveedor y fecha, que es justo su
        # razón de ser. Entran cuando el instrumento tenga identidad canónica.

    # Un símbolo con regla ESCALONADA nunca recibe además la plana (enmienda C026).
    # `AssetProfile` exige `price_range`, así que el perfil auxiliar de USD/MXN declara
    # uno — pero admitirlo aquí reintroduciría el aplanado por la puerta de atrás: el
    # evaluador da precedencia al scoped, y el día que alguien retire el mapa escalonado
    # el plano tomaría el relevo **en silencio**, validando barras de 1990 contra el
    # rango de hoy. Excluirlo hace que esa retirada falle cerrado en vez de degradar.
    con_escalon = scoped_symbols(assets_dir, ranges_path)

    for ruta in sorted(assets_dir.glob("*.yaml")):
        declarado = yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}
        simbolo, rango = declarado.get("symbol"), declarado.get("price_range")
        if not simbolo or not (isinstance(rango, (list, tuple)) and len(rango) == 2):
            continue
        if str(simbolo) in con_escalon or ruta.stem in con_escalon:
            continue
        bajo, alto = Decimal(str(rango[0])), Decimal(str(rango[1]))
        if bajo <= 0 or bajo >= alto:
            raise IngestGuardError(
                f"{ruta.name}: price_range declarado inválido {rango!r}; un rango que no "
                "ordena no puede gobernar una cuarentena"
            )
        rangos[str(simbolo)] = (bajo, alto)

    return rangos


def ruleset_from_spine(
    conn, assets_dir: Path = ASSETS_DIR, ranges_path: Path = QUALITY_RANGES
) -> QualityRuleSet:
    """`QualityRuleSet` con registry canónico y rangos ya traducidos a UUID."""
    rangos = canonical_price_ranges(conn, assets_dir, ranges_path)
    escalonados = canonical_scoped_ranges(conn, ranges_path)
    if not rangos and not escalonados:
        raise IngestGuardError(
            "ningún instrumento de la espina tiene rango declarado (plano ni escalonado): "
            "el guard rechazaría toda barra, que no es una cuarentena sino un apagón"
        )
    version = (yaml.safe_load(ranges_path.read_text(encoding="utf-8")) or {}).get(
        "version", "unversioned"
    )
    return QualityRuleSet(
        price_ranges=rangos,
        scoped_price_ranges=escalonados,
        identity_registry=registry_from_spine(conn),
        version=f"asset-ssot@{version}",
    )


def canonical_scoped_ranges(
    conn, ranges_path: Path = QUALITY_RANGES
) -> dict[str, tuple[ScopedPriceRange, ...]]:
    """Reglas escalonadas traducidas al `instrument_id` canónico vía el registro.

    Misma traducción que las planas (CXD-454): `canonical_symbol → instrument_id` sale
    de `reference.instrument`. Se resuelve también por `asset_id`, porque el YAML
    normativo indexa por ese nombre y la espina por el símbolo.
    """
    por_simbolo = declared_scoped_ranges_by_canonical_symbol(ranges_path)
    if not por_simbolo:
        return {}

    with conn.cursor() as cur:
        cur.execute(
            "SELECT canonical_symbol, asset_id, instrument_id::text FROM reference.instrument"
        )
        filas = cur.fetchall()

    uuid_por_nombre: dict[str, str] = {}
    for simbolo, asset_id, instrument_id in filas:
        uuid_por_nombre[simbolo] = instrument_id
        uuid_por_nombre[asset_id] = instrument_id

    return {
        uuid_por_nombre[nombre]: reglas
        for nombre, reglas in sorted(por_simbolo.items())
        if nombre in uuid_por_nombre
    }


def resolved_instrument_id(
    ruleset: QualityRuleSet, provider_id: str, provider_symbol: str
) -> str | None:
    """UUID canónico del alias, o `None` si el registry no lo resuelve.

    Se pregunta al **registry**, que es quien tiene la respuesta. Dos alternativas que
    parecen funcionar y son trampas:

    * leerlo de `decision.observed_value`: en un rechazo de rango ese campo es
      `{'open': '99999'}` — el valor que ofendió, no la identidad. El evento quedaría
      con FK nula justo cuando el alias **sí** se resolvió (medido, CXD-457).
    * extraerlo del `rule_id`, que literalmente contiene el UUID
      (`bar.range.72f6f7e9-...`): sería recuperar identidad parseando una cadena de
      diagnóstico, es decir una heurística sobre un campo que nadie prometió estable.
    """
    registro = ruleset.identity_registry
    if registro is None:
        return None
    try:
        return registro.resolve(provider_id, provider_symbol)
    except Exception:
        # Alias desconocido: el evento se escribe igual, con FK nula, porque el hecho
        # que registra es precisamente "esto llegó sin identidad canónica".
        return None


def record_quarantine(
    conn,
    *,
    provider_id: str,
    provider_symbol: str,
    row: Mapping[str, Any],
    decision: QualityDecision,
    rule_version: str,
    instrument_id: str | None = None,
    evidence_tier: str | None = None,
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
                instrument_id,
                decision.rule_id or "bar.unknown",
                rule_version,
                json.dumps(_jsonable(decision.observed_value)),
                json.dumps(
                    {
                        "bar": _jsonable(dict(row)),
                        # Con QUE clase de evidencia se juzgo esta barra. Sin esto, un
                        # rechazo por rango plano y uno por regla escalonada son
                        # indistinguibles en la tabla (CLD-459).
                        "range_evidence_tier": evidence_tier
                        or range_evidence_tier(provider_symbol),
                    }
                ),
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
    # Se resuelve UNA vez para todo el lote: el alias no cambia entre barras, y
    # re-preguntar por fila invitaria a resolverlo de otra forma en algun camino.
    instrumento = resolved_instrument_id(ruleset, provider_id, provider_symbol)

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
            instrument_id=instrumento,
            evidence_tier=range_evidence_tier(provider_symbol),
        )

    return ScreeningResult(accepted=aceptadas, quarantined=rechazadas)


#: Columnas OHLCV que viajan al publicador. `symbol`/`source` no van: la identidad la
#: aporta el par `(provider_id, provider_symbol)` que se pasa aparte.
BAR_FIELDS: tuple[str, ...] = ("time", "open", "high", "low", "close", "volume")


def range_evidence_tier(symbol: str, assets_dir: Path = ASSETS_DIR, ranges_path: Path = QUALITY_RANGES) -> str:
    """Qué clase de evidencia respalda el rango con el que se juzga este símbolo.

    Nace de una contradicción que encontré en mi propia entrega (CLD-459): BL-40 dice
    que *"un rango legacy no basta"* y que el gate no debe activarse sin identidad
    verificable, pero yo había cableado USD/COP contra su `price_range` plano. Hoy no
    rechaza nada porque el rango es ancho — y eso es suerte, no diseño.

    En vez de apagar el gate o de enmendar la ficha para que mi entrega encaje, la
    asimetría se hace **visible en el dato**: cada cuarentena declara con qué clase de
    evidencia se juzgó. Un rango plano y uno escalonado dejan de ser indistinguibles en
    la tabla, que era el problema real — no que el gate corriera, sino que corriera sin
    que nadie pudiera saber sobre qué base.

    * ``scoped``    — regla con proveedor y ventana verificables (hoy: usdmxn).
    * ``declared``  — `price_range` del `AssetProfile`: declarado, versionado y
                      revisable, pero sin proveedor ni corte temporal.
    * ``none``      — sin rango: el símbolo no se juzga.
    """
    if symbol in scoped_symbols(assets_dir, ranges_path):
        return "scoped"
    if symbol in declared_ranges_by_canonical_symbol(assets_dir, ranges_path):
        return "declared"
    return "none"


def declared_provider_for(symbol: str, assets_dir: Path = ASSETS_DIR) -> str | None:
    """Proveedor **declarado** para un símbolo canónico, o `None` si nadie lo declara.

    Existe por un defecto medido: las reglas escalonadas se declaran para el
    **proveedor** (`twelvedata`), mientras que los DAGs escribían bajo el nombre de su
    **job** (`twelvedata_multi`, `twelvedata_backfill`). El alias resuelve en los tres
    casos, así que la barra pasaba la identidad y moría después en `bar.range_scope`:
    el 100% de las barras USD/MXN habría acabado en cuarentena — un apagón disfrazado
    de control de calidad.

    La distinción ya estaba en el modelo que construimos: `authoritative_for` sólo la
    tienen los proveedores **declarados**; los jobs quedaron registrados como
    `observed_writer` sin autoridad. Publicar bajo el job contradecía esa separación.
    El job no se pierde: viaja en `source_uri` y en la columna `source` legada, que es
    donde vive el linaje.
    """
    for ruta in sorted(assets_dir.glob("*.yaml")):
        declarado = yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}
        if declarado.get("symbol") != symbol:
            continue
        fuente = declarado.get("data_source")
        if isinstance(fuente, Mapping) and fuente.get("provider"):
            return str(fuente["provider"])
    return None


def publish_or_declare_gap(conn, *, symbol: str, provider_id: str, rows, interval_id: str, source_uri: str):
    """Publica por la frontera Fabric, o devuelve `None` si el símbolo no tiene identidad.

    La **cobertura se mide**: un símbolo está cubierto si su alias `(provider_id, symbol)`
    resuelve en `reference.provider_symbol`. Una lista fija de símbolos cubiertos se
    desincronizaría del catálogo en silencio — un instrumento entraría en la espina y
    seguiría sin gate, o saldría y el gate lo bloquearía sin motivo.

    `None` para un símbolo sin identidad **no es un bypass disfrazado**, y el llamador
    está obligado a declararlo: filtrar sus barras las mandaría todas a cuarentena por
    `unknown_alias` y apagaría una ingesta viva; escribirlas calladamente haría creer
    que pasaron un gate que nunca corrió, y la ausencia de eventos de cuarentena se
    leería como "todo limpio". El aviso es lo único que separa un hueco declarado de un
    bypass.
    """
    from src.market.identity import IdentityError
    from src.market.publication import publish_provider_rows

    # Se publica bajo el proveedor DECLARADO, no bajo el nombre del job que llama. Ver
    # `declared_provider_for`: publicar como `twelvedata_multi` hacía que las reglas
    # escalonadas —declaradas para `twelvedata`— no casaran nunca.
    declarado = declared_provider_for(symbol)
    if declarado is None:
        return None  # símbolo sin activo declarado: fuera de cobertura

    try:
        registry_from_spine(conn).resolve(declarado, symbol)
    except IdentityError:
        return None

    return publish_provider_rows(
        conn,
        provider_id=declarado,
        provider_symbol=symbol,
        interval_id=interval_id,
        rows=rows,
        source_uri=f"{source_uri}?job={provider_id}",
    )


def rows_from_frame(df, columnas: Sequence[str]) -> list[dict[str, Any]]:
    """Convierte un DataFrame a filas planas para el evaluador, sin tocar valores."""
    return [
        {col: fila[col] for col in columnas if col in fila}
        for fila in df.to_dict(orient="records")
    ]
