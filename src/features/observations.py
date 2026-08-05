# -*- coding: utf-8 -*-
"""Construir `observations` para una policy desde su feature-set y el catálogo.

QUÉ CIERRA
----------
La brecha mayor que quedaba de BL-45: **nadie producía `observations::<policy_id>`
ni `decision_cutoff::<policy_id>`**. La cadena gobernada los esperaba por XCom y
ninguna tarea los ponía, así que el grafo mostraba cuatro eslabones que ninguna
corrida podía atravesar.

CÓMO, Y POR QUÉ ASÍ
-------------------
La lista de qué materializar **no se escribe aquí**: se lee del `feature_set_id` que
la policy declara (`ordered_features`), y cada feature se resuelve contra el catálogo
(`CTR-FEATURE-CATALOG-001`). Escribir la lista en este módulo crearía una tercera
fuente de verdad junto al feature-set y el catálogo — justo lo que dejó a `ma_200`
sin productor durante toda su vida.

Función **pura sobre un DataFrame**: recibe las barras ya cargadas, no las consulta.
El motivo es el mismo invariante 5 de `strategy-engines.md` que gobierna las
políticas ("nunca consulta los últimos datos"), y además hace el productor
verificable sin stack — la tarea de Airflow es un envoltorio fino que trae las
barras y llama aquí, igual que `make_validate_inputs` envuelve a
`validate_policy_inputs`.

`available_at` — LO QUE ES Y LO QUE NO
--------------------------------------
Se deriva de `close_time + reconstruction_lag` y **es una RECONSTRUCCIÓN, no un
vintage del proveedor**. No es un detalle: `src/strategies/spx500_regime_gated_v1/
load_real.py` ya lo declara para esta misma serie —"available_at reconstruido
(cierre + 1d), no vintage del proveedor; max status: research_validated"— y esa
limitación **viaja con el dato**, no se pierde al cambiar de capa. Un `available_at`
reconstruido sostiene el corte causal `available_at <= decision_cutoff`, pero **no**
sostiene una afirmación de point-in-time real; por eso cada observación lo declara
explícitamente en `provenance` en vez de dejar que el consumidor lo suponga.

Contract: CTR-FEATURE-CATALOG-001 × CTR-POLICY-BACKEND-001 · decisión C2, CXD-615
"""
from __future__ import annotations

import importlib.util
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO / "config" / "features" / "feature_catalog.yaml"
FEATURE_SET_DIR = REPO / "config" / "features" / "feature_sets"

#: Retraso de disponibilidad RECONSTRUIDO. Un día tras el cierre, que es lo que ya
#: declara el loader del índice oficial. Se nombra la constante en vez de escribir
#: `days=1` en medio del cálculo para que se pueda citar y auditar.
RECONSTRUCTION_LAG = timedelta(days=1)

#: Etiqueta que viaja en cada observación. Si algún día hubiera vintage real del
#: proveedor, este valor cambia y el cambio es visible aguas abajo.
PROVENANCE_RECONSTRUCTED = "available_at_reconstructed:close+P1D"

#: TECHO DE ESTATUS de todo lo que salga de aquí mientras el sello sea reconstruido
#: (co-firmado en CXD-620). No es una nota: es el límite duro. Un `available_at`
#: derivado del cierre demuestra **transporte real y causalidad declarada**, y nada
#: más — no es observación de vintage, así que no puede satisfacer ningún gate que
#: exija evidencia point-in-time productiva.
MAX_STATUS_RECONSTRUCTED = "research_validated"

#: Estatus que este productor NO puede sostener con un sello reconstruido. Se listan
#: para que la prohibición sea comprobable por código y no una frase en un docstring
#: — que es exactamente el error que cometí con el parámetro `window` (CXD-618):
#: escribir la garantía sin implementarla.
FORBIDDEN_STATUSES_RECONSTRUCTED = frozenset({"production", "promoted", "live"})


def status_ceiling(provenance: str) -> str:
    """Techo de estatus alcanzable por una observación con esa `provenance`.

    Existe como FUNCIÓN para que un consumidor pueda preguntarlo en vez de suponerlo,
    y para que la prohibición viaje con el dato igual que la etiqueta. Un sello
    desconocido no se degrada en silencio a "lo mejor posible": se rechaza.
    """
    if provenance == PROVENANCE_RECONSTRUCTED:
        return MAX_STATUS_RECONSTRUCTED
    raise ObservationError(
        f"provenance {provenance!r} sin techo declarado: antes de aceptar un sello "
        f"nuevo hay que decir qué puede y qué no puede sostener"
    )


class ObservationError(RuntimeError):
    """No se puede construir el snapshot declarado. Falla cerrado, nunca a medias."""


def _load_yaml(path: Path) -> Any:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _catalog_index() -> dict[tuple[str, str], dict]:
    doc = _load_yaml(CATALOG_PATH) or {}
    return {
        (str(e["asset_id"]), str(e["feature_id"])): e
        for e in (doc.get("features") or [])
    }


def _feature_set(feature_set_id: str) -> dict:
    for path in sorted(FEATURE_SET_DIR.glob("*.yaml")):
        doc = _load_yaml(path) or {}
        if str(doc.get("feature_set_id")) == feature_set_id:
            return doc
    raise ObservationError(
        f"feature_set_id {feature_set_id!r} no existe en {FEATURE_SET_DIR.name}/: "
        f"una policy que apunta a un set inexistente no tiene contrato de inputs"
    )


#: Contratos de INVOCACION soportados. Espejo de `PRODUCER_CONTRACTS` del validador
#: (`scripts/validation/validate_feature_catalog.py`): el catalogo declara cual usa
#: cada feature y aqui se bifurca por ESE valor. No se infiere por firma — renombrar
#: un argumento de un productor congelado no puede cambiar como se le invoca.
CONTRATO_SERIES = "series_close_v1"
CONTRATO_FRAME = "ohlcv_frame_v1"
CONTRATOS_SOPORTADOS = (CONTRATO_SERIES, CONTRATO_FRAME)

#: Columnas que el contrato de frame promete entregar al productor.
COLUMNAS_FRAME = ("time", "open", "high", "low", "close")


def _resolve_producer(entry: Mapping[str, Any]) -> Callable[[pd.Series], pd.Series] | None:
    """Cargar el productor DECLARADO en `code_reference`, o `None` si es passthrough.

    Se importa por la ruta que declara el catálogo, no por un import estático: si el
    catálogo apunta a otro fichero, se usa ESE. Un import fijo aquí volvería a
    convertir este módulo en una fuente de verdad paralela — el catálogo dejaría de
    mandar y volveríamos al problema que C1 arregló.
    """
    ref = entry.get("code_reference")
    if ref is None:
        return None  # passthrough: la feature ES la columna HOMONIMA del bar canónico
    file_rel, symbol = ref.get("file"), ref.get("symbol")
    if not file_rel or not symbol:
        raise ObservationError(
            f"{entry.get('series_id')}: `code_reference` sin file/symbol; no se puede "
            f"resolver el productor declarado"
        )
    path = REPO / file_rel
    if not path.is_file():
        raise ObservationError(f"{entry.get('series_id')}: no existe {file_rel}")
    spec = importlib.util.spec_from_file_location(f"_feat_{symbol}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, symbol, None)
    if not callable(fn):
        raise ObservationError(f"{file_rel} no expone `{symbol}` como callable")
    return fn


def build_observations(
    policy_spec: Mapping[str, Any],
    bars: pd.DataFrame,
    *,
    decision_cutoff: str,
    close_column: str = "close",
    time_column: str = "time",
) -> dict[str, dict[str, Any]]:
    """`{feature_id: {value, available_at, provenance}}` para la barra de decisión.

    `bars` debe traer la serie COMPLETA hasta la barra de decisión: las features con
    ventana (`ma_200` necesita 200 sesiones) no se pueden calcular sobre un recorte.
    Pasar sólo las últimas N barras daría un número plausible y silenciosamente
    distinto — el peor tipo de error, porque no falla.

    Falla CERRADO ante cualquier hueco: sin barras, sin la columna declarada, con
    `available_at` posterior al cutoff, o si el warm-up deja la feature en `NaN`. En
    particular **no se publica un `NaN`**: una feature requerida sin valor es una
    ausencia, y quien la juzga es el `missing_input_policy` declarado, no este módulo
    inventando un número.
    """
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        raise ObservationError("bars vacío: no se construye un snapshot sin evidencia")
    for col in (time_column, close_column):
        if col not in bars.columns:
            raise ObservationError(f"bars no trae la columna declarada {col!r}")

    asset_id = policy_spec.get("asset")
    if not isinstance(asset_id, str) or not asset_id:
        raise ObservationError(f"{policy_spec.get('id')}: el spec no declara `asset`")

    inputs = policy_spec.get("inputs") or {}
    fs = _feature_set(str(inputs.get("feature_set_id")))
    ordenadas = [o["feature_id"] for o in (fs.get("ordered_features") or [])]
    if not ordenadas:
        raise ObservationError(
            f"{fs.get('feature_set_id')}: sin `ordered_features`; un set vacío no "
            f"declara nada que materializar"
        )

    catalogo = _catalog_index()
    df = bars.sort_values(time_column).reset_index(drop=True)
    cutoff = pd.Timestamp(decision_cutoff)
    if cutoff.tzinfo is None:
        raise ObservationError(
            f"decision_cutoff {decision_cutoff!r} sin timezone: comparar naive contra "
            f"aware no tiene orden definido (data-governance.md)"
        )
    tiempos = pd.to_datetime(df[time_column], utc=True)
    close = df[close_column].astype(float)

    observations: dict[str, dict[str, Any]] = {}
    for feature_id in ordenadas:
        entrada = catalogo.get((asset_id, feature_id))
        if entrada is None:
            raise ObservationError(
                f"{asset_id}.{feature_id} está en el feature-set pero NO en el "
                f"catálogo: no hay productor ni contrato de causalidad declarados"
            )
        productor = _resolve_producer(entrada)
        if productor is None:
            # PASSTHROUGH: la columna que se llama COMO LA FEATURE, no `close`.
            #
            # Aquí ponía `serie = close` para todo passthrough. El catálogo declara
            # `open`, `high` y `low` como passthrough para usdcop, y los sets de
            # smart_simple los ordenan — así que este productor genérico habría
            # publicado el CIERRE bajo las identidades `open/high/low`, con el
            # `series_id` de cada una y sin fallar (CXD-620 §1). No era un riesgo
            # futuro: esas entradas existen hoy. Un valor plausible bajo la identidad
            # equivocada es indistinguible de un dato bueno aguas abajo.
            if feature_id not in df.columns:
                raise ObservationError(
                    f"{asset_id}.{feature_id} es passthrough (sin `code_reference`) "
                    f"pero `bars` no trae la columna {feature_id!r}. Columnas: "
                    f"{sorted(df.columns)}. No se sustituye por otra"
                )
            serie = df[feature_id].astype(float)
        else:
            contrato = entrada.get("producer_contract", CONTRATO_SERIES)
            if contrato not in CONTRATOS_SOPORTADOS:
                raise ObservationError(
                    f"{asset_id}.{feature_id}: producer_contract {contrato!r} no "
                    f"soportado (soportados: {CONTRATOS_SOPORTADOS}). Un contrato sin "
                    f"declarar NO se degrada al de por defecto"
                )
            if contrato == CONTRATO_SERIES:
                # `fn(close) -> Series` — el contrato historico (`compute_ma_200`).
                serie = productor(close)
            else:
                # `fn(df) -> df` — productor de FRAME. Es el caso de
                # `build_daily_features`, el codigo CONGELADO del track BTC que
                # calcula ~10 features de golpe: se le entrega el frame REAL completo
                # y se toma la columna que el catalogo declara. Apuntar al congelado
                # en vez de reescribir su formula es lo que impide que exista una
                # segunda definicion de una feature ya congelada (CXD-628, decision A).
                faltan = [c for c in COLUMNAS_FRAME if c not in df.columns]
                if faltan:
                    raise ObservationError(
                        f"{asset_id}.{feature_id}: el contrato {contrato!r} entrega "
                        f"{list(COLUMNAS_FRAME)} y `bars` no trae {faltan}"
                    )
                columna = entrada.get("output_column")
                if not isinstance(columna, str) or not columna:
                    raise ObservationError(
                        f"{asset_id}.{feature_id}: contrato de frame sin "
                        f"`output_column` declarada; no se adivina cual de las "
                        f"salidas es esta feature"
                    )
                salida = productor(df[list(COLUMNAS_FRAME)])
                if not isinstance(salida, pd.DataFrame):
                    raise ObservationError(
                        f"{asset_id}.{feature_id}: el productor de frame devolvio "
                        f"{type(salida).__name__}, no un DataFrame"
                    )
                if columna not in salida.columns:
                    raise ObservationError(
                        f"{asset_id}.{feature_id}: el productor no emitio la columna "
                        f"declarada {columna!r} (emitio: {sorted(salida.columns)[:12]})"
                    )
                if len(salida) != len(df):
                    raise ObservationError(
                        f"{asset_id}.{feature_id}: el productor devolvio {len(salida)} "
                        f"filas para {len(df)} barras; alinear por posicion series de "
                        f"distinta longitud desplazaria la barra de decision"
                    )
                serie = salida[columna].astype(float).reset_index(drop=True)

        # La barra de decisión es la ÚLTIMA cuyo available_at no excede el cutoff.
        disponibles = tiempos + RECONSTRUCTION_LAG
        elegibles = disponibles <= cutoff
        if not elegibles.any():
            raise ObservationError(
                f"{asset_id}.{feature_id}: ninguna barra está disponible en "
                f"{decision_cutoff} (available_at = cierre + {RECONSTRUCTION_LAG}); no "
                f"se decide con datos que aún no existían"
            )
        idx = int(elegibles.to_numpy().nonzero()[0][-1])
        valor = serie.iloc[idx]
        if pd.isna(valor):
            raise ObservationError(
                f"{asset_id}.{feature_id}: valor NaN en la barra de decisión "
                f"({tiempos.iloc[idx]}). Suele ser warm-up insuficiente — "
                f"`{entrada.get('lookback')}` de historia. NO se publica un NaN: una "
                f"feature requerida sin valor es una AUSENCIA, y la resuelve el "
                f"`missing_input_policy` declarado, no este módulo"
            )
        observations[feature_id] = {
            "value": float(valor),
            "available_at": disponibles.iloc[idx].isoformat(),
            "provenance": PROVENANCE_RECONSTRUCTED,
        }
    return observations


#: Orden de exigencia de los estatus. Hace falta un ORDEN, no un conjunto, para poder
#: hablar del techo MINIMO cuando conviven sellos distintos: sin orden, "el más bajo"
#: no significa nada y la mezcla acabaría heredando el mejor.
STATUS_RANK = {"research_validated": 1, "production": 2, "promoted": 2, "live": 2}

#: Qué estatus RECLAMA cada `migration.status`. `CUTOVER` es el único que significa
#: "esta ES la vía viva" —el legacy ya está apagado—, así que reclama `production`.
#: `PARITY_GREEN` significa "reproduce al legacy", que es investigación validada.
STATUS_CLAIMED_BY_MIGRATION = {
    "CUTOVER": "production",
    "PARITY_GREEN": "research_validated",
}


def assert_observations_support_status(
    observations: Mapping[str, Mapping[str, Any]], *, migration_status: str
) -> None:
    """GATE fail-closed: la evidencia debe sostener lo que el estado reclama.

    POR QUÉ EXISTE ESTA FUNCIÓN Y NO BASTA `status_ceiling`. C2b declaró el techo en
    código —constantes y una función consultable— y **nadie la consultaba** (CXD-620
    §2). Una función que puede preguntarse pero no se pregunta no prohíbe nada; es la
    misma forma exacta del error que ya cometí con el parámetro `window`: escribir el
    mecanismo y no cablearlo. Aquí el techo se **aplica** en la frontera por la que
    una señal escapa.

    La regla: si el `migration.status` de la policy reclama `production` —hoy sólo
    `CUTOVER`— y alguna observación viene con sello RECONSTRUIDO, se bloquea. Un
    `available_at` derivado del cierre demuestra transporte y causalidad declarada;
    no demuestra que el dato estuviera observado en ese instante, y una vía viva no
    puede apoyarse en eso.

    Fail-closed también ante lo desconocido: un `migration.status` sin entrada en
    `STATUS_CLAIMED_BY_MIGRATION` no se degrada al caso benigno — se rechaza. Ese
    "por defecto lo permisivo" es como se cuelan los estados nuevos sin revisar.
    """
    if not observations:
        raise ObservationError("sin observaciones no hay evidencia que juzgar")
    if migration_status not in STATUS_CLAIMED_BY_MIGRATION:
        raise ObservationError(
            f"migration.status {migration_status!r} sin reclamo declarado en "
            f"STATUS_CLAIMED_BY_MIGRATION: antes de dejar pasar un estado nuevo hay "
            f"que decir qué nivel de evidencia reclama"
        )
    reclamado = STATUS_CLAIMED_BY_MIGRATION[migration_status]
    if reclamado not in STATUS_RANK:
        raise ObservationError(
            f"el estatus reclamado {reclamado!r} no tiene rango declarado; sin orden "
            f"no se puede decidir si la evidencia lo sostiene"
        )

    # Se pregunta a la AUTORIDAD DEL TECHO por CADA sello, en vez de comparar contra
    # la constante reconstruida. La version anterior hacia
    # `if o["provenance"] == PROVENANCE_RECONSTRUCTED`, asi que **cualquier sello
    # inventado atravesaba CUTOVER**: bastaba escribir `provenance:
    # "vintage_proveedor"` y la señal salia (CXD-622, reproducido: devolvia None).
    #
    # Lo hiriente es que `status_ceiling` YA era fail-closed ante un sello
    # desconocido — escrita en C2b— y este gate, que nacio para APLICAR el techo,
    # **no la llamaba**. Tercera vez en esta serie que escribo el mecanismo y no lo
    # consulto (`window`, `status_ceiling`, y ahora el gate del propio
    # `status_ceiling`). Por eso ahora la autoridad es la funcion y no una
    # comparacion local: una segunda forma de decidir lo mismo es una segunda forma
    # de equivocarse.
    techos = {fid: status_ceiling(o.get("provenance")) for fid, o in observations.items()}
    desconocidos = [t for t in techos.values() if t not in STATUS_RANK]
    if desconocidos:
        raise ObservationError(f"techos sin rango declarado: {sorted(set(desconocidos))}")

    # MINIMO, no maximo: con sellos mezclados manda el peor. Heredar el mejor seria
    # dejar que una feature bien sellada legitime a las demas.
    fid_minimo = min(techos, key=lambda k: STATUS_RANK[techos[k]])
    techo_min = techos[fid_minimo]
    if STATUS_RANK[reclamado] > STATUS_RANK[techo_min]:
        raise ObservationError(
            f"migration.status={migration_status} reclama {reclamado!r}, pero el techo "
            f"MINIMO de la evidencia es {techo_min!r} (lo impone "
            f"{fid_minimo!r}, provenance={observations[fid_minimo].get('provenance')!r}). "
            f"Techos por feature: {techos}. Hasta que exista `available_at` OBSERVADO, "
            f"esta evidencia no sostiene una via viva"
        )
