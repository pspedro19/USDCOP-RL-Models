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


def _resolve_producer(entry: Mapping[str, Any]) -> Callable[[pd.Series], pd.Series] | None:
    """Cargar el productor DECLARADO en `code_reference`, o `None` si es passthrough.

    Se importa por la ruta que declara el catálogo, no por un import estático: si el
    catálogo apunta a otro fichero, se usa ESE. Un import fijo aquí volvería a
    convertir este módulo en una fuente de verdad paralela — el catálogo dejaría de
    mandar y volveríamos al problema que C1 arregló.
    """
    ref = entry.get("code_reference")
    if ref is None:
        return None  # passthrough: la feature ES una columna del bar canónico
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
        serie = close if productor is None else productor(close)

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
