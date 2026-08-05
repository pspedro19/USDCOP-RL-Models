"""BL-20 — CANDADO DE COMPLETITUD por activo (CXD-569).

POR QUÉ EXISTE. El 2026-08-05 se añadió cobertura de interpretabilidad para Gold y BTC
(`7ac243cd`, 12 artefactos). CODEX aceptó el incremento y encontró el hueco que yo no vi:
**no había ningún juez de esa obligación**. `test_interpretability_schema.py` hace
`ARTIFACTS = glob(...)` y parametriza *sólo lo que exista* —valida bien lo presente, pero no
exige que algo esté presente—, y `test_interpretability_artifacts.py` genera únicamente
COP/ridge y SPX. Consecuencia medida por él: **borrar los 12 artefactos nuevos dejaba la suite
verde**. Un criterio que se cumple porque no hay nada que comprobar es la forma más cara de
falso verde, y es exactamente la que este repo lleva persiguiendo todo el día.

QUÉ FIJA ESTE FICHERO, y qué NO:
  - la MATRIZ ESPERADA se DERIVA de las configs por activo (`ASSET_CONFIGS` +
    `_models_for_asset`), nunca de una lista escrita a mano: si mañana Gold declara un
    booster más, el candado lo exige solo. Una lista fija aquí sería otra fuente de verdad
    que se desincroniza en silencio;
  - la ausencia de cualquier `(asset, model_id)` esperado es ROJO;
  - `_models_for_asset` rechaza alias no declarados y excluye los `hybrid_*` de la ruta de
    árbol (TreeSHAP no es correcto sobre un modelo mitad lineal mitad árbol);
  - el defecto COP no se toca: `usdcop` sin `--asset` resuelve exactamente las constantes de
    siempre.

NO comprueba la CALIDAD de los artefactos (aditividad, folds, no-degeneración): de eso ya se
encargan `test_interpretability_artifacts.py` y el schema. Aquí sólo se juzga **presencia y
vocabulario**, que es justo lo que no tenía juez.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analysis.generate_interpretability import (  # noqa: E402
    ASSET_CONFIGS,
    ZOO_LINEAR_MODELS,
    ZOO_TREE_MODELS,
    _models_for_asset,
)

ARTIFACT_ROOT = REPO / "data" / "interpretability"

#: Activos que DEBEN tener cobertura publicada. `usdcop` incluido a propósito: si alguien
#: rompiera la ruta por defecto al tocar la parametrización, este candado también lo caza.
COVERED_ASSETS = ("usdcop", "xauusd", "btcusdt")


def _expected_matrix() -> list[tuple[str, str]]:
    """(asset, model_id) esperados, DERIVADOS de las configs — nunca escritos a mano."""
    return [
        (asset, mid)
        for asset in COVERED_ASSETS
        for kind in ("linear", "tree", "hybrid")
        for mid in _models_for_asset(asset, kind)
    ]


def _published(asset: str) -> set[str]:
    base = ARTIFACT_ROOT / "zoo" / asset
    if not base.is_dir():
        return set()
    return {p.name for p in base.iterdir() if p.is_dir()
            and any(p.glob("*/summary.json"))}


@pytest.mark.parametrize("asset,model_id", _expected_matrix(),
                         ids=lambda v: str(v))
def test_every_declared_model_has_a_published_artifact(asset: str, model_id: str) -> None:
    """Cada modelo DECLARADO por el activo tiene artefacto publicado.

    Rojo con: borrar `data/interpretability/zoo/xauusd/xgboost_pure/` (o cualquier otro).
    Antes de este fichero esa misma acción dejaba la suite entera VERDE (CXD-569).
    """
    publicados = _published(asset)
    assert model_id in publicados, (
        f"{asset}: el activo DECLARA el modelo {model_id!r} en su config pero no hay "
        f"artefacto de interpretabilidad publicado en data/interpretability/zoo/{asset}/. "
        f"Publicados: {sorted(publicados) or '(ninguno)'}. "
        f"Regenerar con: python scripts/analysis/generate_interpretability.py "
        f"--asset {asset} --skip-rules"
    )


def test_the_expected_matrix_is_not_empty() -> None:
    """Guarda anti-vacuidad del propio candado.

    Si `_models_for_asset` devolviera vacío para todo (config movida, clave `models`
    renombrada), la parametrización de arriba se quedaría SIN CASOS y pasaría verde sin
    comprobar nada. Este test es el que impide que el juez desaparezca en silencio.
    """
    matriz = _expected_matrix()
    assert len(matriz) >= 27, (
        f"la matriz esperada colapsó a {len(matriz)} entradas: con 3 activos x "
        f"(3 lineales + 3 arboles + 3 hibridos) deben ser >= 27. Un colector vacio "
        f"fichero verde sin juzgar nada.")


def test_usdcop_default_vocabulary_is_untouched() -> None:
    """La parametrización por activo NO puede haber movido el defecto COP."""
    assert _models_for_asset("usdcop", "linear") == ZOO_LINEAR_MODELS
    assert _models_for_asset("usdcop", "tree") == ZOO_TREE_MODELS


@pytest.mark.parametrize("asset", ("xauusd", "btcusdt"))
def test_non_cop_assets_use_their_own_declared_vocabulary(asset: str) -> None:
    """Gold/BTC traen su propio vocabulario: arboles `_pure`, y CERO hibridos.

    Los `hybrid_*` mezclan lineal y arbol: TreeSHAP no es correcto sobre ellos y publicar
    esa atribucion seria peor que no publicarla (decision declarada en la ficha).
    """
    trees = _models_for_asset(asset, "tree")
    assert trees, f"{asset}: no resolvió ningún árbol declarado"
    assert all(m.endswith("_pure") for m in trees), (
        f"{asset}: la ruta de árbol debe resolver SOLO boosters puros, resolvió {trees}")
    assert not any("hybrid" in m for m in trees), (
        f"{asset}: un `hybrid_*` entró en la ruta TreeSHAP — atribución incorrecta por "
        f"construcción: {trees}")


def test_undeclared_asset_is_rejected_not_silently_defaulted() -> None:
    """Un activo desconocido es error DURO, jamás un fallback a COP.

    Degradar en silencio a `usdcop` produciría artefactos que dicen `asset: <lo que sea>`
    computados con los datos de COP — una mentira con formato válido.
    """
    with pytest.raises((ValueError, KeyError)):
        _models_for_asset("noexiste", "linear")


def test_every_covered_asset_declares_a_config_path() -> None:
    """Los activos cubiertos están en `ASSET_CONFIGS`; si no, `--asset` los rechazaría."""
    faltan = [a for a in COVERED_ASSETS if a not in ASSET_CONFIGS]
    assert not faltan, f"activos cubiertos sin entrada en ASSET_CONFIGS: {faltan}"
