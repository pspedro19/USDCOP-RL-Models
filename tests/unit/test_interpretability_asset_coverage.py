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


def test_v11_composite_component_has_a_published_artifact() -> None:
    """La superficie `composite` de v11 tiene artefacto, y NO se confunde con el zoo.

    Rojo con: borrar `data/interpretability/composite/usdcop/usdcop_ridge_br/`.
    Sin este test la superficie que cierra BL-20 quedaba sin juez, igual que Gold/BTC
    antes de CXD-569.
    """
    base = ARTIFACT_ROOT / "composite" / "usdcop" / "usdcop_ridge_br"
    assert base.is_dir() and any(base.glob("*/summary.json")), (
        "falta el artefacto del componente `usdcop_ridge_br` (role=decision_input) de "
        "smart_simple_v11. Regenerar con: python -c \"from scripts.analysis."
        "generate_interpretability import generate_composite_v11; generate_composite_v11()\"")


def test_v11_composite_declares_the_recipe25_and_denies_explaining_the_decision() -> None:
    """Las DOS afirmaciones que hacen honesto ese artefacto, exigidas por contrato.

    (1) Que explica la RECETA de 25 y no el snapshot de 23 que persiste el DAG — la
        divergencia esta `declared_not_resolved` en el manifiesto y omitirla haria el
        artefacto ambiguo justo donde importa.
    (2) Que NO explica lo que la estrategia OPERA: gate de regimen, sizing y TP/HS son
        REGLAS y no se atribuyen con SHAP. Sin esta frase, "interpretabilidad de v11" se
        lee como si explicara el PnL.

    Rojo con: quitar cualquiera de las dos del `scope`, o publicar con != 25 features.
    """
    import json
    art = sorted((ARTIFACT_ROOT / "composite" / "usdcop" / "usdcop_ridge_br")
                 .glob("*/summary.json"))
    assert art, "sin artefacto composite que juzgar"
    d = json.loads(art[-1].read_text(encoding="utf-8"))
    assert d["n_features"] == 25, (
        f"el componente de v11 se atribuye sobre la RECETA de 25 features; "
        f"el artefacto declara {d['n_features']}")
    scope = d.get("scope", "")
    assert "recipe25" in scope and "dag_legacy23" in scope, (
        "el scope debe nombrar AMBOS feature sets: explicar uno y callar el otro deja "
        "ambiguo que se esta atribuyendo")
    assert "NO explica" in scope and "REGLAS" in scope, (
        "el scope debe negar explicitamente que esto explique la decision operada "
        "(gate/sizing/TP-HS son reglas, no el modelo)")


def test_every_covered_asset_declares_a_config_path() -> None:
    """Los activos cubiertos están en `ASSET_CONFIGS`; si no, `--asset` los rechazaría."""
    faltan = [a for a in COVERED_ASSETS if a not in ASSET_CONFIGS]
    assert not faltan, f"activos cubiertos sin entrada en ASSET_CONFIGS: {faltan}"
