# -*- coding: utf-8 -*-
"""BL-39 no puede volver a reclamar como pendiente lo que ya está entregado.

Su sección «Qué falta exactamente» pedía tres cosas y **dos ya existían**: el catálogo
estable con sus seis campos (34/34) y los `feature_set` por estrategia. Una ficha que
reclama trabajo hecho hace perder el tiempo al siguiente que la lea y, peor, esconde cuál
es la brecha real — aquí, que el snapshot de normalización está poblado en 1 de 6.

Este fichero fija **el hecho**, no sólo la redacción: además de exigir que la ficha nombre
la brecha correcta, comprueba contra el repo que lo que declara entregado siga entregado.
Un candado que sólo mirase el texto sería una profecía autocumplida — la ficha pasaría por
decir las palabras, aunque el catálogo desapareciera.

Escrito al estilo de `test_bl19_backlog_anchors.py` / `test_bl18_backlog_anchors.py`.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BL39 = ROOT / ".claude/specs/planes/backlog/BL-39-feature-contracts-normalizacion.md"

CATALOGO = ROOT / "config/features/feature_catalog.yaml"
FEATURE_SETS = ROOT / "config/features/feature_sets"
SNAPSHOTS = ROOT / "config/features/normalization_snapshots"

#: Los seis campos que la ficha daba por pendientes y que hoy están completos.
CAMPOS_EXIGIDOS = (
    "feature_id",
    "causality_policy",
    "source_contract",
    "transformation",
    "lookback",
    "code_reference",
)


def _documento() -> tuple[dict[str, object], str]:
    texto = BL39.read_text(encoding="utf-8")
    _, frontmatter, cuerpo = texto.split("---", 2)
    return yaml.safe_load(frontmatter), cuerpo


def _catalogo() -> list[dict]:
    doc = yaml.safe_load(CATALOGO.read_text(encoding="utf-8"))
    features = doc.get("features", doc)
    return features if isinstance(features, list) else list(features.values())


def test_bl39_stays_partial_and_keeps_its_anchors() -> None:
    """El estado no se promueve por reescribir la redacción."""
    frontmatter, _ = _documento()
    assert frontmatter["status"] == "PARTIAL", (
        "corregir la narrativa NO es entregar: la brecha (snapshot 1/6, consumo no "
        "probado, matriz sin fixture) sigue abierta"
    )
    anclas = frontmatter.get("code_anchors") or []
    assert anclas, "BL-39 se quedó sin `code_anchors`"
    for ancla in anclas:
        assert (ROOT / str(ancla)).exists(), f"ancla rota en BL-39: {ancla}"


def test_bl39_no_longer_claims_the_delivered_work_is_pending() -> None:
    """La frase obsoleta no vuelve.

    Rojo con: restaurar la redacción anterior, que pedía "Catálogo estable (...) en Git"
    como si no existiera.
    """
    _, cuerpo = _documento()
    assert "Catálogo estable (feature_id, causality_policy" not in cuerpo, (
        "BL-39 volvió a pedir el catálogo como pendiente. Está completo: 34/34 features "
        "con los seis campos, y `validate_feature_catalog.py` da 0 violations"
    )


def test_bl39_names_the_real_remaining_gap() -> None:
    """La brecha vigente está escrita, no sólo eliminada la vieja.

    Se exige que aparezca la cobertura del snapshot y la matriz como fixture. Sin esto,
    alguien podría borrar la sección entera y el test anterior seguiría verde.
    """
    _, cuerpo = _documento()
    for marca in ("normalización", "1 de 6", "fixture de CI"):
        assert marca in cuerpo, f"BL-39 ya no nombra la brecha real: falta {marca!r}"


def test_what_the_ficha_declares_delivered_is_actually_delivered() -> None:
    """EL candado contra la profecía autocumplida: se mide el repo, no el texto.

    Si el catálogo perdiera un campo o un feature_set desapareciera, la ficha quedaría
    afirmando una entrega falsa — y sin esto, pasaría igual por seguir diciendo las
    palabras correctas.
    """
    catalogo = _catalogo()
    assert len(catalogo) == 34, (
        f"el catálogo tiene {len(catalogo)} features y la ficha declara 34. Es `==` y no "
        f"`>=` a propósito (CXD-737): con `>=`, añadir una feature dejaría la cifra de la "
        f"ficha obsoleta sin que nada lo notase — el mismo defecto que este fichero nació "
        f"para cerrar. Si el catálogo crece, actualizar BL-39 en el mismo commit"
    )
    for campo in CAMPOS_EXIGIDOS:
        faltan = [f.get("feature_id", "?") for f in catalogo if campo not in f]
        assert not faltan, (
            f"{len(faltan)} features del catálogo no declaran {campo!r} (p.ej. "
            f"{faltan[:3]}), pero BL-39 lo da por entregado al 100%"
        )

    sets = sorted(FEATURE_SETS.glob("*.yaml"))
    assert len(sets) == 6, (
        f"BL-39 declara 6 feature_sets entregados y hay {len(sets)}. `==` por la misma "
        f"razón que arriba: la ficha afirma una cardinalidad y debe moverse con ella"
    )
    for ruta in sets:
        doc = yaml.safe_load(ruta.read_text(encoding="utf-8"))
        ordenadas = doc.get("ordered_features") or []
        assert ordenadas, f"{ruta.name} no declara `ordered_features`"
        for entrada in ordenadas:
            assert {"feature_id", "order", "required"} <= set(entrada), (
                f"{ruta.name} tiene una entrada sin feature_id/order/required: {entrada}"
            )


def test_the_normalization_gap_is_exactly_one_of_six() -> None:
    """La cifra de la ficha se mide, no se hereda.

    Si mañana se pobla otro snapshot, este test cae y obliga a actualizar la ficha —que es
    lo que NO pasó con la redacción anterior, y por eso pedía trabajo ya hecho—. También
    cae si alguien vacía el único poblado.

    Se comprueba además que el id poblado tenga su artefacto: la frase honesta es
    "cobertura 1/6", **no** "no hay artefacto detrás del id" (corrección de Codex,
    CXD-AUX-734).
    """
    poblados = {}
    for ruta in sorted(FEATURE_SETS.glob("*.yaml")):
        doc = yaml.safe_load(ruta.read_text(encoding="utf-8"))
        snapshot = doc.get("normalization_snapshot_id")
        if snapshot:
            poblados[ruta.stem] = snapshot

    assert len(poblados) == 1, (
        f"la ficha declara cobertura 1/6 del snapshot de normalización y hay "
        f"{len(poblados)}: {poblados}. Actualizar BL-39 en vez de dejar la cifra vieja"
    )
    (_, snapshot_id), = poblados.items()
    artefacto = SNAPSHOTS / f"{snapshot_id}.yaml"
    assert artefacto.is_file(), (
        f"el único snapshot poblado ({snapshot_id}) no tiene artefacto en {SNAPSHOTS}; "
        f"entonces la brecha NO es sólo de cobertura y la ficha debe decirlo"
    )
