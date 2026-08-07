# -*- coding: utf-8 -*-
"""Un bundle no puede declarar en el manifest métricas que sus propios ficheros contradicen.

EL DEFECTO, medido el 2026-08-06
--------------------------------
`smart_simple_v11` publicaba TRES cifras para el mismo año y la misma versión 2.0.0:

    manifest.backtests[].headline.return_pct        =  +7.35   (correcta, post-arreglo)
    backtests/2.0.0/summary_2025.json               = +25.63   (abril, PRE-arreglo)
    backtests/2.0.0/trades_2025.json  compone       = +25.63   (abril, PRE-arreglo)
    data/approvals/approval_state.json              =  +7.35   (surface del Vote 2)

La causa: dos commits del 2026-07-21 cerraron defectos reales —`cf392508` una fuga de purga
(look-ahead), `03eaa994` los fills de hard stop rellenados a precios por los que el mercado
hizo gap— y bajaron 2025 de +25.63% a +7.35%. Regeneraron `public/data/production/trades/`
y el `headline` del manifest, pero **no** los ficheros del bundle inmutable, que quedaron
fosilizados en abril.

Nadie lo detectó durante meses porque **cada fichero es internamente coherente**: el summary
cuadra con sus trades, el manifest cuadra con el approval. Sólo se ve comparando entre
ellos. El titular de CLAUDE.md sobrevivió a su propia refutación por esta grieta, y la
cartera multi-activo consumía el número pre-fuga porque leía el bundle.

QUÉ FIJA ESTE FICHERO
---------------------
Que `manifest.backtests[].headline` y el `summary` al que ese mismo registro apunta no
puedan divergir en silencio. Es el gate que faltaba: no valida que un número sea correcto
—eso no lo puede saber un test— sino que **dos declaraciones del mismo hecho coincidan**.

POR QUÉ NO SE ARREGLA SOBRESCRIBIENDO EL BUNDLE
-----------------------------------------------
`registry-lifecycle.md` §5: los `backtests[]` son inmutables y NUNCA se sobrescriben.
2.0.0/2025 es el registro histórico de la corrida pre-arreglo y debe seguir existiendo. La
salida correcta es una versión nueva, no una mentira reescrita. Mientras esa versión no se
publique, este test señala la divergencia en vez de dejarla callada.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ESTRATEGIAS = REPO / "usdcop-trading-dashboard" / "public" / "data" / "strategies"

# Divergencias YA MEDIDAS el 2026-08-06. No son perdones: son deuda declarada con dueño.
#
# AVISO SOBRE ESTA LISTA: un gate con 19 excepciones es un gate débil, y quien lo lea debe
# saberlo. Se listan TODAS —en vez de subir la tolerancia hasta que dejen de verse— porque
# subir el umbral las volvería invisibles sin resolverlas, que es exactamente el defecto
# que este fichero existe para impedir. La lista sólo puede encogerse.
#
# DOS CAUSAS DISTINTAS, y conviene no confundirlas:
#
# (A) GRANDES y de mecánica — el arreglo de una fuga quedó a medias. Las dos entradas de
#     COP son IMÁGENES ESPECULARES: `cf392508`+`03eaa994` regeneraron los artefactos de
#     producción (a los que apunta 1.0.0) y el headline del manifest de 2.0.0, pero no el
#     headline de 1.0.0 ni los ficheros del bundle de 2.0.0. Cada mitad quedó actualizada
#     en el sitio contrario. `gold_long_only_b1` es grande y su causa NO está establecida.
#
# (B) PEQUEÑAS y de calendario — BTC/Gold publican summaries de historia completa que los
#     DAGs semanales regeneran, mientras el `headline` del manifest es una foto del día de
#     publicación. Derivan unos pocos puntos por semana sin que nadie mienta. Sigue siendo
#     una violación del contrato: dos declaraciones del mismo hecho que no coinciden.
DIVERGENCIAS_DECLARADAS: dict[tuple[str, str, int], str] = {
    # (A) mecánica
    ("smart_simple_v11", "2.0.0", 2025): (
        "manifest +7.35% (post-arreglo) vs bundle +25.63% (abril, pre-fuga), -18.28 pp. "
        "cf392508 cerró una fuga de purga y 03eaa994 hizo open-aware los fills de hard "
        "stop; ninguno regeneró los ficheros del bundle. Inmutable por "
        "registry-lifecycle.md §5: se salda publicando una versión nueva, NO sobrescribiendo."
    ),
    ("smart_simple_v11", "1.0.0", 2025): (
        "espejo del anterior: manifest +25.63% (pre-fuga) vs production summary +7.35% "
        "(post-arreglo), +18.28 pp. Aquí el fichero se actualizó y el headline no."
    ),
    ("gold_long_only_b1", "1.0.0", 2026): (
        "+24.80 pp, la mayor de Gold. Causa NO establecida — no encaja en la deriva de "
        "calendario del resto, que se queda en 2-5 pp. Pendiente de diagnóstico."
    ),
    # (B) calendario: summary de historia completa regenerado vs headline congelado
    ("btc_hodl_b1", "1.0.0", 2026): "-4.70 pp",
    ("btc_hodl_b1", "1.2.1", 2026): "-3.07 pp",
    ("gold_dxy_tilt", "1.0.0", 2026): "+2.51 pp",
    ("gold_dxy_tilt", "1.3.0", 2026): "+2.18 pp",
    ("gold_dxy_tilt_s05", "1.0.0", 2026): "+2.55 pp",
    ("gold_dxy_tilt_s05", "1.3.0", 2026): "+2.13 pp",
    ("gold_dxy_tilt_s07", "1.0.0", 2026): "+2.47 pp",
    ("gold_dxy_tilt_s07", "1.3.0", 2026): "+2.23 pp",
    ("gold_long_only_b1", "1.3.0", 2026): "+2.72 pp",
    ("gold_regime_gated_v1", "1.0.0", 2026): "+2.53 pp",
    ("gold_regime_gated_v1", "1.3.0", 2026): "-3.60 pp",
    ("gold_trend_b2", "1.0.0", 2026): "+4.90 pp",
    ("gold_trend_b2", "1.3.0", 2026): "-0.91 pp",
    ("gold_trend_ens", "1.0.0", 2026): "-2.05 pp",
    ("gold_trend_ens", "1.3.0", 2026): "-2.05 pp",
    ("smart_simple_v11", "1.1.0", 2026): "-0.67 pp",
}

TOLERANCIA_PP = 0.05  # puntos porcentuales: absorbe redondeo de publicación, no un cambio real


def _bundles():
    """(strategy_id, version, year, headline_declarado, ruta_summary) de cada backtest."""
    if not ESTRATEGIAS.is_dir():
        return
    for d in sorted(ESTRATEGIAS.iterdir()):
        manifiesto = d / "manifest.json"
        if not manifiesto.is_file():
            continue
        try:
            m = json.loads(manifiesto.read_text(encoding="utf-8"))
        except Exception:
            continue
        for b in m.get("backtests", []):
            headline = b.get("headline") or {}
            # DOS nombres para el mismo hecho: BTC/Gold/COP usan `return_pct` (62 backtests)
            # y SPX usa `total_return_pct` (8). Mirar sólo el primero saltaba las 8 de SPX
            # EN SILENCIO — mi propio gate tenía el verde por vacuidad que existe para
            # cazar. Se descubrió persiguiendo un supuesto NaN de SPX que no era NaN: era
            # esta divergencia de esquema leída por un consumidor que pedía `return_pct`.
            declarado = headline.get("return_pct")
            if declarado is None:
                declarado = headline.get("total_return_pct")
            if declarado is None:
                continue
            ruta = b.get("summary")
            if not isinstance(ruta, str):
                continue
            # Las rutas del manifest cuelgan de `public/data/`.
            fichero = ESTRATEGIAS.parent / ruta
            yield d.name, str(b.get("model_version")), b.get("year"), float(declarado), fichero


def _retorno_del_summary(fichero: Path, strategy_id: str) -> float | None:
    try:
        s = json.loads(fichero.read_text(encoding="utf-8"))
    except Exception:
        return None
    estrategias = s.get("strategies")
    if isinstance(estrategias, dict):
        propia = estrategias.get(strategy_id)
        if isinstance(propia, dict) and propia.get("total_return_pct") is not None:
            return float(propia["total_return_pct"])
    if s.get("total_return_pct") is not None:
        return float(s["total_return_pct"])
    return None


@pytest.mark.skipif(not ESTRATEGIAS.is_dir(), reason="bundles no presentes en este checkout")
def test_no_bundle_declares_a_return_its_own_summary_contradicts() -> None:
    """El `headline` del manifest y el summary al que apunta deben coincidir."""
    divergen = []
    for sid, version, year, declarado, fichero in _bundles():
        if not fichero.is_file():
            continue
        real = _retorno_del_summary(fichero, sid)
        if real is None:
            continue
        if abs(real - declarado) > TOLERANCIA_PP:
            clave = (sid, version, year)
            if clave in DIVERGENCIAS_DECLARADAS:
                continue
            divergen.append(
                f"{sid} v{version} {year}: manifest declara {declarado:+.2f}% pero "
                f"{fichero.name} dice {real:+.2f}%"
            )
    assert not divergen, (
        "hay bundles cuyo manifest contradice sus propios ficheros. Cada fichero es "
        "internamente coherente, así que esto sólo se ve comparándolos — que es como el "
        "titular de COP sobrevivió meses a su refutación:\n  " + "\n  ".join(divergen)
    )


@pytest.mark.skipif(not ESTRATEGIAS.is_dir(), reason="bundles no presentes en este checkout")
def test_the_declared_divergences_still_exist() -> None:
    """Cada perdón de la lista debe seguir describiendo una divergencia REAL.

    Sin esto, la lista se convierte en un cementerio: entradas que ya no corresponden a nada
    y que seguirían silenciando una divergencia futura del mismo bundle. Un perdón que ya no
    hace falta es un agujero abierto, no un residuo inofensivo.
    """
    vistos = {
        (sid, version, year): (declarado, fichero)
        for sid, version, year, declarado, fichero in _bundles()
    }
    obsoletas = []
    for clave in DIVERGENCIAS_DECLARADAS:
        if clave not in vistos:
            obsoletas.append(f"{clave}: ya no existe ese backtest en el manifest")
            continue
        declarado, fichero = vistos[clave]
        if not fichero.is_file():
            obsoletas.append(f"{clave}: el summary declarado no existe en disco")
            continue
        real = _retorno_del_summary(fichero, clave[0])
        if real is None or abs(real - declarado) <= TOLERANCIA_PP:
            obsoletas.append(
                f"{clave}: ya NO diverge (manifest {declarado:+.2f}% vs summary "
                f"{real if real is None else f'{real:+.2f}'}%) — retirar la entrada"
            )
    assert not obsoletas, (
        "divergencias declaradas que ya no corresponden a la realidad; retirarlas o el "
        "perdón silenciará una divergencia nueva:\n  " + "\n  ".join(obsoletas)
    )


@pytest.mark.skipif(not ESTRATEGIAS.is_dir(), reason="bundles no presentes en este checkout")
def test_the_headline_return_key_divergence_between_assets_is_pinned() -> None:
    """SPX nombra el retorno `total_return_pct`; el resto, `return_pct`. Queda fijado.

    NO es un NaN, aunque así se reportó primero. Los headlines de SPX traen números reales
    (9.85, 4.96, 7.68, …) y sus manifests no contienen ni un `NaN` literal. Lo que ocurre es
    que un consumidor que pida `headline.return_pct` recibe `None` en las 8 entradas de SPX
    — y en Python eso se convierte en NaN en cuanto pasa por `float()` o por numpy. El
    síntoma se parecía a una violación de `strategy-contract.md` §2; la causa era esquema.

    Se fija el estado MEDIDO en vez de imponer un nombre: renombrar en un lado rompería a
    quien ya lee el otro, y esa decisión no la toma un test. Lo que este test impide es que
    la divergencia crezca a un tercer nombre o se extienda a más activos sin que nadie lo
    note — que es como el consumidor acabó viendo NaN.
    """
    familias: dict[str, set[str]] = {}
    if not ESTRATEGIAS.is_dir():
        pytest.skip("sin bundles")
    for d in sorted(ESTRATEGIAS.iterdir()):
        manifiesto = d / "manifest.json"
        if not manifiesto.is_file():
            continue
        try:
            m = json.loads(manifiesto.read_text(encoding="utf-8"))
        except Exception:
            continue
        for b in m.get("backtests", []):
            h = b.get("headline") or {}
            for clave in ("return_pct", "total_return_pct"):
                if clave in h:
                    familias.setdefault(d.name.split("_")[0], set()).add(clave)

    esperado = {
        "btc": {"return_pct"},
        "gold": {"return_pct"},
        "smart": {"return_pct"},
        "spx500": {"total_return_pct"},
    }
    for fam, claves in sorted(familias.items()):
        assert claves == esperado.get(fam, {"return_pct"}), (
            f"la familia {fam} cambió de nombre para el retorno del headline: {claves}. "
            f"Un consumidor que pida la clave antigua recibirá None, y eso se vuelve NaN "
            f"en cuanto pase por float()"
        )
    for fam, claves in familias.items():
        assert len(claves) == 1, (
            f"{fam} usa DOS nombres a la vez para el mismo hecho: {claves}"
        )


@pytest.mark.skipif(not ESTRATEGIAS.is_dir(), reason="bundles no presentes en este checkout")
def test_the_cop_divergence_is_exactly_the_one_documented() -> None:
    """El caso que originó el gate se fija con sus dos cifras, no en abstracto.

    Si alguien "arregla" esto sobrescribiendo el bundle inmutable, este test cae y obliga a
    justificarlo — sobrescribir viola `registry-lifecycle.md` §5 aunque el número resultante
    sea el bueno.
    """
    cop = ESTRATEGIAS / "smart_simple_v11" / "manifest.json"
    if not cop.is_file():
        pytest.skip("smart_simple_v11 no publicado en este checkout")
    m = json.loads(cop.read_text(encoding="utf-8"))
    entradas = [
        b for b in m.get("backtests", [])
        if str(b.get("model_version")) == "2.0.0" and b.get("year") == 2025
    ]
    assert entradas, "desapareció el backtest 2.0.0/2025: el registro histórico no se borra"
    headline = entradas[0].get("headline") or {}
    assert abs(float(headline.get("return_pct")) - 7.35) < 0.01, (
        "el headline post-arreglo de COP 2025 dejó de ser +7.35%: si cambió el número, "
        "cambió el backtest, y eso es una versión nueva"
    )
    gates = entradas[0].get("gates") or {}
    assert gates.get("recommendation") == "REVIEW" and gates.get("passed") == 4, (
        "COP 2025 dejó de estar en 4/6 REVIEW sin una versión nueva que lo justifique"
    )
