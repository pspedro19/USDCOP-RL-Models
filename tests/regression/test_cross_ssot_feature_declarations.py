# -*- coding: utf-8 -*-
"""Gate cross-SSOT: lo que una policy EXIGE debe estar declarado como input.

POR QUE EXISTE
--------------
Al ir a escribir el productor de `observations::` para BL-45 aparecio que los dos
SSOT se contradicen sobre que es un input:

    config/policies/<id>.yaml          -> `inputs.required_features`   (lo que la policy EXIGE)
    config/features/feature_sets/*.yaml -> `ordered_features`          (lo que ALGUIEN materializa)

Nadie los habia cruzado nunca. Consecuencia operativa, que no es teorica: un
productor construido desde el feature set —que es el contrato de QUE materializar—
entregaria menos features de las que la policy exige, y la policy fallaria SIEMPRE
por `missing`. La cadena gobernada de BL-45 no se puede atravesar por CONTRATO,
no por bug.

Y no se puede alegar `derived_in_policy`: `src/contracts/policy_dsl.py` **no tiene
ningun operador de ventana** (medido: cero coincidencias de `rolling|window|sma|mean`),
asi que una policy declarativa no puede derivar una media movil; y las dos policies
CODED lo dicen en su propio codigo —`gold.py`: *"the policy only consumes them"*;
`btc.py`: `required = ("realized_vol_20",)`—. **La prosa de `derived_in_policy` no
exime a nadie**: solo eximiria un contrato estructurado mas codigo que demuestre que
la Policy recibe inputs crudos y deriva dentro (CXD-609).

DOS CLASES, NO UNA (correccion de CODEX en CXD-609 a mi conteo inicial de "9
huerfanas homogeneas")
----------------------------------------------------------------------------------
1. **6 huerfanas EJECUTABLES** en 3 policies construibles (SPX 1, Gold 4, BTC 1).
   Son deuda real: la policy corre y le falta el input.
2. **3 componentes de `smart_simple_v11`**, que es `SPEC_ONLY`, sin implementacion, y
   declara `required_features_verified: false`. Sus 25 `ordered_features` son la
   receta UPSTREAM del predictor y sus 3 requeridas son componentes de decision
   DOWNSTREAM: meterlas en el mismo subconjunto mezclaria capas y daria un rojo
   falso. Tienen test SEPARADO, que fija su estado declarado en vez de exigir
   coherencia entre capas distintas.

Contract: interfaz BL-39 (catalogo/feature-set) x BL-45 (snapshot productivo).
Estado: deuda DECLARADA, no arreglada. El remedio para SPX es la decision C
co-firmada en CXD-608 (feature-set propio + catalogo + productor unico + democion
de `PARITY_GREEN`), y todavia no esta ejecutado.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
POLICY_DIR = REPO / "config" / "policies"
FEATURE_SET_DIR = REPO / "config" / "features" / "feature_sets"

#: Las 6 huerfanas EJECUTABLES, pinneadas por policy. Pinnear el conjunto exacto
#: —y no solo el conteo— es lo que hace que aparezca una septima ponga rojo: un
#: `len(...) == 6` pasaria igual si se arreglara una y se rompiera otra.
DEUDA_EJECUTABLE: dict[str, frozenset[str]] = {
    "spx500_daily_ma200_v1": frozenset({"ma_200"}),
    "gold_trend_simple": frozenset({"sma_63", "sma_126", "sma_252", "realized_vol_20"}),
    "btc_hodl_b1": frozenset({"realized_vol_20"}),
}

#: `SPEC_ONLY`, otra clase: ver el docstring.
POLICY_SPEC_ONLY = "smart_simple_v11"

MOTIVO = (
    "BLOCKED_OPERATOR_DECISION — la feature que la policy exige no esta declarada como "
    "input en su feature_set ni catalogada. El remedio (decision C, CXD-608) crea "
    "feature-set propio + entrada de catalogo + productor unico y DEMUEVE la policy de "
    "PARITY_GREEN mientras se revalida: es gobierno de modelado, no fontaneria. Cuando "
    "se cierre, este xfail XPASSa y pone rojo, y hay que borrarlo en el MISMO commit."
)


def _specs() -> dict[str, dict]:
    out = {}
    for path in sorted(POLICY_DIR.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        out[str(doc["id"])] = doc
    return out


def _feature_sets() -> dict[str, dict]:
    out = {}
    for path in sorted(FEATURE_SET_DIR.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        out[str(doc.get("feature_set_id"))] = doc
    return out


def _huerfanas(policy_id: str) -> set[str]:
    """`required_features` que su feature_set declarado NO ordena como input."""
    spec = _specs()[policy_id]
    inputs = spec.get("inputs") or {}
    fs = _feature_sets().get(str(inputs.get("feature_set_id")))
    assert fs is not None, (
        f"{policy_id}: declara feature_set_id={inputs.get('feature_set_id')!r} y ese "
        f"feature set NO EXISTE. Una referencia colgando no es una declaracion"
    )
    ordenadas = {o["feature_id"] for o in (fs.get("ordered_features") or [])}
    return {f for f in (inputs.get("required_features") or []) if f not in ordenadas}


@pytest.mark.parametrize("policy_id", sorted(DEUDA_EJECUTABLE))
@pytest.mark.xfail(strict=True, reason=MOTIVO)
def test_every_required_feature_is_declared_as_an_input(policy_id: str) -> None:
    """El invariante que DEBERIA cumplirse, marcado como deuda mientras no se cumple.

    `strict=True` a proposito: el dia que la discrepancia de esta policy se cierre,
    el test XPASSa y pytest lo cuenta como FALLO. Es la unica forma de que la deuda
    no sobreviva a su propio remedio — un `xfail` no estricto se quedaria ahi para
    siempre diciendo "esto esta roto" sobre algo ya arreglado, que es la version
    lenta del mismo problema que este fichero denuncia.
    """
    huerfanas = _huerfanas(policy_id)
    assert not huerfanas, (
        f"{policy_id} exige {sorted(huerfanas)}, que su feature_set declarado no ordena "
        f"como input. Ningun productor las materializaria y la policy fallaria por "
        f"`missing` en toda corrida"
    )


def test_the_declared_debt_is_exactly_what_was_pinned_no_more_no_less() -> None:
    """Aparece una septima huerfana => ROJO inmediato, sin esperar a nadie.

    Este test **no** es xfail: el conjunto pinneado arriba es la deuda ACEPTADA, y
    cualquier desviacion —una nueva, o una que cambie de nombre— es regresion nueva,
    no deuda conocida. Sin el, los `xfail` de arriba servirian de paraguas: se
    podrian añadir huerfanas nuevas y nadie lo notaria porque "eso ya estaba en rojo".
    """
    observada = {pid: frozenset(_huerfanas(pid)) for pid in DEUDA_EJECUTABLE}
    assert observada == DEUDA_EJECUTABLE, (
        f"la deuda cross-SSOT cambio.\n  pinneada: {DEUDA_EJECUTABLE}\n  observada: "
        f"{observada}\nSi es un arreglo, quita la entrada AQUI y el xfail de esa policy "
        f"en el mismo commit. Si es nueva, no la aceptes en silencio."
    )


def test_no_runnable_policy_outside_the_pinned_set_has_orphans() -> None:
    """Y ninguna OTRA policy construible tiene huerfanas sin declarar.

    Cubre el hueco que dejan los dos tests anteriores: los dos miran solo las tres
    policies ya conocidas. Una policy NUEVA con huerfanas entraria sin que ninguno
    se enterase — un candado que solo vigila a los sospechosos de siempre.
    """
    sorpresas = {}
    for policy_id, spec in _specs().items():
        if policy_id in DEUDA_EJECUTABLE or policy_id == POLICY_SPEC_ONLY:
            continue
        if (spec.get("migration") or {}).get("status") == "SPEC_ONLY":
            continue
        huerfanas = _huerfanas(policy_id)
        if huerfanas:
            sorpresas[policy_id] = sorted(huerfanas)
    assert not sorpresas, (
        f"policies construibles con huerfanas NO declaradas en la deuda: {sorpresas}"
    )


def test_spec_only_policy_is_pinned_by_its_declared_state_not_by_layer_mixing() -> None:
    """`smart_simple_v11` es otra clase y se juzga por lo que DECLARA.

    Sus 25 `ordered_features` son la receta UPSTREAM del predictor; sus 3
    `required_features` son componentes de decision DOWNSTREAM. Exigir
    `required ⊆ ordered` aqui mezclaria dos capas y daria un rojo FALSO — un rojo
    falso gasta la misma credibilidad que un verde falso, y a la larga hace que
    nadie mire el gate.

    Lo que si se fija es su estado declarado: mientras diga `SPEC_ONLY` y
    `required_features_verified: false`, sus tres componentes no cuentan como deuda
    ejecutable. Si alguien lo promueve o marca las features como verificadas sin
    resolver antes el contrato de componentes de decision (R8-D1), esto cae.
    """
    spec = _specs()[POLICY_SPEC_ONLY]
    estado = (spec.get("migration") or {}).get("status")
    # OJO: `required_features_verified` vive bajo `governance`, no bajo `inputs`
    # (medido). Lo asumi en `inputs` y el test fallo con None -- el MISMO error de
    # nivel que ya cometi con `engine.retrain`. Leer la clave donde no esta produce
    # un None que se parece muchisimo a "no declarado".
    verificadas = (spec.get("governance") or {}).get("required_features_verified")
    assert estado == "SPEC_ONLY", (
        f"{POLICY_SPEC_ONLY} paso a {estado!r}: sus 3 componentes de decision dejan de "
        f"estar exentos y hay que declarar el contrato de componentes antes de promover"
    )
    assert verificadas is False, (
        f"{POLICY_SPEC_ONLY} declara required_features_verified={verificadas!r}: si estan "
        f"verificadas, entran en el gate cross-SSOT como el resto"
    )


def test_the_dsl_still_cannot_derive_a_rolling_feature() -> None:
    """La premisa del gate, fijada en vez de comentada.

    Todo esto se apoya en que una policy declarativa NO puede derivar una media
    movil. Si mañana el DSL gana un operador de ventana, la exencion
    `derived_in_policy` pasaria a ser defendible para las declarativas y este gate
    habria que repensarlo entero. Mejor enterarse por un rojo que por una discusion.
    """
    dsl = (REPO / "src" / "contracts" / "policy_dsl.py").read_text(encoding="utf-8")
    ventana = [t for t in ("rolling", "sma(", "moving_average", "window_mean") if t in dsl]
    assert not ventana, (
        f"el DSL parece tener operadores de ventana {ventana}: revisar si "
        f"`derived_in_policy` ya exime a las policies declarativas"
    )
