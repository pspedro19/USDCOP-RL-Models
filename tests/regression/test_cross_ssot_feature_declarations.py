# -*- coding: utf-8 -*-
"""Gate cross-SSOT: lo que una policy EXIGE debe estar declarado como input.

POR QUE EXISTE
--------------
Los dos SSOT se contradecian sobre que es un input:

    config/policies/<id>.yaml            -> `inputs.required_features`  (lo que EXIGE)
    config/features/feature_sets/*.yaml  -> `ordered_features`          (lo que ALGUIEN materializa)

Nadie los habia cruzado nunca. Consecuencia medida, no teorica: un productor construido
desde el feature-set —que es el contrato de QUE materializar— entregaba menos features
de las que la policy exige, y la policy fallaba por `missing` en **toda** corrida. La
cadena gobernada de BL-45 no se podia atravesar por CONTRATO, no por bug.

No se puede alegar `derived_in_policy`: `src/contracts/policy_dsl.py` no tiene ningun
operador de ventana (medido), asi que una policy declarativa no puede derivar una media;
y las dos policies CODED lo dicen en su propio codigo — `gold.py`: *"the policy only
consumes them"*; `btc.py`: `required = ("realized_vol_20",)`. **La prosa no exime.**

DE ALLOWLIST A JUEZ DIRECTO (CXD-631 §5)
----------------------------------------
La primera version llevaba una **allowlist** de deuda aceptada con `xfail(strict=True)`
por policy. Era lo correcto mientras habia 6 huerfanas ejecutables que nadie podia
cerrar de golpe. Cerradas SPX, BTC y Gold, esa forma se vuelve peligrosa: **una lista
vacia pasa por vacuidad**, y un `xfail` parametrizado sin sujeto no juzga nada. Peor
aun, la excepcion quedaria ahi para que una regresion futura la reutilizara — "esto ya
estaba tolerado".

Asi que la allowlist DESAPARECE. Queda un juez directo que recorre **toda** policy
construible y falla si cualquier `required_feature` no esta ordenada en su feature-set.
Sin lista, sin skip, sin excepcion heredable.

`smart_simple_v11` conserva su candado SEPARADO: es `SPEC_ONLY`, sin implementacion, y
sus 25 `ordered_features` son la receta UPSTREAM del predictor mientras sus 3 requeridas
son componentes de decision DOWNSTREAM. Exigir subconjunto ahi mezclaria capas y daria
un **rojo falso**, que gasta la misma credibilidad que un verde falso.

Contract: interfaz BL-39 (catalogo/feature-set) x BL-45 (snapshot productivo).
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
POLICY_DIR = REPO / "config" / "policies"
FEATURE_SET_DIR = REPO / "config" / "features" / "feature_sets"

#: `SPEC_ONLY`, otra clase — ver el docstring.
POLICY_SPEC_ONLY = "smart_simple_v11"


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


def _huerfanas(policy_id: str, spec: dict) -> set[str]:
    """`required_features` que su feature_set declarado NO ordena como input."""
    inputs = spec.get("inputs") or {}
    fs = _feature_sets().get(str(inputs.get("feature_set_id")))
    assert fs is not None, (
        f"{policy_id}: declara feature_set_id={inputs.get('feature_set_id')!r} y ese "
        f"feature set NO EXISTE. Una referencia colgando no es una declaracion"
    )
    ordenadas = {o["feature_id"] for o in (fs.get("ordered_features") or [])}
    return {f for f in (inputs.get("required_features") or []) if f not in ordenadas}


def _construibles() -> dict[str, dict]:
    """Toda policy que NO sea `SPEC_ONLY` declarado."""
    return {
        pid: spec for pid, spec in _specs().items()
        if (spec.get("migration") or {}).get("status") != "SPEC_ONLY"
    }


def test_every_required_feature_of_every_buildable_policy_is_declared_as_an_input() -> None:
    """EL juez. Recorre TODAS las construibles; ninguna excepcion, ninguna lista.

    Antes esto vivia parametrizado con una allowlist de deuda y `xfail(strict=True)`.
    Cerradas las tres policies, la allowlist se habria quedado vacia —y una lista vacia
    **pasa sin juzgar nada**—, ademas de dejar una excepcion que una regresion futura
    podria reutilizar. Aqui no hay donde apuntarse.

    Rojo con: devolver cualquier feature requerida a `derived_in_policy` sin ordenarla,
    o anadir una policy nueva que exija algo que su set no declare.
    """
    sucias = {
        pid: sorted(h) for pid, spec in _construibles().items()
        if (h := _huerfanas(pid, spec))
    }
    assert not sucias, (
        f"policies que EXIGEN features que su feature-set no declara como input: "
        f"{sucias}. Ningun productor las materializaria y la policy fallaria por "
        f"`missing` en toda corrida. `derived_in_policy` NO exime: solo eximiria un "
        f"contrato estructurado mas codigo que demuestre derivacion interna"
    )


def test_the_judge_actually_has_policies_to_judge() -> None:
    """Anti-vacuidad del juez mismo.

    Si `_construibles()` devolviera vacio —un `status` renombrado, un glob roto— el test
    de arriba pasaria **sin mirar nada**. Es exactamente la forma de falso verde que este
    fichero nacio para desmontar, asi que se vigila tambien aqui.
    """
    construibles = _construibles()
    assert len(construibles) >= 3, (
        f"solo {len(construibles)} policies construibles ({sorted(construibles)}): el "
        f"juez se quedo casi sin sujeto y su verde dejaria de significar algo"
    )
    total_req = sum(
        len((s.get("inputs") or {}).get("required_features") or [])
        for s in construibles.values()
    )
    assert total_req >= 6, (
        f"las construibles solo exigen {total_req} features en total: con tan poco que "
        f"cruzar, el juez no distingue 'todo declarado' de 'no hay nada que declarar'"
    )


def test_spec_only_policy_is_pinned_by_its_declared_state_not_by_layer_mixing() -> None:
    """`smart_simple_v11` es otra clase y se juzga por lo que DECLARA.

    Sus 25 `ordered_features` son la receta UPSTREAM del predictor; sus 3
    `required_features` son componentes de decision DOWNSTREAM. Exigir
    `required subset ordered` aqui mezclaria dos capas y daria un rojo FALSO — que gasta
    la misma credibilidad que un verde falso, y a la larga hace que nadie mire el gate.

    Lo que si se fija es su estado declarado: mientras diga `SPEC_ONLY` y
    `required_features_verified: false`, sus tres componentes no entran al juez. Si
    alguien lo promueve o los marca verificados sin resolver antes el contrato de
    componentes de decision, esto cae.
    """
    spec = _specs()[POLICY_SPEC_ONLY]
    estado = (spec.get("migration") or {}).get("status")
    # OJO: `required_features_verified` vive bajo `governance`, no bajo `inputs`.
    verificadas = (spec.get("governance") or {}).get("required_features_verified")
    assert estado == "SPEC_ONLY", (
        f"{POLICY_SPEC_ONLY} paso a {estado!r}: sus 3 componentes de decision dejan de "
        f"estar exentos y hay que declarar el contrato de componentes antes de promover"
    )
    assert verificadas is False, (
        f"{POLICY_SPEC_ONLY} declara required_features_verified={verificadas!r}: si "
        f"estan verificadas, entran en el juez como el resto"
    )


def test_the_dsl_still_cannot_derive_a_rolling_feature() -> None:
    """La premisa del gate, fijada en vez de comentada.

    Todo esto se apoya en que una policy declarativa NO puede derivar una media movil.
    Si manana el DSL gana un operador de ventana, la exencion `derived_in_policy` pasaria
    a ser defendible para las declarativas y este gate habria que repensarlo entero.
    Mejor enterarse por un rojo que por una discusion.
    """
    dsl = (REPO / "src" / "contracts" / "policy_dsl.py").read_text(encoding="utf-8")
    ventana = [t for t in ("rolling", "sma(", "moving_average", "window_mean") if t in dsl]
    assert not ventana, (
        f"el DSL parece tener operadores de ventana {ventana}: revisar si "
        f"`derived_in_policy` ya exime a las policies declarativas"
    )
