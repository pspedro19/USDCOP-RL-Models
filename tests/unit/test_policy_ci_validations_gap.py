"""BL-45 §11 — las validaciones CI que aún no tenían candado.

`tests/unit/test_policy_specs.py` ya cubre buena parte de la lista de §11 (retrain,
feature_set/resample, whitelist, fallbacks, policy_hash, determinismo, equivalencia con
los productores congelados). Al auditarla contra el texto de la spec quedaban cuatro
sin candado propio, y no son las decorativas:

* **conflictos con prioridad o resolución declarada** — sin esto, dos reglas que
  disparan a la vez deciden por el orden en que alguien las escribió en el YAML;
* **salidas dentro de los caps de dirección y exposición** — una regla puede declarar
  `target_exposure: 3.0` y nadie lo mira hasta que se ejecuta;
* **`rule_trace` con TODAS las condiciones relevantes** — trazar sólo la ganadora hace
  irreconstruible por qué las otras no dispararon, que es la mitad de una explicación;
* **`required_features` presentes** — una policy que referencia una feature que su
  `feature_set` no declara falla en producción, no en CI.

Cada uno se comprueba contra los specs **reales** de `config/policies/`, no contra
fixtures: un candado sobre un spec inventado prueba el candado, no el sistema.
"""

from __future__ import annotations

import pytest

from src.contracts.policy import PolicyContext
from src.contracts.policy_dsl import ALLOWED_OPERATORS, referenced_features
from src.strategies.policies.loader import build_policy, load_all_policy_specs

SPECS = {s["id"]: s for s in load_all_policy_specs()}

#: Techo de exposición admisible. No es una preferencia: `strategy-engines.md` exige que
#: la salida respete caps declarados, y una política rule-based que pidiera apalancamiento
#: por encima de 1x lo haría sin ningún gate de riesgo detrás.
EXPOSURE_CAP = 1.0
VALID_DIRECTIONS = {"LONG", "SHORT", "FLAT", "HOLD"}


def _politica(spec) -> dict:
    """El bloque `policy:` del spec — las reglas NO viven en la raiz.

    Lo aprendi rompiendo este mismo test: mi primera version leia `spec["rules"]` y
    saltaba los cuatro specs, con lo que la bateria daba verde sin ejercitar nada. Un
    test que se salta a si mismo es peor que no tenerlo, porque parece cobertura.
    """
    bloque = spec.get("policy")
    return bloque if isinstance(bloque, dict) else {}


def _reglas(spec) -> list:
    return list(_politica(spec).get("rules") or [])


def _es_declarativa(spec) -> bool:
    """¿El spec declara modo `declarative`? Se lee el MODO, no la presencia de reglas.

    Saltar por "no tiene reglas" haría que una política declarativa que las olvidara se
    saltara sola y en silencio — un skip que oculta justo el fallo que el candado busca.
    Las tres `coded_policy` del repo tienen su lógica en un módulo Python y no deben
    tener reglas; ésas sí es correcto saltarlas.
    """
    modo = ((spec.get("engine") or {}).get("implementation") or {}).get("mode")
    return modo != "coded_policy"


def _saltar_si_codificada(spec, policy_id: str) -> None:
    if not _es_declarativa(spec):
        pytest.skip(f"{policy_id} es coded_policy: sus reglas viven en un módulo, no en el spec")
    assert _reglas(spec), (
        f"{policy_id} se declara declarativa pero no trae reglas: no puede decidir nada"
    )


@pytest.mark.parametrize("policy_id", sorted(SPECS))
def test_rule_conflicts_are_resolved_by_declaration_not_by_yaml_order(policy_id) -> None:
    """Dos reglas que disparan a la vez no pueden decidir por orden de escritura.

    O la política declara `resolution.mode` (hoy `first_match`) **y** cada regla lleva
    `priority`, o el resultado depende de cómo alguien ordenó el YAML — que no es una
    decisión de gobierno sino un accidente de edición.
    """
    spec = SPECS[policy_id]
    reglas = _reglas(spec)
    if len(reglas) < 2:
        pytest.skip(f"{policy_id} tiene {len(reglas)} regla(s): no hay conflicto posible")

    resolucion = _politica(spec).get("resolution") or {}
    assert resolucion.get("mode"), f"{policy_id}: sin `resolution.mode` declarado"

    prioridades = [r.get("priority") for r in reglas]
    assert all(p is not None for p in prioridades), (
        f"{policy_id}: reglas sin `priority` — el desempate lo haría el orden del fichero"
    )
    assert len(set(prioridades)) == len(prioridades), (
        f"{policy_id}: prioridades repetidas {prioridades}; el empate vuelve a "
        "resolverse por posición"
    )


@pytest.mark.parametrize("policy_id", sorted(SPECS))
def test_every_declared_output_respects_direction_and_exposure_caps(policy_id) -> None:
    """Ninguna salida declarada puede pedir una dirección desconocida ni pasarse de cap.

    Se comprueba en el **spec**, no en ejecución: una regla con `target_exposure: 3.0`
    sólo se descubriría el día que dispara, y para entonces ya pidió 3x.
    """
    spec = SPECS[policy_id]

    salidas = [r["output"] for r in _reglas(spec) if "output" in r]
    resolucion = _politica(spec).get("resolution") or {}
    if "default_target_exposure" in resolucion:
        salidas.append(
            {
                "direction": resolucion.get("default_direction", "FLAT"),
                "target_exposure": resolucion["default_target_exposure"],
            }
        )

    for salida in salidas:
        direccion = salida.get("direction")
        assert direccion in VALID_DIRECTIONS, (
            f"{policy_id}: dirección {direccion!r} fuera de {sorted(VALID_DIRECTIONS)}"
        )
        exposicion = salida.get("target_exposure")
        assert isinstance(exposicion, (int, float)) and not isinstance(exposicion, bool), (
            f"{policy_id}: `target_exposure` {exposicion!r} no es un número"
        )
        assert abs(float(exposicion)) <= EXPOSURE_CAP, (
            f"{policy_id}: exposición {exposicion} supera el cap {EXPOSURE_CAP}; una "
            "rule-based no puede pedir apalancamiento sin un gate de riesgo detrás"
        )


@pytest.mark.parametrize("policy_id", sorted(SPECS))
def test_every_rule_references_only_whitelisted_operators(policy_id) -> None:
    """Recorrido explícito del AST de cada condición real, no sólo del caso feliz."""
    spec = SPECS[policy_id]

    def _operadores(nodo) -> list[str]:
        if not isinstance(nodo, dict):
            return []
        encontrados = [nodo["operator"]] if "operator" in nodo else []
        for valor in nodo.values():
            if isinstance(valor, dict):
                encontrados += _operadores(valor)
            elif isinstance(valor, list):
                for item in valor:
                    encontrados += _operadores(item)
        return encontrados

    for regla in _reglas(spec):
        for operador in _operadores(regla.get("when")):
            assert operador in ALLOWED_OPERATORS, (
                f"{policy_id}/{regla.get('id')}: operador {operador!r} fuera del "
                f"whitelist {sorted(ALLOWED_OPERATORS)}"
            )


@pytest.mark.parametrize("policy_id", sorted(SPECS))
def test_the_trace_explains_every_rule_not_only_the_winner(policy_id) -> None:
    """`rule_trace` debe traer TODAS las reglas evaluadas, no sólo la que ganó.

    Trazar sólo la ganadora deja irreconstruible **por qué las otras no dispararon**,
    que es la mitad de una explicación. Y el frontend renderiza esa traza (invariante 7):
    si no está completa, la única forma de completarla sería que la UI reevaluara las
    condiciones — exactamente lo que la regla prohíbe.
    """
    spec = SPECS[policy_id]
    _saltar_si_codificada(spec, policy_id)
    reglas = _reglas(spec)

    politica = build_policy(spec)
    requeridas = sorted(
        {f for regla in reglas for f in referenced_features(regla["when"])}
    )
    snapshot = {nombre: 1.0 for nombre in requeridas}

    decision = politica.evaluate(
        snapshot, PolicyContext(as_of="2026-01-05", mode="DECISION")
    )
    trazadas = {entrada.rule_id for entrada in decision.rule_trace.rules}

    assert trazadas == {str(r["id"]) for r in reglas}, (
        f"{policy_id}: la traza cubre {sorted(trazadas)} de {len(reglas)} reglas; las "
        "no disparadas también deben explicarse"
    )
    for entrada in decision.rule_trace.rules:
        assert entrada.observed, (
            f"{policy_id}/{entrada.rule_id}: la traza no registra los valores observados"
        )


@pytest.mark.parametrize("policy_id", sorted(SPECS))
def test_required_features_are_declared_by_the_policy_itself(policy_id) -> None:
    """Toda feature referenciada por una regla está en `required_features`.

    Si una regla usa una feature que la política no declara requerir, el snapshot puede
    llegar sin ella y el fallo aparece **en producción**, no en CI. Es la diferencia
    entre un contrato y una esperanza.
    """
    spec = SPECS[policy_id]
    _saltar_si_codificada(spec, policy_id)
    reglas = _reglas(spec)

    usadas = {f for regla in reglas for f in referenced_features(regla["when"])}
    declaradas = set(build_policy(spec).required_features())

    assert usadas <= declaradas, (
        f"{policy_id}: usa {sorted(usadas - declaradas)} sin declararlas como "
        "required_features; el snapshot podría llegar sin ellas"
    )


@pytest.mark.parametrize("policy_id", sorted(SPECS))
def test_a_rule_based_policy_never_declares_the_train_capability(policy_id) -> None:
    """§11: `rule_based` no declara `capability=train`.

    No es redundante con `retrain: never`: ese campo dice que NO se reentrena, y
    `capabilities` dice que tareas se le pueden PEDIR. Declarar `train` en una policy
    que no entrena haria que el factory le generase una tarea de entrenamiento vacia —
    una tarea verde que no hace nada es peor que una que falta, porque el tablero la
    cuenta como cobertura.
    """
    spec = SPECS[policy_id]
    if (spec.get("engine") or {}).get("type") != "rule_based":
        pytest.skip(f"{policy_id} no es rule_based")

    capacidades = list(spec.get("capabilities") or [])
    assert capacidades, f"{policy_id}: sin `capabilities` no se sabe que se le puede pedir"
    assert "train" not in capacidades, (
        f"{policy_id} es rule_based y declara capability 'train': el factory le crearia "
        "una tarea de entrenamiento que no entrena nada"
    )


@pytest.mark.parametrize("policy_id", sorted(SPECS))
def test_every_policy_declares_its_decision_cutoff(policy_id) -> None:
    """§11: ningun input puede superar el `decision_cutoff` — y para eso hay que declararlo.

    La validacion de §11 es sobre datos en ejecucion, pero su condicion previa es
    verificable aqui: sin `decision_point` declarado no existe cutoff contra el que
    comparar, y "ningun input lo supera" seria cierto por vacuidad. Una politica sin
    punto de decision consumiria implicitamente "lo ultimo que haya", que es el
    look-ahead mas facil de cometer y el mas dificil de ver.
    """
    entradas = SPECS[policy_id].get("inputs") or {}
    cutoff = entradas.get("decision_point")

    assert cutoff, (
        f"{policy_id}: sin `inputs.decision_point`, la regla 'ningun input supera el "
        "cutoff' se cumple por vacuidad y la policy leeria lo ultimo disponible"
    )
    assert isinstance(cutoff, str) and cutoff.strip()


@pytest.mark.parametrize("policy_id", sorted(SPECS))
def test_a_withdrawn_policy_keeps_its_manifest_and_hashes(policy_id) -> None:
    """§11: `WITHDRAWN` conserva bundles y manifiesto.

    Retirar una estrategia no puede borrar su rastro: sin `ssot_manifest` ni
    `policy_hash` seria imposible reconstruir que decidia cuando estaba viva, y una
    retirada se volveria indistinguible de un borrado. Es la misma razon por la que la
    cuarentena guarda el `source_record` de la barra que rechaza.

    Se exige de TODA policy, no solo de las retiradas: si el manifiesto se declarara al
    retirar, no habria nada que conservar.
    """
    gobierno = SPECS[policy_id].get("governance") or {}

    assert gobierno.get("ssot_manifest"), (
        f"{policy_id}: sin `ssot_manifest`, retirarla borraria la unica forma de saber "
        "que decidia"
    )
    assert str(gobierno.get("policy_hash", "")).startswith("sha256:"), (
        f"{policy_id}: sin `policy_hash` sellado, un bundle retirado no se puede "
        "verificar contra lo que corrio"
    )
    assert gobierno.get("frozen_at"), (
        f"{policy_id}: sin `frozen_at` no se sabe desde cuando el manifiesto es el que "
        "gobernaba"
    )
