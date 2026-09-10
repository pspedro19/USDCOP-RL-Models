# -*- coding: utf-8 -*-
"""La cadena, atravesada de punta a punta con datos REALES (BL-45).

QUÉ PRUEBA Y QUÉ NO
-------------------
Prueba que `produce_observations → resolve_snapshot → validate_inputs → evaluate`
se **atraviesa entera** partiendo del índice oficial en disco y llegando a una
decisión con su `rule_trace`. Cada eslabón consume literalmente lo que produjo el
anterior: nada se fabrica a mano por el camino, que es la diferencia entre probar la
cadena y probar cuatro funciones que casualmente encajan.

**NO prueba que los VALORES sean correctos.** Esta es la frontera menos obvia y la que
más fácil se lee de más. Cada test compara la decisión contra **la regla aplicada al
mismo snapshot que recibió**: si `ma_200` llegara corrupto —digamos, cero—, el test
calcularía `close > 0 ⇒ LONG` y la policy diría LONG también. **Self-consistente, y
verde.** Medido: sabotear `compute_ma_200` o `build_trend_smas` a `0.0` **no pone rojo
este fichero**; sí ponen rojo los tests de productor.

Es decir: aquí se verifica **tránsito y aplicación de la regla**, no exactitud numérica.
La exactitud vive —y debe seguir viviendo— en los tests de productor, que comparan la
serie completa contra una referencia legacy **independiente**. Duplicar esa comparación
aquí crearía una segunda fuente de verdad sobre la fórmula, que es justo lo que los
slices de SPX/BTC/Gold eliminaron.

**NO prueba que la cadena corra en Airflow.** No hay contenedor de Airflow en este
entorno (sí hay `postgres`, `redis`, `trading-api` y `signalbridge`), así que lo que
se ejercita es la lógica de los cuatro eslabones sin el scheduler y sin el fetch de
DB — el quinto, `publish`, necesita `reference.instrument` viva y queda fuera. Esa
limitación está declarada en la ficha de BL-45 y **este fichero no la cierra**.

POR QUÉ HACE FALTA, HABIENDO YA CANDADOS POR ESLABÓN
----------------------------------------------------
Los candados existentes prueban cada eslabón contra entradas construidas a mano en el
test. Eso deja vivo el fallo más caro de esta serie: **cada pieza correcta y el
conjunto inatravesable**. Fue literalmente el caso en R3 —cuatro eslabones en el
grafo, tres de ellos incapaces de ejecutarse— y ningún test por-eslabón lo vio,
porque cada uno recibía justo lo que necesitaba. Aquí nadie le da nada a nadie:
la salida de uno ES la entrada del siguiente.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.contracts.policy import PolicyContext  # noqa: E402
from src.features.observations import build_observations  # noqa: E402
from src.orchestration.feature_snapshot import resolve_feature_snapshot  # noqa: E402
from src.policy_engine import evaluate_policy, validate_policy_inputs  # noqa: E402
from src.strategies.policies.loader import (  # noqa: E402
    build_policy,
    canonical_policy_hash,
    load_policy_spec,
)

SEED = REPO / "seeds" / "latest" / "spx500_daily_ohlcv.parquet"
SPEC_PATH = REPO / "config" / "policies" / "spx500_daily_ma200_v1.yaml"
CUTOFF = "2026-07-29T00:00:00+00:00"


#: Los tres seeds estan VERSIONADOS (`git ls-files` lo confirma). Su ausencia por
#: tanto NO es "no se puede probar aqui": es que la cobertura que estos tests afirman
#: **desaparecio**, y eso tiene que ponerse rojo. Un `skip` sobre un fichero que el
#: repo garantiza es exactamente el falso verde que esta suite existe para desmontar
#: (CXD-634).
def _leer_seed(ruta: Path) -> pd.DataFrame:
    assert ruta.is_file(), (
        f"falta el seed VERSIONADO {ruta.name}. No se salta: sin el, esta cadena deja "
        f"de estar probada y el verde de la suite dejaria de significar lo que dice"
    )
    return pd.read_parquet(ruta)


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return _leer_seed(SEED)


@pytest.fixture(scope="module")
def spec() -> dict:
    """El spec REAL, promovido **sólo en memoria** para tener sujeto.

    `spx500_daily_ma200_v1` está en `PARITY_PENDING` por la democión deliberada de la
    decisión C: v1.1.0 es otra identidad y no hereda el veredicto de paridad de
    v1.0.0. Así que hoy no hay ninguna policy elegible en el repo.

    Se promueve una COPIA y se re-congela su hash. **No se toca el fichero**:
    re-promover el spec real para que una suite tenga sujeto sería exactamente el
    acto que la democión existe para impedir, y además la promoción es del operador.
    """
    import copy

    doc = copy.deepcopy(load_policy_spec(SPEC_PATH))
    doc.setdefault("inputs", {})["max_snapshot_age"] = "P7D"
    doc["migration"]["status"] = "PARITY_GREEN"
    doc["governance"]["policy_hash"] = canonical_policy_hash(doc)
    return doc


def test_the_chain_is_traversable_from_real_bars_to_a_real_decision(spec, bars) -> None:
    """Los cuatro eslabones, encadenados de verdad.

    Rojo si CUALQUIERA de ellos deja de aceptar lo que produce el anterior — que es
    el defecto que R3 tenía y que ningún test por-eslabón podía ver.
    """
    # 1. producir: del parquet a `{feature: {value, available_at, provenance}}`
    observations = build_observations(spec, bars, decision_cutoff=CUTOFF)
    assert set(observations) == {"close", "ma_200"}

    # 2. resolver: aplica el corte causal y proyecta SÓLO valores
    snapshot = resolve_feature_snapshot(observations, decision_cutoff=CUTOFF)
    assert set(snapshot) == {"close", "ma_200"}
    assert all(isinstance(v, float) for v in snapshot.values())
    # La metadata se queda en la frontera de lectura: si el snapshot la arrastrara,
    # la policy podría decidir con ella y el corte dejaría de ser el único juez.
    assert "available_at" not in str(snapshot)

    # 3. validar: con los fallbacks DECLARADOS por el spec, no con los del runner
    policy = build_policy(spec)
    ctx = PolicyContext(as_of=CUTOFF, extras={"snapshot_is_stale": False})
    fallbacks = {
        "missing_input_policy": spec["policy"]["missing_input_policy"],
        "stale_input_policy": spec["policy"]["stale_input_policy"],
    }
    assert validate_policy_inputs(policy, snapshot, ctx, **fallbacks) is None

    # 4. evaluar: una decisión REAL, con su traza
    decision = evaluate_policy(policy, snapshot, ctx, **fallbacks)
    assert decision.direction in {"LONG", "FLAT"}
    assert decision.rule_trace is not None, "una decisión sin traza no es auditable"
    assert decision.engine_ref.policy_hash == spec["governance"]["policy_hash"], (
        "la decisión no lleva la identidad de la policy que la produjo"
    )


def test_the_decision_matches_the_rule_applied_to_the_real_numbers(spec, bars) -> None:
    """La decisión NO se acepta por venir del motor: se comprueba contra la regla.

    `close > ma_200 ⇒ LONG (exposición 1.0)`, si no `FLAT (0.0)`. Es aritmética de
    dos números que el propio test ya tiene delante, así que puede juzgar el
    veredicto en vez de confiar en él. Un test que sólo comprobara "devuelve algo
    con `direction`" pasaría con el motor cableado a `FLAT` para siempre.
    """
    observations = build_observations(spec, bars, decision_cutoff=CUTOFF)
    snapshot = resolve_feature_snapshot(observations, decision_cutoff=CUTOFF)
    ctx = PolicyContext(as_of=CUTOFF, extras={"snapshot_is_stale": False})
    decision = evaluate_policy(
        build_policy(spec), snapshot, ctx,
        missing_input_policy="FAIL_CLOSED", stale_input_policy="FLAT",
    )

    por_encima = snapshot["close"] > snapshot["ma_200"]
    esperado = ("LONG", 1.0) if por_encima else ("FLAT", 0.0)
    assert (decision.direction, float(decision.target_exposure)) == esperado, (
        f"close={snapshot['close']} vs ma_200={snapshot['ma_200']} ⇒ esperado "
        f"{esperado}, obtenido ({decision.direction}, {decision.target_exposure})"
    )


def test_a_stale_snapshot_degrades_the_whole_chain_to_the_declared_flat(spec, bars) -> None:
    """Con el umbral apretado, la MISMA cadena y los MISMOS datos degradan a FLAT.

    Ejercita el fallback declarado (`stale_input_policy: FLAT`) de punta a punta y no
    en una llamada aislada: se cambia UNA cosa —el umbral de frescura— y se comprueba
    que el veredicto cambia. Sin este contraste, "degrada a FLAT" no se distingue de
    "siempre da FLAT".
    """
    import copy
    from datetime import timedelta

    apretado = copy.deepcopy(spec)
    apretado["inputs"]["max_snapshot_age"] = "PT1M"          # un minuto: todo es viejo
    apretado["governance"]["policy_hash"] = canonical_policy_hash(apretado)

    observations = build_observations(apretado, bars, decision_cutoff=CUTOFF)
    snapshot = resolve_feature_snapshot(observations, decision_cutoff=CUTOFF)

    # La frescura se DERIVA de la evidencia, igual que hace el factory.
    from src.features.observations import RECONSTRUCTION_LAG  # noqa: F401

    mas_vieja = min(pd.Timestamp(o["available_at"]) for o in observations.values())
    stale = (pd.Timestamp(CUTOFF) - mas_vieja) > timedelta(minutes=1)
    assert stale is True, "el umbral apretado debería declarar stale este snapshot"

    ctx = PolicyContext(as_of=CUTOFF, extras={"snapshot_is_stale": stale})
    degradada = validate_policy_inputs(
        build_policy(apretado), snapshot, ctx,
        missing_input_policy="FAIL_CLOSED", stale_input_policy="FLAT",
    )
    assert degradada is not None and degradada.direction == "FLAT"
    assert "INPUT_STALE" in degradada.reason_codes


# --- La cadena, sobre las TRES policies construibles --------------------------
#
# Hasta aquí sólo se probaba con `spx500_daily_ma200_v1`, que es DECLARATIVA y usa el
# contrato de productor `series_close_v1`. Eso dejaba sin cubrir la mitad del sistema:
# Gold y BTC son `coded_policy` y sus features salen por `ohlcv_frame_v1`. Una cadena
# que sólo se ejercita con un motor y un contrato no está probada, está muestreada.

TRES = [
    # (policy_id, seed, motor esperado)
    ("spx500_daily_ma200_v1", "spx500", "declarative"),
    ("btc_hodl_b1", "btcusdt", "coded_policy"),
    ("gold_trend_simple", "xauusd", "coded_policy"),
]
CUTOFF_3 = "2026-07-30T00:00:00+00:00"


def _promovida(policy_id: str) -> dict:
    """El spec REAL, promovido y re-congelado **sólo en memoria**.

    Las tres están en `PARITY_PENDING` por decisión de gobierno. Promover el fichero
    para que una suite tenga sujeto sería justo lo que la democión existe para impedir.
    """
    import copy

    doc = copy.deepcopy(
        load_policy_spec(REPO / "config" / "policies" / f"{policy_id}.yaml")
    )
    doc["migration"]["status"] = "PARITY_GREEN"
    doc["governance"]["policy_hash"] = canonical_policy_hash(doc)
    return doc


@pytest.mark.parametrize("policy_id,seed,motor", TRES, ids=[t[0] for t in TRES])
def test_the_chain_traverses_for_every_buildable_policy(policy_id, seed, motor) -> None:
    """`produce → resolve → validate → evaluate` en las tres, con datos reales.

    Cada eslabón consume lo que produjo el anterior. Y se comprueba el MOTOR además del
    resultado: si mañana alguien colapsara las tres al camino declarativo, el test
    seguiría verde mirando sólo la decisión.
    """
    ruta = REPO / "seeds" / "latest" / f"{seed}_daily_ohlcv.parquet"
    spec = _promovida(policy_id)
    assert spec["engine"]["implementation"]["mode"] == motor, (
        f"{policy_id}: el motor declarado cambió a "
        f"{spec['engine']['implementation']['mode']!r}; este candado cubre AMBOS "
        f"caminos y hay que revisar cuál quedó sin ejercitar"
    )

    bars = _leer_seed(ruta).sort_values("time").reset_index(drop=True)
    observations = build_observations(spec, bars, decision_cutoff=CUTOFF_3)
    requeridas = set(spec["inputs"]["required_features"])
    assert requeridas <= set(observations), (
        f"{policy_id}: el productor no materializó {sorted(requeridas - set(observations))}"
    )

    snapshot = resolve_feature_snapshot(observations, decision_cutoff=CUTOFF_3)
    ctx = PolicyContext(as_of=CUTOFF_3, extras={"snapshot_is_stale": False})
    fallbacks = {
        "missing_input_policy": spec["policy"]["missing_input_policy"],
        "stale_input_policy": spec["policy"]["stale_input_policy"],
    }
    assert validate_policy_inputs(build_policy(spec), snapshot, ctx, **fallbacks) is None

    decision = evaluate_policy(build_policy(spec), snapshot, ctx, **fallbacks)
    assert decision.direction in {"LONG", "FLAT", "SHORT"}
    assert decision.reason_codes, "una decisión sin reason code no es auditable"
    # Y NO se acepta un FLAT degradado colado como decisión de la regla.
    assert "INPUT_MISSING" not in decision.reason_codes
    assert "INPUT_STALE" not in decision.reason_codes


def test_the_gold_vote_decides_the_direction_not_a_fallback() -> None:
    """La decisión de Gold se juzga contra SU regla: voto 2-de-3 sobre las SMA.

    Con los datos de hoy el cierre está por DEBAJO de las tres medias ⇒ 0/3 votos ⇒
    FLAT con `SMA_VOTES_LT_MIN`. Se comprueba el conteo y el reason code, no sólo que
    "devuelva algo": un FLAT puede venir de la regla o de un fallback degradado, y
    confundirlos es exactamente cómo una cadena rota parece sana.
    """
    ruta = REPO / "seeds" / "latest" / "xauusd_daily_ohlcv.parquet"
    spec = _promovida("gold_trend_simple")
    bars = _leer_seed(ruta).sort_values("time").reset_index(drop=True)
    snapshot = resolve_feature_snapshot(
        build_observations(spec, bars, decision_cutoff=CUTOFF_3), decision_cutoff=CUTOFF_3
    )
    votos = sum(snapshot["close"] > snapshot[f"sma_{w}"] for w in (63, 126, 252))
    decision = evaluate_policy(
        build_policy(spec), snapshot,
        PolicyContext(as_of=CUTOFF_3, extras={"snapshot_is_stale": False}),
        missing_input_policy="FAIL_CLOSED", stale_input_policy="FLAT",
    )
    esperado = "LONG" if votos >= 2 else "FLAT"
    assert decision.direction == esperado, (
        f"votos={votos}/3 ⇒ esperado {esperado}, obtenido {decision.direction} "
        f"({decision.reason_codes})"
    )
    if votos < 2:
        assert "SMA_VOTES_LT_MIN" in decision.reason_codes, (
            f"FLAT sin el reason code de la regla: {decision.reason_codes}. Un FLAT "
            f"degradado y un FLAT decidido no son lo mismo"
        )


def test_the_three_policy_matrix_is_exactly_what_it_claims_to_cover() -> None:
    """Anti-vacuidad de la parametrizacion (CXD-634).

    `TRES` podria quedarse en una entrada —o en ninguna— y los tests de arriba
    seguirian "pasando" habiendo cubierto la mitad del sistema. Se exige el conjunto
    EXACTO de policies y activos, y que **ambos motores** esten representados: el valor
    de este fichero es precisamente cubrir `declarative` y `coded_policy`, y una matriz
    que perdiera uno de los dos dejaria de justificar su propia existencia.
    """
    ids = {p for p, _, _ in TRES}
    activos = {a for _, a, _ in TRES}
    motores = {m for _, _, m in TRES}
    assert ids == {"spx500_daily_ma200_v1", "btc_hodl_b1", "gold_trend_simple"}, ids
    assert activos == {"spx500", "btcusdt", "xauusd"}, activos
    assert motores == {"declarative", "coded_policy"}, (
        f"la matriz solo cubre {motores}: el sentido de este fichero es ejercitar los "
        f"DOS motores y los dos contratos de productor"
    )


def test_the_frontier_this_file_does_NOT_cross_is_declared() -> None:
    """Lo que estos tests NO prueban, escrito y comprobable.

    Es facil leer "cadena end-to-end en verde" como "la cadena corre en produccion".
    No corre: falta `publish` —necesita `reference.instrument` viva— y falta Airflow.
    Este candado existe para que esa frontera no se erosione en silencio: si alguien
    importara o llamara el eslabon de publicacion aqui, el fichero estaria afirmando
    mas de lo que hace.

    Se inspecciona el AST y no el texto: la primera version buscaba subcadenas y se
    ponia roja por su PROPIO docstring, que nombra `publish` para declarar la
    frontera. Confundir prosa con codigo es la version tonta del mismo error que este
    fichero persigue -- afirmar sobre lo que no se midio.
    """
    import ast

    arbol = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    importados: set[str] = set()
    llamados: set[str] = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.ImportFrom):
            importados |= {a.name for a in nodo.names}
            if nodo.module:
                importados.add(nodo.module)
        elif isinstance(nodo, ast.Import):
            importados |= {a.name for a in nodo.names}
        elif isinstance(nodo, ast.Call):
            f = nodo.func
            nombre = getattr(f, "id", None) or getattr(f, "attr", None)
            if nombre:
                llamados.add(nombre)

    prohibidos = {"make_publish_signal", "publish_signal", "make_produce_observations"}
    cruzados = (importados | llamados) & prohibidos
    assert not cruzados, (
        f"este fichero importa o llama {sorted(cruzados)}: si de verdad cruza la "
        f"frontera de publicacion, hay que reescribir su docstring y la ficha de "
        f"BL-45, que la declaran ABIERTA"
    )
    assert "airflow" not in {i.split(".")[0] for i in importados}, (
        "este fichero importa airflow: entonces ya no es 'sin scheduler'"
    )
