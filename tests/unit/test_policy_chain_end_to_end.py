# -*- coding: utf-8 -*-
"""La cadena, atravesada de punta a punta con datos REALES (BL-45).

QUÉ PRUEBA Y QUÉ NO
-------------------
Prueba que `produce_observations → resolve_snapshot → validate_inputs → evaluate`
se **atraviesa entera** partiendo del índice oficial en disco y llegando a una
decisión con su `rule_trace`. Cada eslabón consume literalmente lo que produjo el
anterior: nada se fabrica a mano por el camino, que es la diferencia entre probar la
cadena y probar cuatro funciones que casualmente encajan.

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


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    if not SEED.is_file():
        pytest.skip(f"falta el seed del índice oficial: {SEED}")
    return pd.read_parquet(SEED)


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
