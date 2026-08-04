"""BL-45 — el spec MA200 declarativo debe decidir IGUAL que el `coded_policy` vivo.

Es un criterio literal de la ficha (*"spec MA200 declarativo evalúa idéntico al
coded_policy actual"*) y el que decide si el DSL sirve para algo: un motor declarativo
que no reproduce la política que ya corre no es una migración, es una segunda política.

El coded vivo es `src/strategies/spx500_regime_gated_v1/benchmarks.py::_ma200`:

    ma = close.rolling(200, min_periods=200).mean()
    return (close > ma).astype(float)

Lo que hace interesante esta paridad no es el caso central —que un cierre por encima de
su media da exposición 1.0— sino los bordes, que es donde dos implementaciones "obvias"
divergen:

* **el warm-up**: las primeras 199 barras no tienen media. El coded las convierte en
  `0.0` porque `close > NaN` es `False` y `astype(float)` lo aplana; un declarativo que
  tratara el `NaN` como "dato ausente → error" divergiría exactamente ahí, en el 6% de
  la serie que nadie mira.
* **la igualdad estricta**: `close == ma` debe dar FLAT en ambos. `greater_than` y `>=`
  se comportan igual el 99,9% de las barras y distinto justo en la que importa.

Por eso la comparación se hace **barra a barra sobre la serie real completa**, no sobre
tres casos escogidos: los casos escogidos son precisamente los que uno no sabe elegir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED_SPX = ROOT / "seeds" / "latest" / "spx500_daily_ohlcv.parquet"

#: El mismo spec declarativo que vive en la documentación del DSL
#: (`src/contracts/policy_dsl.py`), no una copia inventada para el test.
MA200_SPEC = {
    "id": "spx500_daily_ma200_v1",
    "version": "2.0.0",
    "policy_hash": "sha256:" + "ab" * 32,
    "resolution": {
        "mode": "first_match",
        "default_target_exposure": 0.0,
        "default_direction": "FLAT",
        "default_reason_code": "CLOSE_BELOW_MA200",
        # Fallback DECLARADO para el warm-up (CXD-462): la MA de 200 sesiones no existe
        # hasta la barra 200. Declararlo conserva exactamente la conducta del coded
        # vivo, pero como afirmación auditable en vez de un `0.0` aritmético.
        "feature_fallbacks": {
            "ma_200": {
                "direction": "FLAT",
                "target_exposure": 0.0,
                "reason_code": "MA200_WARMUP_NO_SIGNAL",
            }
        },
    },
    "rules": [
        {
            "id": "trend_on",
            "label": "Precio sobre MA200",
            "priority": 100,
            "when": {
                "operator": "greater_than",
                "left": "feature.close",
                "right": "feature.ma_200",
            },
            "output": {
                "direction": "LONG",
                "target_exposure": 1.0,
                "reason_code": "CLOSE_ABOVE_MA200",
            },
        }
    ],
}


def _coded_ma200(df):
    """El benchmark productivo, importado — no reimplementado.

    Reimplementarlo aquí convertiría el test en "mi lectura del coded coincide con mi
    declarativo", que no prueba nada sobre el código que corre.
    """
    from src.strategies.spx500_regime_gated_v1.benchmarks import _ma200

    return _ma200(df)


def _load_spx():
    pd = pytest.importorskip("pandas")
    if not SEED_SPX.is_file():
        pytest.skip(f"sin seed SPX real ({SEED_SPX.name}): la paridad exige datos vivos")
    df = pd.read_parquet(SEED_SPX)
    columnas = {c.lower(): c for c in df.columns}
    if "close" not in columnas:
        pytest.skip(f"el seed SPX no expone 'close': {list(df.columns)}")
    return df.rename(columns={columnas["close"]: "close"})


def _decisiones(spec, df, media):
    """Exposición declarativa barra a barra, incluido el warm-up."""
    from src.contracts.policy import PolicyContext
    from src.contracts.policy_dsl import DeclarativePolicy

    politica = DeclarativePolicy(spec)
    for i in range(len(df)):
        valor = float(media.iloc[i])
        snapshot = {"close": float(df["close"].iloc[i]), "ma_200": valor}
        yield i, politica.evaluate(
            snapshot, PolicyContext(as_of=str(df["time"].iloc[i]), mode="DECISION")
        )


def test_declarative_ma200_matches_the_live_coded_policy_on_every_single_bar() -> None:
    """Paridad **total** sobre la serie real, warm-up incluido (CXD-462).

    El fallback declarado es lo que hace comparable el tramo inicial. Sin él las dos
    implementaciones no eran equivalentes ahí, y la diferencia no era cosmética: el
    coded convertía "no hay media" en exposición `0.0`.
    """
    df = _load_spx()
    codificado = _coded_ma200(df)
    media = df["close"].rolling(200, min_periods=200).mean()

    divergencias = [
        (str(df["time"].iloc[i]), float(codificado.iloc[i]), float(d.target_exposure))
        for i, d in _decisiones(MA200_SPEC, df, media)
        if float(d.target_exposure) != float(codificado.iloc[i])
    ]

    assert len(df) == 7943, f"el seed cambió de tamaño ({len(df)}): re-verifica la paridad"
    assert not divergencias, (
        f"{len(divergencias)} de {len(df)} barras divergen; primeras: {divergencias[:5]}"
    )


def test_removing_the_declared_fallback_fails_closed_during_warm_up() -> None:
    """Retirar el fallback debe **romper**, no volver a decidir en silencio.

    Es el candado que convierte la paridad en una afirmación gobernada: sin la
    declaración, el DSL se niega a decidir sin `ma_200` — que es su conducta correcta —
    y el tramo de warm-up deja de existir en vez de rellenarse con un `0.0` plausible.
    """
    from src.contracts.policy import PolicyContext
    from src.contracts.policy_dsl import DeclarativePolicy

    sin_fallback = {
        **MA200_SPEC,
        "resolution": {
            k: v for k, v in MA200_SPEC["resolution"].items() if k != "feature_fallbacks"
        },
    }
    politica = DeclarativePolicy(sin_fallback)

    with pytest.raises(ValueError, match="not finite"):
        politica.evaluate(
            {"close": 4000.0, "ma_200": float("nan")},
            PolicyContext(as_of="1995-01-03", mode="DECISION"),
        )


def test_the_fallback_is_local_and_does_not_relax_the_engine() -> None:
    """Otra política SIN fallback sigue rechazando `NaN`: no se tocó el default global.

    Es el límite estricto que pidió CXD-462. Un fallback global habría convertido una
    excepción declarada para un caso en una laxitud para todos, que es exactamente cómo
    una garantía fuerte se erosiona sin que nadie decida erosionarla.
    """
    from src.contracts.policy import PolicyContext
    from src.contracts.policy_dsl import DeclarativePolicy

    otra = {
        **MA200_SPEC,
        "id": "otra_politica_v1",
        "policy_hash": "sha256:" + "cd" * 32,
        "resolution": {
            k: v for k, v in MA200_SPEC["resolution"].items() if k != "feature_fallbacks"
        },
    }

    with pytest.raises(ValueError, match="not finite"):
        DeclarativePolicy(otra).evaluate(
            {"close": 1.0, "ma_200": float("inf")},
            PolicyContext(as_of="2026-01-05", mode="DECISION"),
        )


def test_a_fallback_decision_is_marked_as_such_and_never_leaks_a_nan() -> None:
    """La decisión por ausencia se distingue de una decisión por regla, y es exportable.

    Dos propiedades en una: `fallback_applied` con su `reason_code` propio impide
    confundir "faltaba el dato" con "ninguna regla disparó"; y la traza registra `null`
    en vez del `NaN` crudo, porque los contratos de export prohíben `NaN`/`Infinity` —
    colarlo por la traza habría reintroducido justo lo que el DSL rechaza por la entrada.
    """
    from src.contracts.policy import PolicyContext
    from src.contracts.policy_dsl import DeclarativePolicy

    decision = DeclarativePolicy(MA200_SPEC).evaluate(
        {"close": 4000.0, "ma_200": float("nan")},
        PolicyContext(as_of="1995-01-03", mode="DECISION"),
    )

    assert float(decision.target_exposure) == 0.0
    assert decision.direction == "FLAT"
    assert decision.reason_codes == ("MA200_WARMUP_NO_SIGNAL",)
    assert decision.rule_trace.fallback_applied is True
    assert decision.decision_components == {"ma_200": None}
    assert decision.rule_trace.rules[0].observed == {"ma_200": None}


def test_a_corrupt_value_is_not_treated_as_an_absence() -> None:
    """`"abc"` o `True` NO activan el fallback: no son ausencia, son contrato roto.

    La distinción es el filo de todo el mecanismo. Un fallback que se tragara valores
    corruptos volvería a fabricar decisiones sobre datos basura — con la agravante de
    que ahora llevarían un `reason_code` que las hace parecer deliberadas.
    """
    from src.contracts.policy import PolicyContext
    from src.contracts.policy_dsl import DeclarativePolicy

    politica = DeclarativePolicy(MA200_SPEC)
    for corrupto in ("abc", True, None):
        with pytest.raises(ValueError):
            politica.evaluate(
                {"close": 4000.0, "ma_200": corrupto},
                PolicyContext(as_of="1995-01-03", mode="DECISION"),
            )


def test_equality_is_not_above_the_average() -> None:
    """`close == ma_200` es FLAT en los dos lados.

    `greater_than` frente a `>=` coinciden en casi todas las barras y difieren justo en
    la que decide; fijarlo evita que una futura "simplificación" del operador cambie la
    política sin que nada se ponga rojo.
    """
    from src.contracts.policy import PolicyContext
    from src.contracts.policy_dsl import DeclarativePolicy

    politica = DeclarativePolicy(MA200_SPEC)
    decision = politica.evaluate(
        {"close": 4000.0, "ma_200": 4000.0},
        PolicyContext(as_of="2026-01-05", mode="DECISION"),
    )

    assert float(decision.target_exposure) == 0.0, "la igualdad no es 'por encima'"


def test_the_same_input_always_yields_the_same_decision() -> None:
    """Determinismo, el otro criterio de verificación de la ficha.

    Se evalúa dos veces con el mismo snapshot y se exige idéntica exposición, dirección
    y `policy_hash`: si la política dependiera de estado oculto o de la hora, aquí se
    vería.
    """
    from src.contracts.policy import PolicyContext
    from src.contracts.policy_dsl import DeclarativePolicy

    politica = DeclarativePolicy(MA200_SPEC)
    snapshot = {"close": 4200.0, "ma_200": 4100.0}
    contexto = PolicyContext(as_of="2026-01-05", mode="DECISION")

    primera = politica.evaluate(snapshot, contexto)
    segunda = politica.evaluate(snapshot, contexto)

    assert (primera.target_exposure, primera.direction) == (
        segunda.target_exposure,
        segunda.direction,
    )
    assert DeclarativePolicy(MA200_SPEC).policy_hash == politica.policy_hash
