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


def test_declarative_ma200_matches_the_live_coded_policy_bar_by_bar() -> None:
    """Paridad sobre la serie real completa, **donde la media existe**.

    El warm-up se excluye a propósito y no por comodidad: ahí las dos implementaciones
    no son comparables, y el porqué está fijado en
    `test_the_warm_up_is_where_the_two_implementations_genuinely_disagree`.
    """
    from src.contracts.policy import PolicyContext
    from src.contracts.policy_dsl import DeclarativePolicy

    df = _load_spx()
    codificado = _coded_ma200(df)
    media = df["close"].rolling(200, min_periods=200).mean()
    politica = DeclarativePolicy(MA200_SPEC)

    divergencias, evaluadas = [], 0
    for i in range(len(df)):
        if media.iloc[i] != media.iloc[i]:  # NaN: warm-up, tramo no comparable
            continue
        evaluadas += 1
        snapshot = {"close": float(df["close"].iloc[i]), "ma_200": float(media.iloc[i])}
        decision = politica.evaluate(
            snapshot, PolicyContext(as_of=str(df["time"].iloc[i]), mode="DECISION")
        )
        if float(decision.target_exposure) != float(codificado.iloc[i]):
            divergencias.append(
                (str(df["time"].iloc[i]), float(codificado.iloc[i]),
                 float(decision.target_exposure))
            )

    assert evaluadas > 7000, f"sólo {evaluadas} barras comparadas: serie insuficiente"
    assert not divergencias, (
        f"{len(divergencias)} de {evaluadas} barras divergen entre el coded y el "
        f"declarativo; primeras: {divergencias[:5]}"
    )


def test_the_warm_up_is_where_the_two_implementations_genuinely_disagree() -> None:
    """Las 199 barras sin media NO son equivalentes, y el declarativo es el honesto.

    Hallazgo de esta paridad, no un ajuste del test:

    * el **coded** hace `(close > ma).astype(float)`. Como `close > NaN` es `False` en
      Python, las barras sin media salen `0.0` — es decir, "no hay dato" se convierte
      silenciosamente en "la política dice estar plano", que son cosas distintas y
      quedan indistinguibles en la serie.
    * el **declarativo** rechaza el `NaN` con `NaN/Infinity forbidden` y **falla
      cerrado**: se niega a decidir sin el dato.

    El criterio literal de BL-45 ("evalúa idéntico") **no se cumple aquí**, y forzarlo
    sería el error: haría falta enseñar al DSL a tragarse `NaN`, degradando la garantía
    fuerte para imitar el defecto de la implementación vieja. Cuál de las dos semánticas
    gobierna es decisión de gobierno; este candado impide que se resuelva por descuido.
    """
    from src.contracts.policy import PolicyContext
    from src.contracts.policy_dsl import DeclarativePolicy

    politica = DeclarativePolicy(MA200_SPEC)

    with pytest.raises(ValueError, match="not finite"):
        politica.evaluate(
            {"close": 4000.0, "ma_200": float("nan")},
            PolicyContext(as_of="1995-01-03", mode="DECISION"),
        )

    # Y el coded, en cambio, produce un 0.0 indistinguible de una decisión real.
    pd = pytest.importorskip("pandas")
    serie = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
    assert float(_coded_ma200(serie).iloc[0]) == 0.0


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
