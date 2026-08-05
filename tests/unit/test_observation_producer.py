# -*- coding: utf-8 -*-
"""`build_observations` — el productor que faltaba (BL-45 C2).

Cierra la brecha mayor que quedaba: **nadie producía `observations::<policy_id>`**.
La cadena gobernada los esperaba por XCom y ninguna tarea los ponía, así que el grafo
mostraba cuatro eslabones que ninguna corrida podía atravesar.

Lo que estos candados vigilan NO es la aritmética —de eso ya se encarga
`test_spx500_ma200_producer.py`— sino las tres formas en que un productor miente sin
fallar: publicar una feature que nadie declaró, publicar un valor que aún no existía,
y publicar un `NaN` como si fuera un número.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.features.observations import (  # noqa: E402
    PROVENANCE_RECONSTRUCTED,
    ObservationError,
    build_observations,
)

SEED = REPO / "seeds" / "latest" / "spx500_daily_ohlcv.parquet"
SPEC_PATH = REPO / "config" / "policies" / "spx500_daily_ma200_v1.yaml"
CUTOFF = "2026-07-29T00:00:00+00:00"


@pytest.fixture(scope="module")
def spec() -> dict:
    import yaml

    return yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    if not SEED.is_file():
        pytest.skip(f"falta el seed del índice oficial: {SEED}")
    return pd.read_parquet(SEED)


def test_it_publishes_exactly_what_the_feature_set_declares(spec, bars) -> None:
    """Ni una feature de más ni una de menos, y la lista NO se escribe aquí.

    Se compara contra las `ordered_features` del feature-set leídas aparte: si el
    productor tuviera su propia lista, sería una tercera fuente de verdad junto al
    set y el catálogo — exactamente lo que dejó a `ma_200` sin productor durante toda
    su vida.
    """
    import yaml

    fs = yaml.safe_load(
        (REPO / "config" / "features" / "feature_sets"
         / "spx500_daily_ma200_v1.yaml").read_text(encoding="utf-8")
    )
    declaradas = {o["feature_id"] for o in fs["ordered_features"]}

    obs = build_observations(spec, bars, decision_cutoff=CUTOFF)
    assert set(obs) == declaradas, (
        f"publicadas {sorted(obs)} vs declaradas {sorted(declaradas)}"
    )
    assert declaradas == {"close", "ma_200"}, (
        "el feature-set cambió: revisa este candado a conciencia en vez de ajustarlo"
    )


def test_every_observation_carries_a_causal_available_at(spec, bars) -> None:
    """`available_at <= decision_cutoff`, sin excepción.

    Es el invariante que `resolve_feature_snapshot` aplica aguas abajo; producirlo
    ya cumplido evita que la cadena entera muera en el eslabón siguiente por algo que
    el productor podía haber respetado.
    """
    obs = build_observations(spec, bars, decision_cutoff=CUTOFF)
    corte = pd.Timestamp(CUTOFF)
    for fid, o in obs.items():
        assert pd.Timestamp(o["available_at"]) <= corte, (
            f"{fid}: available_at={o['available_at']} excede el cutoff {CUTOFF}"
        )


def test_the_reconstructed_provenance_travels_with_every_value(spec, bars) -> None:
    """El `available_at` es RECONSTRUIDO y cada observación lo dice.

    No es cosmética. `load_real.py` declara para esta misma serie que el sello es
    "cierre + 1d, no vintage del proveedor" y que por eso el status máximo alcanzable
    es `research_validated`, nunca `production`. Esa limitación **viaja con el dato**:
    si se pierde al cambiar de capa, aguas abajo alguien leerá un timestamp exacto y
    creerá que es point-in-time real.

    Rojo con: quitar `provenance` del payload, o cambiarlo por una etiqueta que
    afirme vintage sin haberlo conseguido.
    """
    obs = build_observations(spec, bars, decision_cutoff=CUTOFF)
    assert obs, "sin observaciones no hay nada que juzgar"
    for fid, o in obs.items():
        assert o.get("provenance") == PROVENANCE_RECONSTRUCTED, (
            f"{fid}: provenance={o.get('provenance')!r}. Un sello reconstruido que se "
            f"presenta como vintage es una afirmación de causalidad que no se tiene"
        )


def test_a_naive_cutoff_is_rejected_instead_of_compared_by_luck(spec, bars) -> None:
    """Sin timezone no hay orden definido — es la regla de oro de este repo."""
    with pytest.raises(ObservationError, match="timezone"):
        build_observations(spec, bars, decision_cutoff="2026-07-29T00:00:00")


def test_insufficient_warmup_is_an_absence_not_an_invented_number(spec, bars) -> None:
    """Con menos de 200 sesiones, `ma_200` es NaN y NO se publica: se falla cerrado.

    Publicar el NaN sería peor que fallar: `float('nan')` sobrevive a un JSON mal
    hecho, atraviesa la cadena y compara `False` contra cualquier cosa, así que la
    policy tomaría una decisión "válida" sobre un valor que no existe. Quien juzga
    una ausencia es el `missing_input_policy` declarado, no el productor.
    """
    recorte = bars.sort_values("time").head(50)
    with pytest.raises(ObservationError, match="NaN"):
        build_observations(spec, recorte, decision_cutoff=CUTOFF)


def test_a_cutoff_before_the_first_bar_refuses_to_decide(spec, bars) -> None:
    """Ninguna barra disponible => error, jamás una decisión con datos del futuro."""
    with pytest.raises(ObservationError, match="disponible"):
        build_observations(spec, bars, decision_cutoff="1990-01-01T00:00:00+00:00")


def test_the_decision_bar_moves_with_the_cutoff(spec, bars) -> None:
    """Dos cutoffs distintos => barras de decisión distintas.

    Sin esto, el productor podría estar devolviendo siempre la última fila y pasaría
    todos los tests anteriores: publicaría el dato de hoy para un backfill de 2020 y
    nadie se enteraría. Es el candado que separa "respeta el corte" de "casualmente
    el corte es el final de la serie".
    """
    reciente = build_observations(spec, bars, decision_cutoff=CUTOFF)
    antiguo = build_observations(spec, bars, decision_cutoff="2020-06-15T00:00:00+00:00")
    assert antiguo["close"]["available_at"] < reciente["close"]["available_at"]
    assert antiguo["close"]["value"] != reciente["close"]["value"], (
        "el valor no cambió al mover el cutoff seis años: el productor no está "
        "respetando el corte, está devolviendo la última fila"
    )


def test_an_undeclared_feature_set_fails_closed(bars) -> None:
    """Una policy que apunta a un feature-set inexistente no tiene contrato de inputs."""
    roto = {"id": "x", "asset": "spx500", "inputs": {"feature_set_id": "no_existe_v9"}}
    with pytest.raises(ObservationError, match="no existe"):
        build_observations(roto, bars, decision_cutoff=CUTOFF)


def test_a_feature_outside_the_catalog_fails_closed(bars, monkeypatch) -> None:
    """Declarada en el set pero SIN catalogar => error, no un valor improvisado.

    Es el caso que originó todo esto: `ma_200` estaba en la práctica en esa
    situación. Si el productor la resolviera igualmente —adivinando la fórmula, o
    saltándosela— volveríamos a tener una feature sin contrato de causalidad ni
    `code_reference`, que es lo que C1 arregló.
    """
    import src.features.observations as mod

    real = mod._feature_set

    def _con_fantasma(fsid):
        fs = dict(real(fsid))
        fs["ordered_features"] = list(fs["ordered_features"]) + [
            {"feature_id": "fantasma_x", "order": 9, "required": True}
        ]
        return fs

    monkeypatch.setattr(mod, "_feature_set", _con_fantasma)
    spec = {
        "id": "spx500_daily_ma200_v1",
        "asset": "spx500",
        "inputs": {"feature_set_id": "spx500_daily_ma200_v1_action_v1"},
    }
    with pytest.raises(ObservationError, match="catálogo"):
        build_observations(spec, bars, decision_cutoff=CUTOFF)


def test_a_reconstructed_stamp_caps_the_status_at_research_validated(spec, bars) -> None:
    """CXD-620: el techo es EJECUTABLE, no una frase en un docstring.

    Un `available_at` derivado del cierre demuestra transporte real y causalidad
    declarada. Nada más. **No es observación de vintage**, así que no puede
    satisfacer ningún gate que exija evidencia point-in-time productiva, y ninguna
    promoción a `production` puede apoyarse en él.

    Se comprueba por código y no por prosa a propósito: con `window` (CXD-618)
    escribí la garantía en un docstring y no la implementé, y todo siguió verde.
    Una prohibición que sólo existe en un comentario no prohíbe nada.

    Rojo con: subir `MAX_STATUS_RECONSTRUCTED` a `production`, o hacer que
    `status_ceiling` acepte un sello desconocido devolviendo el techo bueno.
    """
    from src.features.observations import (
        FORBIDDEN_STATUSES_RECONSTRUCTED,
        MAX_STATUS_RECONSTRUCTED,
        status_ceiling,
    )

    assert MAX_STATUS_RECONSTRUCTED == "research_validated"
    assert MAX_STATUS_RECONSTRUCTED not in FORBIDDEN_STATUSES_RECONSTRUCTED
    assert {"production", "live"} <= FORBIDDEN_STATUSES_RECONSTRUCTED

    obs = build_observations(spec, bars, decision_cutoff=CUTOFF)
    for fid, o in obs.items():
        assert status_ceiling(o["provenance"]) == MAX_STATUS_RECONSTRUCTED, (
            f"{fid}: una observación con sello reconstruido no puede sostener un "
            f"estatus por encima de {MAX_STATUS_RECONSTRUCTED}"
        )


def test_an_unknown_stamp_is_rejected_instead_of_degraded_to_the_best_ceiling() -> None:
    """Un sello nuevo no hereda el techo del viejo: hay que declararlo.

    Sin esto, alguien podría inventar `provenance: "vintage_proveedor"` —como hice yo
    en la mutación M35— y el consumidor le daría el techo por defecto sin que nadie
    hubiera comprobado que hay vintage de verdad detrás. Antes de aceptar un sello
    nuevo hay que decir qué puede y qué no puede sostener.
    """
    from src.features.observations import ObservationError, status_ceiling

    with pytest.raises(ObservationError, match="sin techo declarado"):
        status_ceiling("vintage_proveedor")


def test_passthrough_publishes_the_column_named_by_the_feature_not_close(monkeypatch) -> None:
    """OHLC DISTINTOS: cada passthrough publica LO SUYO (CXD-620 §1).

    El productor hacía `serie = close` para todo `code_reference: null`. El catálogo
    declara `open`, `high` y `low` como passthrough para usdcop y los sets de
    smart_simple los ordenan, así que habría publicado el CIERRE bajo las identidades
    `open/high/low` — con su `series_id` propio, sin fallar, y sin forma de notarlo
    aguas abajo. No era un riesgo futuro: esas entradas existen hoy.

    La fixture usa cuatro valores DISTINTOS a propósito. Con OHLC iguales —que es lo
    que suele salir de un generador perezoso— este test pasaría con el bug puesto.

    Rojo con: volver a `serie = close` en la rama passthrough.
    """
    import src.features.observations as mod

    real = mod._feature_set

    def _ohlc(fsid):
        fs = dict(real("spx500_daily_ma200_v1_action_v1"))
        fs["ordered_features"] = [
            {"feature_id": f, "order": i, "required": True}
            for i, f in enumerate(("open", "high", "low", "close"))
        ]
        return fs

    monkeypatch.setattr(mod, "_feature_set", _ohlc)
    spec = {
        "id": "x",
        "asset": "usdcop",          # el único activo con open/high/low catalogados
        "inputs": {"feature_set_id": "usdcop_smart_simple_v11_recipe25"},
    }
    bars = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-07-20", "2026-07-21"], utc=True),
            "open": [10.0, 11.0],
            "high": [30.0, 31.0],
            "low": [1.0, 2.0],
            "close": [20.0, 21.0],
        }
    )
    obs = build_observations(spec, bars, decision_cutoff=CUTOFF)
    assert {k: v["value"] for k, v in obs.items()} == {
        "open": 11.0, "high": 31.0, "low": 2.0, "close": 21.0
    }, f"un passthrough publicó el valor de otra columna: {obs}"


def test_a_passthrough_without_its_column_fails_closed(monkeypatch) -> None:
    """Y si la columna no llega, es error — nunca una sustitución silenciosa."""
    import src.features.observations as mod

    real = mod._feature_set

    def _solo_open(fsid):
        fs = dict(real("spx500_daily_ma200_v1_action_v1"))
        fs["ordered_features"] = [{"feature_id": "open", "order": 0, "required": True}]
        return fs

    monkeypatch.setattr(mod, "_feature_set", _solo_open)
    spec = {"id": "x", "asset": "usdcop",
            "inputs": {"feature_set_id": "usdcop_smart_simple_v11_recipe25"}}
    bars = pd.DataFrame(
        {"time": pd.to_datetime(["2026-07-21"], utc=True), "close": [20.0]}
    )
    with pytest.raises(ObservationError, match="no trae la columna"):
        build_observations(spec, bars, decision_cutoff=CUTOFF)


def test_a_live_status_cannot_be_backed_by_reconstructed_evidence(spec, bars) -> None:
    """El techo se APLICA, no sólo se declara (CXD-620 §2).

    C2b puso constantes y una función consultable, y **nadie la consultaba**: la misma
    observación reconstruida seguía atravesando `publish` si la policy se volvía
    elegible. Una función que puede preguntarse pero no se pregunta no prohíbe nada —
    es la forma exacta del error del parámetro `window`.

    `CUTOVER` es el único `migration.status` que significa "esta ES la vía viva", así
    que reclama `production`; con sello reconstruido, se bloquea.

    Rojo con: quitar la llamada del eslabón `publish`, o mapear `CUTOVER` a
    `research_validated` para que deje de doler.
    """
    from src.features.observations import assert_observations_support_status

    obs = build_observations(spec, bars, decision_cutoff=CUTOFF)

    # PARITY_GREEN reclama investigación validada: la evidencia reconstruida basta.
    assert_observations_support_status(obs, migration_status="PARITY_GREEN")

    # Se afirma sobre "no sostiene una via viva" y no sobre la palabra RECONSTRUIDO:
    # desde C2d el mensaje habla del techo MINIMO y de quien lo impone, porque el
    # gate ya no compara contra una constante sino que pregunta al techo por cada
    # sello. Anclar un candado a una palabra concreta del mensaje lo vuelve fragil
    # ante mejoras del propio mensaje.
    with pytest.raises(ObservationError, match="no sostiene una via viva"):
        assert_observations_support_status(obs, migration_status="CUTOVER")


def test_an_unmapped_migration_status_is_rejected_not_waved_through(spec, bars) -> None:
    """Un estado nuevo no pasa por defecto: hay que declarar qué evidencia reclama.

    "Por defecto lo permisivo" es como se cuelan los estados nuevos sin revisar; y el
    coste de equivocarse aquí es publicar una señal viva sobre evidencia que no la
    sostiene.
    """
    from src.features.observations import assert_observations_support_status

    obs = build_observations(spec, bars, decision_cutoff=CUTOFF)
    with pytest.raises(ObservationError, match="sin reclamo declarado"):
        assert_observations_support_status(obs, migration_status="ESTADO_INVENTADO")


def test_an_invented_provenance_cannot_sail_through_a_live_status() -> None:
    """El bypass que encontró CODEX (CXD-622), cerrado y con candado.

    El gate comparaba contra la constante reconstruida
    (`if provenance == PROVENANCE_RECONSTRUCTED`), así que **cualquier sello
    inventado atravesaba `CUTOVER`**: bastaba escribir `provenance:
    "vintage_proveedor"` y la señal salía. Reproducido: devolvía `None`.

    Lo hiriente es que `status_ceiling` ya era fail-closed ante un sello desconocido
    —escrita en C2b— y el gate, que nació para APLICAR el techo, no la llamaba.
    Tercera vez en esta serie que escribo el mecanismo y no lo consulto. Ahora la
    autoridad es la función: una segunda forma de decidir lo mismo es una segunda
    forma de equivocarse.

    Rojo con: volver a comparar contra la constante en vez de preguntar al techo.
    """
    from src.features.observations import assert_observations_support_status

    inventado = {"close": {"value": 1.0, "available_at": "2026-07-28T20:00:00+00:00",
                           "provenance": "vintage_proveedor"}}
    # Ni siquiera para el estatus benigno: un sello que nadie registró no se juzga.
    for estado in ("CUTOVER", "PARITY_GREEN"):
        with pytest.raises(ObservationError, match="sin techo declarado"):
            assert_observations_support_status(inventado, migration_status=estado)


def test_mixed_provenances_take_the_WORST_ceiling_not_the_best(monkeypatch) -> None:
    """Con sellos mezclados manda el PEOR techo.

    Sin esto, una feature bien sellada legitimaría a las demás: bastaría con que UNA
    tuviera vintage real para que el conjunto —reconstruidas incluidas— pasara a
    `production`. El techo de un snapshot es el de su evidencia más débil, igual que
    su frescura es la de su observación más vieja (CXD-603).

    El segundo sello se inyecta en la AUTORIDAD por monkeypatch en vez de añadir una
    constante `PROVENANCE_OBSERVED` al código productivo: hoy nada produce vintage
    real, y meter una constante que nadie produce sería otra vez un mecanismo sin
    llamador — justo el patrón que este fichero lleva tres rondas corrigiendo.
    """
    import src.features.observations as mod

    real = mod.status_ceiling

    def _con_vintage(prov: str) -> str:
        return "production" if prov == "vintage_de_prueba" else real(prov)

    monkeypatch.setattr(mod, "status_ceiling", _con_vintage)

    mezcla = {
        "close": {"value": 1.0, "available_at": "2026-07-28T20:00:00+00:00",
                  "provenance": "vintage_de_prueba"},          # techo production
        "ma_200": {"value": 1.0, "available_at": "2026-07-28T20:00:00+00:00",
                   "provenance": PROVENANCE_RECONSTRUCTED},    # techo research_validated
    }
    # Sólo-vintage sí sostiene una vía viva…
    mod.assert_observations_support_status(
        {"close": mezcla["close"]}, migration_status="CUTOVER"
    )
    # …pero en cuanto entra una reconstruida, el conjunto no.
    with pytest.raises(ObservationError, match="MINIMO"):
        mod.assert_observations_support_status(mezcla, migration_status="CUTOVER")


def test_an_unknown_producer_contract_fails_closed_at_the_resolver(monkeypatch) -> None:
    """Un contrato de invocación desconocido NO se degrada al de por defecto.

    Lo encontré midiendo mis propias mutaciones: quitar la guarda de
    `CONTRATOS_SOPORTADOS` dejaba **toda la suite verde**, porque con un contrato
    válido declarado la guarda nunca se ejerce. Era una guarda sin candado — el mismo
    patrón (mecanismo escrito, nadie lo prueba) que ya costó tres rechazos en este
    fichero, esta vez en forma de rama no cubierta.

    El validador del catálogo también lo rechaza, pero eso protege el YAML; esto
    protege el RESOLVER, que es quien acabaría invocando al productor de la forma
    equivocada si alguien construyera una entrada en memoria.

    Rojo con: quitar la guarda, o hacer que caiga al contrato de Series por defecto.
    """
    import src.features.observations as mod

    real = mod._catalog_index

    def _con_contrato_raro():
        idx = dict(real())
        fila = dict(idx[("spx500", "ma_200")])
        fila["producer_contract"] = "convencion_inventada_v9"
        idx[("spx500", "ma_200")] = fila
        return idx

    monkeypatch.setattr(mod, "_catalog_index", _con_contrato_raro)
    spec = {
        "id": "spx500_daily_ma200_v1", "asset": "spx500",
        "inputs": {"feature_set_id": "spx500_daily_ma200_v1_action_v1"},
    }
    bars = pd.DataFrame({
        "time": pd.to_datetime(["2026-07-20", "2026-07-21"], utc=True),
        "open": [1.0, 2.0], "high": [3.0, 4.0], "low": [0.5, 0.6], "close": [2.0, 3.0],
    })
    with pytest.raises(ObservationError, match="no soportado"):
        build_observations(spec, bars, decision_cutoff=CUTOFF)
