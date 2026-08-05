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
