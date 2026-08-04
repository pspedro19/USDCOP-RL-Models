"""BL-40 — el choque de namespaces que impide cablear la cuarentena.

`QualityRuleSet.evaluate_provider_bar` no tenía llamador productivo porque exigía un
`ProviderSymbolRegistry` y `reference.provider_symbol` estaba vacía. Poblada la espina
(BL-37), el guard ya es construible... y al probarlo contra datos reales aparece un
choque que lo bloquea:

* la migración 072 define `reference.instrument.instrument_id` como **UUID**, y
  `quality.quarantine_event.instrument_id` tiene FK a esa columna;
* pero `QualityRuleSet` usa ese mismo valor como **clave de `price_ranges`**, y ese YAML
  está indexado por *slugs* (`usdmxn`, `usdclp`).

Estos candados existen para que ese desacuerdo no se olvide ni se parchee a escondidas.
Van a ponerse **rojos** cuando alguien lo resuelva — y eso es lo que se busca: obligan a
revisitar este módulo en vez de dejar un guard muerto que aparenta cubrir la ingesta.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_quality.ingest_guard import QUALITY_RANGES  # noqa: E402

MIGRACION_072 = ROOT / "database" / "migrations" / "072_reference_identity.sql"
MIGRACION_073 = ROOT / "database" / "migrations" / "073_market_quality.sql"


def test_the_canonical_instrument_id_is_a_uuid() -> None:
    """La identidad canónica es un UUID generado, no un slug elegible a mano."""
    ddl = MIGRACION_072.read_text(encoding="utf-8")
    assert "instrument_id UUID PRIMARY KEY DEFAULT gen_random_uuid()" in ddl


def test_the_price_ranges_are_keyed_by_slugs_not_uuids() -> None:
    """El YAML de rangos usa slugs: ninguna clave puede ser un UUID."""
    config = yaml.safe_load(QUALITY_RANGES.read_text(encoding="utf-8")) or {}
    claves = list(config.get("price_ranges", {}))

    assert claves, "sin rangos declarados no hay nada que comparar"
    for clave in claves:
        assert "-" not in clave and len(clave) < 32, (
            f"'{clave}' parece un UUID: si los rangos ya se indexan por identidad "
            "canónica, este candado sobra y el guard de ingesta puede cablearse"
        )


def test_the_two_namespaces_still_disagree() -> None:
    """El candado que se pone rojo el día que esto se arregle. **Ese es su trabajo.**

    Mientras las claves de `price_ranges` y los `instrument_id` canónicos vivan en
    espacios distintos, cablear el guard pondría en cuarentena barras válidas. Cuando
    alguien alinee ambos, este test fallará y obligará a revisitar
    `src/data_quality/ingest_guard.py` para conectarlo de verdad.
    """
    config = yaml.safe_load(QUALITY_RANGES.read_text(encoding="utf-8")) or {}
    claves = set(config.get("price_ranges", {}))

    # Los activos de la espina, por su `asset_id`, tampoco aparecen como claves: los dos
    # únicos rangos declarados son de instrumentos que ni siquiera están en la espina.
    activos_espina = {"usdcop", "btcusdt", "xauusd", "spx500"}
    assert not (claves & activos_espina), (
        "ya hay rangos declarados para activos de la espina: revisa si el guard de "
        "ingesta puede cablearse (ver el docstring de ingest_guard.py)"
    )


def test_quarantine_event_points_at_the_canonical_instrument() -> None:
    """La cuarentena referencia la identidad canónica: el guard debe hablar ese idioma.

    Es lo que descarta la salida fácil de traducir UUID a slug dentro del guard: el
    evento que se escribe lleva FK al UUID, así que la traducción tendría que existir
    en las dos direcciones y ninguna capa la declara.
    """
    ddl = MIGRACION_073.read_text(encoding="utf-8")
    assert "instrument_id UUID REFERENCES reference.instrument(instrument_id)" in ddl
