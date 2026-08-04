"""BL-40 — la cuarentena habla identidad canónica sin que nadie invente la traducción.

`QualityRuleSet.evaluate_provider_bar` no tenía llamador porque exigía un
`ProviderSymbolRegistry` y `reference.provider_symbol` estaba vacía. Poblada la espina
(BL-37), el guard es construible — y aparece un choque: el evaluador indexa
`price_ranges` por el `instrument_id` del registry, que es un **UUID**, mientras el YAML
de rangos los indexa por **asset_id**.

Parecía una decisión de contrato. No lo era: la traducción ya está **declarada** en dos
sitios (la espina mapea `instrument → asset`; cada activo declara su `price_range`), así
que re-clavar los rangos es derivación, no heurística — y no hace falta tocar `rules.py`.

Estos candados fijan justo eso: que la traducción salga de lo declarado, que el fallo
siga siendo cerrado, y que el límite conocido (`usdmxn`/`usdclp` sin perfil) quede
escrito en vez de descubrirse otra vez dentro de seis semanas.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_quality.ingest_guard import ASSETS_DIR, QUALITY_RANGES  # noqa: E402

MIGRACION_072 = ROOT / "database" / "migrations" / "072_reference_identity.sql"
MIGRACION_073 = ROOT / "database" / "migrations" / "073_market_quality.sql"

#: Activos de la espina, es decir los que tienen `AssetProfile`.
ACTIVOS_ESPINA = ("usdcop", "btcusdt", "xauusd", "spx500")


def test_the_canonical_instrument_id_is_a_uuid() -> None:
    """La identidad canónica es un UUID generado, no un slug elegible a mano."""
    ddl = MIGRACION_072.read_text(encoding="utf-8")
    assert "instrument_id UUID PRIMARY KEY DEFAULT gen_random_uuid()" in ddl


def test_quarantine_event_points_at_the_canonical_instrument() -> None:
    """La cuarentena referencia el UUID: por eso el registry NO puede devolver slugs.

    Es lo que descarta la salida fácil —hacer que el registry resuelva a `asset_id` para
    que encajen los rangos—: el evento que se escribe lleva FK al UUID.
    """
    ddl = MIGRACION_073.read_text(encoding="utf-8")
    assert "instrument_id UUID REFERENCES reference.instrument(instrument_id)" in ddl


def test_every_spine_asset_declares_its_own_price_range() -> None:
    """La fuente de los rangos es el SSOT del activo, no una tabla aparte.

    Sin esta declaración la traducción no existiría y habría que inventar un rango, que
    es exactamente lo que la espina prohíbe.
    """
    for asset_id in ACTIVOS_ESPINA:
        declarado = yaml.safe_load(
            (ASSETS_DIR / f"{asset_id}.yaml").read_text(encoding="utf-8")
        )
        rango = declarado.get("price_range")
        assert isinstance(rango, list) and len(rango) == 2, (
            f"{asset_id} no declara price_range: sus barras no podrían evaluarse sin "
            "inventarle un rango"
        )
        bajo, alto = float(rango[0]), float(rango[1])
        assert 0 < bajo < alto, f"{asset_id}: price_range {rango} no ordena"


def test_the_translation_is_declared_not_invented() -> None:
    """`asset_id → instrument_id` la declara la espina; el guard sólo la lee.

    Se comprueba contra la base cuando la hay: cada activo con perfil tiene exactamente
    un instrumento canónico, así que la traducción es una función, no una elección.
    """
    try:
        from scripts.data.ingest_asset_ohlcv import _db_conn

        conn = _db_conn()
    except Exception:  # pragma: no cover - CI sin base de datos
        pytest.skip("sin base de datos: la traducción se verifica en entorno con DB")

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT asset_id, COUNT(*) FROM reference.instrument GROUP BY 1")
            por_activo = dict(cur.fetchall())
    finally:
        conn.close()

    for asset_id in ACTIVOS_ESPINA:
        assert por_activo.get(asset_id) == 1, (
            f"{asset_id} tiene {por_activo.get(asset_id)} instrumentos canónicos: la "
            "traducción dejaría de ser una función y habría que elegir, es decir inventar"
        )


def test_a_bar_outside_the_declared_range_is_quarantined_and_a_good_one_is_not() -> None:
    """La propiedad que BL-40 pide: el evaluador ya discrimina de verdad.

    Antes de re-clavar los rangos, esta misma llamada devolvía `bar.unknown_instrument`
    para una barra perfectamente válida — el guard habría puesto en cuarentena el 100%
    de la ingesta.
    """
    from datetime import datetime, timezone

    try:
        from scripts.data.ingest_asset_ohlcv import _db_conn

        from src.data_quality.ingest_guard import ruleset_from_spine

        conn = _db_conn()
    except Exception:  # pragma: no cover - CI sin base de datos
        pytest.skip("sin base de datos: el guard se verifica en entorno con DB")

    try:
        reglas = ruleset_from_spine(conn)
    finally:
        conn.close()

    momento = datetime(2026, 1, 5, tzinfo=timezone.utc)
    buena = {"time": momento, "open": 4000, "high": 4010, "low": 3990, "close": 4005, "volume": 10}
    fuera = {"time": momento, "open": 99999, "high": 99999, "low": 99999, "close": 99999, "volume": 10}

    assert reglas.evaluate_provider_bar("twelvedata", "USD/COP", buena, observed_at=momento).accepted
    decision = reglas.evaluate_provider_bar("twelvedata", "USD/COP", fuera, observed_at=momento)
    assert not decision.accepted and decision.rule_id.startswith("bar.range.")

    # Y el alias no registrado sigue muriendo: fail-closed intacto.
    desconocida = reglas.evaluate_provider_bar("nadie", "XXX/YYY", buena, observed_at=momento)
    assert not desconocida.accepted and desconocida.rule_id == "bar.unknown_alias"


def test_a_symbol_with_a_scoped_rule_never_gets_the_flat_range_too() -> None:
    """El escalón **excluye** al plano (enmienda C026), y no sólo lo precede.

    Este candado sustituye al canario que avisaba de que `usdmxn` no tenía perfil: el
    perfil ya existe, así que el riesgo cambió de "no se evalúa" a "se evalúa con la
    regla equivocada".

    `AssetProfile` **exige** `price_range`, de modo que el perfil auxiliar de USD/MXN
    declara uno. Admitirlo en el mapa plano reintroduciría el aplanado por la puerta de
    atrás: hoy el evaluador da precedencia al escalón, pero el día que alguien retirara
    el mapa escalonado el plano tomaría el relevo **en silencio** y una barra de 1990 se
    validaría contra el rango de hoy. Excluyéndolo, esa retirada falla cerrado.
    """
    from src.data_quality.ingest_guard import (
        declared_ranges_by_canonical_symbol,
        declared_scoped_ranges_by_canonical_symbol,
        scoped_symbols,
    )

    escalonados = declared_scoped_ranges_by_canonical_symbol()
    assert escalonados, "el YAML normativo ya no declara ningún rango escalonado"

    planos = set(declared_ranges_by_canonical_symbol())
    for nombre in scoped_symbols():
        assert nombre not in planos, (
            f"{nombre} tiene regla escalonada Y entrada plana: si se retira el escalón, "
            "el plano validaría en silencio fuera de su alcance"
        )


def test_the_scoped_rule_keeps_its_provider_and_cutoff() -> None:
    """El escalón conserva proveedor y `valid_from`: es lo único que lo hace escalón.

    El corte de Banxico (SIE CF373, 1993-01-01) marca un cambio de unidad monetaria.
    Aplanarlo a un `(low, high)` sin fecha no es una simplificación: borra el hecho
    económico que la regla codifica.
    """
    from src.data_quality.ingest_guard import declared_scoped_ranges_by_canonical_symbol

    reglas = declared_scoped_ranges_by_canonical_symbol()["usdmxn"]
    assert len(reglas) >= 1
    for regla in reglas:
        assert regla.provider_id, "un escalón sin proveedor aplica a cualquiera: no es un escalón"
        assert regla.valid_from.tzinfo is not None, "'valid_from' sin zona horaria es ambiguo"
        assert regla.low < regla.high


def test_a_bar_outside_the_scope_is_quarantined_not_silently_accepted() -> None:
    """Fuera del alcance —otro proveedor o antes del corte— la barra NO pasa.

    Es la propiedad que el aplanado destruía. Se ejercita contra el evaluador real con
    el mapa escalonado construido por el guard, sin base de datos: la traducción a UUID
    se prueba aparte.
    """
    from datetime import datetime, timezone

    from src.data_quality.ingest_guard import declared_scoped_ranges_by_canonical_symbol
    from src.data_quality.rules import QualityRuleSet

    reglas = QualityRuleSet(
        price_ranges={},
        scoped_price_ranges={"mxn": declared_scoped_ranges_by_canonical_symbol()["usdmxn"]},
    )
    barra = {
        "time": datetime(2020, 1, 5, tzinfo=timezone.utc),
        "open": 20, "high": 21, "low": 19, "close": 20, "volume": 1,
    }

    dentro = reglas._evaluate_bar(
        "mxn", barra, provider_id="twelvedata",
        observed_at=datetime(2020, 1, 5, tzinfo=timezone.utc),
    )
    assert dentro.accepted, "una barra dentro del alcance y del rango debería pasar"

    antes = reglas._evaluate_bar(
        "mxn", barra, provider_id="twelvedata",
        observed_at=datetime(1990, 1, 5, tzinfo=timezone.utc),
    )
    assert not antes.accepted and antes.rule_id == "bar.range_scope"

    otro = reglas._evaluate_bar(
        "mxn", barra, provider_id="otro_proveedor",
        observed_at=datetime(2020, 1, 5, tzinfo=timezone.utc),
    )
    assert not otro.accepted and otro.rule_id == "bar.range_scope"

    sin_contexto = reglas._evaluate_bar("mxn", barra, provider_id=None, observed_at=None)
    assert not sin_contexto.accepted and sin_contexto.rule_id == "bar.range_context", (
        "sin proveedor ni instante no se puede saber qué escalón aplica: debe fallar cerrado"
    )


def test_a_known_range_rejection_keeps_its_canonical_fk() -> None:
    """El evento de cuarentena NO puede perder el instrumento que sí se resolvió.

    Defecto real que midió Codex (CXD-457): la primera versión sacaba el UUID de
    `decision.observed_value`, que en un rechazo de rango es `{'open': '99999'}` — el
    valor que ofendió, no la identidad. El evento quedaba con FK nula justo cuando el
    alias se había resuelto perfectamente, dejando la cuarentena huérfana del catálogo.

    Y hay una trampa que este candado también cierra: el `rule_id` **contiene** el UUID
    (`bar.range.72f6f7e9-...`). Sacarlo de ahí sería recuperar identidad parseando una
    cadena de diagnóstico que nadie prometió estable. Se pregunta al registry.
    """
    from datetime import datetime, timezone

    try:
        from scripts.data.ingest_asset_ohlcv import _db_conn

        from src.data_quality.ingest_guard import resolved_instrument_id, ruleset_from_spine

        conn = _db_conn()
    except Exception:  # pragma: no cover - CI sin base de datos
        pytest.skip("sin base de datos: la FK canónica se verifica en entorno con DB")

    try:
        reglas = ruleset_from_spine(conn)
    finally:
        conn.close()

    momento = datetime(2026, 1, 5, tzinfo=timezone.utc)
    fuera = {"time": momento, "open": 99999, "high": 99999, "low": 99999, "close": 99999, "volume": 1}

    decision = reglas.evaluate_provider_bar("twelvedata", "USD/COP", fuera, observed_at=momento)
    assert not decision.accepted and decision.rule_id.startswith("bar.range.")

    uuid_resuelto = resolved_instrument_id(reglas, "twelvedata", "USD/COP")
    assert uuid_resuelto, (
        "rango conocido sin instrument_id: el evento se escribiría con FK nula y la "
        "cuarentena quedaría huérfana del catálogo canónico"
    )
    # La identidad NO sale del diagnóstico...
    assert not isinstance(decision.observed_value, dict) or "instrument_id" not in decision.observed_value
    # ...y un alias no registrado sí puede quedar nulo: ahí el hecho ES la ausencia.
    assert resolved_instrument_id(reglas, "nadie", "XXX/YYY") is None
