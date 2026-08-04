"""BL-40 + BL-17 — la espina de identidad se DERIVA de los SSOT, nunca se inventa.

`reference.provider_symbol` (BL-40) no podía poblarse porque exigía `instrument`, que
exigía `asset`, que exigía `calendar`, y las cuatro estaban vacías: BL-40 y BL-17 son
el mismo trabajo. Poblarlas es fácil; poblarlas **sin inventar nada** es el punto.

Estos candados atacan el modo de fallo real de un seed de identidad: que alguien
rellene un hueco con un valor plausible. Un `calendar_id` inventado o un `provider` que
nunca escribió una fila son peores que la tabla vacía, porque la tabla vacía no miente.
"""

from __future__ import annotations

import copy
import dataclasses
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.seed_reference_spine import (  # noqa: E402
    INSTRUMENT_TYPE_BY_CLASS,
    ProviderFact,
    SpineError,
    asset_rows,
    calendar_id_for,
    calendar_rows,
    declared_assets,
    instrument_rows,
    split_facts,
)

ASSETS_DIR = ROOT / "config" / "assets"


def _asset_dir_con(tmp_path: Path, asset_id: str, mutar) -> Path:
    """Copia un SSOT real a un directorio temporal aplicando `mutar` al dict."""
    raw = yaml.safe_load((ASSETS_DIR / f"{asset_id}.yaml").read_text(encoding="utf-8"))
    mutado = mutar(copy.deepcopy(raw))
    destino = tmp_path / f"{asset_id}.yaml"
    destino.write_text(yaml.safe_dump(mutado, sort_keys=False), encoding="utf-8")
    return tmp_path


# ------------------------------------------------------------------ derivación real


def test_every_asset_field_comes_from_the_declared_profile() -> None:
    """Ningún campo de `reference.asset` es una elección del script.

    Se compara fila a fila contra el YAML: si alguien introdujera un default "razonable"
    en el seed, este candado lo vería como divergencia con el SSOT.
    """
    perfiles = declared_assets(ASSETS_DIR)
    filas = {f["asset_id"]: f for f in asset_rows(perfiles)}

    for asset_id, perfil in perfiles.items():
        declarado = yaml.safe_load(
            (ASSETS_DIR / f"{asset_id}.yaml").read_text(encoding="utf-8")
        )
        fila = filas[asset_id]
        assert fila["display_name"] == declarado["display_name"]
        assert fila["asset_class"] == declarado["asset_class"]
        assert fila["quote_currency"] == declarado["quote_ccy"]
        # La anualización sale del perfil, no de un 252 de conveniencia.
        assert fila["annualization"] == perfil.session.trading_days_per_year


def test_the_calendar_identity_is_derived_not_minted(tmp_path: Path) -> None:
    """Dos activos con el mismo modo y zona COMPARTEN calendario.

    Es la propiedad que distingue derivar de inventar: si el `calendar_id` se fabricara
    por activo, habría cuatro calendarios idénticos con nombres distintos y la tabla
    dejaría de significar "calendario".
    """
    perfiles = declared_assets(ASSETS_DIR)
    ids = {aid: calendar_id_for(p) for aid, p in perfiles.items()}

    # XAU/USD (metals, UTC) y BTC/USDT (24x7, UTC) comparten zona pero NO modo: son
    # calendarios distintos de verdad, y el id lo refleja.
    assert ids["xauusd"] != ids["btcusdt"]
    assert "utc" in ids["xauusd"] and "utc" in ids["btcusdt"]

    filas = calendar_rows(perfiles)
    assert len(filas) == len(set(ids.values())), "un calendario por (modo, zona)"
    for fila in filas:
        assert fila["timezone"], "el calendario debe declarar zona horaria"
        assert fila["session_definition"], "sin definición de sesión no hay calendario"


def test_an_undeclared_asset_class_never_reaches_the_spine(tmp_path: Path) -> None:
    """Una `asset_class` desconocida muere ANTES de llegar a la espina.

    Medido al escribir este candado: quien la rechaza es el propio `AssetProfile`
    (`require_valid()`), no el seed. Se deja aquí la evidencia de esa capa **y** el
    `SpineError` como defensa en profundidad, que sí se ejercita abajo con un perfil
    construido a mano.
    """
    directorio = _asset_dir_con(
        tmp_path, "usdcop", lambda d: {**d, "asset_class": "perpetual_future"}
    )
    with pytest.raises(ValueError, match="asset_class"):
        declared_assets(directorio)


def test_the_class_to_instrument_type_map_covers_exactly_the_contract(tmp_path: Path) -> None:
    """La traducción clase→`instrument_type` debe cubrir las clases que el contrato admite.

    Es el candado que impide la deriva real: si mañana el contrato acepta una clase
    nueva, este test se pone rojo en vez de que el seed le asigne un tipo genérico. Y
    con un perfil de clase desconocida, el `SpineError` del seed dispara de verdad.
    """
    from src.contracts.asset_profile import AssetProfile

    perfiles = declared_assets(ASSETS_DIR)
    clases_reales = {p.asset_class for p in perfiles.values()}
    assert clases_reales <= set(INSTRUMENT_TYPE_BY_CLASS), (
        f"clases sin traducción declarada: {clases_reales - set(INSTRUMENT_TYPE_BY_CLASS)}"
    )

    # Defensa en profundidad: saltándose `require_valid`, el seed sigue negándose.
    impostor = dataclasses.replace(perfiles["usdcop"], asset_class="perpetual_future")
    with pytest.raises(SpineError, match="instrument_type"):
        asset_rows({"impostor": impostor})


def test_an_asset_without_declared_annualization_never_receives_a_default(
    tmp_path: Path,
) -> None:
    """Sin anualización declarada, el seed ABORTA — no hereda el default silencioso.

    Este candado nació de un fallo propio: `AssetProfile` aplica `250` por defecto
    (`d.get("trading_days_per_year", 250)`), así que el perfil llega "válido" y
    `AnnualizationRegistry` —que sólo rechaza no-positivos— nunca ve el hueco. USD/COP
    vale 261: el default habría anualizado un 4% mal sin avisar a nadie.
    """

    def sin_anualizacion(d: dict) -> dict:
        d["session"] = {k: v for k, v in d["session"].items() if k != "trading_days_per_year"}
        return d

    directorio = _asset_dir_con(tmp_path, "usdcop", sin_anualizacion)

    with pytest.raises(SpineError, match="trading_days_per_year"):
        declared_assets(directorio)


def test_the_default_that_this_guard_exists_for_is_still_there(tmp_path: Path) -> None:
    """Si el contrato dejara de aplicar el default, este guard sobra — avisa, no falla.

    Sin este candado el guard anterior podría volverse decorativo sin que nadie lo note:
    seguiría verde tanto si el default existe como si no. Aquí queda registrado que la
    razón de ser del guard es un comportamiento REAL y medible del contrato.
    """
    from src.contracts.asset_profile import load_asset_profile

    directorio = _asset_dir_con(
        tmp_path,
        "usdcop",
        lambda d: {
            **d,
            "session": {k: v for k, v in d["session"].items() if k != "trading_days_per_year"},
        },
    )
    perfil = load_asset_profile("usdcop", assets_dir=directorio)
    assert perfil.session.trading_days_per_year == 250, (
        "el contrato ya no aplica el default de 250: revisa si el guard del seed sigue "
        "haciendo falta (USD/COP declara 261)"
    )


# --------------------------------------------------------------- evidencia, no fe


def test_a_provider_only_exists_if_it_wrote_real_rows() -> None:
    """Los proveedores se MIDEN de la base; no hay lista de proveedores en ningún YAML.

    Es deliberado: un proveedor declarado pero inactivo afirmaría una capacidad que el
    sistema no tiene. La única prueba admisible de que un proveedor existe es que haya
    escrito filas.
    """
    for ruta in sorted(ASSETS_DIR.glob("*.yaml")):
        declarado = yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}
        if "asset_id" not in declarado:
            continue
        assert not {"provider", "providers"} & set(declarado), (
            f"{ruta.name} declara proveedores: habría dos fuentes de verdad. O el SSOT "
            "los declara y el seed deja de medirlos, o los mide y el SSOT calla"
        )

    # Y la evidencia sí tiene que venir de tablas de datos reales, no de una constante.
    from scripts.data.seed_reference_spine import OHLCV_SOURCES

    assert OHLCV_SOURCES, "sin tablas de evidencia no hay forma de medir un proveedor"


def test_symbols_with_rows_but_no_profile_stay_out() -> None:
    """USD/BRL, USD/MXN y SPY tienen filas reales y aun así NO entran.

    Es la regla funcionando en su caso incómodo: hay datos, pero nadie declaró el
    activo. Meterlos "porque hay filas" invertiría la dirección de la verdad —- sería
    la base definiendo el catálogo en vez del SSOT.
    """
    perfiles = declared_assets(ASSETS_DIR)
    hechos = [
        ProviderFact("twelvedata", "USD/COP", 86770, "usdcop_m5_ohlcv"),
        ProviderFact("twelvedata_backfill", "USD/BRL", 236730, "usdcop_m5_ohlcv"),
        ProviderFact("twelvedata_daily_deep", "SPY", 8430, "asset_daily_ohlcv"),
    ]
    dentro, fuera = split_facts(hechos, perfiles)

    assert [h.symbol for h in dentro] == ["USD/COP"]
    assert sorted(h.symbol for h in fuera) == ["SPY", "USD/BRL"]


def test_spy_is_never_collapsed_into_the_spx_index() -> None:
    """SPY no se mapea a SPX/500 aunque "sea lo mismo" coloquialmente.

    La propia migración 072 lo declara: *"SPX, SPY and ES are distinct instruments;
    aliases never collapse identity"*. Colapsarlos para que la espina "cuadre" con los
    datos existentes destruiría justo la identidad que la tabla protege.
    """
    perfiles = declared_assets(ASSETS_DIR)
    simbolos = {f["canonical_symbol"] for f in instrument_rows(perfiles)}

    assert "SPX/500" in simbolos
    assert "SPY" not in simbolos

    _, fuera = split_facts(
        [ProviderFact("twelvedata_daily_deep", "SPY", 8430, "asset_daily_ohlcv")], perfiles
    )
    assert [h.symbol for h in fuera] == ["SPY"]


def test_instrument_currencies_come_from_the_pair_declared_in_the_ssot() -> None:
    """`base`/`quote` salen del par declarado, no de partir el símbolo por la barra.

    Partir `BTC/USDT` funcionaría; partir `SPX/500` daría `quote='500'`, que no es una
    moneda. El SSOT ya declara `quote_ccy: USD` — leerlo evita el error entero.
    """
    perfiles = declared_assets(ASSETS_DIR)
    por_simbolo = {f["canonical_symbol"]: f for f in instrument_rows(perfiles)}

    assert por_simbolo["SPX/500"]["quote_currency"] == "USD"
    assert por_simbolo["SPX/500"]["quote_currency"] != "500"
    assert por_simbolo["BTC/USDT"]["quote_currency"] == "USDT"
