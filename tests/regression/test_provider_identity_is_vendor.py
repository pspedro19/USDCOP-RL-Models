"""C038 — `reference.provider` es el VENDEDOR, no la ruta de ingesta.

Por que existe este fichero
---------------------------
`reference.provider` llego a tener 14 filas de las que solo 3 eran proveedores: binance,
investing y twelvedata, las unicas con `authoritative_for` no vacio. Las otras 11 eran
RUTAS DE INGESTA ascendidas a identidad de primera clase — `twelvedata_backfill`,
`twelvedata_gap_fill`, `twelvedata_daily_deep`, `twelvedata_multi`, `binance_daily`,
`investing_daily`... y la que lo delata, **`twelvedata_manual_test`**: una prueba manual
con FK entrante desde `market.raw_bar`.

La tabla se veia verde —poblada, con FKs, sin nulos— y su contenido no significaba lo que
decia la columna. Ninguna prueba lo detectaba, porque todas median PRESENCIA. Causa
probable: poblarla derivando de la columna `source` de los seeds, que registra QUE PROCESO
escribio la fila, no QUE VENDEDOR la sirvio.

Lo que se fija aqui, y lo que deliberadamente NO
------------------------------------------------
Se fija la CLASE de error, no las 11 filas concretas: un `provider_id` no puede ser la
extension con sufijo de otro `provider_id`. Enumerar los 11 nombres habria dejado pasar el
numero 12 (`twelvedata_intraday`, `binance_futures`...), que es exactamente como llego el
11 despues del 10.

NO se afirma aqui que toda fila de `provider` tenga `authoritative_for` no vacio, y la
razon es honestidad: `daily_native` etiqueta el origen del seed diario de USD/COP
(`scripts/ops/seed_session_calendar.py:146`) y ese parquet **no lleva columna de
procedencia** — sus columnas son time/open/high/low/close. El vendedor detras de esa serie
no es medible desde el repo, y un test que forzara a declararlo obligaria a inventarlo.
Queda como pregunta abierta declarada, no como invariante fingida.

El perimetro de huerfanos es DERIVADO, no una lista a mano: se compara contra los activos
que tienen AssetProfile en `config/assets/*.yaml`. USD/BRL tiene barras vivas y NO tiene
perfil —se ingiere como serie correlacionada de COP, nunca se onboardeo como activo—, asi
que su ausencia de `provider_symbol` no es un fallo de identidad sino de onboarding, y se
declara en vez de taparse insertando un activo fabricado.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ASSET_CONFIG_DIR = ROOT / "config" / "assets"

# Mismo contrato que BL-42 (CXD-012): un skip silencioso jamas debe leerse como verde
# donde la DB se supone que existe.
REQUIRE_DB = os.environ.get("C038_REQUIRE_DB") == "1"


def _db_unavailable(msg: str):
    if REQUIRE_DB:
        pytest.fail(f"C038_REQUIRE_DB=1 but {msg}")
    pytest.skip(
        f"{msg} — la identidad de proveedor solo es comprobable contra la DB viva: "
        f"driver + credenciales + esquema reference.* cargado"
    )


def _conn():
    try:
        import psycopg2
    except ImportError:
        _db_unavailable("psycopg2 not installed")
    try:
        return psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
            user=os.environ.get("POSTGRES_USER", "admin"),
            password=os.environ.get("POSTGRES_PASSWORD", ""),
            connect_timeout=3,
        )
    except Exception as e:  # noqa: BLE001
        _db_unavailable(f"postgres unreachable ({e.__class__.__name__})")


def _query(sql: str):
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()
    finally:
        conn.close()


def test_db_available_when_required():
    """Canario: con C038_REQUIRE_DB=1 la indisponibilidad es roja, no un skip."""
    if not REQUIRE_DB:
        pytest.skip("C038_REQUIRE_DB no armado — modo advisory")
    rows = _query("SELECT count(*) FROM reference.provider;")
    assert rows[0][0] > 0, "reference.provider vacia: el esquema de identidad no esta cargado"


def test_no_provider_is_an_ingestion_route_of_another():
    """LA CLASE: ningun provider_id puede ser la extension con sufijo de otro.

    `twelvedata_backfill` bajo `twelvedata` es una ruta, no un vendedor. Se comprueba la
    forma, no una lista de nombres, para que el siguiente `<vendedor>_<loquesea>` tampoco
    entre.
    """
    ids = sorted(r[0] for r in _query("SELECT provider_id FROM reference.provider;"))
    offenders = [
        (child, parent)
        for child in ids
        for parent in ids
        if child != parent and child.startswith(parent + "_")
    ]
    assert not offenders, (
        "rutas de ingesta ascendidas a proveedor: "
        + ", ".join(f"{c} (ruta de {p})" for c, p in offenders)
    )


def test_vendor_symbol_pair_resolves_to_exactly_one_instrument():
    """Un mismo (vendedor, simbolo) no puede apuntar a dos instrumentos.

    Es la precondicion que hizo el colapso 20->N inequivoco; si se rompe, cualquier
    consolidacion futura tendria que ELEGIR, y elegir aqui es inventar identidad.
    """
    rows = _query(
        "SELECT provider_id, provider_symbol, count(DISTINCT instrument_id) "
        "FROM reference.provider_symbol GROUP BY 1,2 HAVING count(DISTINCT instrument_id) > 1;"
    )
    assert not rows, f"(vendedor, simbolo) ambiguos: {rows}"


def _onboarded_canonical_symbols() -> set[str]:
    """Simbolos de activos con AssetProfile, DERIVADO del directorio de configs."""
    import re

    symbols = set()
    for path in ASSET_CONFIG_DIR.glob("*.yaml"):
        if path.stem.endswith("_forecasting") or path.stem in {"pipelines", "fabric_factories"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in re.findall(r"^\s*(?:canonical_)?symbol\s*:\s*['\"]?([A-Z]{3}/[A-Z]{3,4})", text, re.M):
            symbols.add(match)
    return symbols


def test_onboarded_assets_have_no_orphan_symbols():
    """Todo simbolo de un activo ONBOARDEADO debe tener fila en provider_symbol.

    El perimetro sale de `config/assets/*.yaml`, no de una lista escrita a mano: si manana
    se onboardea USD/BRL, este test empieza a exigirlo solo. Hoy USD/BRL tiene barras vivas
    y NO tiene perfil, asi que su huerfandad es deuda de onboarding declarada — no se tapa
    insertando un activo fabricado.
    """
    onboarded = _onboarded_canonical_symbols()
    if not onboarded:
        pytest.fail("perimetro vacio: no se derivo ningun simbolo de config/assets/*.yaml")
    registered = {r[0] for r in _query("SELECT DISTINCT provider_symbol FROM reference.provider_symbol;")}
    missing = sorted(s for s in onboarded if s not in registered)
    assert not missing, f"activos onboardeados sin provider_symbol: {missing}"
