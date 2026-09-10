"""
Regression: ningun activo puede desaparecer en silencio de la capa canonica.

Contract: CTR-MKT-CANON-001 · Date: 2026-08-24

## El defecto que motiva este guard

`market_ohlcv_daily` — la capa canonica que alimenta las vistas wide y el dashboard —
resuelve el activo con un JOIN, no con una lista:

    FROM asset_daily_ohlcv d JOIN dim_asset da ON da.symbol = d.symbol

Un JOIN que no casa **no da error**: simplemente devuelve menos filas. `dim_asset` tenia
`spx500` con `symbol = 'SPX500'` mientras `asset_daily_ohlcv` guardaba **'SPX/500'**, y
las **7.962 filas** de SPX (1995-2026) llevaban meses invisibles. El sintoma llego por un
camino largo: el dashboard reportaba `missing` para el 2026-07-20, un dia en que NYSE
estuvo abierto, la fila existia y el calendario decia `is_trading_day = t`.

Corregido por la migracion `089_dim_asset_spx_symbol_alignment.sql`. Este test evita que
vuelva a pasar con cualquier activo: el modo de fallo es silencioso, asi que la unica
defensa es una comprobacion explicita.

DB-backed; hace `skip` limpio cuando el stack esta abajo — como sus vecinos
`test_wide_views.py` y `test_data_quality_floor.py`.
"""

from __future__ import annotations

import os

import pytest

# Simbolos presentes en `asset_daily_ohlcv` que a proposito NO son activos tradeables del
# control plane y por tanto no necesitan fila en `dim_asset`.
#
# El universo tradeable son EXACTAMENTE cuatro, y las dos fuentes coinciden:
#     config/assets/*.yaml            -> btcusdt, spx500, usdcop, xauusd
#     public/data/registry.json       -> btcusdt, spx500, usdcop, xauusd
#
# Lo de abajo son SERIES DE REFERENCIA: alimentan features y contexto cross-asset, pero no
# se operan, no tienen AssetProfile y no salen en el dashboard como activo. Que no lleguen
# a `market_ohlcv_daily` es correcto, no un fallo.
#
# Anadir un simbolo aqui es una decision, no un atajo para silenciar el test: si el
# simbolo SI es tradeable, lo que toca es darle su fila en `dim_asset`.
NON_TRADEABLE_SYMBOLS: set[str] = {
    "USD/MXN",   # feature #15 del FEATURE_ORDER (`usdmxn_change_1d`), no se opera
    "USD/BRL",   # contexto LATAM para el regimen, no se opera
    "SPY",       # mismo subyacente que SPX/500; `xasset_signals.py` lo excluye del
                 # z-score cross-sectional a proposito ("incluir ambos duplicaria el
                 # peso equity en el ranking")
}

MIN_EXPECTED_ASSETS = 4   # usdcop, xauusd, btcusdt, spx500


def _conn():
    psycopg2 = pytest.importorskip("psycopg2")
    try:
        return psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
            user=os.environ.get("POSTGRES_USER", "admin"),
            password=os.environ.get("POSTGRES_PASSWORD", ""),
            connect_timeout=3)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"postgres unreachable ({e.__class__.__name__})")


@pytest.fixture(scope="module")
def db():
    conn = _conn()
    conn.cursor().execute("SET max_parallel_workers_per_gather = 0")
    yield conn
    conn.close()


def test_every_daily_symbol_resolves_to_an_asset(db):
    """Un simbolo sin fila en `dim_asset` se cae del JOIN sin decir nada."""
    cur = db.cursor()
    cur.execute("""
        SELECT d.symbol, count(*)
        FROM asset_daily_ohlcv d
        LEFT JOIN dim_asset da ON da.symbol = d.symbol
        WHERE da.asset_id IS NULL
        GROUP BY 1 ORDER BY 2 DESC
    """)
    orphans = [(s, n) for s, n in cur.fetchall() if s not in NON_TRADEABLE_SYMBOLS]
    assert not orphans, (
        f"simbolos de asset_daily_ohlcv sin fila en dim_asset: {orphans}.\n"
        "Esas filas NO aparecen en market_ohlcv_daily ni en el dashboard, y el JOIN no "
        "levanta ningun error. Anade el activo a dim_asset con el simbolo del DATO "
        "(el campo `symbol` de config/assets/<id>.yaml, no `chart_symbol`), o declaralo "
        "en NON_TRADEABLE_SYMBOLS con su razon."
    )


def test_canonical_layer_carries_every_declared_asset(db):
    """Todo `dim_asset` con datos debe llegar a la capa canonica."""
    cur = db.cursor()
    cur.execute("""
        SELECT da.asset_id, da.symbol
        FROM dim_asset da
        WHERE EXISTS (SELECT 1 FROM asset_daily_ohlcv d WHERE d.symbol = da.symbol)
          AND NOT EXISTS (SELECT 1 FROM market_ohlcv_daily m WHERE m.asset_id = da.asset_id)
    """)
    missing = cur.fetchall()
    assert not missing, (
        f"activos con datos que no llegan a market_ohlcv_daily: {missing}"
    )


def test_canonical_layer_has_at_least_the_onboarded_assets(db):
    """Si el conteo baja, un activo se cayo — exactamente el fallo de SPX."""
    cur = db.cursor()
    cur.execute("SELECT count(DISTINCT asset_id) FROM market_ohlcv_daily")
    n = cur.fetchone()[0]
    assert n >= MIN_EXPECTED_ASSETS, (
        f"market_ohlcv_daily solo tiene {n} activos; se esperaban >= "
        f"{MIN_EXPECTED_ASSETS} (usdcop, xauusd, btcusdt, spx500). Un activo dejo de "
        "casar por simbolo."
    )


def test_dim_asset_symbols_are_unique(db):
    """Dos activos con el mismo simbolo duplicarian filas en el JOIN."""
    cur = db.cursor()
    cur.execute("SELECT symbol, count(*) FROM dim_asset GROUP BY 1 HAVING count(*) > 1")
    dupes = cur.fetchall()
    assert not dupes, f"simbolos duplicados en dim_asset: {dupes}"


def test_spx_uses_the_data_symbol_not_the_chart_symbol(db):
    """Candado sobre el caso concreto que costo meses de invisibilidad."""
    cur = db.cursor()
    cur.execute("SELECT symbol FROM dim_asset WHERE asset_id = 'spx500'")
    row = cur.fetchone()
    if row is None:
        pytest.skip("spx500 no esta en dim_asset en este entorno")
    assert row[0] == "SPX/500", (
        f"dim_asset.spx500 usa symbol={row[0]!r}. Debe ser 'SPX/500' (el `symbol` del "
        "AssetProfile). 'SPX500' es el `chart_symbol`, para el dashboard. Ver la "
        "migracion 083."
    )


# ---------------------------------------------------------------------------
# Barras en dias que el mercado estuvo cerrado (CTR-DQ-ASSET-DAILY-002)
# ---------------------------------------------------------------------------
# `market_ohlcv_daily_wide` etiqueta `off_session` cuando el calendario dice cerrado pero
# hay barra. Fallaba en 10 fines de semana con `cop=off_session`: la vista tenia razon, la
# barra sobraba. Se borraron 57 barras de fin de semana (usdcop 10 + xauusd 47) con
# `scripts/ops/fix_non_session_daily_bars.py` el 2026-08-24.
#
# DEUDA DECLARADA: quedan 95 barras en FESTIVOS (usdcop 92 de `daily_native`, 2020-2026,
# + 3 deep). Son igual de sinteticas —el mercado colombiano estuvo cerrado— pero llevan
# seis anos en la serie diaria que usa produccion y borrarlas reescribe la historia del
# track H5. Es decision del operador: `--include-holidays` + recalcular el forward
# publicado. Se congela el conteo para que no crezca sin que nadie se entere.
MAX_HOLIDAY_BARS = 95


def _non_session(cur, weekend_only: bool):
    scope = ("AND EXTRACT(dow FROM (d.time AT TIME ZONE 'UTC')::date) IN (0, 6)"
             if weekend_only else
             "AND EXTRACT(dow FROM (d.time AT TIME ZONE 'UTC')::date) NOT IN (0, 6)")
    cur.execute(f"""
        SELECT da.asset_id, count(*)
        FROM asset_daily_ohlcv d
        JOIN dim_asset da ON da.symbol = d.symbol
        JOIN market_session_calendar c
          ON c.asset_id = da.asset_id
         AND c.session_date = (d.time AT TIME ZONE 'UTC')::date
        WHERE da.calendar_kind <> 'utc_24_7'
          AND c.is_trading_day IS NOT TRUE
          {scope}
        GROUP BY 1 ORDER BY 2 DESC
    """)
    return cur.fetchall()


def test_no_weekend_bars_for_session_bound_assets(db):
    """Ni USD/COP ni el oro spot cotizan sabado o domingo: una barra ahi es fabricacion."""
    offenders = _non_session(db.cursor(), weekend_only=True)
    assert not offenders, (
        f"barras diarias en fin de semana: {offenders}. Son relleno del proveedor; "
        "borralas con `python scripts/ops/fix_non_session_daily_bars.py --apply`."
    )


def test_holiday_bar_debt_does_not_grow(db):
    """La deuda de festivos esta congelada; que no crezca en silencio."""
    offenders = _non_session(db.cursor(), weekend_only=False)
    total = sum(n for _a, n in offenders)
    assert total <= MAX_HOLIDAY_BARS, (
        f"barras en festivo: {total} > {MAX_HOLIDAY_BARS} congeladas ({offenders}). "
        "Aparecieron festivos NUEVOS con barra. Si la deuda se pago, baja "
        "MAX_HOLIDAY_BARS en el mismo commit."
    )


# ---------------------------------------------------------------------------
# El filtro de ingesta, a nivel de LOGICA (no de datos)
# ---------------------------------------------------------------------------
from pathlib import Path as _Path  # noqa: E402

ROOT = _Path(__file__).resolve().parents[2]

# Los tests de arriba miran la BD: cazan las barras de fin de semana DESPUES de que
# alguien las escriba. Este mira la funcion que decide si escribirlas, y no necesita
# esperar a que corra un DAG.
#
# Historia: hicieron falta TRES intentos. (1) borrar las barras — volvieron en horas
# porque la ingesta diaria no filtraba sesion, solo la de 5 minutos. (2) anadir
# `_weekday_only` convirtiendo a Bogota — una barra DIARIA del sabado llega a las 00:00
# UTC, que en COT es VIERNES 19:00, asi que pasaba igual. (3) evaluar la fecha UTC, que
# es como la leen el calendario y los tests de arriba.


def test_weekend_filter_uses_the_utc_date_not_the_bogota_one():
    """Una barra diaria del sabado sellada a 00:00 UTC debe filtrarse."""
    import ast
    import io
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    src_path = ROOT / "scripts" / "ops" / "backfill_max_history.py"
    tree = ast.parse(io.open(src_path, encoding="utf-8").read())
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "_weekday_only"), None)
    assert fn is not None, (
        "`_weekday_only` desapareció de backfill_max_history.py: sin él, la ingesta "
        "diaria vuelve a escribir sábados y domingos."
    )
    ns = {"UTC": timezone.utc, "BOG": ZoneInfo("America/Bogota"), "datetime": datetime}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<f>", "exec"), ns)
    weekday_only = ns["_weekday_only"]

    cases = {"2026-08-21": True,    # viernes
             "2026-08-22": False,   # sabado  <- el que se colaba
             "2026-08-23": False,   # domingo
             "2026-08-24": True}    # lunes
    for day, expected in cases.items():
        ts = datetime.fromisoformat(f"{day}T00:00:00+00:00")
        assert weekday_only(ts) is expected or weekday_only(ts) == expected, (
            f"{day} 00:00 UTC: el filtro devuelve {weekday_only(ts)}, se esperaba "
            f"{expected}. Si convierte a Bogotá, el sábado se lee como viernes 19:00."
        )


def test_the_daily_ingestion_call_passes_the_filter():
    """El filtro solo sirve si la llamada DIARIA lo pasa; la de 5 min ya lo hacía."""
    src_lines = (ROOT / "scripts" / "ops" / "backfill_max_history.py").read_text(
        encoding="utf-8").splitlines()
    # La llamada concreta a td_series con interval "1day" sobre asset_daily_ohlcv.
    # Se localiza por linea y no por regex: el patron tendria que cruzar el ")" de
    # `timedelta(days=1)` que vive dentro de los propios argumentos.
    idx = next((i for i, ln in enumerate(src_lines)
                if '"1day"' in ln and "today_plus" in ln), None)
    assert idx is not None, "ya no existe la llamada de td_series con interval 1day"
    window = "\n".join(src_lines[idx:idx + 8])
    assert "session_filter" in window and "_weekday_only" in window, (
        "la llamada de `td_series` para asset_daily_ohlcv no pasa `session_filter`. "
        "Sin él, el proveedor devuelve sábados y domingos y entran tal cual."
    )
