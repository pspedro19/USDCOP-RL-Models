"""
Regression: las SEEDS de OHLCV son integras — sin depender de Postgres.

Contract: CTR-DQ-TZ-001 (mismo contrato que `scripts/ops/fix_tz_wall_cot_{rows,seed}.py`)

## El hueco que este modulo cierra

`test_ohlcv_timestamps_are_instants.py` valida la **BD** y hace `skip` cuando Postgres no
responde — que es el caso en cualquier maquina sin el stack levantado, y en CI. Resultado: la
seed, que es lo que leen la tesis, cualquier corrida local y un clon limpio, **no la validaba
nadie**. Por eso un desfase de timezone vivio meses en `seeds/latest/usdcop_m5_ohlcv.parquet`
sin que saltara una sola alarma:

    225 sesiones (13,4%) con las barras 5 horas antes de la ventana real,
    y 183 de ellas dentro del hold-out de la tesis (28,4% de ese bloque).

Causa raiz: `airflow/dags/l0_ohlcv_backfill.py` insertaba naive-COT en una columna TIMESTAMPTZ
con `PGTZ: UTC` (ya corregido, 2026-07-21). Reparado en la seed el 2026-08-24 con
`scripts/ops/fix_tz_wall_cot_seed.py`. Este guard evita que vuelva.

**No importa nada que necesite Postgres. Ese es el punto del modulo.**
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ROOT / "seeds" / "latest"
CALENDAR = ROOT / "config" / "trading_calendar.json"

# Pares con sesion 08:00-12:55 COT (`data-governance.md`, regla de oro).
SESSION_SYMBOLS = {"USDCOP", "USDMXN", "USDBRL"}

# XAU/USD y BTC/USDT guardan TIMESTAMPTZ basado en INSTANTE, 24h, no COT localizado
# (`data-governance.md` + `_asbuilt-implementation.md`). Su histograma horario es plano por
# diseno, asi que la ventana de sesion no les aplica.
EXEMPT_SYMBOLS = {"XAUUSD", "BTCUSDT"}

SESSION_LO, SESSION_HI = 8, 12
BARS_PER_SESSION = 60          # 08:00-12:55 COT en pasos de 5 minutos

SEED_FILES = ("usdcop_m5_ohlcv.parquet", "fx_multi_m5_ohlcv.parquet")

# ---------------------------------------------------------------------------
# Deuda declarada: festivos colombianos que el proveedor sirve con sesion COMPLETA.
#
# Mercado cerrado + 60 barras "perfectas" = relleno sintetico, no negociacion real. No se
# borran aqui (borrar datos es decision del operador, y la mascara de evaluacion de F1 es el
# sitio correcto para excluirlos del computo), pero se CONGELAN: si aparece un festivo nuevo
# con sesion completa, este test falla y obliga a mirarlo.
#
# Medido el 2026-08-24 tras reparar la timezone.
#
# Patron observado en los festivos de 2025 (lista canonica de `config/trading_calendar.json`):
# la mayoria trae un residuo parcial de 2-18 barras, y solo tres traen la sesion COMPLETA
# (01-01, 01-06, 12-25). Ambos casos son mercado cerrado, asi que la mascara de evaluacion de
# F1 debe excluir TODOS los festivos, no solo los completos; este guard solo vigila que no
# aparezca uno completo NUEVO, que es la senal de que el proveedor empezo a fabricar sesiones.
# ---------------------------------------------------------------------------
SYNTHETIC_HOLIDAY_SESSIONS = {
    "2022-04-15",  # Viernes Santo
    "2023-04-07",  # Viernes Santo
    "2023-12-25",  # Navidad
    "2024-01-01",  # Ano Nuevo
    "2024-03-29",  # Viernes Santo
    "2024-05-01",  # Dia del Trabajo
    "2024-12-25",  # Navidad
    "2025-01-01",  # Ano Nuevo
    "2025-01-06",  # Reyes — lo cazo ESTE guard al cruzar con la lista canonica del config
    "2025-12-25",  # Navidad
    "2026-01-01",  # Ano Nuevo
}


def _load(name: str) -> "pd.DataFrame":
    path = SEEDS / name
    if not path.is_file():
        pytest.skip(f"{name} no existe")
    raw = path.open("rb").read(64)
    if raw.startswith(b"version https://git-lfs"):
        pytest.skip(f"{name} es un puntero LFS sin materializar")
    df = pd.read_parquet(path)
    df["_t"] = pd.to_datetime(df["time"])
    df["_h"] = df["_t"].dt.hour
    df["_d"] = df["_t"].dt.date
    df["_sym"] = df["symbol"].astype(str).str.upper().str.replace("/", "", regex=False)
    return df


def _session_rows(df: "pd.DataFrame") -> "pd.DataFrame":
    return df[df["_sym"].isin(SESSION_SYMBOLS)]


ALL_SEEDS = pytest.mark.parametrize("seed_name", SEED_FILES)


@ALL_SEEDS
def test_session_pairs_live_only_in_the_cot_window(seed_name: str):
    """La invariante que se rompio: 08:00-12:55 COT, nada mas.

    Una barra en horas 3-7 es la ventana de sesion desplazada -5h — el sintoma exacto de
    naive-COT interpretado como UTC.
    """
    df = _load(seed_name)
    sess = _session_rows(df)
    if sess.empty:
        pytest.skip("sin pares de sesion en esta seed")
    bad = sess[~sess["_h"].between(SESSION_LO, SESSION_HI)]
    if len(bad):
        hist = bad.groupby(["_sym", "_h"]).size().to_dict()
        shifted = int(bad["_h"].between(3, 7).sum())
        raise AssertionError(
            f"{seed_name}: {len(bad)} barras fuera de 08:00-12:55 COT -> {hist}\n"
            f"  {shifted} de ellas en horas 3-7 = la ventana desplazada -5h "
            f"(naive-COT leido como UTC).\n"
            f"  Repara con: python scripts/ops/fix_tz_wall_cot_seed.py --apply"
        )


@ALL_SEEDS
def test_exempt_symbols_are_not_forced_into_the_cot_window(seed_name: str):
    """Guard del guard: XAU/BTC son 24h por diseno; 'arreglarlos' seria el error contrario."""
    df = _load(seed_name)
    ex = df[df["_sym"].isin(EXEMPT_SYMBOLS)]
    if ex.empty:
        pytest.skip("esta seed no trae XAU/BTC")
    assert ex["_h"].nunique() > (SESSION_HI - SESSION_LO + 1), (
        f"{seed_name}: {sorted(ex['_sym'].unique())} solo tiene "
        f"{ex['_h'].nunique()} horas distintas. Guardan INSTANTES 24h "
        "(data-governance.md); si se les aplico un filtro de sesion COT, revertirlo."
    )


@ALL_SEEDS
def test_no_duplicate_timestamps(seed_name: str):
    df = _load(seed_name)
    dup = df.duplicated(subset=["_sym", "_t"]).sum()
    assert dup == 0, (
        f"{seed_name}: {dup} timestamps duplicados por (symbol, time). El upsert por "
        "(time, symbol) de `data-governance.md` deberia hacerlos imposibles."
    )


@ALL_SEEDS
def test_no_weekend_sessions_for_session_pairs(seed_name: str):
    df = _load(seed_name)
    sess = _session_rows(df)
    if sess.empty:
        pytest.skip("sin pares de sesion")
    wk = sorted({str(d) for d in sess["_d"] if d.weekday() >= 5})
    assert not wk, f"{seed_name}: sesiones en fin de semana: {wk[:10]}"


@ALL_SEEDS
def test_no_session_exceeds_the_bar_budget(seed_name: str):
    """Mas de 60 barras en una sesion = solapamiento de convenciones o duplicado."""
    df = _load(seed_name)
    sess = _session_rows(df)
    if sess.empty:
        pytest.skip("sin pares de sesion")
    counts = sess.groupby(["_sym", "_d"]).size()
    over = counts[counts > BARS_PER_SESSION]
    assert over.empty, (
        f"{seed_name}: {len(over)} sesiones con mas de {BARS_PER_SESSION} barras: "
        f"{over.head(5).to_dict()}"
    )


def test_no_new_synthetic_holiday_sessions():
    """Festivo colombiano + sesion completa = relleno del proveedor, no mercado.

    Los conocidos estan congelados arriba. Uno nuevo hace fallar el test: hay que mirarlo y,
    si es relleno, anadirlo a la lista Y a la mascara de evaluacion (F1, test 15 del plan).
    """
    df = _load("usdcop_m5_ohlcv.parquet")
    cop = df[df["_sym"] == "USDCOP"]
    counts = cop.groupby("_d").size()
    full = {str(d) for d, n in counts.items() if n == BARS_PER_SESSION}

    holidays = set()
    if CALENDAR.is_file():
        cal = json.loads(CALENDAR.read_text(encoding="utf-8"))
        for key, vals in cal.items():
            if key.startswith("holidays_") and key.endswith("_colombia"):
                holidays |= set(vals)
    # La lista canonica de `config/trading_calendar.json` solo cubre 2025; el resto de anios
    # viven en `ADDITIONAL_HOLIDAYS` dentro del DAG de backfill (importarlo exigiria Airflow).
    # Por eso la deuda congelada abajo actua tambien como cobertura multi-anio.
    holidays |= SYNTHETIC_HOLIDAY_SESSIONS

    offenders = sorted((full & holidays) - SYNTHETIC_HOLIDAY_SESSIONS)
    assert not offenders, (
        f"festivos colombianos NUEVOS con sesion completa de {BARS_PER_SESSION} barras: "
        f"{offenders}. El mercado estaba cerrado; son relleno del proveedor. Revisalos y, si "
        "se confirman, anadelos a SYNTHETIC_HOLIDAY_SESSIONS y a la mascara de evaluacion."
    )


def test_frozen_synthetic_holidays_still_exist():
    """Si la deuda se paga, este guard debe dejar de mentir."""
    df = _load("usdcop_m5_ohlcv.parquet")
    counts = df[df["_sym"] == "USDCOP"].groupby("_d").size()
    full = {str(d) for d, n in counts.items() if n == BARS_PER_SESSION}
    gone = sorted(SYNTHETIC_HOLIDAY_SESSIONS - full)
    assert not gone, (
        f"estos festivos ya NO tienen sesion completa: {gone}. Se limpiaron — borralos de "
        "SYNTHETIC_HOLIDAY_SESSIONS en el mismo commit."
    )


@ALL_SEEDS
def test_ohlc_is_coherent(seed_name: str):
    """low <= min(open, close) y high >= max(open, close), o la barra es inventada."""
    df = _load(seed_name)
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    bad = df[(df["low"] > df[["open", "close"]].min(axis=1) + 1e-9)
             | (df["high"] < df[["open", "close"]].max(axis=1) - 1e-9)
             | (df["high"] < df["low"])]
    assert bad.empty, (
        f"{seed_name}: {len(bad)} barras con OHLC incoherente, p.ej.\n"
        f"{bad[['symbol', 'time', 'open', 'high', 'low', 'close']].head(3).to_string()}"
    )


@ALL_SEEDS
def test_seeds_agree_on_the_shared_cop_rows(seed_name: str):
    """`usdcop_m5_ohlcv` y `fx_multi_m5_ohlcv` comparten las filas de COP: no pueden divergir.

    Reparar una y olvidar la otra fue un riesgo real de la migracion de timezone.
    """
    single = _load("usdcop_m5_ohlcv.parquet")
    multi = _load("fx_multi_m5_ohlcv.parquet")
    a = set(single[single["_sym"] == "USDCOP"]["_t"])
    b = set(multi[multi["_sym"] == "USDCOP"]["_t"])
    assert a == b, (
        "las dos seeds discrepan en las filas de USD/COP: "
        f"solo en usdcop_m5={len(a - b)}, solo en fx_multi={len(b - a)}. "
        "Aplica `fix_tz_wall_cot_seed.py` a AMBAS."
    )
