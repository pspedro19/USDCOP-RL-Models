"""Escribir un seed es un UPSERT, nunca una sobrescritura.

`.claude/rules/data-governance.md` lo fija como invariante 3 -- "Siempre UPSERT por
`(time, symbol)`, nunca INSERT plano" -- y su DO NOT anade que los seeds no se editan a mano
porque son el fallback de restore.

`_write_seed` hacia `df.to_parquet(path)` a secas. Para Gold y BTC no se notaba: son activos
recien incorporados cuya historia cabe entera en la ventana que devuelve el proveedor. Para
USD/COP, cuyo seed arranca en 2019-12 y cuya API devuelve unas semanas por llamada, el
comando que el propio script documenta

    python scripts/data/ingest_asset_ohlcv.py --asset usdcop --no-db

reemplazaba 99.714 filas por 4.533 y borraba seis anos de historia. Ejecutado y medido el
2026-09-11: el gate de calidad habia dado `PASS rows=4533 errors=0` un segundo antes, porque
valida lo que llega, no lo que desaparece. Un gate verde sobre datos correctos convivia con
la perdida del activo mas critico del repositorio.

Estos tests fijan las tres propiedades que faltaban: la historia sobrevive, la ventana fresca
gana el conflicto, y el conteo no puede bajar.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "data" / "ingest_asset_ohlcv.py"


@pytest.fixture(scope="module")
def module():
    if not SCRIPT.is_file():
        pytest.skip("ingest_asset_ohlcv.py no existe en este checkout")
    spec = importlib.util.spec_from_file_location("ingest_asset_ohlcv_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"no se pudo importar el script: {type(exc).__name__}: {exc}")
    if not hasattr(mod, "_merge_into_existing_seed"):
        pytest.fail(
            "ingest_asset_ohlcv.py no expone _merge_into_existing_seed: la escritura de "
            "seeds volvio a ser plana y puede borrar historia."
        )
    return mod


def _bars(start: str, periods: int, symbol: str = "USD/COP", close: float = 4000.0):
    times = pd.date_range(start, periods=periods, freq="5min", tz="America/Bogota")
    return pd.DataFrame({
        "time": times,
        "symbol": symbol,
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "volume": 0,
    })


def test_history_survives_a_fresh_window(module, tmp_path: Path) -> None:
    """El caso real: historia larga, ventana corta que solo cubre el final."""
    history = _bars("2020-01-02 08:00", 500)
    path = tmp_path / "usdcop_m5_ohlcv.parquet"
    history.to_parquet(path, index=False)

    fresh = _bars("2020-01-04 08:00", 40)      # solapa y extiende
    merged = module._merge_into_existing_seed(fresh, path)

    assert len(merged) >= len(history), (
        "el merge perdio filas: es exactamente el defecto que borro seis anos de USD/COP"
    )
    assert merged["time"].min() == history["time"].min(), (
        "la barra mas antigua desaparecio; la ventana fresca no puede truncar la historia"
    )
    assert merged["time"].is_monotonic_increasing
    assert not merged.duplicated(subset=["time", "symbol"]).any()


def test_fresh_bars_win_the_conflict(module, tmp_path: Path) -> None:
    """Una correccion del proveedor debe imponerse sobre la barra vieja."""
    history = _bars("2020-01-02 08:00", 10, close=4000.0)
    path = tmp_path / "seed.parquet"
    history.to_parquet(path, index=False)

    fresh = _bars("2020-01-02 08:00", 3, close=4321.0)
    merged = module._merge_into_existing_seed(fresh, path)

    overlapping = merged[merged["time"].isin(fresh["time"])]
    assert (overlapping["close"] == 4321.0).all(), (
        "la barra corregida no gano el UPSERT; el seed conservaria el valor viejo"
    )
    assert len(merged) == len(history), "un solape puro no debe anadir filas"


def test_a_shrinking_merge_raises_instead_of_writing(module, tmp_path: Path) -> None:
    """Si el resultado encoge, se falla: escribir seria perder datos en silencio."""
    history = _bars("2020-01-02 08:00", 100)
    path = tmp_path / "seed.parquet"
    history.to_parquet(path, index=False)

    # Un seed sin clave compartida no autoriza a sobrescribir.
    sin_clave = pd.DataFrame({"foo": [1, 2, 3]})
    with pytest.raises(RuntimeError):
        module._merge_into_existing_seed(sin_clave, path)


def test_unreadable_seed_is_not_overwritten(module, tmp_path: Path) -> None:
    path = tmp_path / "corrupto.parquet"
    path.write_bytes(b"esto no es un parquet")
    with pytest.raises(RuntimeError):
        module._merge_into_existing_seed(_bars("2020-01-02 08:00", 5), path)


def test_missing_seed_is_created_from_the_fresh_window(module, tmp_path: Path) -> None:
    """Un activo nuevo no tiene historia que preservar; ese caso sigue funcionando."""
    fresh = _bars("2026-01-02 08:00", 20, symbol="XAU/USD")
    merged = module._merge_into_existing_seed(fresh, tmp_path / "nuevo.parquet")
    assert len(merged) == len(fresh)
