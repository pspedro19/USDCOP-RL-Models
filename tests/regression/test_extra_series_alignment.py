"""Un brazo que cubre menos sesiones que el bloque tiene que alinearse por FECHA.

`thesis_statistics.py` sabia leer solo `{config}_seed{n}.json`, asi que un brazo LLM o el hibrido
no cabian mas que disfrazados de PPO — lo que ademas los habria colado en el PBO, que barre
cualquier clave con `_seed`.

El riesgo al abrirles la puerta es de alineacion, no de formato. Una liquidacion puede cubrir
**menos** sesiones que el bloque: una sesion sin sus 59 barras se excluye y **no se rellena con
cero**, porque rellenar convertiria "no decidio" en "decidio no operar". Si esa serie corta se
comparase posicionalmente contra las 226 del bloque, el contraste pareado emparejaria el dia 5 de
un brazo con el dia 5 del otro **sin que sean el mismo dia**, y saldria un numero con toda la
apariencia de un resultado.

Estos tests fijan que la alineacion sea por fecha, que la cobertura se declare, y que un brazo
sin fechas en comun falle en vez de devolver ceros.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class _Spec:
    date: date


def _specs(days: list[int]) -> list[_Spec]:
    return [_Spec(date=date(2023, 1, d)) for d in days]


def _settlement(tmp_path: Path, rows: list[tuple[str, float]], **extra) -> Path:
    payload = {"sessions": [{"session_date": d, "daily_return": r} for d, r in rows], **extra}
    path = tmp_path / "settlement.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def extra_series():
    pytest.importorskip("pandas")
    from scripts.analysis.thesis_statistics import extra_series as fn
    return fn


def test_values_land_on_their_own_dates_not_on_their_own_order(extra_series, tmp_path) -> None:
    """El caso que rompe todo: el brazo cubre el dia 3 y el 5, no los dos primeros."""
    specs = _specs([2, 3, 4, 5, 6])
    path = _settlement(tmp_path, [("2023-01-03", 0.01), ("2023-01-05", -0.02)])

    values, mask, meta = extra_series(path, specs)

    np.testing.assert_array_equal(mask, [False, True, False, True, False])
    assert values[1] == 0.01 and values[3] == -0.02
    assert meta["sessions_covered"] == 2
    assert meta["sessions_in_block"] == 5
    assert meta["coverage"] == 0.4


def test_dates_outside_the_block_are_ignored(extra_series, tmp_path) -> None:
    """Una liquidacion puede traer fechas de otro bloque; no pueden colarse."""
    specs = _specs([2, 3])
    path = _settlement(tmp_path, [("2023-01-03", 0.01), ("2024-06-01", 9.99)])

    values, mask, _ = extra_series(path, specs)

    assert mask.tolist() == [False, True]
    assert 9.99 not in values


def test_no_common_date_is_an_error_not_a_zero_series(extra_series, tmp_path) -> None:
    """Devolver ceros aqui seria publicar un brazo plano inventado."""
    specs = _specs([2, 3])
    path = _settlement(tmp_path, [("2025-05-05", 0.01)])

    with pytest.raises(ValueError, match="ninguna de sus fechas"):
        extra_series(path, specs)


def test_a_settlement_without_sessions_is_refused(extra_series, tmp_path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"sessions": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="sessions"):
        extra_series(path, _specs([2]))


def test_scope_defaults_to_retrospective_and_confirmatory_to_false(extra_series, tmp_path) -> None:
    """Nada entra por esta puerta declarandose confirmatorio por omision."""
    specs = _specs([3])
    path = _settlement(tmp_path, [("2023-01-03", 0.0)])
    _, _, meta = extra_series(path, specs)
    assert meta["scope"] == "retrospective_diagnostic"
    assert meta["confirmatory"] is False


def test_an_explicit_confirmatory_flag_is_carried_through(extra_series, tmp_path) -> None:
    """Si algun dia un brazo SI es confirmatorio, el informe tiene que decirlo, no ocultarlo."""
    specs = _specs([3])
    path = _settlement(tmp_path, [("2023-01-03", 0.0)],
                       confirmatory=True, scope="forward_confirmatory")
    _, _, meta = extra_series(path, specs)
    assert meta["confirmatory"] is True
    assert meta["scope"] == "forward_confirmatory"
