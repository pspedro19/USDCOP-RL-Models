"""BL-18 — el DSR de SPX delega en el SSOT constitucional, sin perder precision.

`quant-constitution.md` nombra `services/common/metrics.py::deflated_sharpe_ratio` como
**SSOT y gate de release**: es el numero con el que se promueve una estrategia. Que
`spx500_regime_gated_v1` tuviera su propia copia hacia que el veredicto dependiera de
que fichero se importara.

Medi la equivalencia ANTES de delegar (deltas ~5e-05, por un redondeo a 4 decimales del
SSOT que Codex retiro en `ffd88146` tras el hallazgo de CLD-462). Tras delegar la delta
es **exactamente cero**, porque ya no hay dos implementaciones: hay una.

El borde del gate se ata aparte porque es donde un redondeo importaria: el bar es
`DSR > 0.95`, y un valor entre 0.94995 y 0.95005 podia caer a un lado u otro segun quien
lo calculara. Con precision completa, la comparacion es la misma en ambos lados.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASOS = [
    (0.08, 1000, 10, 0.05, 0.0, 3.0),
    (0.05, 500, 100, 0.03, -0.5, 5.0),
    (0.12, 2000, 989, 0.04, 0.2, 4.0),
    (0.02, 250, 42, 0.02, 0.0, 3.0),
]


def _pareja(sr, n, k, sd, skew, kurt):
    from services.common.metrics import deflated_sharpe_ratio as gobernado
    from src.strategies.spx500_regime_gated_v1.deflated_sharpe import deflated_sharpe as local

    b = gobernado(sr, n, k, sd, skew, kurt)
    return local(sr, sd**2, k, n, skew, kurt), (b.get("dsr") if isinstance(b, dict) else b)


@pytest.mark.parametrize("caso", CASOS)
def test_the_local_dsr_is_bit_identical_to_the_governed_one(caso) -> None:
    """Delta EXACTAMENTE cero: no hay dos implementaciones, hay una delegando."""
    a, b = _pareja(*caso)
    assert a == b, (
        f"el DSR local difiere del gobernado en {abs(a - b):.2e}: si vuelve a haber dos "
        "implementaciones, el veredicto de promocion depende de que fichero se importe"
    )


def test_no_rounding_survives_at_the_gate_boundary() -> None:
    """En el borde `DSR > 0.95`, ambos lados deben decidir IGUAL.

    Es el escenario que motivo el hallazgo: un redondeo a 4 decimales podia mover un
    valor de 0.94996 a 0.9500 y voltear un PROMOTE. Se barre el borde en pasos finos y
    se exige que ninguna entrada caiga a lados distintos del bar.
    """
    discrepancias = []
    for i in range(2000):
        sr = 0.05 + i * 0.00005
        a, b = _pareja(sr, 1000, 42, 0.03, 0.0, 3.0)
        if (a > 0.95) != (b > 0.95):
            discrepancias.append((sr, a, b))

    assert not discrepancias, (
        f"{len(discrepancias)} entradas caen a lados distintos del bar 0.95; primeras: "
        f"{discrepancias[:3]}"
    )


def test_the_local_module_no_longer_reimplements_the_statistic() -> None:
    """La delegacion es real: el modulo IMPORTA el SSOT, no reproduce su formula.

    Sin este candado, alguien podria "reintroducir por comodidad" el calculo local y la
    equivalencia numerica seguiria verde el dia que se escribiera — y divergiria la
    primera vez que el SSOT cambiara.
    """
    import inspect

    from src.strategies.spx500_regime_gated_v1 import deflated_sharpe as modulo

    for nombre in ("expected_max_sharpe", "probabilistic_sharpe"):
        fuente = inspect.getsource(getattr(modulo, nombre))
        assert "services.common.metrics" in fuente, (
            f"{nombre} dejo de delegar en el SSOT: volveria a haber dos estadisticos"
        )
