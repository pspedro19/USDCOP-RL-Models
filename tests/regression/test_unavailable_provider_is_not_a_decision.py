"""Una caida de red no es una decision del modelo, y no puede liquidarse como si lo fuera.

El protocolo congelado dice que una respuesta invalida conserva la exposicion previa
(`retain_previous_weight`). Esa politica se diseno para un modelo que responde **mal** -- texto
fuera de contrato, JSON roto --, que es comportamiento del modelo y por tanto medible.

Un modelo al que **no se llego** es otra cosa. `APIConnectionError` / `APITimeoutError` dejan la
fila con `unavailable: true`, `tokens_used: 0` y el sha256 de la cadena vacia
(`e3b0c442...b7852b855`). Si esa sesion se liquida, el informe registra una caida de la conexion
como una jornada entera de "mantener la posicion", indistinguible de una conviccion del modelo —
y el brazo pasa a medir la red ademas del modelo.

Medido el 2026-09-12 durante la corrida de seleccion: una caida local tumbo **882 barras en
DeepSeek y 882 en Azure a la vez**, 14 sesiones completas por brazo. Los dos proveedores fallaron
en el mismo instante, que es la firma de un problema local y no de un proveedor.

Estas sesiones se excluyen y se vuelven a pedir. Excluir pierde muestra; liquidarlas inventa
decisiones.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EMPTY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _row(date: str, bar: int, *, weight: float = 0.5, unavailable: bool = False) -> dict:
    return {
        "decision_id": f"{date}::llm::{bar}",
        "session_date": date,
        "bar": bar,
        "weight": weight,
        "valid_json": not unavailable,
        "unavailable": unavailable,
        "error": "APIConnectionError" if unavailable else None,
        "tokens_used": 0 if unavailable else 500,
        "raw_response_sha256": EMPTY_SHA if unavailable else "a" * 64,
    }


def _ledger(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "ledger.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _read_weights(ledger: Path, dates: set[str]):
    from scripts.analysis.thesis_hybrid import _llm_weights
    return _llm_weights(ledger, dates)


@pytest.fixture(autouse=True)
def _needs_numpy():
    pytest.importorskip("numpy")


def test_a_session_with_any_unreachable_bar_is_excluded() -> None:
    """Basta UNA barra sin respuesta: la senda de 59 ya no es del modelo."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        rows = [_row("2023-04-19", b) for b in range(59)]
        rows[50] = _row("2023-04-19", 50, unavailable=True)
        weights, excluded = _read_weights(_ledger(tmp_path, rows), {"2023-04-19"})

        assert weights == {}
        assert excluded["2023-04-19"].startswith("provider_unavailable")
        assert "1_bars" in excluded["2023-04-19"], "el motivo debe decir cuantas barras se cayeron"


def test_a_clean_session_still_settles() -> None:
    """La regla no puede llevarse por delante las sesiones buenas."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        rows = [_row("2023-04-18", b) for b in range(59)]
        weights, excluded = _read_weights(_ledger(tmp_path, rows), {"2023-04-18"})

        assert excluded == {}
        assert "2023-04-18" in weights
        assert len(weights["2023-04-18"]) == 59


def test_an_invalid_but_ANSWERED_response_is_not_excluded() -> None:
    """Responder mal SI es comportamiento del modelo: se conserva el peso previo y se liquida.

    Es la distincion que da sentido a todo esto. `valid_json: false` con `unavailable: false`
    significa que el modelo contesto algo fuera de contrato -- un dato sobre el modelo. La
    politica congelada lo resuelve reteniendo la exposicion previa, y esa sesion cuenta.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        rows = [_row("2023-04-17", b) for b in range(59)]
        rows[10]["valid_json"] = False
        rows[10]["error"] = "response_not_object"
        rows[10]["tokens_used"] = 480          # contesto: gasto tokens
        weights, excluded = _read_weights(_ledger(tmp_path, rows), {"2023-04-17"})

        assert excluded == {}, "una respuesta mala no es una ausencia de respuesta"
        assert "2023-04-17" in weights


def test_the_settlement_applies_the_same_rule() -> None:
    """La liquidacion y el hibrido tienen que excluir lo mismo, o compararian muestras distintas."""
    source = (ROOT / "scripts" / "analysis" / "settle_thesis_llm.py").read_text(encoding="utf-8")
    assert 'bars[i].get("unavailable")' in source
    assert "provider_unavailable" in source
