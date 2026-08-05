"""C-031 (carril CLAUDE): `agent_tools` no puede volver a exponer tono sin gobierno.

Origen: residual A4 de la review CLD-490 sobre BL-40. El gate de disponibilidad C028
convierte un sentimiento no medido en `null + reason`, pero gobierna la COLUMNA DE DB.
`data/news/gdelt_daily_sentiment.csv` no tiene `feature_status`, ni cutoff causal, ni cota
de frescura, ni provenance — y `agent_tools.load_gdelt_sentiment()` lo leía y devolvía sus
columnas `tone_*` directamente, sin un solo llamador. Un bypass cargado y sin gatillo.

Decisión de gobierno (CXD-513): el CSV no se declara disponible sin productor/cutoff/
provenance; el fallback numérico muere y la superficie permanece `UNAVAILABLE`.

Estos candados son de `src/analysis/agent_tools.py` únicamente. El gemelo de
`weekly_generator._load_news_context` pertenece al carril CODEX de C-031; cuando caiga,
`test_no_module_under_analysis_reads_the_ungoverned_csv` se puede ampliar a todo
`src/analysis/` borrando la excepción declarada abajo.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import src.analysis.agent_tools as agent_tools

MODULE_PATH = Path(agent_tools.__file__)
ANALYSIS_DIR = MODULE_PATH.parent

# Ruta del CSV sin gobierno, partida para que este archivo no sea su propio contraejemplo.
UNGOVERNED_CSV = "gdelt_daily" + "_sentiment.csv"
TONE_COLUMN = "tone" + "_avg"


def _code(path: Path) -> str:
    """Fuente sin lineas de comentario: la nota que explica POR QUE el lector murio
    nombra el CSV y el tono, y no debe contar como una lectura."""
    return "\n".join(
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )


def test_the_ungoverned_sentiment_loader_stays_deleted() -> None:
    assert not hasattr(agent_tools, "load_gdelt_sentiment"), (
        "load_gdelt_sentiment devolvia las columnas tone_* del CSV sin feature_status, "
        "cutoff ni provenance; si vuelve, vuelve el bypass del gate C028"
    )


def test_agent_tools_does_not_reference_the_ungoverned_csv() -> None:
    code = _code(MODULE_PATH)
    assert UNGOVERNED_CSV not in code
    assert TONE_COLUMN not in code


def test_the_article_loader_that_remains_carries_no_sentiment_number() -> None:
    """`load_gdelt_articles` es la via viva: titulos, nunca tono."""
    body = _code(MODULE_PATH).split("def load_gdelt_articles")[1].split("\ndef ")[0]
    assert 'usecols=["date", "title", "source", "domain", "language"]' in body
    for numeric in ("tone", "sentiment", "score"):
        assert numeric not in body.lower().replace("gdelt_articles", ""), (
            f"load_gdelt_articles no debe traer '{numeric}': el tono se gobierna por "
            "quality.feature_status, no por un CSV"
        )


def test_no_module_under_analysis_reads_the_ungoverned_csv() -> None:
    """Candado repo-parcial: hoy solo `weekly_generator` queda, y es el carril CODEX."""
    pendiente_codex = {"weekly_generator.py"}
    culpables = {
        path.name for path in ANALYSIS_DIR.glob("*.py") if UNGOVERNED_CSV in _code(path)
    }
    assert culpables <= pendiente_codex, (
        f"lectores nuevos del CSV sin gobierno: {sorted(culpables - pendiente_codex)}"
    )


def test_removing_the_loader_did_not_break_its_importers() -> None:
    """`agent_graph` importa el loader de articulos, no el de sentimiento."""
    graph_source = (ANALYSIS_DIR / "agent_graph.py").read_text(encoding="utf-8")
    assert "load_gdelt_articles" in graph_source
    assert "load_gdelt_sentiment" not in graph_source


@pytest.mark.parametrize("nombre", ["load_daily_ohlcv", "load_gdelt_articles", "get_cop_series"])
def test_the_rest_of_the_public_surface_survives(nombre: str) -> None:
    assert callable(getattr(agent_tools, nombre))
