# -*- coding: utf-8 -*-
"""El bundle que se evalúa es el que el manifest declara PRODUCTION.

EL DEFECTO, medido el 2026-08-06
--------------------------------
`production_backtest_dir` elegía por los dígitos del nombre del directorio. Para
`smart_simple_v11` eso da `3.0.0-A`/`3.0.0-B` → [3,0,0], por encima de `2.0.0`. Pero
3.0.0-A/B son variantes de **investigación**: el manifest las lista sólo como `backtests`,
no tienen rol de producción y **no contienen 2026**. El manifest declara
`production.model_version = 2.0.0`, que sí trae los dos años.

Aguas abajo eso se veía como "USD/COP no opera en 2026": la cartera walk-forward medía
0.0% de días en mercado y la ponderación inverse-vol le daba peso casi infinito a una
sleeve sin posición (CXD-807). El síntoma parecía una propiedad del mercado; la causa era
que se leía un bundle que nunca se desplegó.

DIRECCIÓN DEL EFECTO
--------------------
Seguir el manifest **sube** COP 2025 de +18.73% a +25.63% y crea un 2026 de +1.77%. Que un
arreglo mueva el número a favor obliga a demostrar que la regla no es "elegir el mejor":
`3.0.0-A` da +26.58% en 2025 y **tampoco** se elige. Eso es lo que fija
`test_the_declared_version_wins_even_when_another_scores_better`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pandas")

from scripts.analysis import asset_year_metrics as metrics  # noqa: E402
from scripts.analysis.asset_year_metrics import production_backtest_dir  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ESTRATEGIAS = REPO / "usdcop-trading-dashboard" / "public" / "data" / "strategies"


def _bundle(base: Path, version: str, anios: tuple[int, ...]) -> None:
    d = base / "backtests" / version
    d.mkdir(parents=True)
    for a in anios:
        (d / f"trades_{a}.json").write_text("[]", encoding="utf-8")


def _trades(path: Path, timestamps: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"timestamp": stamp} for stamp in timestamps]), encoding="utf-8")


def test_the_manifest_production_pointer_wins_over_the_higher_number(tmp_path) -> None:
    """`2.0.0` declarada gana a `3.0.0-B`, que sólo es más alta.

    Rojo con la versión anterior: ordenaba por dígitos y devolvía 3.0.0-B.
    """
    s = tmp_path / "smart_simple_v11"
    _bundle(s, "2.0.0", (2025, 2026))
    _bundle(s, "3.0.0-A", (2025,))
    _bundle(s, "3.0.0-B", (2025,))
    (s / "manifest.json").write_text(
        json.dumps({"production": {"model_version": "2.0.0", "year": 2026}}), encoding="utf-8"
    )
    assert production_backtest_dir(s).name == "2.0.0"


def test_the_declared_version_wins_even_when_another_scores_better(tmp_path) -> None:
    """La regla sigue al manifest, NO al resultado.

    Se declara `1.0.0` teniendo `9.9.9` disponible. Si la función devolviera la "mejor" o
    la más alta, este test caería — y sería la prueba de que el arreglo anterior sólo
    estaba eligiendo el número que más conviene.
    """
    s = tmp_path / "cualquiera"
    _bundle(s, "1.0.0", (2025, 2026))
    _bundle(s, "9.9.9", (2025, 2026))
    (s / "manifest.json").write_text(
        json.dumps({"production": {"model_version": "1.0.0"}}), encoding="utf-8"
    )
    assert production_backtest_dir(s).name == "1.0.0"


def test_without_a_production_pointer_nothing_changes(tmp_path) -> None:
    """Las 17 sleeves que no declaran producción conservan el orden por dígitos."""
    s = tmp_path / "gold_trend_b2"
    _bundle(s, "1.2.0", (2025,))
    _bundle(s, "1.3.0", (2025,))
    (s / "manifest.json").write_text(json.dumps({"strategy_id": "gold_trend_b2"}), encoding="utf-8")
    assert production_backtest_dir(s).name == "1.3.0"


def test_a_declared_version_missing_on_disk_falls_back_and_says_so(tmp_path, capsys) -> None:
    """Si la declarada no existe, se avisa en voz alta y se cae al orden.

    Callarlo dejaría exactamente el defecto que este fichero cierra, sólo que invisible.
    """
    s = tmp_path / "rota"
    _bundle(s, "1.0.0", (2025,))
    (s / "manifest.json").write_text(
        json.dumps({"production": {"model_version": "7.7.7"}}), encoding="utf-8"
    )
    assert production_backtest_dir(s).name == "1.0.0"
    assert "7.7.7" in capsys.readouterr().out


def test_a_corrupt_manifest_does_not_break_the_loader(tmp_path) -> None:
    """Un manifest ilegible degrada al orden por versión, no revienta la evaluación."""
    s = tmp_path / "corrupta"
    _bundle(s, "2.0.0", (2025,))
    (s / "manifest.json").write_text("{ esto no es json", encoding="utf-8")
    assert production_backtest_dir(s).name == "2.0.0"


def test_a_generic_partial_extract_does_not_claim_its_whole_year(tmp_path, monkeypatch) -> None:
    """One orphan trade cannot erase the rest of a published year."""
    strategy = tmp_path / "partial_strategy"
    _bundle(strategy, "1.0.0", (2025,))
    _trades(strategy / "backtests" / "1.0.0" / "trades_2025.json", ["2025-01-02", "2025-06-02"])
    (strategy / "manifest.json").write_text(
        json.dumps({"strategy_id": "partial_strategy", "production": None}), encoding="utf-8"
    )
    production = tmp_path / "production"
    _trades(production / "partial_strategy.json", ["2025-01-02"])
    monkeypatch.setattr(metrics, "PROD_TRADES", production)

    assert [t["timestamp"] for t in metrics.load_trades(strategy)] == ["2025-01-02", "2025-06-02"]


def test_an_explicit_year_artifact_is_authoritative_even_with_one_trade(tmp_path, monkeypatch) -> None:
    """The ``_YYYY`` contract, rather than a count threshold, owns the full year."""
    strategy = tmp_path / "declared_strategy"
    _bundle(strategy, "1.0.0", (2025,))
    _trades(strategy / "backtests" / "1.0.0" / "trades_2025.json", ["2025-01-02", "2025-06-02"])
    (strategy / "manifest.json").write_text(
        json.dumps({"strategy_id": "declared_strategy", "production": None}), encoding="utf-8"
    )
    production = tmp_path / "production"
    _trades(production / "declared_strategy_2025.json", ["2025-01-02"])
    monkeypatch.setattr(metrics, "PROD_TRADES", production)

    assert [t["timestamp"] for t in metrics.load_trades(strategy)] == ["2025-01-02"]


def test_a_manifest_production_pointer_makes_a_generic_artifact_authoritative(
    tmp_path, monkeypatch
) -> None:
    strategy = tmp_path / "promoted_strategy"
    _bundle(strategy, "1.0.0", (2025,))
    _trades(strategy / "backtests" / "1.0.0" / "trades_2025.json", ["2025-01-02", "2025-06-02"])
    (strategy / "manifest.json").write_text(
        json.dumps({"production": {"model_version": "1.0.0"}}), encoding="utf-8"
    )
    production = tmp_path / "production"
    _trades(production / "promoted_strategy.json", ["2025-01-02"])
    monkeypatch.setattr(metrics, "PROD_TRADES", production)

    assert [t["timestamp"] for t in metrics.load_trades(strategy)] == ["2025-01-02"]


@pytest.mark.skipif(not ESTRATEGIAS.is_dir(), reason="bundles no presentes en este checkout")
def test_on_the_real_bundles_cop_resolves_to_the_production_version() -> None:
    """Sobre el disco real: COP debe resolver a 2.0.0, la única con 2026.

    Es el caso que originó todo, y se comprueba contra los ficheros publicados y no contra
    un tmp_path — un fixture sintético no habría detectado nunca el defecto.
    """
    cop = ESTRATEGIAS / "smart_simple_v11"
    if not cop.is_dir():
        pytest.skip("smart_simple_v11 no publicado en este checkout")
    elegida = production_backtest_dir(cop)
    assert elegida is not None and elegida.name == "2.0.0", (
        f"resolvió a {elegida.name if elegida else None}: la cartera volvería a ver COP "
        f"sin operar en 2026"
    )
    assert (elegida / "trades_2026.json").is_file(), (
        "la versión productiva no publica 2026: el hueco sería real y no del loader"
    )
