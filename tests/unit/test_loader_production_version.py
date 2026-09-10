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


def test_asset_resolution_uses_manifest_identity_not_a_misleading_prefix(tmp_path) -> None:
    strategy = tmp_path / "btc_name_but_gold_contract"
    strategy.mkdir()
    (strategy / "manifest.json").write_text(
        json.dumps({
            "strategy_id": "btc_name_but_gold_contract",
            "asset_id": "xauusd",
            "symbol": "XAU/USD",
        }),
        encoding="utf-8",
    )

    assert metrics.resolve_asset(strategy.name, strategy) == ("XAU/USD", "xauusd", 250)


@pytest.mark.parametrize(
    ("manifest", "warning"),
    [
        ({"strategy_id": "another_id", "asset_id": "btcusdt"}, "identidad divergente"),
        ({"strategy_id": "btc_candidate", "asset_id": "unknown"}, "asset_id desconocido"),
    ],
)
def test_asset_resolution_fails_closed_on_invalid_manifest_identity(
    tmp_path, capsys, manifest, warning
) -> None:
    strategy = tmp_path / "btc_candidate"
    strategy.mkdir()
    (strategy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert metrics.resolve_asset(strategy.name, strategy) == (None, None, None)
    assert warning in capsys.readouterr().out


def test_asset_resolution_fails_closed_on_a_corrupt_manifest(tmp_path, capsys) -> None:
    strategy = tmp_path / "btc_candidate"
    strategy.mkdir()
    (strategy / "manifest.json").write_text("{broken", encoding="utf-8")

    assert metrics.resolve_asset(strategy.name, strategy) == (None, None, None)
    assert "manifest invalido" in capsys.readouterr().out


def test_asset_resolution_keeps_prefix_fallback_only_without_a_manifest(tmp_path) -> None:
    strategy = tmp_path / "btc_legacy"
    strategy.mkdir()

    assert metrics.resolve_asset(strategy.name, strategy) == ("BTC/USDT", "btcusdt", 365)


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


# ---------------------------------------------------------------------------
# CLD-706 §2 — el gatillo de PROMOCION. Anadido bajo lease CLAUDE 2026-08-11,
# autorizado por el operador; ACK conceptual de Codex en CXD-829.
# ---------------------------------------------------------------------------
def _trades_con_rango(path: Path, timestamps: list[str], inicio: str, fin: str) -> None:
    """Artefacto de produccion que DECLARA su cobertura, como hacen los reales."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "strategy_id": path.stem,
        "date_range": {"start": inicio, "end": fin},
        "trades": [{"timestamp": stamp} for stamp in timestamps],
    }), encoding="utf-8")


def test_declared_coverage_survives_the_asset_being_promoted(tmp_path, monkeypatch) -> None:
    """Promover un activo NO puede borrarle historia publicada.

    El gatillo, medido antes de este candado: `gold_dynamic_exit.json` declara
    `date_range 2026-01-01..2026-07-21` pero contiene un trade entrado el 2025-12-30 —la
    posicion abierta al inicio de su ventana—. Con la autoridad inferida de los SELLOS, ese
    huerfano reclamaba TODO 2025 en cuanto el manifest ganaba `production.model_version`, y
    la cartera pasaba de 19.48%/3.22% a 14.79%/0.89% re-destruyendo 11 trades. Sin error y
    sin aviso.

    El artefacto ya publica su cobertura; leerla es invariante al puntero del manifest, al
    nombre del fichero y al calendario. Este test fija esa invariancia: **el mismo artefacto
    con y sin puntero de produccion debe cargar exactamente los mismos trades.**
    """
    estrategia = tmp_path / "gold_like"
    _bundle(estrategia, "1.0.0", (2025,))
    _trades(estrategia / "backtests" / "1.0.0" / "trades_2025.json",
            ["2025-03-01", "2025-06-01", "2025-09-01"])
    produccion = tmp_path / "production"
    # Trade huerfano de dic-2025 dentro de un artefacto que declara cubrir SOLO 2026.
    _trades_con_rango(produccion / "gold_like.json", ["2025-12-30", "2026-02-02"],
                      "2026-01-01", "2026-07-21")
    monkeypatch.setattr(metrics, "PROD_TRADES", produccion)

    esperado = ["2025-12-30", "2026-02-02", "2025-03-01", "2025-06-01", "2025-09-01"]

    for puntero in (None, {"model_version": "1.0.0"}):
        (estrategia / "manifest.json").write_text(
            json.dumps({"strategy_id": "gold_like", "production": puntero}), encoding="utf-8")
        cargado = [t["timestamp"] for t in metrics.load_trades(estrategia)]
        assert cargado == esperado, (
            f"con production={puntero!r} la historia de 2025 cambio: {cargado}")


def test_declared_coverage_still_suppresses_the_year_it_owns(tmp_path, monkeypatch) -> None:
    """La invariancia no puede lograrse ignorando la autoridad: DENTRO de su rango declarado,
    la ausencia de un trade en produccion sigue siendo INFORMACION, no un hueco que tapar."""
    estrategia = tmp_path / "cop_like"
    _bundle(estrategia, "1.0.0", (2025,))
    _trades(estrategia / "backtests" / "1.0.0" / "trades_2025.json",
            ["2025-01-05", "2025-03-01"])
    produccion = tmp_path / "production"
    _trades_con_rango(produccion / "cop_like_2025.json", ["2025-03-01"],
                      "2025-01-01", "2026-01-02")
    (estrategia / "manifest.json").write_text(
        json.dumps({"strategy_id": "cop_like", "production": {"model_version": "1.0.0"}}),
        encoding="utf-8")
    monkeypatch.setattr(metrics, "PROD_TRADES", produccion)

    # 2025 lo cubre produccion: el trade de enero del bundle NO se cuela.
    assert [t["timestamp"] for t in metrics.load_trades(estrategia)] == ["2025-03-01"]
