"""
Regression: `config/trading_config.yaml` y los literales del codigo no divergen.

Contract: CTR-TRADING-CONFIG-001 · Date: 2026-08-24

## Contexto

Hasta el 2026-08-24 ese fichero **no existia**, pero OCHO sitios lo citaban como SSOT —dos
DAGs, el risk manager, `date_ranges.yaml`, un ADR y la guia de onboarding—. Los valores
vivian duplicados en literales de Python, y ya habian derivado: el docstring de
`RiskLimits` decia `cooldown_after_losses=3 / cooldown_minutes=30` mientras el codigo
devolvia 5/60, y el unico test que lo habria cazado estaba en `src/tests/`, fuera de
`testpaths`, sin ejecutarse nunca.

El fichero se creo con los valores REALES del codigo. Todavia no lo lee nadie: cablearlo
cambiaria el `code_hash` de modulos dentro del muro de congelacion de los manifiestos de
estrategia, y eso exige aprobacion del operador (ver el xfail BLOCKED_OPERATOR_DECISION en
`test_approval_mutual_exclusion.py`).

Mientras el cableado no ocurra, **este test es lo unico que impide que vuelvan a divergir**.
No necesita Postgres ni Airflow: lee el YAML y el AST de los modulos.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "trading_config.yaml"
RISK_MANAGER = ROOT / "src" / "risk" / "risk_manager.py"
L5_DAG = ROOT / "airflow" / "dags" / "l5_multi_model_inference.py"


@pytest.fixture(scope="module")
def cfg() -> dict:
    if not CONFIG.is_file():
        pytest.fail(
            f"falta {CONFIG.relative_to(ROOT).as_posix()}. Lo citan ocho sitios como SSOT; "
            "borrarlo devuelve el ancla colgante."
        )
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _dataclass_defaults(path: Path, class_name: str) -> dict[str, float]:
    """Extrae los defaults de un dataclass sin importar el modulo."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            out: dict[str, float] = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                    if isinstance(stmt.target, ast.Name) and isinstance(
                            stmt.value, ast.Constant):
                        out[stmt.target.id] = stmt.value.value
            return out
    raise AssertionError(f"{class_name} no encontrado en {path}")


def test_config_declares_its_status_honestly(cfg):
    """Si algun dia se cablea, el `status` debe dejar de decir 'declarative'."""
    assert cfg["status"] in {"declarative", "authoritative"}, (
        f"status desconocido: {cfg['status']!r}"
    )


def test_risk_limits_match_the_code(cfg):
    """Los defaults de `RiskLimits` son hoy la verdad operativa."""
    code = _dataclass_defaults(RISK_MANAGER, "RiskLimits")
    for key in ("max_drawdown_pct", "max_daily_loss_pct", "max_trades_per_day",
                "cooldown_after_losses", "cooldown_minutes"):
        assert key in code, f"RiskLimits ya no define {key}"
        assert cfg["risk"][key] == code[key], (
            f"divergencia en {key}: trading_config.yaml={cfg['risk'][key]} vs "
            f"RiskLimits={code[key]}. Uno de los dos cambio sin el otro — que es "
            "exactamente el fallo que este test existe para cazar."
        )


def test_signal_thresholds_match_the_dag(cfg):
    """Los cortes LONG/SHORT viven como literales en el DAG L5."""
    if not L5_DAG.is_file():  # pragma: no cover
        pytest.skip("l5_multi_model_inference.py no existe")
    text = L5_DAG.read_text(encoding="utf-8")
    long_t, short_t = cfg["thresholds"]["long"], cfg["thresholds"]["short"]
    assert f"threshold_long: float = {long_t}" in text, (
        f"el DAG L5 ya no usa threshold_long={long_t}; actualiza trading_config.yaml"
    )
    assert f"threshold_short: float = {short_t}" in text, (
        f"el DAG L5 ya no usa threshold_short={short_t}; actualiza trading_config.yaml"
    )
    assert short_t < 0 < long_t, "la banda muerta LONG/SHORT debe rodear el cero"


def test_it_does_not_duplicate_the_date_ranges(cfg):
    """Los rangos de fechas tienen su propio SSOT; duplicarlos crea la segunda verdad."""
    assert "date_ranges" not in cfg, (
        "trading_config.yaml no debe llevar rangos de fechas: su SSOT es "
        "config/date_ranges.yaml (que se declara a si mismo 'the AUTHORITATIVE source')."
    )
    for key, target in (("date_ranges_ssot", "config/date_ranges.yaml"),
                        ("research_partition_ssot", "config/research/partition.yaml")):
        assert cfg.get(key) == target, f"{key} debe apuntar a {target}"
        assert (ROOT / target).is_file(), f"{key} apunta a un fichero inexistente: {target}"
