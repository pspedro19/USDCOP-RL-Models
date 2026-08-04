"""El servidor MCP de noticias es development-only (decisión del operador, 2026-08-04).

Contexto medido antes de la decisión (CLD-378/380/382):

- `news_articles` (migración 045) es el almacén canónico: lo escribe
  `src/news_engine/storage/database.py`, invocado por `airflow/dags/news_daily_pipeline.py`,
  `news_alert_monitor.py`, `news_weekly_digest.py` y `news_maintenance.py`.
- `news_articles_search` es el índice de esta herramienta MCP. Su DDL vive **fuera de
  todo plan revisado**: la crean `scripts/ops/migrate_csv_to_pg.py` y
  `mcp_server.py --init-db`, ambos con la misma forma.
- El MCP no aparece en ningún compose/Dockerfile/DAG y la librería `mcp` no está
  instalada en los contenedores: no puede ejecutarse en producción.

La decisión fue **declarar la deuda, no borrar la herramienta**: borrar
`migrate_csv_to_pg.py` destruiría la única vía de recarga del histórico GDELT.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: La tabla dev que NO debe entrar en ningún plan de migración gobernado.
TABLA_DEV = "news_articles_search"


def _doc_normalizado(texto: str) -> str:
    """Docstrings con saltos de línea: las frases del contrato los cruzan."""
    return re.sub(r"\s+", " ", texto or "").lower()


def _cargar_migrador():
    spec = importlib.util.spec_from_file_location(
        "db_migrate_dev_only", ROOT / "scripts" / "ops" / "db_migrate.py"
    )
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _planes_que_declaran_la_tabla_dev(migrador) -> dict[str, list[str]]:
    """Planes cuyo DDL declara la tabla dev.

    Itera ``MIGRATION_PLANS``, que es **la fuente de planes**. Iterar
    ``REQUIRED_TABLES_BY_PLAN`` fue un falso verde real (CXD-386): un plan puede
    existir —incluso estar en ``REVIEW_GATED_PLANS``— sin declarar required tables,
    y entonces quedaba sin inspeccionar pese a que el candado prometía "ningún plan".
    """
    infractores = {}
    for plan in migrador.MIGRATION_PLANS:
        declaradas = sorted(
            t for t in migrador.created_tables_for_plan(plan) if TABLA_DEV in t
        )
        if declaradas:
            infractores[plan] = declaradas
    return infractores


# ---------------------------------------------------------------------------
# Candados ESTRUCTURALES — la garantía dura
# ---------------------------------------------------------------------------

def test_dev_table_is_not_declared_by_any_governed_plan() -> None:
    """`news_articles_search` no puede aparecer en el DDL de ningún plan.

    Complementa el guard de `6858b8d2`: aquél exige que toda required table tenga
    DDL; éste exige que esta tabla de desarrollo **no entre** en un plan. Si alguien
    la gobierna, que sea por decisión y no por deriva.
    """
    migrador = _cargar_migrador()

    infractores = _planes_que_declaran_la_tabla_dev(migrador)
    assert not infractores, (
        f"{TABLA_DEV} entró en el DDL de un plan gobernado: {infractores}. "
        "Es esquema de una herramienta dev-only; gobernarla exige decisión del operador"
    )


def test_dev_table_lock_inspects_plans_without_required_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prueba causal versionada del falso verde de CXD-386.

    Un plan **sin entrada en `REQUIRED_TABLES_BY_PLAN`** debe seguir inspeccionándose.
    Si alguien vuelve a iterar el mapa de required en vez de la fuente de planes, este
    test cae — que es justo lo que una mutación manual no garantiza a futuro.
    """
    migrador = _cargar_migrador()

    sql = tmp_path / "999_synthetic_dev_plan.sql"
    sql.write_text(
        f"CREATE TABLE IF NOT EXISTS public.{TABLA_DEV} (id SERIAL PRIMARY KEY);\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(migrador.MIGRATION_PLANS, "synthetic-dev-plan", (sql,))
    assert "synthetic-dev-plan" not in migrador.REQUIRED_TABLES_BY_PLAN

    infractores = _planes_que_declaran_la_tabla_dev(migrador)
    assert "synthetic-dev-plan" in infractores, (
        "un plan sin mapa de required quedó sin inspeccionar: el candado promete "
        "'ningún plan' y debe recorrer MIGRATION_PLANS, no REQUIRED_TABLES_BY_PLAN"
    )


def test_dev_table_is_not_a_required_table_of_any_plan() -> None:
    """Tampoco puede exigirse como required table: nadie debe bloquear por ella."""
    migrador = _cargar_migrador()

    exigida = {
        plan: [t for t in requeridas if TABLA_DEV in t]
        for plan, requeridas in migrador.REQUIRED_TABLES_BY_PLAN.items()
    }
    con_req = {plan: t for plan, t in exigida.items() if t}
    assert not con_req, (
        f"{TABLA_DEV} figura como required table: {con_req}. "
        "Una herramienta dev-only no puede hacer fallar la validación de un plan"
    )


# ---------------------------------------------------------------------------
# Candados DOCUMENTALES — tripwire contra la deriva del texto, no prueba de semántica
# ---------------------------------------------------------------------------

#: Proposiciones POSITIVAS exigidas. Se exigen afirmaciones, nunca se prohíben frases:
#: un candado por frase prohibida se sortea reescribiendo el texto (lección de `a9abfc7e`).
_PROPOSICIONES_MCP = (
    ("dev-only", "development-only", "debe declararse development-only"),
    (
        "fuera-de-plan",
        "fuera de todo plan",
        "debe decir que su DDL vive fuera de todo plan revisado",
    ),
)


def test_mcp_server_docstring_declares_development_only() -> None:
    """El artefacto debe declarar su propio estado, no depender de una nota externa.

    Tripwire textual: no demuestra semántica. La garantía dura son los candados
    estructurales de arriba.
    """
    doc = _doc_normalizado(
        (ROOT / "src" / "news_engine" / "mcp_server.py").read_text(encoding="utf-8")[:2500]
    )
    faltan = [
        f"{nombre} ({motivo})"
        for nombre, frase, motivo in _PROPOSICIONES_MCP
        if frase not in doc
    ]
    assert not faltan, "el docstring de mcp_server.py no declara: " + "; ".join(faltan)


def test_csv_migration_script_declares_its_ddl_is_out_of_plan() -> None:
    """El segundo creador de la tabla dev debe declararlo también."""
    doc = _doc_normalizado(
        (ROOT / "scripts" / "ops" / "migrate_csv_to_pg.py").read_text(encoding="utf-8")[:2000]
    )
    assert "fuera de todo plan" in doc, (
        "migrate_csv_to_pg.py crea news_articles_search y debe declarar que su DDL "
        "no pertenece a ningún plan de migración revisado"
    )
