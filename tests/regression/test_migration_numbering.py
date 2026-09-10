"""
Regression: numeracion y orden de las migraciones de base de datos.

Contexto (auditoria de limpieza 2026-08-24). `database/migrations/` tenia CUATRO
numeros duplicados — dos series fusionadas en algun momento:

    001_add_macro_table.sql       /  001_initial_setup.sql
    002_add_foreign_keys.sql      /  002_rename_usdmxn_column.sql
    025_forecast_experiments.sql  /  025_lineage_tables.sql
    056_admin_console_is_test.sql /  056_rbac_dynamic_roles.sql

POR QUE NO SE RENOMBRAN.  `scripts/ops/db_migrate.py` registra cada migracion **por
nombre de fichero** (`_migrations.filename VARCHAR(255) NOT NULL UNIQUE`) con guard
de checksum inmutable ("immutable migration checksum drift"). Renombrar un fichero
ya aplicado lo convierte en una migracion NUEVA a ojos del runner y se re-ejecutaria
contra toda base existente. `.claude/rules/data-freshness.md` lo dice sin rodeos:
*"Una migracion aplicada es inmutable; crear otra."*

COMO SE EJECUTAN REALMENTE (verificado, no asumido).  El runner NO descubre
`database/migrations/*.sql` por glob: usa **planes explicitos** (`MIGRATION_PLANS`),
y su propio comentario lo justifica — *"adding a file does not make it deployable"*.
El unico plan que si usa `sorted(glob("*.sql"))` es `legacy-init`, sobre
`init-scripts/`. Por eso este modulo vigila dos cosas distintas:

  - `database/migrations/`: el numero duplicado no dispara hoy un orden incorrecto,
    pero el nombre es la clave de registro inmutable, asi que la unica ventana para
    renumerar es ANTES de aplicar. El guard cierra esa ventana a tiempo.
  - `init-scripts/`: aqui el orden lo decide el alfabeto de verdad. Un numero
    duplicado si puede colar un script antes que su dependencia.

Y comprueba que todo fichero nombrado en un plan exista: un plan que apunta a un
fichero borrado falla en despliegue, no en CI.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = ROOT / "database" / "migrations"
INIT_SCRIPTS = ROOT / "init-scripts"
RUNNER = ROOT / "scripts" / "ops" / "db_migrate.py"

NUMBER_RE = re.compile(r"^(\d+)[_-]")

# Companion de rollback de la 033. No lleva prefijo numerico A PROPOSITO: no es una
# migracion hacia adelante y ningun plan la incluye. Se declara para que el guard no
# la confunda con un descuido, y para que un SEGUNDO rollback suelto si salte.
NON_FORWARD_ALLOWED = {"rollback_033_event_triggers.sql"}

# Colisiones heredadas. NO se anaden entradas aqui para silenciar un fallo nuevo:
# la migracion nueva se renumera ANTES de aplicarse, que es cuando aun se puede.
GRANDFATHERED: dict[str, set[str]] = {
    "001": {"001_add_macro_table.sql", "001_initial_setup.sql"},
    "002": {"002_add_foreign_keys.sql", "002_rename_usdmxn_column.sql"},
    "025": {"025_forecast_experiments.sql", "025_lineage_tables.sql"},
    "056": {"056_admin_console_is_test.sql", "056_rbac_dynamic_roles.sql"},
}


def _group_by_number(paths) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        m = NUMBER_RE.match(path.name)
        if m:
            groups[m.group(1).lstrip("0") or "0"] = groups[
                m.group(1).lstrip("0") or "0"
            ] | {path.name}
    return groups


def _migration_sql():
    return sorted(MIGRATIONS.glob("*.sql"))


def test_migrations_directory_is_populated():
    assert MIGRATIONS.is_dir(), f"falta {MIGRATIONS}"
    assert _migration_sql(), "no hay migraciones .sql"


def test_only_declared_files_skip_the_numeric_prefix():
    unnumbered = {
        p.name for p in _migration_sql() if not NUMBER_RE.match(p.name)
    } - NON_FORWARD_ALLOWED
    assert not unnumbered, (
        f"ficheros .sql sin prefijo numerico en database/migrations/: {sorted(unnumbered)}. "
        "Si es una migracion, numerala. Si es un rollback o un script auxiliar, "
        "declaralo en NON_FORWARD_ALLOWED explicando por que no es una migracion."
    )


def test_no_new_duplicate_migration_numbers():
    """Las 4 colisiones heredadas se toleran; una quinta es un fallo."""
    groups = _group_by_number(_migration_sql())
    expected = {k.lstrip("0") or "0": v for k, v in GRANDFATHERED.items()}
    offenders = {
        number: sorted(names)
        for number, names in groups.items()
        if len(names) > 1 and names != expected.get(number)
    }
    assert not offenders, (
        f"numero de migracion duplicado NUEVO: {offenders}.\n"
        "Renumera el fichero AHORA — una vez aplicado, su nombre es la clave "
        "inmutable en `_migrations.filename` y renombrarlo lo re-ejecuta contra "
        "las bases existentes."
    )


def test_grandfathered_collisions_still_exist_as_declared():
    """Si la deuda se paga (o cambia), este guard debe dejar de mentir."""
    groups = _group_by_number(_migration_sql())
    for number, expected in GRANDFATHERED.items():
        key = number.lstrip("0") or "0"
        actual = groups.get(key, set())
        assert actual == expected, (
            f"la colision heredada {number} ya no coincide con lo declarado.\n"
            f"  declarado: {sorted(expected)}\n"
            f"  real:      {sorted(actual)}\n"
            "Si se resolvio, borra su entrada de GRANDFATHERED en el MISMO commit."
        )


def test_globbed_plan_directory_has_no_duplicate_numbers():
    """`init-scripts/` es el unico plan descubierto por `sorted(glob('*.sql'))`.

    Aqui el orden lo decide el alfabeto de verdad, asi que un numero duplicado si
    puede ejecutar un script antes que su dependencia.
    """
    if not INIT_SCRIPTS.is_dir():  # pragma: no cover
        pytest.skip("init-scripts/ no existe")
    groups = _group_by_number(sorted(INIT_SCRIPTS.glob("*.sql")))
    dupes = {n: sorted(names) for n, names in groups.items() if len(names) > 1}
    assert not dupes, (
        f"numeros duplicados en init-scripts/*.sql: {dupes}. Este directorio SI se "
        "recorre por glob ordenado (MIGRATION_PLANS['legacy-init']), asi que el "
        "empate lo rompe el alfabeto."
    )


def test_every_file_named_in_a_migration_plan_exists():
    """Un plan que apunta a un fichero borrado falla en despliegue, no en CI."""
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    named: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.endswith(".sql") and NUMBER_RE.match(node.value):
                named.add(node.value)
    assert named, "no se extrajo ningun nombre de migracion de MIGRATION_PLANS"
    on_disk = {p.name for p in _migration_sql()}
    missing = sorted(named - on_disk)
    assert not missing, (
        f"db_migrate.py nombra migraciones que no existen en disco: {missing}"
    )


def test_ordering_satisfies_the_known_cross_file_dependency():
    """`002_add_foreign_keys` referencia una tabla que crea un `001_`."""
    creator = "001_add_macro_table.sql"
    consumer = "002_add_foreign_keys.sql"
    names = sorted(p.name for p in _migration_sql())
    if creator not in names or consumer not in names:  # pragma: no cover
        pytest.skip("las migraciones de la premisa ya no existen")
    assert names.index(creator) < names.index(consumer), (
        f"{consumer} referencia `macro_indicators_daily`, creada por {creator}, "
        "pero ahora ordena antes. Una instalacion limpia fallaria."
    )
    sql = (MIGRATIONS / consumer).read_text(encoding="utf-8", errors="ignore")
    assert "macro_indicators_daily" in sql, (
        f"{consumer} ya no referencia `macro_indicators_daily`: la premisa de este "
        "test cambio, revisalo en vez de borrarlo."
    )
