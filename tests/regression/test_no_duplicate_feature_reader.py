"""
Regression: un solo `FeatureReader` canonico (auditoria de limpieza 2026-08-24).

Estado que motivo el guard: `src/` definia la clase `FeatureReader` TRES veces, y
dos eran alcanzables a la vez por rutas de import distintas:

  1. `src/feature_store/feature_reader.py`         (420 L) <- lo importan los DAGs
  2. `src/feature_store/readers/feature_reader.py` (519 L) <- lo exportaba `__init__`
  3. `src/features/feature_reader.py`              (321 L) <- doble EN MEMORIA

(1) y (2) leen la MISMA tabla `inference_features_5m` y ambos docstrings decian ser
"single source of truth". Produccion usaba (1) por ruta directa mientras
`from src.feature_store import FeatureReader` resolvia a (2) — dos clases distintas
bajo el mismo nombre. Eso hace imposible el `feature_schema_hash` que el plan de
tesis exige como artefacto por corrida, y viola la regla DRY de CLAUDE.md
("same feature code for training and inference; never duplicate").

Resolucion: (1) es el canonico del paquete; (2) se exporta con nombre propio
(`ObservationFeatureReader`) porque cumple otro rol — construir la observacion
np.ndarray; (3) es un doble en memoria, no un lector de base de datos.

Las aserciones se derivan del contrato, no de un snapshot: si alguien anade un
cuarto lector o vuelve a exportar (2) como `FeatureReader` generico, esto falla.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

# El lector de la ruta de produccion. Los DAGs importan ESTA ruta.
CANONICAL = SRC / "feature_store" / "feature_reader.py"

# Lectores hermanos con rol declarado y distinto. Cualquier OTRO fichero que
# defina `class FeatureReader` es una regresion.
KNOWN_SIBLINGS = {
    SRC / "feature_store" / "readers" / "feature_reader.py",
    SRC / "features" / "feature_reader.py",
}

# Metodos que la ruta de produccion invoca sobre el lector canonico.
# Derivados de los call-sites reales, no de la implementacion:
#   airflow/dags/sensors/feature_sensor.py  -> get_latest_features, check_norm_stats_hash
#   airflow/dags/tasks/l5_inference_task.py -> get_features, get_feature_reader
PRODUCTION_API = {
    "has_features",
    "get_features",
    "get_latest_features",
    "check_norm_stats_hash",
}

DAG_CALLSITES = {
    ROOT / "airflow" / "dags" / "sensors" / "feature_sensor.py",
    ROOT / "airflow" / "dags" / "tasks" / "l5_inference_task.py",
}


def _classes_defined(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}


def _methods_of(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                n.name
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return set()


def test_canonical_reader_exists():
    assert CANONICAL.is_file(), f"falta el lector canonico: {CANONICAL}"
    assert "FeatureReader" in _classes_defined(CANONICAL)


def test_no_fourth_feature_reader_appears():
    """Solo el canonico y los dos hermanos declarados definen `FeatureReader`."""
    offenders = []
    for path in SRC.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if path == CANONICAL or path in KNOWN_SIBLINGS:
            continue
        try:
            if "FeatureReader" in _classes_defined(path):
                offenders.append(path.relative_to(ROOT).as_posix())
        except SyntaxError:  # pragma: no cover - fichero roto, otro test lo cazara
            continue
    assert not offenders, (
        "Nueva clase `FeatureReader` fuera de las rutas declaradas: "
        f"{offenders}. Reusa el canonico "
        "(src/feature_store/feature_reader.py) o declara el rol nuevo en este guard."
    )


def test_package_root_exports_the_production_reader():
    """`from src.feature_store import FeatureReader` debe dar el de produccion.

    Se comprueba sobre el AST del `__init__.py`, sin importar el paquete: importarlo
    arrastra psycopg2/feast y este guard debe correr en cualquier entorno.
    """
    init = SRC / "feature_store" / "__init__.py"
    tree = ast.parse(init.read_text(encoding="utf-8"))

    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                local = alias.asname or alias.name
                bound[local] = f"{node.module}.{alias.name}"

    assert bound.get("FeatureReader") == "feature_reader.FeatureReader", (
        "La raiz del paquete debe exportar el lector de PRODUCCION "
        "(.feature_reader), no el constructor de observacion (.readers). "
        f"Hoy resuelve a: {bound.get('FeatureReader')!r}"
    )
    assert bound.get("ObservationFeatureReader") == "readers.FeatureReader", (
        "El lector constructor de observacion debe exportarse con nombre propio "
        "`ObservationFeatureReader`, no como `FeatureReader` generico."
    )


def test_canonical_reader_keeps_the_api_the_dags_call():
    """Los DAGs son la ruta de produccion: su API no puede desaparecer en un merge."""
    methods = _methods_of(CANONICAL, "FeatureReader")
    missing = PRODUCTION_API - methods
    assert not missing, (
        f"El lector canonico perdio metodos que usan los DAGs: {sorted(missing)}. "
        "Consolidar hacia otra implementacion sin portarlos rompe "
        "l5_inference_task.py y feature_sensor.py."
    )


@pytest.mark.parametrize("dag_file", sorted(DAG_CALLSITES))
def test_dags_import_the_canonical_path(dag_file: Path):
    """Ningun DAG debe colgarse del hermano constructor de observacion."""
    if not dag_file.is_file():  # pragma: no cover - el DAG fue movido
        pytest.skip(f"{dag_file} no existe")
    text = dag_file.read_text(encoding="utf-8")
    assert "src.feature_store.feature_reader" in text, (
        f"{dag_file.name} deberia importar el lector canonico"
    )
    assert not re.search(r"from\s+src\.feature_store\.readers\s+import", text), (
        f"{dag_file.name} importa el hermano `.readers`, que NO implementa "
        f"{sorted(PRODUCTION_API)}."
    )


def test_siblings_declare_their_distinct_role():
    """Ningun hermano puede volver a reclamar ser el 'single source of truth'."""
    for sibling in sorted(KNOWN_SIBLINGS):
        if not sibling.is_file():
            continue
        head = sibling.read_text(encoding="utf-8")[:4000].lower()
        assert "single source of truth" not in head, (
            f"{sibling.relative_to(ROOT).as_posix()} vuelve a declararse "
            "'single source of truth'. El canonico es "
            "src/feature_store/feature_reader.py; este fichero debe declarar su rol."
        )
