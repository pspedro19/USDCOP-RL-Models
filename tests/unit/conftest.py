"""
Unit Tests conftest.py
Common fixtures and configuration for unit tests.
"""

import sys
from pathlib import Path

# `src` en el path — pero SIN adelantarlo, y esa distincion es el fondo del asunto.
#
# Aqui habia un `sys.path.insert(0, src)` incondicional. Parecia inofensivo: el conftest
# padre ya anade `src`, asi que esta linea solo lo movia al frente. El problema es que
# hay DOS paquetes llamados `contracts` en el repo —`src/contracts/` y
# `airflow/dags/contracts/`— y adelantar `src` decide cual gana, justo al reves de lo que
# `tests/conftest.py:30-36` declara CRITICO por escrito ("airflow/dags must come BEFORE
# ... MUST BE FIRST", y nombra a `test_all_layer_contracts` como el beneficiario).
#
# Medido neutralizando esta unica linea y restaurando por bytes:
#     CON la linea:  import contracts -> src/contracts/__init__.py
#     SIN la linea:  import contracts -> airflow/dags/contracts/__init__.py
# y `tests/unit/test_all_layer_contracts.py` pasaba de `37 skipped, EXIT=0` a `37 passed`
# al correrlo en foco.
#
# Por que no se notaba en CI: `tests/unit/airflow/` se colecta antes que los
# `tests/unit/test_*.py` y `tests/unit/airflow/test_sensors.py:20` vuelve a insertar
# `airflow/dags` al frente durante la coleccion, compensando esto por casualidad. Tres
# ficheros —uno declara el orden, este lo rompia, un tercero lo arreglaba sin saberlo— y
# nadie habia declarado esa cadena. En la suite completa el delta es cero; lo que se
# arregla es que el resultado deje de depender de que se colecte antes.
#
# El conftest padre ya pone `src` en el path. Esta guarda solo cubre el caso de que
# alguien corra este directorio sin el, y entonces **anade al final**: estar disponible
# es lo que hace falta; estar delante es lo que rompe.
project_root = Path(__file__).parent.parent.parent
_src = str(project_root / "src")
if _src not in sys.path:
    sys.path.append(_src)


# =============================================================================
# COLLECT IGNORE: Skip files that cause hard crashes
# =============================================================================

def check_onnxruntime_available():
    """
    Check if onnxruntime can be imported without crashing.
    On Windows, onnxruntime can cause access violations.
    """
    try:
        # This is a lightweight check - don't actually import onnxruntime
        # which would cause the crash. Instead, check if the package exists.
        import importlib.util
        spec = importlib.util.find_spec("onnxruntime")
        if spec is None:
            return False

        # On Windows, even if the package exists, it may crash on import
        # Check platform and skip on Windows to avoid the access violation
        if sys.platform == "win32":
            return False  # Skip on Windows to avoid access violation

        return True
    except Exception:
        return False


# Files to skip if onnxruntime is not available
_onnxruntime_dependent_files = [
    "test_onnx_converter.py",
]

# Build collect_ignore list
collect_ignore = []

if not check_onnxruntime_available():
    collect_ignore.extend(_onnxruntime_dependent_files)
