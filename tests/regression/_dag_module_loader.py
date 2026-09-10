"""Importador de modulos que viven bajo `airflow/dags/`, inmune al `sys.path`.

POR QUE (auditoria de limpieza 2026-08-24). Los tests que necesitan
`contracts.dag_registry` o `utils.data_quality` hacian
`sys.path.insert(0, "airflow/dags")` + `__import__("contracts.dag_registry")`.
Eso funciona SOLO si nadie mas ha puesto antes en el path otro paquete con ese
nombre — y el repo tiene **siete** paquetes llamados `contracts`:

    airflow/dags/contracts   config/contracts   services/inference_api/contracts
    src/contracts            src/core/contracts tests/contracts
    usdcop-trading-dashboard/lib/contracts

Sintoma real: `pytest tests/regression/` pasaba, pero `pytest tests/` (lo que hace
`make test`) daba `ModuleNotFoundError: No module named 'contracts.dag_registry'`,
porque incluir cualquier ruta de `tests/` mete `tests/` en el path y `contracts`
resuelve a `tests/contracts`, que no tiene `dag_registry`. CI no lo veia porque
corre `pytest tests/unit/` y `pytest tests/integration/` por separado.

Cargar por RUTA DE FICHERO elimina la dependencia del orden del path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

DAGS_ROOT = Path(__file__).resolve().parents[2] / "airflow" / "dags"


def import_dag_module(dotted: str) -> ModuleType:
    """Importa `paquete.modulo` desde `airflow/dags/` por ruta, no por `sys.path`.

    Registra los paquetes padre en `sys.modules` bajo un nombre con prefijo propio
    (`_dagpkg_<paquete>`) para no pisar ningun `contracts`/`utils` real del proceso.
    """
    parts = dotted.split(".")
    path = DAGS_ROOT.joinpath(*parts).with_suffix(".py")
    if not path.is_file():
        raise ModuleNotFoundError(f"no existe {path}")

    # El modulo puede hacer imports relativos a su paquete, asi que el paquete debe
    # existir. Se registra bajo un alias privado para no colisionar.
    pkg_alias = "_dagpkg_" + "_".join(parts[:-1]) if len(parts) > 1 else None
    if pkg_alias and pkg_alias not in sys.modules:
        pkg_init = DAGS_ROOT.joinpath(*parts[:-1], "__init__.py")
        if pkg_init.is_file():
            spec = importlib.util.spec_from_file_location(
                pkg_alias, pkg_init, submodule_search_locations=[str(pkg_init.parent)]
            )
            pkg = importlib.util.module_from_spec(spec)
            sys.modules[pkg_alias] = pkg
            # `airflow/dags` debe estar en el path SOLO mientras se ejecuta el
            # __init__, que puede importar hermanos por nombre corto.
            dags = str(DAGS_ROOT)
            added = dags not in sys.path
            if added:
                sys.path.insert(0, dags)
            try:
                spec.loader.exec_module(pkg)
            finally:
                if added:
                    sys.path.remove(dags)

    mod_alias = f"{pkg_alias}.{parts[-1]}" if pkg_alias else f"_dagmod_{parts[-1]}"
    if mod_alias in sys.modules:
        return sys.modules[mod_alias]

    spec = importlib.util.spec_from_file_location(mod_alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_alias] = module
    dags = str(DAGS_ROOT)
    added = dags not in sys.path
    if added:
        sys.path.insert(0, dags)
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_alias, None)
        raise
    finally:
        if added:
            sys.path.remove(dags)
    return module
