# -*- coding: utf-8 -*-
"""Resolver el namespace `services` cuando el paquete raiz gana el nombre.

Por que existe
--------------
`docker-compose.yml` (enterprise) monta `./services:/opt/airflow/services` en
`airflow-scheduler` y `airflow-webserver`. Con
`PYTHONPATH=/opt/airflow:/opt/airflow/dags`, el paquete **raiz** `services/` gana
el nombre `services`, y entonces:

* `from services.l2_data_quality_report import ...` -> `ModuleNotFoundError`
* `from services import BacktestRunnerFactory, ...`  -> `ImportError` (el
  `services/__init__.py` raiz no exporta esos simbolos)

`docker-compose.compact.yml` NO monta ese directorio, asi que ahi `services`
resuelve directo a `dags/services` y nada falla. El defecto es
**condicional al despliegue**, y por eso no lo ve el gate de importacion: los DAGs
afectados importan **dentro de las funciones de tarea**, luego reventarian en
*task runtime*, no al parsear.

Este helper es el punto unico. Se llama a nivel de modulo en el DAG: Airflow
re-importa el fichero del DAG en el proceso que ejecuta la tarea, asi que para
cuando corre el callable el `__path__` ya esta extendido.

Nota de duplicacion consciente: `utils/retry_policy.py` y `l0_macro_backfill.py`
llevan su propia copia de esta logica y NO se migran aqui. `retry_policy` se
carga **por ruta** en su test (con la raiz del repo fuera de `sys.path`), asi que
depender de `utils.dags_namespace` lo volveria no cargable en ese escenario, que
es justo el que su candado ejercita. Migrarlos exige rehacer ese candado; queda
declarado en vez de escondido.
"""
from __future__ import annotations

import sys
from pathlib import Path

DAGS_DIR = Path(__file__).resolve().parents[1]


def ensure_dags_namespace() -> list[str]:
    """Hacer que `services.<mod>` resuelva contra `dags/services`.

    Idempotente. Devuelve el `__path__` resultante del paquete `services` para
    que un test pueda afirmar sobre el resultado y no sobre el efecto secundario.
    """
    local_services = str(DAGS_DIR / "services")
    try:
        import services
    except ImportError:
        # Nadie gano el nombre todavia: lo que debe ser importable es el
        # directorio de DAGs, no el subdirectorio `services` -- si se añade el
        # subdirectorio, `dlq_service` quedaria como modulo TOP-LEVEL y
        # `services.dlq_service` seguiria fallando.
        dags_dir = str(DAGS_DIR)
        if dags_dir not in sys.path:
            sys.path.append(dags_dir)
        import services  # ahora si resuelve

    if local_services not in services.__path__:
        services.__path__.append(local_services)
    return list(services.__path__)
