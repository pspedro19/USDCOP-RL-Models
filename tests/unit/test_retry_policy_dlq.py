"""El DLQ de L0 debe recibir de verdad las extracciones agotadas (CTR-L0-DLQ-001).

Por que existe este candado: `utils/retry_policy._save_to_dlq` importa
`services.dlq_service`. Cuando el paquete **raiz** `services/` esta en el path,
gana el nombre y `services.dlq_service` no resuelve; el `except ImportError` de
`_save_to_dlq` lo degrada a un `warning`, asi que el registro DLQ **no se escribe
y nadie se entera**.

Es CONDICIONAL AL DESPLIEGUE, comprobado contra contenedores vivos (2026-08-03):
`docker-compose.yml` monta `./services:/opt/airflow/services` en scheduler Y
webserver => el defecto muerde ahi (es el stack del fallo de `health_check` del
2026-07-27 que documenta `l0_macro_backfill._ensure_dags_services`).
`docker-compose.compact.yml` NO lo monta => alli `services.__path__` ya era
`['/opt/airflow/dags/services']` y el import funcionaba. Este candado reproduce la
condicion adversa a proposito, para que el arreglo no dependa del compose elegido.

Matiz importante que este test NO afirma: la tarea si fallaba. Ambos llamadores
propagan la excepcion original (`raise last_exception` y un `__exit__` que
devuelve `False`), asi que la senal de fallo nunca se perdio. Lo que se perdia en
silencio era el registro forense y, con el, cualquier reproceso basado en DLQ.
"""
import importlib
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RETRY_POLICY = REPO / "airflow" / "dags" / "utils" / "retry_policy.py"


def _load_retry_policy():
    """Cargar por ruta: `utils` tambien esta ensombrecido por `src/utils`."""
    spec = importlib.util.spec_from_file_location("dags_utils_retry_policy", RETRY_POLICY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dlq_module_resolves_even_when_root_services_wins_the_name():
    """Reproduce la condicion de Airflow: root `services` importado primero."""
    import services  # el paquete raiz que gana el nombre

    assert services.__path__, "services debe ser un paquete"

    _load_retry_policy()  # debe reparar la resolucion al importarse

    # Si esto lanza, `_save_to_dlq` caeria en su rama ImportError en produccion.
    importlib.import_module("services.dlq_service")


def test_save_to_dlq_reaches_the_service_instead_of_warning(monkeypatch):
    """El camino real de persistencia se ejecuta, no la degradacion silenciosa."""
    import services  # noqa: F401  (mismo orden de import que en el DAG)
    retry_policy = _load_retry_policy()

    dlq_module = importlib.import_module("services.dlq_service")

    saved = []

    class _FakeDLQ:
        def save_failed_extraction(self, **kwargs):
            saved.append(kwargs)

    monkeypatch.setattr(dlq_module, "get_dlq_service", lambda: _FakeDLQ())

    retry_policy._save_to_dlq(
        "fred",
        "FEDFUNDS",
        RuntimeError("upstream agotado"),
        {"function": "extract", "stats": {"attempts": 5}},
    )

    assert saved, "la extraccion agotada no llego al DLQ"
    assert saved[0]["source"] == "fred"
    assert saved[0]["variable"] == "FEDFUNDS"
    assert saved[0]["error_type"] == "RuntimeError"
