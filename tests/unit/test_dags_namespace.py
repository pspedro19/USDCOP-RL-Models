# -*- coding: utf-8 -*-
"""El namespace `services` de los DAGs debe resolver bajo el compose enterprise.

Condicion que se reproduce aqui: `docker-compose.yml` monta
`./services:/opt/airflow/services` en `airflow-scheduler` y `airflow-webserver`, y
`PYTHONPATH=/opt/airflow:/opt/airflow/dags` pone la RAIZ primero. Entonces el
paquete raiz `services/` gana el nombre y los DAGs que importan `services.*`
fallan **en task runtime** — no al parsear, porque esos imports viven dentro de
las funciones de tarea. `docker-compose.compact.yml` no monta ese directorio, asi
que ahi nunca se ve. Ver CXD-303 / CLD-326.

Dos garantias distintas, y la segunda existe porque la primera NO basta:

1. `ensure_dags_namespace()` extiende `services.__path__`, con lo que los
   SUBMODULOS (`services.l2_data_quality_report`, `services.backtest_factory`...)
   vuelven a resolver.
2. Extender `__path__` **no re-ejecuta** el `__init__.py` del paquete que gano, asi
   que `from services import BacktestRunnerFactory` sigue fallando: esos simbolos
   son atributos del `__init__` raiz, que no los define. Por eso
   `l4_backtest_validation` importa por submodulo y no por paquete.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DAGS = REPO / "airflow" / "dags"
HELPER = DAGS / "utils" / "dags_namespace.py"


def _run_under_enterprise_layout() -> dict:
    """Ejecutar en subproceso con la RAIZ del repo por delante de dags/."""
    program = f"""
import json, sys, importlib, importlib.util
sys.path.insert(0, {str(DAGS)!r})
sys.path.insert(0, {str(REPO)!r})   # la raiz gana, como /opt/airflow en enterprise
import services
out = {{"winner": services.__file__}}
spec = importlib.util.spec_from_file_location("dags_namespace_probe", {str(HELPER)!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.ensure_dags_namespace()

def probe(fn):
    try:
        fn()
        return True
    except Exception as exc:
        return f"{{type(exc).__name__}}: {{exc}}"

out["submodule_l2"] = probe(lambda: importlib.import_module("services.l2_data_quality_report"))
out["submodule_backtest"] = probe(lambda: importlib.import_module("services.backtest_factory"))
out["submodule_validation"] = probe(lambda: importlib.import_module("services.validation_strategies"))
out["submodule_alerts"] = probe(lambda: importlib.import_module("services.alert_service"))
out["submodule_metrics"] = probe(lambda: importlib.import_module("services.metrics_exporter"))
out["root_services_common"] = probe(lambda: importlib.import_module("services.common.prometheus_metrics"))
def _bare():
    from services import BacktestRunnerFactory  # noqa: F401
out["bare_package_import"] = probe(_bare)
print(json.dumps(out))
"""
    proc = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env={**__import__("os").environ, "POSTGRES_PASSWORD": "test-only"},
    )
    assert proc.stdout.strip(), f"sin salida del subproceso: {proc.stderr[-800:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_enterprise_layout_is_actually_reproduced():
    """Si la raiz no gana el nombre, el resto del fichero no prueba nada."""
    result = _run_under_enterprise_layout()
    winner = Path(result["winner"]).resolve()
    assert winner == (REPO / "services" / "__init__.py").resolve(), (
        f"la condicion enterprise no se reprodujo; gano {winner}"
    )


def test_helper_restores_every_submodule_the_dags_need():
    result = _run_under_enterprise_layout()
    for key in (
        "submodule_l2",
        "submodule_backtest",
        "submodule_validation",
        "submodule_alerts",
        "submodule_metrics",
    ):
        assert result[key] is True, f"{key} no resolvio: {result[key]}"


def test_bare_package_import_still_fails_which_is_why_dags_use_submodules():
    """Documenta el limite del helper: `__path__` no re-ejecuta el `__init__`."""
    result = _run_under_enterprise_layout()
    assert result["bare_package_import"] is not True, (
        "si `from services import X` funcionara, la razon para importar por "
        "submodulo en l4 habria desaparecido y este candado debe revisarse"
    )


def test_dags_do_not_import_symbols_from_the_ambiguous_package():
    """Candado de fuente: volver a `from services import X` reintroduce el fallo."""
    offenders = []
    for dag in ("l4_backtest_validation.py", "l2_dataset_builder.py"):
        for number, line in enumerate((DAGS / dag).read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if "from services import" in code:
                offenders.append(f"{dag}:{number}")
    assert not offenders, (
        "importar simbolos del paquete `services` es ambiguo bajo el compose "
        f"enterprise; usar `from services.<modulo> import ...`: {offenders}"
    )


def test_root_services_common_is_preserved():
    """Extender el namespace NO debe robarle submodulos al paquete raiz.

    `services/macro_extraction_strategies.py` importa a proposito
    `services.common.prometheus_metrics` del paquete RAIZ. Como el `__path__` del
    ganador conserva su propia ruta en primera posicion, los submodulos raiz
    mantienen prioridad; este candado lo fija para que una futura "mejora" que
    reemplace el `__path__` en vez de extenderlo no rompa esa importacion.
    """
    result = _run_under_enterprise_layout()
    assert result["root_services_common"] is True, (
        f"se perdio el submodulo raiz: {result['root_services_common']}"
    )
