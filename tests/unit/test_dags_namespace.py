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


# Modulos que DEBEN resolver el namespace antes de sus imports de `services`.
# La clave es la razon, no el fichero: cada uno importa `services.*` en un punto
# donde el paquete raiz puede haber ganado el nombre.
_MUST_WIRE_NAMESPACE = {
    "utils/circuit_breaker.py": "carga services.metrics_exporter con `except ImportError: pass`",
    "l2_dataset_builder.py": "importa services.l2_data_quality_report dentro de la tarea",
    "l4_backtest_validation.py": "importa services.<submodulo> dentro de las tareas",
}


def test_modules_that_need_the_namespace_actually_call_the_helper():
    """Que el helper funcione no prueba que alguien lo llame.

    Este candado nace de un hueco propio: la bateria anterior seguia en 5P cuando
    CODEX retiro `ensure_dags_namespace()` de `circuit_breaker.py` en su
    cross-review adversarial de `835f836b`. Los tests ejercitaban el helper
    llamandolo ELLOS, asi que un modulo que dejara de invocarlo era invisible --
    exactamente el patron "mecanismo correcto sin llamador" que este repo ya
    tiene medido en `integration/AUDIT-CLAUDE-wiring-gap.md`.
    """
    missing = []
    for relative, reason in _MUST_WIRE_NAMESPACE.items():
        source = (DAGS / relative).read_text(encoding="utf-8")
        called = any(
            line.strip().startswith("ensure_dags_namespace()")
            for line in source.splitlines()
        )
        if not called:
            missing.append(f"{relative} ({reason})")
    assert not missing, (
        "estos modulos importan `services.*` pero ya no resuelven el namespace, "
        f"asi que bajo el compose enterprise fallarian en runtime: {missing}"
    )


def test_importing_circuit_breaker_alone_resolves_the_metrics_module():
    """Comprobacion END-TO-END. NO es el candado causal del caller.

    Importa `utils.circuit_breaker` bajo el layout enterprise sin precargar ni
    llamar el helper a mano, y exige que despues `services.metrics_exporter`
    resuelva. Eso es cierto y util: fija que el sistema, tal como se importa de
    verdad, deja las metricas resolubles.

    LIMITACION, medida y no supuesta: **este test NO atribuye ese efecto a
    circuit_breaker**. `utils/__init__.py:64` importa `retry_policy`, que resuelve
    el namespace por su cuenta, asi que cualquier `import utils.<algo>` arrastra
    el wiring. Verificado aislado: importar SOLO `utils.retry_policy` ya deja
    `services.metrics_exporter` resoluble. Consecuencia comprobada: al retirar la
    llamada de circuit_breaker, este test **sigue en verde** y el unico que cae es
    `test_modules_that_need_the_namespace_actually_call_the_helper`.

    El candado causal del caller es ese, el de fuente. Este no lo sustituye.
    """
    program = f"""
import json, sys, importlib
sys.path.insert(0, {str(DAGS)!r})
sys.path.insert(0, {str(REPO)!r})   # la raiz gana, como en enterprise
import services                      # el marker raiz se queda con el nombre
out = {{"winner": services.__file__}}
try:
    importlib.import_module("utils.circuit_breaker")   # unico wiring permitido
    out["imported_cb"] = True
except Exception as exc:
    out["imported_cb"] = f"{{type(exc).__name__}}: {{exc}}"
try:
    importlib.import_module("services.metrics_exporter")
    out["metrics_after_cb_import"] = True
except Exception as exc:
    out["metrics_after_cb_import"] = f"{{type(exc).__name__}}: {{exc}}"
print(json.dumps(out))
"""
    proc = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env={**__import__("os").environ, "POSTGRES_PASSWORD": "test-only"},
    )
    assert proc.stdout.strip(), f"sin salida: {proc.stderr[-800:]}"
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert Path(result["winner"]).resolve() == (REPO / "services" / "__init__.py").resolve()
    assert result["imported_cb"] is True, f"no se pudo importar circuit_breaker: {result['imported_cb']}"
    assert result["metrics_after_cb_import"] is True, (
        "importar `utils.circuit_breaker` ya no deja resuelto "
        f"`services.metrics_exporter`: {result['metrics_after_cb_import']}"
    )
