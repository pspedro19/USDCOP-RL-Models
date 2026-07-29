"""El watchdog no debe apilar copias de un trabajo pesado sobre si mismo.

Origen (2026-07-28, evidencia en vivo): `docker exec usdcop-airflow-scheduler ps` mostro
`generate_weekly_forecasts.py` con PPID=1, **1276% de CPU y 58 minutos de reloj**, sin fila
en `task_instance` — o sea invisible desde la UI de Airflow. El healthcheck del scheduler
(10s) empezo a expirar y el contenedor quedo `unhealthy`. Causa: `core_watchdog` corre cada
hora y lanzaba estos trabajos con un `subprocess.Popen` a fuego y olvido, sin ninguna guarda
de duplicado; el trabajo tarda 30-45 min y una corrida lenta se solapa con la siguiente.

Estos tests protegen `_spawn_singleton`. La mutacion que los pone rojos es quitar la
comprobacion de "ya hay uno vivo" y volver al `Popen` directo.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WATCHDOG_PATH = REPO_ROOT / "airflow" / "dags" / "core_watchdog.py"


def _load_watchdog():
    """Carga el DAG con Airflow stubbeado.

    El repo tiene su propio directorio `airflow/`, que ensombrece al paquete real; por eso
    el modulo se carga por RUTA y las dependencias de Airflow se inyectan en `sys.modules`
    antes, en vez de confiar en el import normal.
    """
    class _StubDAG:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _StubOperator:
        def __init__(self, *args, **kwargs):
            pass

        # El fichero encadena tareas con `>>`; devolver el otro extremo permite `a >> b >> c`.
        # Las variantes reflejadas cubren `[t1, t2] >> t3`, que una lista no sabe resolver.
        def __rshift__(self, other):
            return other

        def __lshift__(self, other):
            return other

        def __rrshift__(self, other):
            return self

        def __rlshift__(self, other):
            return self

    stub_airflow = types.ModuleType("airflow")
    stub_airflow.DAG = _StubDAG
    stub_operators = types.ModuleType("airflow.operators")
    stub_python = types.ModuleType("airflow.operators.python")
    stub_python.PythonOperator = _StubOperator

    saved = {k: sys.modules.get(k) for k in ("airflow", "airflow.operators", "airflow.operators.python")}
    sys.modules["airflow"] = stub_airflow
    sys.modules["airflow.operators"] = stub_operators
    sys.modules["airflow.operators.python"] = stub_python
    try:
        spec = importlib.util.spec_from_file_location("_core_watchdog_under_test", WATCHDOG_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


@pytest.fixture
def watchdog(tmp_path, monkeypatch):
    monkeypatch.setenv("WATCHDOG_LOCK_DIR", str(tmp_path / "locks"))
    module = _load_watchdog()
    module.SPAWN_LOCK_DIR = tmp_path / "locks"
    return module


def _sleeper(seconds: int = 30) -> list:
    return [sys.executable, "-c", f"import time; time.sleep({seconds})"]


def test_second_spawn_is_refused_while_the_first_is_alive(watchdog, tmp_path):
    """Rojo si se quita la comprobacion de PID vivo y se vuelve al Popen directo."""
    log = str(tmp_path / "job.log")

    assert watchdog._spawn_singleton("job", _sleeper(), log) is True
    pid = int((watchdog.SPAWN_LOCK_DIR / "job.pid").read_text())

    try:
        # Segunda llamada con el primero AUN VIVO: es el caso real del watchdog horario.
        assert watchdog._spawn_singleton("job", _sleeper(), log) is False
        # Y no se ha sustituido el lock: el dueño sigue siendo el primero.
        assert int((watchdog.SPAWN_LOCK_DIR / "job.pid").read_text()) == pid
    finally:
        os.kill(pid, 9)


def test_a_stale_lock_from_a_dead_run_is_recycled(watchdog, tmp_path):
    """Un lock huerfano no puede bloquear para siempre: si el PID murio, se recicla."""
    log = str(tmp_path / "job.log")
    watchdog.SPAWN_LOCK_DIR.mkdir(parents=True, exist_ok=True)

    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (watchdog.SPAWN_LOCK_DIR / "job.pid").write_text(str(dead.pid))

    assert watchdog._spawn_singleton("job", _sleeper(), log) is True
    pid = int((watchdog.SPAWN_LOCK_DIR / "job.pid").read_text())
    assert pid != dead.pid
    os.kill(pid, 9)


def test_a_corrupt_lock_does_not_block_forever(watchdog, tmp_path):
    """Un lock ilegible es un fallo de escritura, no una corrida viva: no debe bloquear."""
    log = str(tmp_path / "job.log")
    watchdog.SPAWN_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    (watchdog.SPAWN_LOCK_DIR / "job.pid").write_text("no-soy-un-pid")

    assert watchdog._spawn_singleton("job", _sleeper(), log) is True
    os.kill(int((watchdog.SPAWN_LOCK_DIR / "job.pid").read_text()), 9)


def test_different_jobs_do_not_block_each_other(watchdog, tmp_path):
    """La guarda es POR TRABAJO: el forecasting corriendo no puede bloquear el analisis."""
    log = str(tmp_path / "job.log")

    assert watchdog._spawn_singleton("forecasts", _sleeper(), log) is True
    assert watchdog._spawn_singleton("analysis", _sleeper(), log) is True

    for name in ("forecasts", "analysis"):
        os.kill(int((watchdog.SPAWN_LOCK_DIR / f"{name}.pid").read_text()), 9)


def test_every_heavy_job_goes_through_the_guard():
    """Ningun `Popen` suelto puede volver al fichero: el perimetro es el fichero entero.

    Rojo si alguien añade una accion de auto-heal nueva con `subprocess.Popen` directo,
    que es exactamente como nacio el defecto.
    """
    source = WATCHDOG_PATH.read_text(encoding="utf-8")
    popen_sites = source.count("subprocess.Popen(")
    assert popen_sites == 1, (
        f"core_watchdog.py tiene {popen_sites} llamadas a subprocess.Popen; solo puede haber "
        "UNA, la de _spawn_singleton. Un Popen suelto vuelve a permitir que dos copias del "
        "mismo trabajo pesado se solapen y ahoguen al scheduler."
    )
