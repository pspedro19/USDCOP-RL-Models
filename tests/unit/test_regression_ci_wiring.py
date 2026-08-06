# -*- coding: utf-8 -*-
"""El job de regresión de CI está cableado, y sus excepciones son dos y declaradas.

POR QUÉ EXISTE
--------------
`tests/regression/` tiene 66 ficheros. Hasta 2026-08-06, CI sólo nombraba 13 —repartidos
entre `fabric-contracts.yml` y `specs-gate.yml`— y los otros 53 no los ejecutaba **nadie**.
Gates escritos, funcionando, verdes en local, y sin llamador.

No es un problema teórico: `test_macro_clean_fx_scale.py` llevaba detectando un empalme de
escala real en USDMXN (×10⁴) y USDCLP (×10²) desde 2026-06-26/29, y nadie lo vio porque
ningún workflow lo corría. El detector funcionaba; le faltaba quien lo llamara.

QUÉ FIJA ESTE FICHERO
---------------------
Que el cableado no se erosione en silencio, que es como se erosiona siempre. En concreto:

  * existe un job cuyo step **bloqueante** corre el **directorio** `tests/regression/`
    —no una lista de ficheros, que es como se vuelve a quedar fuera la mitad—;
  * ese step no se ablanda con `continue-on-error` ni con `|| true`;
  * hay **exactamente dos** exclusiones, y son las dos declaradas;
  * las dos se **EJECUTAN** en steps propios, nombrados y visibles. Una exclusión que no
    se ejecuta es un fichero borrado del mapa: nadie vuelve a mirar si sigue roja.

POR QUÉ CUARENTENA VISIBLE Y NO `xfail`
---------------------------------------
Se discutió con Codex y su argumento ganó: un `xfail` dentro del test se traga **cualquier
otro** fallo del mismo test y pierde el diagnóstico exacto. El step en rojo visible
conserva la salida completa y obliga a mirarla.

LO QUE ESTE CANDADO **NO** GARANTIZA
------------------------------------
Es un candado de **regresión**, no de imposibilidad. Comprueba la forma del YAML; alguien
decidido puede esquivarlo (un `-k` que filtre, un `pytest.ini` con `addopts`, mover el job
a otro fichero). No se vende como más de lo que es.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CI = REPO / ".github" / "workflows" / "ci.yml"
JOB = "regression-test"

#: Las DOS únicas exclusiones admitidas, con el motivo por el que existen. Añadir una
#: tercera hace fallar este fichero a propósito: la conversación tiene que pasar por aquí.
CUARENTENA = {
    "tests/regression/test_macro_clean_fx_scale.py":
        "empalme de escala real en MACRO_DAILY_CLEAN/MASTER (USDMXN x10^4, USDCLP x10^2) "
        "desde 2026-06-26/29; repair bloqueado porque la DB viva tiene 0 filas y no hay "
        "fuente autoritativa de la que reparar sin inventar numeros",
    "tests/regression/test_action_threshold_ssot.py":
        "importar `src.training.config` dispara `src/training/__init__.py`, que arrastra "
        "el stack RL y con el `stable_baselines3`; ese paquete vive solo en el extra `ml`, "
        "que este job no instala",
}


def _job() -> dict:
    doc = yaml.safe_load(CI.read_text(encoding="utf-8"))
    jobs = doc.get("jobs") or {}
    assert JOB in jobs, (
        f"el job `{JOB}` no existe en ci.yml. Sin el, los 66 gates de "
        f"`tests/regression/` vuelven a no ejecutarse en CI y su ausencia no la nota nadie"
    )
    return jobs[JOB]


def _steps_con_run() -> list[dict]:
    return [s for s in (_job().get("steps") or []) if "run" in s]


def _bloqueante() -> dict:
    candidatos = [
        s for s in _steps_con_run()
        if "pytest tests/regression/" in s["run"] and not s.get("continue-on-error")
    ]
    assert len(candidatos) == 1, (
        f"se esperaba EXACTAMENTE un step bloqueante que corra `tests/regression/`, hay "
        f"{len(candidatos)}: {[s.get('name') for s in candidatos]}"
    )
    return candidatos[0]


def test_the_quarantined_files_actually_exist() -> None:
    """Anti-vacuidad: no se excluye lo que no existe.

    Si un fichero en cuarentena se borra o se renombra, su exclusión pasa a proteger a
    nadie y su step a no ejecutar nada — verde por vacío. Se exige que el sujeto exista
    antes de dar por bueno el resto del candado.
    """
    for ruta in CUARENTENA:
        assert (REPO / ruta).is_file(), (
            f"{ruta} esta en cuarentena pero NO existe. Una exclusion sin sujeto es ruido "
            f"heredado: borrarla del workflow en vez de dejarla"
        )


def test_the_blocking_step_runs_the_whole_directory_not_a_handpicked_list() -> None:
    """El sujeto es el DIRECTORIO. Una lista de ficheros es cómo nacieron los 53 huérfanos.

    Rojo con: sustituir `pytest tests/regression/ ...` por una enumeración de ficheros.
    """
    run = _bloqueante()["run"]
    assert "pytest tests/regression/ " in run or run.rstrip().endswith("pytest tests/regression/"), (
        f"el step bloqueante no invoca el directorio completo:\n{run}\n"
        f"Enumerar ficheros es exactamente como 53 gates se quedaron sin llamador"
    )


def test_the_blocking_step_is_not_softened() -> None:
    """Nada de `continue-on-error` ni `|| true` en el step que debe bloquear.

    Un job que no puede fallar es un job decorativo, y encima uno que se lee como verde.
    """
    step = _bloqueante()
    assert not step.get("continue-on-error"), "el step bloqueante lleva continue-on-error"
    assert "|| true" not in step["run"], (
        "el step bloqueante esconde su codigo de salida con `|| true`: fallaria y CI lo "
        "leeria como verde"
    )


def test_there_are_exactly_two_ignores_and_they_are_the_declared_ones() -> None:
    """Dos exclusiones, ni una más, y las dos declaradas aquí con su motivo.

    Rojo con: añadir un tercer `--ignore`, o cambiar uno por otro fichero. La lista de
    arriba es el sitio donde esa decisión se discute, no el YAML de tapadillo.
    """
    run = _bloqueante()["run"]
    ignorados = {
        t.split("--ignore=", 1)[1].strip().rstrip("\\").strip()
        for t in run.split()
        if t.startswith("--ignore=")
    }
    # `--ignore=x` puede venir pegado a un salto de linea con backslash; se normaliza.
    ignorados = {i for i in ignorados if i}
    assert ignorados == set(CUARENTENA), (
        f"las exclusiones del step bloqueante son {sorted(ignorados)} y las declaradas son "
        f"{sorted(CUARENTENA)}. Toda exclusion nueva pasa por la lista de este fichero, "
        f"con motivo, owner y condicion que la cancela"
    )


@pytest.mark.parametrize("ruta", sorted(CUARENTENA))
def test_each_quarantined_file_is_still_executed_in_a_visible_step(ruta: str) -> None:
    """Cada excluido se EJECUTA en su propio step nombrado y no bloqueante.

    Ésta es la aserción que separa "cuarentena" de "barrido bajo la alfombra". Excluir sin
    ejecutar es borrar el fichero del mapa: se queda roto para siempre y nadie se entera,
    porque no hay salida que mirar.

    Rojo con: borrar el step de cuarentena dejando el `--ignore` puesto.
    """
    steps = [s for s in _steps_con_run() if ruta in s["run"] and s.get("continue-on-error")]
    assert len(steps) == 1, (
        f"{ruta} esta excluido del step bloqueante pero no tiene UN step propio "
        f"`continue-on-error` que lo ejecute (encontrados: {len(steps)}). Una exclusion "
        f"que ademas no corre es un gate borrado, no un gate en cuarentena.\n"
        f"Motivo declarado: {CUARENTENA[ruta]}"
    )
    nombre = steps[0].get("name") or ""
    assert "QUARANTINE" in nombre.upper(), (
        f"el step que ejecuta {ruta} se llama {nombre!r}: sin la marca en el nombre, un "
        f"rojo esperado se confunde con un rojo nuevo en la lista de checks"
    )
