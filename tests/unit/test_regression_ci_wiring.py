# -*- coding: utf-8 -*-
"""El job de regresión de CI está cableado, y toda excepción suya está declarada.

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
  * las exclusiones son **exactamente** las declaradas en `CUARENTENA`, ni una más;
  * cada una se **EJECUTA** en su step propio, nombrado y visible. Una exclusión que no
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

import tomllib
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CI = REPO / ".github" / "workflows" / "ci.yml"
JOB = "regression-test"

#: Las únicas exclusiones admitidas. **Hoy está vacía, y eso es el estado deseado**: el
#: step bloqueante corre `tests/regression/` entero, sin `--ignore`. Añadir una entrada
#: aquí es lo que autoriza a excluir un fichero del YAML, y exige motivo, owner y
#: condición de cancelación.
CUARENTENA: dict[str, str] = {}

#: Cuarentenas RETIRADAS al cumplirse su condición. Se dejan escritas —no borradas— porque
#: el valor de una lista de excepciones está en que se vea que se vacía:
#:
#:   test_action_threshold_ssot.py — importar `src.training.config` (dataclass pura)
#:       disparaba el `__init__` del paquete y con él `stable_baselines3`. Cerrada en
#:       `9b67ffa8` volviendo los reexports lazy (PEP 562); el gate pasa 2/2 con sb3
#:       genuinamente ausente.
#:
#:   test_macro_clean_fx_scale.py — empalme de escala real (USDMXN ×10⁴, USDCLP ×10²)
#:       desde 2026-06-26/29. Cerrada el 2026-08-06: se cargó `macro_indicators_daily`
#:       con la reparación declarada, el backup se regeneró desde esa base sana y
#:       `MACRO_DAILY_CLEAN` a su vez; el detector pasa con 0 empalmes. El dato no se
#:       perdonó, se arregló — que es la única forma legítima de retirar una cuarentena.


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

    Con `CUARENTENA` vacía este bucle no recorre nada, así que **no basta con él**: el
    régimen sin cuarentenas lo cubre `test_with_no_quarantines_the_blocking_step_has_no_ignores`.
    Se dice aquí porque un test que itera una lista vacía y pasa es justamente la forma de
    verde que este fichero persigue.
    """
    for ruta in CUARENTENA:
        assert (REPO / ruta).is_file(), (
            f"{ruta} esta en cuarentena pero NO existe. Una exclusion sin sujeto es ruido "
            f"heredado: borrarla del workflow en vez de dejarla"
        )


def test_with_no_quarantines_the_blocking_step_has_no_ignores() -> None:
    """El régimen SIN cuarentenas se comprueba explícitamente, no por ausencia de tests.

    Al vaciar `CUARENTENA` el 2026-08-06, dos comprobaciones se quedaron sin sujeto: el
    bucle de existencia no recorría nada y el test parametrizado se **saltaba** con «got
    empty parameter set». Tres tests dejaron de juzgar y el fichero seguía en verde — el
    defecto exacto que persigue.

    Así que el estado «cero cuarentenas» se afirma en positivo: si la lista está vacía, el
    YAML no puede llevar NINGÚN `--ignore` ni ningún step de cuarentena.
    """
    if CUARENTENA:
        pytest.skip("hay cuarentenas declaradas: las cubren los tests de su régimen")

    run = _bloqueante()["run"]
    assert "--ignore=" not in run, (
        f"`CUARENTENA` está vacía pero el step bloqueante excluye ficheros: {run!r}. "
        f"Una exclusión sin entrada en la lista es exactamente lo que este candado impide"
    )
    pasos_cuarentena = [
        s for s in _steps_con_run() if (s.get("name") or "").upper().find("QUARANTINE") >= 0
    ]
    assert not pasos_cuarentena, (
        f"quedan steps de cuarentena sin entrada en `CUARENTENA`: "
        f"{[s.get('name') for s in pasos_cuarentena]}"
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


def test_the_ignores_are_exactly_the_declared_quarantines() -> None:
    """Las exclusiones del YAML == `CUARENTENA`, en los dos sentidos.

    El nombre es **agnóstico del conteo** a propósito. Se llamaba
    `..._exactly_two_ignores_...` y al cerrar la cuarentena de `action_threshold` en
    `4a27b74a` pasó a haber una: el nombre quedó mintiendo el mismo día que se escribió.
    Un test cuyo nombre afirma un número obliga a recordar renombrarlo, y eso no se
    recuerda — es la misma narrativa obsoleta que este repo arrastra en otros sitios.

    Rojo con: añadir un `--ignore` que no esté declarado arriba, o quitar uno que sí.
    La lista de `CUARENTENA` es donde esa decisión se discute, no el YAML de tapadillo.
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


def _install_step() -> dict:
    pasos = [s for s in _steps_con_run() if "pip install -e" in s["run"]]
    assert len(pasos) == 1, (
        f"se esperaba un unico step de instalacion en `{JOB}`, hay {len(pasos)}"
    )
    return pasos[0]


def test_the_job_installs_the_extra_that_provides_the_parquet_engine() -> None:
    """El job instala la extra que trae `pyarrow`. Sin ella, rojo por el motivo equivocado.

    SEIS ficheros de `tests/regression/` tocan parquet y **tres están en el step
    bloqueante** (`test_cop_features_pit`, `test_feature_contracts`,
    `test_macro_features_are_live`). Sin engine, `pandas.read_parquet` lanza ImportError:
    el job caería, sí, pero por una dependencia ausente y no por el contrato que vigila.
    Peor todavía en la cuarentena FX, que moriría **antes** de llegar al assert de escala
    — o sea, perdiendo exactamente el diagnóstico por el que ese step existe.

    Lo señaló Codex (CXD-689) revisando `1fb0f44c`, donde este candado dio 6 verdes pese
    al defecto: no miraba las dependencias, así que su verde no cubría esto.

    NO se fija el nombre "data" a mano: se busca **qué extra declara pyarrow** y se exige
    ésa. Si mañana `pyarrow` se mueve de extra, este test sigue vigilando lo correcto en
    vez de proteger un nombre obsoleto.

    Rojo con: quitar esa extra de la línea de instalación.
    """
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    proveedoras = sorted(
        nombre for nombre, paquetes in extras.items()
        if any("pyarrow" in str(p).lower() for p in paquetes)
    )
    assert proveedoras, (
        "ninguna extra de pyproject declara `pyarrow`: o se movio a las dependencias base "
        "—y entonces este test sobra— o el repo se quedo sin engine de parquet"
    )
    run = _install_step()["run"]
    assert any(f"{n}]" in run or f"{n}," in run for n in proveedoras), (
        f"el job `{JOB}` no instala ninguna de las extras que traen pyarrow "
        f"({proveedoras}); su linea es:\n{run}\n"
        f"Sin engine de parquet, tres gates del step BLOQUEANTE y la cuarentena FX "
        f"fallan por ImportError en vez de por lo que vigilan"
    )


def test_the_heavy_ml_extra_is_not_installed() -> None:
    """La contrapartida: `ml` NO entra, y eso también se fija.

    `ml` arrastra `stable-baselines3` y con él torch a cada corrida de CI. Instalarlo
    haría pasar la cuarentena de `action_threshold`, sí — pero es una decisión de coste
    que este slice no toma, y que se tomaría a la vista, no colándola al añadir extras.

    Si alguien decide meterlo, este test cae y obliga a retirar también esa cuarentena:
    las dos cosas van juntas o el YAML queda mintiendo.
    """
    run = _install_step()["run"]
    assert "ml]" not in run and "ml," not in run, (
        f"el job instala la extra `ml` (torch en cada corrida). Si es intencionado, "
        f"retirar tambien la cuarentena de test_action_threshold_ssot.py, que existe "
        f"precisamente porque `ml` NO se instala:\n{run}"
    )


@pytest.mark.parametrize("ruta", sorted(CUARENTENA) or [None])
def test_each_quarantined_file_is_still_executed_in_a_visible_step(ruta) -> None:
    """Cada excluido se EJECUTA en su propio step nombrado y no bloqueante.

    Ésta es la aserción que separa "cuarentena" de "barrido bajo la alfombra". Excluir sin
    ejecutar es borrar el fichero del mapa: se queda roto para siempre y nadie se entera,
    porque no hay salida que mirar.

    Rojo con: borrar el step de cuarentena dejando el `--ignore` puesto.

    Con la lista vacía `parametrize` recibiría un conjunto vacío y pytest **saltaría** el
    test sin decir nada útil; se le pasa `[None]` para que exista un caso y se declare por
    qué no juzga, en vez de desaparecer.
    """
    if ruta is None:
        pytest.skip("sin cuarentenas declaradas — lo cubre el test del régimen vacío")
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
