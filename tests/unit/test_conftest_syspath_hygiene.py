# -*- coding: utf-8 -*-
"""`sys.path` de los tests: quién gana cuando DOS paquetes se llaman `contracts`.

EL REPO TIENE DOS
-----------------
    src/contracts/            → policy, strategy_schema, approval_store…
    airflow/dags/contracts/   → dag_registry, l0_data_contracts, l*_contracts

Sólo uno puede responder a `import contracts`. `tests/conftest.py:30-36` lo decide por
escrito y lo marca CRÍTICO:

    # CRITICAL: airflow/dags must come BEFORE services/inference_api because both
    # have a 'contracts' subpackage and test_all_layer_contracts needs the airflow one

CÓMO SE INCUMPLÍA
-----------------
`tests/unit/conftest.py` hacía después un `sys.path.insert(0, src)` **incondicional**,
que adelantaba `src` por delante de `airflow/dags` y le daba la vuelta a esa decisión.
Aislado neutralizando esa única línea y restaurando por bytes:

    CON la línea:  contracts → src/contracts/__init__.py
    SIN la línea:  contracts → airflow/dags/contracts/__init__.py

LO QUE COSTABA, DICHO CON PRECISIÓN
-----------------------------------
`test_all_layer_contracts.py` protege su import con un guard que hace `skip` si
`contracts.l0_data_contracts` no está, y con el orden invertido ese guard saltaba. Pero
**sólo al ejecutar ese fichero por su cuenta**:

    sólo ese fichero ............ 37 skipped, EXIT=0
    suite `tests/unit` completa .. 37 passed

O sea que **en CI corrían y pasaban**. Una versión anterior de este mismo docstring decía
"37 tests reportando éxito sin ejecutarse, y un CI que mire el código de salida los ve
verdes": era falso, lo escribí yo, y queda aquí dicho para que nadie lo vuelva a citar.

Por qué en la suite sí: `tests/unit/airflow/` se colecta antes que los
`tests/unit/test_*.py` —los directorios ordenan primero— y
`tests/unit/airflow/test_sensors.py:20` inserta `airflow/dags` al frente durante la
colección, compensando por casualidad. Tres ficheros: uno declara el orden, otro lo
rompía, un tercero lo arreglaba sin saberlo, y nadie había declarado esa cadena.

El daño real, entonces, no era "tests muertos" sino **dependencia del alcance**: el mismo
fichero se ejecutaba o no según con quién se le corriera, y el modo que fallaba es
justamente el que usa quien está depurándolo. Su mensaje de skip ofrecía encima una
salida que no existe —"run this file in isolation"—: en aislamiento
`tests/unit/conftest.py` se carga igual, así que saltaba igual.

QUÉ FIJA ESTE FICHERO
---------------------
Dos cosas, y ninguna mirando el `sys.path` vivo — ver el docstring de
`test_the_unit_conftest_does_not_push_src_to_the_front` para por qué eso no funcionaba:

  1. que `tests/unit/conftest.py` no vuelva a adelantar `src` (estático, sobre la fuente);
  2. que ejecutar `test_all_layer_contracts.py` en solitario no salte tests (subproceso);

más la higiene de la fixture `feature_builder`. No juzga a `FeatureBuilder` — eso es de
sus propios tests.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

# API privada de pytest a proposito: no hay equivalente publico para distinguir "la
# fixture fallo" de "la fixture no existe", y esa distincion es justo lo que separa este
# test de uno vacuo. Si un pytest futuro la mueve, el import revienta en la coleccion —
# ruidoso y localizado, que es como se quiere descubrir esto.
from _pytest.fixtures import FixtureLookupError

REPO = Path(__file__).resolve().parents[2]
# Nota: aquí había `SRC` y `DAGS` para comparar índices de `sys.path` en vivo. Se fueron
# con ese candado — ver el docstring de `test_the_unit_conftest_does_not_push_src_to_the_front`.


def test_both_contracts_packages_exist() -> None:
    """La premisa, fijada en vez de contada en un comentario.

    Si algún día uno de los dos se renombra, la colisión desaparece y este fichero hay
    que revisarlo a conciencia en vez de dejarlo vigilando por inercia.
    """
    assert (REPO / "src" / "contracts" / "__init__.py").is_file()
    assert (REPO / "airflow" / "dags" / "contracts" / "__init__.py").is_file()


def test_the_unit_conftest_does_not_push_src_to_the_front() -> None:
    """El candado de orden, hecho ESTÁTICO sobre la fuente. Y por qué no es dinámico.

    La primera versión comparaba `sys.path.index(DAGS) < sys.path.index(SRC)` sobre el
    path VIVO. Pasaba en foco y **fallaba en la suite completa**: allí `src` acababa en
    el índice 4 y `airflow/dags` en el 66, porque decenas de módulos de test insertan
    rutas al importarse. O sea, un candado cuyo veredicto dependía de con qué otros
    ficheros se le ejecutara — la misma clase de fragilidad que este fichero denuncia,
    reproducida por mí al escribirlo.

    `sys.path` es un global que casi todo el mundo muta; no es donde se fija un contrato.
    Lo que sí es estable es la FUENTE: `tests/unit/conftest.py` no debe adelantar `src`.
    Eso se comprueba aquí, y la consecuencia observable —que los 37 se ejecuten— se
    comprueba abajo en un subproceso limpio.

    Rojo con: devolver cualquier `sys.path.insert(0, ...)` a `tests/unit/conftest.py`.
    """
    fuente = (REPO / "tests" / "unit" / "conftest.py").read_text(encoding="utf-8")
    ofensivas = [
        linea.strip()
        for linea in fuente.splitlines()
        if "sys.path.insert" in linea and not linea.lstrip().startswith("#")
    ]
    assert not ofensivas, (
        f"`tests/unit/conftest.py` vuelve a adelantar rutas: {ofensivas}. "
        f"`tests/conftest.py` declara CRÍTICO que `airflow/dags` vaya ANTES que `src` "
        f"(dos paquetes `contracts`); insertar al frente aquí lo invierte y los 37 tests "
        f"de test_all_layer_contracts.py vuelven a saltar en bloque con EXIT=0. Si hace "
        f"falta que `src` esté disponible, `append` basta: disponible ≠ delante"
    )


def test_the_layer_contract_tests_run_when_invoked_on_their_own() -> None:
    """La consecuencia observable, medida en un SUBPROCESO limpio.

    Por qué subproceso y no `find_spec` aquí: dentro de esta sesión, `contracts` ya está
    resuelto y cacheado en `sys.modules` por quien haya importado antes, así que
    comprobarlo en proceso mide el estado ambiente, no el contrato. Medido: en la suite
    completa esa comprobación pasaba por caché mientras el orden real estaba invertido.

    Lo que se fija es lo que le ocurre a una persona que abre ese fichero para depurarlo:
    ejecutarlo SOLO debe correr sus tests, no saltárselos. Antes daba `37 skipped` con
    `EXIT=0` —y su mensaje sugería "run this file in isolation", que es justo lo que no
    funcionaba—; sólo se salvaba si `tests/unit/airflow/` se colectaba antes y arreglaba
    el path por casualidad.

    No se fija el número 37: eso es cuántos tests tiene hoy ese fichero y cambiará. Se
    fija que **no haya saltados** y que haya un número razonable de ejecutados.

    Rojo con: `sys.path.insert(0, src)` de vuelta en `tests/unit/conftest.py` (medido:
    vuelve a `37 skipped`).
    """
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/unit/test_all_layer_contracts.py",
            "-q", "--no-header", "-p", "no:randomly",
        ],
        cwd=str(REPO), capture_output=True, text=True, timeout=300,
    )
    salida = proc.stdout + proc.stderr
    resumen = next(
        (l for l in reversed(salida.strip().splitlines()) if " in " in l and "=" in l),
        salida[-200:],
    )
    assert "skipped" not in resumen, (
        f"ejecutado en solitario, `test_all_layer_contracts.py` salta tests: {resumen!r}. "
        f"Un skip masivo se ve idéntico a un verde en el código de salida — EXIT=0 no "
        f"significaría que pasan, significaría que no corren"
    )
    ejecutados = re.search(r"(\d+) passed", resumen)
    assert ejecutados and int(ejecutados.group(1)) >= 20, (
        f"apenas se ejecutaron tests: {resumen!r}. Sin sujeto, este candado dejaría de "
        f"significar algo (ese fichero tenía 37 cuando se escribió esto)"
    )


def test_the_feature_builder_fixture_leaves_sys_path_as_it_found_it(
    request: pytest.FixtureRequest,
) -> None:
    """Higiene de la fixture: no debe mutar el path de forma permanente.

    HONESTIDAD SOBRE EL ALCANCE: hoy la fixture no inserta nada, porque `src` ya está en
    el path cuando corre; su bloque de limpieza es inerte y mutarlo no pone rojo a nadie.
    Lo que este test SÍ atrapa es la regresión que importa —que alguien vuelva a meter un
    `insert` sin restauración en el cuerpo de la fixture—, verificado por mutación.

    Se compara la lista completa y en orden, no la pertenencia de `src`: una fixture que
    reordenara el path sin añadir nada causaría el mismo problema de resolución y una
    comprobación de pertenencia no lo vería.
    """
    antes = list(sys.path)
    try:
        request.getfixturevalue("feature_builder")
    except FixtureLookupError:
        # ANTI-VACUIDAD, y no es hipotetico: sin esta rama, renombrar la fixture dejaba
        # el test en VERDE. Medido —`feature_builder` → `feature_builder_RENOMBRADA`
        # daba `1 passed` en 0.21s— porque el `except` de abajo se tragaba el fallo de
        # lookup, `sys.path` no cambiaba y la comparacion final se cumplia sin haber
        # ejercitado nada. Lo señaló Codex (CXD-668) antes de que yo lo viera.
        pytest.fail(
            "la fixture `feature_builder` no existe: este test no ha ejercitado nada y "
            "su verde no significaria nada. Si se renombro, actualizar aqui; si se "
            "borro, borrar tambien este test en vez de dejarlo pasando en vacio"
        )
    except (Exception, pytest.skip.Exception):
        # Se traga CUALQUIER desenlace de la fixture a propósito. Hoy `FeatureBuilder()`
        # revienta con `norm_stats missing required features` —drift de contrato RL,
        # ajeno a esto— y una versión anterior de este test se ponía roja por eso: roja
        # por el motivo equivocado, que engaña igual que un verde falso.
        #
        # `pytest.skip.Exception` va EXPLÍCITA porque hereda de `BaseException`, no de
        # `Exception`: con sólo `except Exception` el skip de la fixture se propagaba y
        # este test se SALTABA en silencio — o sea, dejaba de juzgar sin decirlo. Medido,
        # no supuesto: pasó al escribirlo.
        #
        # Lo único que este test juzga es el `sys.path`, y los caminos de error/skip son
        # justo donde es más fácil olvidar restaurarlo.
        pass
    assert list(sys.path) == antes, (
        "la fixture `feature_builder` dejó `sys.path` modificado; con dos paquetes "
        "`contracts` en el repo eso hace que la resolución dependa del orden de colección"
    )
