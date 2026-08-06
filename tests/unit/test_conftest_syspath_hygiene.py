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

LO QUE COSTABA (la razón de que esto sea un candado y no un detalle de estilo)
-----------------------------------------------------------------------------
`test_all_layer_contracts.py` protege su import con un guard que hace `skip` si
`contracts.l0_data_contracts` no está. Con el orden invertido, ese guard saltaba
**siempre**: la suite daba `37 skipped` con `EXIT=0`. Treinta y siete tests reportando
éxito sin ejecutarse ni una vez, y un CI que mire el código de salida los ve verdes.
Su propio mensaje de skip ofrecía además una salida que no existe —"run this file in
isolation"—: en aislamiento `tests/unit/conftest.py` se carga igual, así que salta igual.

Sin la línea, esos mismos 37 pasan. No se relajó ninguna aserción para conseguirlo.

QUÉ FIJA ESTE FICHERO
---------------------
El orden, la resolución que ese orden produce, y que los 37 sigan siendo ejecutables.
No juzga a `FeatureBuilder` — eso es de sus propios tests.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = str((REPO / "src").resolve())
DAGS = str((REPO / "airflow" / "dags").resolve())


def test_both_contracts_packages_exist() -> None:
    """La premisa, fijada en vez de contada en un comentario.

    Si algún día uno de los dos se renombra, la colisión desaparece y este fichero hay
    que revisarlo a conciencia en vez de dejarlo vigilando por inercia.
    """
    assert (REPO / "src" / "contracts" / "__init__.py").is_file()
    assert (REPO / "airflow" / "dags" / "contracts" / "__init__.py").is_file()


def test_both_paths_are_actually_on_sys_path() -> None:
    """Anti-vacuidad del candado de orden.

    El test de abajo compara dos índices. Si una de las dos rutas no estuviera en
    `sys.path`, ese test no tendría nada que comparar y habría que decidir qué hacer
    —no pasar de largo—. Así que la precondición se exige aquí, aparte y explícita.
    """
    assert DAGS in sys.path, (
        "`airflow/dags` no está en sys.path: `import contracts` no puede resolver al "
        "paquete que `tests/conftest.py` declara necesario"
    )
    assert SRC in sys.path, "`src` no está en sys.path: los imports de dominio fallarán"


def test_airflow_dags_precedes_src_as_the_parent_conftest_declares() -> None:
    """EL candado. `airflow/dags` delante de `src`, que es lo declarado CRÍTICO.

    Rojo con: devolver el `sys.path.insert(0, src)` incondicional a
    `tests/unit/conftest.py` (medido: invierte los dos índices).
    """
    i_dags, i_src = sys.path.index(DAGS), sys.path.index(SRC)
    assert i_dags < i_src, (
        f"`src` ({i_src}) precede a `airflow/dags` ({i_dags}) en sys.path, al revés de "
        f"lo que `tests/conftest.py` declara CRÍTICO. `import contracts` resolverá a "
        f"`src/contracts` y los 37 tests de test_all_layer_contracts.py volverán a "
        f"saltar en bloque con EXIT=0"
    )


def test_import_contracts_resolves_to_the_airflow_package() -> None:
    """La consecuencia del orden, comprobada donde de verdad se nota.

    El test anterior mira índices; éste mira lo único que le importa a quien escribe
    `import contracts`. Se fijan los dos porque un cambio en `importmode` de pytest, o
    un `.pth` nuevo, podría respetar el orden y aun así cambiar la resolución.
    """
    spec = importlib.util.find_spec("contracts")
    assert spec is not None and spec.origin is not None, "`contracts` no resuelve a nada"
    assert Path(spec.origin).resolve().parent.parent == Path(DAGS), (
        f"`import contracts` resuelve a {spec.origin} en vez de al paquete de "
        f"`airflow/dags`, que es el que `tests/conftest.py` declara necesario"
    )


def test_the_thirty_seven_layer_contract_tests_are_executable() -> None:
    """Que los 37 no vuelvan a estar verdes por no ejecutarse.

    `test_all_layer_contracts.py` hace `skip` si este import falla. Un `skip` masivo se
    ve idéntico a un verde en el código de salida, así que el import se comprueba aquí
    **sin guard**: si un día vuelve a fallar, se entera alguien.
    """
    assert importlib.util.find_spec("contracts.l0_data_contracts") is not None, (
        "`contracts.l0_data_contracts` no es importable: los 37 tests de "
        "test_all_layer_contracts.py están saltando en bloque y su EXIT=0 no significa "
        "que pasen, significa que no corren"
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
