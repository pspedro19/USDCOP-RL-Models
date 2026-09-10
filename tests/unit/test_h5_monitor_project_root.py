# -*- coding: utf-8 -*-
"""La raíz del proyecto se resuelve por contenido, no contando directorios.

EL DEFECTO, encontrado en producción el 2026-08-06
--------------------------------------------------
`forecast_h5_l6_weekly_monitor.py` calculaba la raíz con
`Path(__file__).resolve().parents[2]`. Eso acierta en el repo —el fichero está en
`<repo>/airflow/dags/`— y **falla en el contenedor**, donde vive en `/opt/airflow/dags/` y
el mismo cálculo devuelve `/opt`.

Síntoma real, no hipotético: al despausar el DAG, `persist_governed_sharpe` murió con

    FileNotFoundError: [Errno 2] No such file or directory: '/opt/config/metrics/catalog.yaml'

mientras `load_results`, `compute_metrics`, `check_gates` y `persist_evaluation` salían en
verde. Es decir: el DAG parecía funcionar y el único paso que **escribe hechos** no
llegaba a ejecutarse. `control.metric_event` llevaba 3 filas por eso.

Lo que lo hace interesante: el mismo fichero ya esquivaba el problema 350 líneas más
abajo, con `cwd="/opt/airflow" if Path("/opt/airflow/scripts").exists() else …`. El
defecto se conocía en un sitio y no en el otro — que es como sobreviven estas cosas.

QUÉ FIJA ESTE FICHERO
---------------------
Que la resolución dependa de **encontrar el fichero**, no de la profundidad del árbol, y
que falle cerrado cuando no está. Contar directorios es una suposición sobre el layout;
buscar el marcador es una comprobación.

NO cubre los otros 6 DAGs con el mismo `parents[2]` — están declarados como hallazgo
pendiente, no arreglados aquí.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DAG = REPO / "airflow" / "dags" / "forecast_h5_l6_weekly_monitor.py"
FUENTE = DAG.read_text(encoding="utf-8")


def _project_root_fn():
    """Extrae `_project_root` sin ejecutar el DAG entero (importaría Airflow)."""
    ns: dict = {"Path": Path, "__file__": str(DAG)}
    ini = FUENTE.index("def _project_root()")
    fin = FUENTE.index("def persist_governed_metric_events")
    exec(compile(FUENTE[ini:fin], str(DAG), "exec"), ns)
    return ns["_project_root"]


def test_the_root_is_found_in_the_repo_layout() -> None:
    """En el repo, la raíz resuelta contiene el catálogo que se va a leer."""
    root = _project_root_fn()()
    assert (root / "config" / "metrics" / "catalog.yaml").is_file(), (
        f"la raíz resuelta ({root}) no contiene el catálogo: es el fallo original"
    )


def test_the_container_layout_is_resolved_too(tmp_path, monkeypatch) -> None:
    """Con el layout del contenedor (`/opt/airflow/dags/…`) también acierta.

    Se simula creando `<tmp>/airflow/dags/x.py` con el catálogo en `<tmp>/config/…`: son
    DOS niveles desde el fichero, no tres. Con `parents[2]` esto daría el padre de `<tmp>`
    y no encontraría nada — que es exactamente lo que pasaba en producción.
    """
    dags = tmp_path / "airflow" / "dags"
    dags.mkdir(parents=True)
    catalogo = tmp_path / "config" / "metrics"
    catalogo.mkdir(parents=True)
    (catalogo / "catalog.yaml").write_text("metrics: []\n", encoding="utf-8")
    falso = dags / "monitor.py"
    falso.write_text("", encoding="utf-8")

    ini = FUENTE.index("def _project_root()")
    fin = FUENTE.index("def persist_governed_metric_events")
    ns: dict = {"Path": Path, "__file__": str(falso)}
    exec(compile(FUENTE[ini:fin], str(falso), "exec"), ns)

    assert ns["_project_root"]() == tmp_path, (
        "no resolvió la raíz en un layout de dos niveles: contar directorios vuelve a "
        "decidir en vez de buscar el marcador"
    )


def test_a_missing_catalog_fails_closed(tmp_path) -> None:
    """Sin catálogo en ningún ancestro, se para con un mensaje que dice qué falta.

    La alternativa —devolver una ruta calculada— es lo que produjo `/opt/config/...`: una
    ruta sintácticamente válida que no existe, y un fallo lejos de su causa.
    """
    hondo = tmp_path / "a" / "b" / "c"
    hondo.mkdir(parents=True)
    falso = hondo / "monitor.py"
    falso.write_text("", encoding="utf-8")

    ini = FUENTE.index("def _project_root()")
    fin = FUENTE.index("def persist_governed_metric_events")
    ns: dict = {"Path": Path, "__file__": str(falso)}
    exec(compile(FUENTE[ini:fin], str(falso), "exec"), ns)

    with pytest.raises(RuntimeError, match="no encuentro"):
        ns["_project_root"]()


def test_the_dag_no_longer_counts_directories_for_the_catalog() -> None:
    """El uso concreto que fallaba ya no calcula la raíz contando padres.

    Rojo con: devolver `Path(__file__).resolve().parents[2]` a la línea del catálogo.
    """
    assert 'catalog = MetricCatalog.load(root / "config" / "metrics" / "catalog.yaml")' in FUENTE
    assert "root = _project_root()" in FUENTE, (
        "la raíz del catálogo volvió a calcularse por profundidad de árbol"
    )


def test_no_path_in_this_dag_is_derived_by_counting_parents() -> None:
    """Ninguna ruta de este fichero se calcula por profundidad de árbol.

    Había TRES sitios: el catálogo (roto), el ancla del paper ledger (roto, y por eso el
    gate se declaraba «decorativo» sin encontrar un fichero que sí existe) y un `cwd` que
    ya llevaba un apaño ad-hoc. Convivían el defecto y su parche en el mismo fichero.

    Se admite la mención en prosa —el docstring explica el defecto— pero no en código.
    """
    codigo = [
        l for l in FUENTE.splitlines()
        if "parents[2]" in l and not l.lstrip().startswith(("#", "`", '"'))
        and "`Path(__file__)" not in l
    ]
    assert not codigo, (
        f"vuelven a calcularse rutas contando directorios: {codigo}. En el contenedor "
        f"eso resuelve a /opt y el fichero buscado no aparece"
    )
