"""
Regression: las particiones de PRODUCCION e INVESTIGACION no se mezclan.

Contract: CTR-RESEARCH-PARTITION-001 · Date: 2026-08-24

`airflow/dags/l2_dataset_builder.py::load_date_ranges(partition)` sirve dos tracks desde
dos SSOT distintos:

    production -> config/date_ranges.yaml            (H1/H5, barras diarias)
    research   -> config/research/partition.yaml     (tesis RL/LLM/hibrido, 5 minutos)

Confundirlos no daria error: devolveria rangos plausibles del track equivocado y el
hold-out de la tesis —que se abre UNA vez con manifiesto firmado, Regla B— quedaria
entrenado sobre si mismo. El modo de fallo es silencioso, asi que se comprueba explicito.

Sin Airflow ni Postgres: se extrae la funcion por AST.
"""
from __future__ import annotations

import ast
import dataclasses
import io
import logging
from datetime import datetime
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[2]
DAG = ROOT / "airflow" / "dags" / "l2_dataset_builder.py"


@pytest.fixture(scope="module")
def loader():
    src = io.open(DAG, encoding="utf-8").read()
    tree = ast.parse(src)
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "load_date_ranges"), None)
    cls = next((n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "DateRanges"), None)
    assert fn and cls, "load_date_ranges/DateRanges no encontrados en el DAG"
    ns = {"yaml": yaml, "logging": logging, "datetime": datetime,
          "dataclass": dataclasses.dataclass, "CONFIG_DIR": ROOT / "config"}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "<c>", "exec"), ns)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<f>", "exec"), ns)
    return ns["load_date_ranges"]


def test_default_is_production(loader):
    """Cambiar el default silenciosamente reapuntaria todo el track H1/H5."""
    prod = yaml.safe_load((ROOT / "config/date_ranges.yaml").read_text(encoding="utf-8"))
    assert loader().training_start == prod["training"]["start"]


def test_research_maps_the_blocks_in_order(loader):
    """development->train, selection->val, holdout->test. Un cruce entrena en el juicio."""
    cfg = yaml.safe_load((ROOT / "config/research/partition.yaml").read_text(encoding="utf-8"))
    b, r = cfg["blocks"], loader("research")
    assert (r.training_start, r.training_end) == (b["development"]["start"], b["development"]["end"])
    assert (r.validation_start, r.validation_end) == (b["selection"]["start"], b["selection"]["end"])
    assert (r.test_start, r.test_end) == (b["holdout"]["start"], b["holdout"]["end"])


def test_research_blocks_are_ordered_and_disjoint(loader):
    r = loader("research")
    assert r.training_end < r.validation_start, "desarrollo y seleccion se solapan"
    assert r.validation_end < r.test_start, "seleccion y hold-out se solapan"


def test_unknown_partition_is_rejected(loader):
    """Un typo debe fallar, no caer al default sin avisar."""
    with pytest.raises(ValueError, match="particion desconocida"):
        loader("prod")


def test_the_two_tracks_are_not_the_same_ranges(loader):
    """Si algun dia coincidieran, alguien fusiono los SSOT sin querer."""
    p, r = loader("production"), loader("research")
    assert (p.training_start, p.training_end) != (r.training_start, r.training_end)
    assert (p.test_start, p.test_end) != (r.test_start, r.test_end)
