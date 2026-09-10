"""Rutas del carril forward, en un solo sitio.

Contract: CTR-RESEARCH-FORWARD-001 · Date: 2026-08-25

El arnés venía como repositorio propio y cada módulo calculaba su `ROOT` con
`Path(__file__).parent.parent`. Al vendorizarlo bajo `src/research/llm_forward/` eso apunta a
`src/research/`, y el primer arranque murió buscando `src/research/config/preregistration.yaml`.

Se centraliza en vez de parchear cuatro módulos: una ruta duplicada en cuatro sitios se
desincroniza en el primer movimiento de ficheros, y el síntoma —un ledger escrito en un
directorio distinto del que se lee— es de los que no dan error, solo dan cero filas.

Todo es configurable por entorno para que Airflow (que monta `/opt/airflow/data`) y el host
usen el mismo código sin ramas.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _p(env_var: str, default: Path) -> Path:
    return Path(os.environ.get(env_var, default))


# El pre-registro vive con el resto de config del carril de investigacion, no dentro del
# paquete: es el COMPROMISO, y se audita junto a `partition.yaml` y `evaluation_mask.json`.
PREREG_PATH = _p("FWD_PREREG",
                 REPO / "config" / "research" / "preregistration_forward.yaml")

# `data/` y no `outputs/`: `outputs/` esta gitignorado y estos ledgers son la EVIDENCIA de
# que la decision existio antes del resultado. Un artefacto de auditoria que no sobrevive a
# un clon limpio no sirve para auditar nada.
FWD_DATA = _p("FWD_DATA", REPO / "data" / "forward")
LEDGER_DIR = FWD_DATA / "ledger"
CORPUS_STORE = FWD_DATA / "corpus"

DECISIONS_PATH = LEDGER_DIR / "decisions.jsonl"
SETTLEMENTS_PATH = LEDGER_DIR / "settlements.jsonl"
