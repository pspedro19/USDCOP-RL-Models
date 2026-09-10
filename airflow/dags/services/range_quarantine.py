"""Cuarentena de valores macro fuera de rango, EN EL PUNTO DE ESCRITURA.

Contract: CTR-L0-QUARANTINE-001 · Date: 2026-08-25

## Por qué aquí y no en una tarea del DAG

El 2026-08-25 entraron **59 días de Brent a 21-23 USD** (real: 60-70) con el rango
`[30, 150]` declarado en `config/l0_macro_sources.yaml:509` y `validation.enabled: true`.

El diagnóstico fácil —«falta cablear el validador»— era **falso**: `RangeValidator` ya está
en la lista por defecto de `ValidationPipeline` y el backfill la construye. Los huecos eran
cuatro, y todos tienen la misma forma: **la validación estaba en un sitio y la escritura en
otro**.

- El DAG de ingesta diaria (`l0_macro_update`) no tenía tarea de validación en absoluto.
- La del backfill nunca lanzaba (`fail_fast=False` + `logger.warning`).
- La rama de restore desde seeds se la saltaba entera.
- Y si el YAML no aparecía, el validador cargaba cero reglas en silencio.

Añadir una quinta tarea de validación al grafo habría repetido el patrón: una tarea se puede
limpiar, saltar o dejar fuera de una rama nueva. **Este módulo se llama desde el punto donde
las filas se escriben**, así que cubre a la vez el DAG diario, las dos ramas del backfill y
cualquier script que use el servicio de upsert.

## Cuarentena, no bloqueo

`on_invalid: error` tumbaría el DAG entero por una variable mala entre cuarenta, y eso deja la
macro stale — que es **otro** fallo, y bloqueante: `data-freshness.md` corta el training con
macro de más de 7 días. Cambiar un dato malo por ningún dato no es una mejora.

Así que el valor fuera de rango **no se escribe**, se aparta a un registro de cuarentena con
su motivo, y **las demás variables y las demás filas siguen su curso**. Ni un DAG rojo por un
dato malo, ni un dato malo dentro.

`on_invalid` del YAML sigue mandando y es reversible:

| `on_invalid` | Comportamiento |
|---|---|
| `quarantine` (por defecto) | aparta las filas malas, escribe el resto |
| `warn` | loguea y escribe TODO — el comportamiento viejo, que no impidió nada |
| `error` | lanza y aborta la escritura entera |

## Lo que NO hace

No inventa valores ni interpola. Una fila apartada es una fila ausente, y la ausencia la
detecta el gate de frescura, que es quien debe hacerlo.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_POLICY = "quarantine"
VALID_POLICIES = ("quarantine", "warn", "error")


def _repo_root() -> Path:
    """Raiz desde la que colgar `data/` y `config/`.

    En el HOST este fichero vive en `<repo>/airflow/dags/services/`, asi que `parents[3]`
    es la raiz. En el CONTENEDOR vive en `/opt/airflow/dags/services/` y `parents[3]` da
    `/opt`, que no existe como raiz escribible — el intento de escribir el registro moria
    con `Permission denied: '/opt/data'` y la cuarentena se quedaba sin constancia.

    `AIRFLOW_HOME` desambigua las dos disposiciones, y es lo que el contenedor define.
    """
    airflow_home = os.environ.get("AIRFLOW_HOME")
    if airflow_home and Path(airflow_home).is_dir():
        return Path(airflow_home)
    return Path(__file__).resolve().parents[3]


def _quarantine_dir() -> Path:
    return Path(os.environ.get(
        "MACRO_QUARANTINE_DIR", _repo_root() / "data" / "quarantine" / "macro"))


class QuarantineBlocked(RuntimeError):
    """`on_invalid: error` y hubo filas fuera de rango."""


@dataclass
class QuarantineResult:
    """Qué se apartó y por qué. Se devuelve siempre, aunque no se aparte nada."""

    variable: str
    policy: str
    rows_in: int
    rows_kept: int
    rows_quarantined: int = 0
    expected_range: Optional[Tuple[float, float]] = None
    samples: List[Dict[str, Any]] = field(default_factory=list)
    ledger_path: Optional[str] = None

    @property
    def clean(self) -> bool:
        return self.rows_quarantined == 0

    def to_dict(self) -> dict:
        return {
            "variable": self.variable, "policy": self.policy,
            "rows_in": self.rows_in, "rows_kept": self.rows_kept,
            "rows_quarantined": self.rows_quarantined,
            "expected_range": list(self.expected_range) if self.expected_range else None,
            "samples": self.samples, "ledger_path": self.ledger_path,
        }


_RANGES_CACHE: Optional[Dict[str, Tuple[float, float]]] = None
_POLICY_CACHE: Optional[str] = None


def _load_config() -> Tuple[Dict[str, Tuple[float, float]], str]:
    """Rangos y politica desde el YAML, via el RangeValidator que ya existe.

    Se reutiliza `RangeValidator._load_ranges_from_config` a proposito: es la unica
    definicion de donde viven los rangos, y desde 2026-08-25 **lanza** si no encuentra el
    config en vez de devolver `{}` en silencio.
    """
    global _RANGES_CACHE, _POLICY_CACHE
    if _RANGES_CACHE is not None and _POLICY_CACHE is not None:
        return _RANGES_CACHE, _POLICY_CACHE

    try:
        from validators.data_validators import RangeValidator
    except ImportError:  # pragma: no cover - fuera de Airflow
        import sys
        sys.path.insert(0, str(_repo_root() / "airflow" / "dags"))
        from validators.data_validators import RangeValidator

    _RANGES_CACHE = RangeValidator()._ranges

    policy = DEFAULT_POLICY
    import yaml
    for candidate in (Path("/opt/airflow/config/l0_macro_sources.yaml"),
                      _repo_root() / "config" / "l0_macro_sources.yaml"):
        if candidate.exists():
            cfg = (yaml.safe_load(candidate.read_text(encoding="utf-8")) or {})
            declared = (cfg.get("validation", {}) or {}).get("on_invalid")
            if declared in VALID_POLICIES:
                policy = declared
            elif declared:
                # `warn` era el valor historico y NO impidio nada. Se respeta si esta
                # escrito, pero un valor desconocido cae al default seguro, no al viejo.
                logger.warning("on_invalid=%r no reconocido; se usa %r", declared,
                               DEFAULT_POLICY)
            break
    _POLICY_CACHE = policy
    return _RANGES_CACHE, _POLICY_CACHE


def reset_config_cache() -> None:
    """Para los tests: vuelve a leer el YAML en la proxima llamada."""
    global _RANGES_CACHE, _POLICY_CACHE
    _RANGES_CACHE = None
    _POLICY_CACHE = None


def _write_ledger(variable: str, offending: pd.DataFrame, lo: float, hi: float,
                  date_col: Optional[str]) -> Optional[str]:
    """Un JSONL por variable. Append-only: una cuarentena no se edita, se anade."""
    try:
        d = _quarantine_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{variable}.jsonl"
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with path.open("a", encoding="utf-8") as fh:
            for _, row in offending.iterrows():
                fh.write(json.dumps({
                    "quarantined_at": stamp,
                    "variable": variable,
                    "date": str(row[date_col]) if date_col and date_col in row else None,
                    "value": float(row[variable]),
                    "expected_range": [lo, hi],
                }, ensure_ascii=False) + "\n")
        return str(path)
    except Exception as e:  # el registro no puede tumbar la ingesta
        logger.error("[QUARANTINE] no se pudo escribir el registro de %s: %s", variable, e)
        return None


def filter_out_of_range(variable: str, df: pd.DataFrame,
                        date_col: Optional[str] = None,
                        policy: Optional[str] = None) -> Tuple[pd.DataFrame, QuarantineResult]:
    """Aparta las filas de `variable` cuyo valor cae fuera del rango declarado.

    Devuelve `(df_limpio, resultado)`. Si la variable no tiene rango declarado o la columna
    no esta en el DataFrame, devuelve el original sin tocar: **no tener regla no es motivo
    para apartar datos**.
    """
    ranges, default_policy = _load_config()
    policy = policy or default_policy
    rows_in = 0 if df is None else len(df)

    if df is None or df.empty or variable not in df.columns or variable not in ranges:
        return df, QuarantineResult(variable=variable, policy=policy, rows_in=rows_in,
                                    rows_kept=rows_in,
                                    expected_range=ranges.get(variable))

    lo, hi = ranges[variable]
    values = pd.to_numeric(df[variable], errors="coerce")
    # NaN NO es fuera de rango: es ausencia, y la ausencia la gestiona el gate de frescura.
    bad = values.notna() & ((values < lo) | (values > hi))

    if not bad.any():
        return df, QuarantineResult(variable=variable, policy=policy, rows_in=rows_in,
                                    rows_kept=rows_in, expected_range=(lo, hi))

    offending = df.loc[bad]
    samples = []
    for _, row in offending.head(5).iterrows():
        samples.append({
            "date": str(row[date_col]) if date_col and date_col in row else None,
            "value": float(pd.to_numeric(row[variable], errors="coerce")),
        })

    msg = (f"[QUARANTINE] {variable}: {int(bad.sum())} de {rows_in} filas fuera de "
           f"[{lo}, {hi}] -> {samples[:3]}")

    if policy == "error":
        logger.error(msg)
        raise QuarantineBlocked(msg)
    if policy == "warn":
        # El comportamiento historico, conservado por compatibilidad: avisa y escribe todo.
        # No impidio los 59 dias de Brent; solo se usa si alguien lo pide explicitamente.
        logger.warning(msg + " -- POLITICA `warn`: se escriben IGUALMENTE")
        return df, QuarantineResult(variable=variable, policy=policy, rows_in=rows_in,
                                    rows_kept=rows_in, rows_quarantined=0,
                                    expected_range=(lo, hi), samples=samples)

    logger.error(msg + " -- se APARTAN; el resto de filas sigue su curso")
    ledger = _write_ledger(variable, offending, lo, hi, date_col)
    clean = df.loc[~bad]
    return clean, QuarantineResult(
        variable=variable, policy=policy, rows_in=rows_in, rows_kept=len(clean),
        rows_quarantined=int(bad.sum()), expected_range=(lo, hi), samples=samples,
        ledger_path=ledger)


def filter_frame(df: pd.DataFrame, columns: List[str],
                 date_col: Optional[str] = None,
                 policy: Optional[str] = None) -> Tuple[pd.DataFrame, List[QuarantineResult]]:
    """Aplica la cuarentena a varias columnas del mismo DataFrame, una por una.

    Cada columna se evalua con SU rango: apartar la fila entera porque una de cuarenta
    variables se salio seria tirar 39 datos buenos.
    """
    results: List[QuarantineResult] = []
    out = df
    for col in columns:
        out, res = filter_out_of_range(col, out, date_col=date_col, policy=policy)
        if res.rows_quarantined or res.samples:
            results.append(res)
    return out, results
