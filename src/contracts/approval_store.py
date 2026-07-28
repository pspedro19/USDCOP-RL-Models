"""
Approval-state store (CXD-057) — SSOT de la ubicación PRIVADA del estado de aprobación.
=======================================================================================

Los ``approval_state*.json`` ya NO viven bajo ``usdcop-trading-dashboard/public/``.
Bajo ``public/`` el fichero se sirve por el estático ``/data/**``, que el middleware del
dashboard gatea con *una sesión* y nada más, mientras el SSOT reserva ``gates``, el gate
``deflated_sharpe`` (DSR trial-aware) y ``backtest_metrics`` a ``research:read``:

  * ``.claude/specs/platform/frontend-backend-contract.md`` §6 — Backtest/Experimentos
    ✖ para ``free`` y ``subscriber``, ✔ ``admin``/``developer``.
  * ``.claude/specs/platform/ux-navigation.md`` P3 — "cliente jamás ve jerga interna
    (L4/votos/gates)".
  * ``docs/rbac/VISUAL-SPEC-CHECKLIST.md`` §B — "subscriber sin gates/votos/jerga
    L4/PENDING".

Precedente idéntico ya aplicado dos veces en el repo: artefactos SHAP →
``data/interpretability/`` (C-006) y proyección de gobernanza → ``data/control-tower/``.

Este módulo es el ÚNICO sitio donde se decide la ruta. Espejo TypeScript:
``usdcop-trading-dashboard/lib/approvals/store.py`` → ``lib/approvals/store.ts``.

**La resolución multi-estrategia es load-bearing y debe coincidir en las tres
implementaciones** (este módulo, ``lib/approvals/store.ts`` y el DAG H5-L4b): el export
escribe la estrategia ACTIVA en el fichero SIN sufijo mientras el dashboard siempre manda
``strategy_id``; sin el fallback, ``smart_simple_v11`` resolvía a un fichero que nunca se
escribe y el Voto 2 devolvía 404. El fallback solo aplica cuando el singleton ES esa
estrategia, así que una id obsoleta jamás aprueba/despliega el bundle de otra.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

#: Nombre del artefacto de la estrategia ACTIVA (singleton, hoy COP).
DEFAULT_APPROVAL_FILE = "approval_state.json"

#: Whitelist de ``strategy_id`` (bloquea ``..``, separadores y absolutos).
_SID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

#: Raíz del repo: ``src/contracts/approval_store.py`` → ``<repo>``.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def approvals_dir() -> Path:
    """Directorio PRIVADO de artefactos de aprobación.

    Default ``<repo>/data/approvals``; override con ``APPROVALS_DATA_DIR``
    (contenedores/tests), igual que ``CONTROL_TOWER_DATA_DIR``.
    """
    env = os.getenv("APPROVALS_DATA_DIR", "").strip()
    return Path(env).resolve() if env else _REPO_ROOT / "data" / "approvals"


def is_valid_strategy_id(sid: Optional[str]) -> bool:
    return bool(sid) and bool(_SID_RE.match(str(sid))) and ".." not in str(sid)


def approval_path(strategy_id: Optional[str] = None) -> Path:
    """Ruta del artefacto de ``strategy_id`` (sin resolver existencia).

    Sin ``strategy_id`` (o con uno inválido) ⇒ el singleton.
    """
    if not is_valid_strategy_id(strategy_id):
        return approvals_dir() / DEFAULT_APPROVAL_FILE
    return approvals_dir() / f"approval_state_{strategy_id}.json"


def resolve_approval_path(strategy_id: Optional[str] = None) -> Optional[Path]:
    """Resuelve el artefacto REAL de una estrategia con el fallback al singleton.

    Devuelve ``None`` cuando no hay artefacto legible — **fail-closed**: el caller
    debe fallar con motivo declarado, nunca degradar en silencio a "sin datos".
    """
    if not is_valid_strategy_id(strategy_id):
        singleton = approvals_dir() / DEFAULT_APPROVAL_FILE
        return singleton if singleton.is_file() else None

    scoped = approvals_dir() / f"approval_state_{strategy_id}.json"
    if scoped.is_file():
        return scoped

    singleton = approvals_dir() / DEFAULT_APPROVAL_FILE
    try:
        if singleton.is_file():
            data = json.loads(singleton.read_text(encoding="utf-8"))
            if data.get("strategy") == strategy_id:
                return singleton
    except Exception:  # noqa: BLE001 — singleton ilegible: la ruta scoped mantiene el fallo honesto
        pass
    return None


def read_approval(strategy_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Lee el artefacto ÍNTEGRO. ``None`` si no existe o no parsea (fail-closed)."""
    path = resolve_approval_path(strategy_id)
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def write_approval(state: Dict[str, Any], strategy_id: Optional[str] = None) -> Path:
    """Escribe el artefacto (crea el directorio privado si falta). Devuelve la ruta."""
    path = approval_path(strategy_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


#: Campos publicables a una superficie de CLIENTE — ALLOWLIST, jamás blacklist.
#: Espejo exacto de ``PUBLIC_APPROVAL_FIELDS`` en ``lib/approvals/store.ts``.
PUBLIC_APPROVAL_FIELDS = (
    "status",
    "strategy",
    "strategy_name",
    "approved_at",
    "created_at",
    "last_updated",
)


def to_public_approval(state: Dict[str, Any]) -> Dict[str, Any]:
    """Proyección pública por allowlist: RECONSTRUYE, no filtra."""
    return {k: state.get(k) for k in PUBLIC_APPROVAL_FIELDS}
