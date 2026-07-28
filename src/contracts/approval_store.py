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
import math
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

#: Nombre del artefacto de la estrategia ACTIVA (singleton, hoy COP).
DEFAULT_APPROVAL_FILE = "approval_state.json"

#: Whitelist de ``strategy_id`` (bloquea ``..``, separadores y absolutos).
#: Se evalúa con ``fullmatch``: ``re.match`` + ``$`` acepta un salto de línea FINAL
#: (``"ok_id\n"`` pasaba y producía un nombre de fichero con ``\n`` embebido).
_SID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

#: Estados válidos — espejo de ``ProductionStatus`` en
#: ``lib/contracts/production-approval.contract.ts``.
VALID_STATUSES = ("PENDING_APPROVAL", "APPROVED", "REJECTED", "LIVE")

#: Cap de tamaño del artefacto. Espejo de ``readJson`` en ``lib/approvals/store.ts``.
MAX_APPROVAL_BYTES = 2 * 1024 * 1024

#: Sufijo del lock interproceso. **Debe coincidir con ``LOCK_SUFFIX`` de
#: ``lib/approvals/store.ts``**: el Voto 2 (Node) y el export (Python) escriben el MISMO
#: artefacto y solo se excluyen si nombran el mismo lock.
LOCK_SUFFIX = ".lock"
_LOCK_STALE_S = 30.0
_LOCK_WAIT_S = 4.0

#: Raíz del repo: ``src/contracts/approval_store.py`` → ``<repo>``.
_REPO_ROOT = Path(__file__).resolve().parents[2]


class ApprovalLockTimeout(RuntimeError):
    """No se pudo adquirir el lock del artefacto — **no se escribe a ciegas**."""


def approvals_dir() -> Path:
    """Directorio PRIVADO de artefactos de aprobación.

    Default ``<repo>/data/approvals``; override con ``APPROVALS_DATA_DIR``
    (contenedores/tests), igual que ``CONTROL_TOWER_DATA_DIR``.
    """
    env = os.getenv("APPROVALS_DATA_DIR", "").strip()
    return Path(env).resolve() if env else _REPO_ROOT / "data" / "approvals"


def is_valid_strategy_id(sid: Optional[str]) -> bool:
    return bool(sid) and bool(_SID_RE.fullmatch(str(sid))) and ".." not in str(sid)


def approval_path(strategy_id: Optional[str] = None) -> Path:
    """Ruta del artefacto de ``strategy_id`` (sin resolver existencia).

    ``None`` ⇒ el singleton (la estrategia ACTIVA). Una id **inválida NO es lo mismo
    que ausente**: lanza ``ValueError`` (fail-closed). Antes degradaba al singleton, de
    modo que ``write_approval(state, '../../x')`` pisaba el estado de la estrategia
    activa mientras el espejo TypeScript rechazaba la misma id — paridad rota en la
    dirección peligrosa (P1, CODEX).
    """
    if strategy_id is None:
        return approvals_dir() / DEFAULT_APPROVAL_FILE
    if not is_valid_strategy_id(strategy_id):
        raise ValueError("invalid strategy_id (whitelist ^[A-Za-z0-9][A-Za-z0-9_-]*$)")
    return approvals_dir() / f"approval_state_{strategy_id}.json"


def resolve_approval_path(strategy_id: Optional[str] = None) -> Optional[Path]:
    """Resuelve el artefacto REAL de una estrategia con el fallback al singleton.

    Devuelve ``None`` cuando no hay artefacto legible — **fail-closed**: el caller
    debe fallar con motivo declarado, nunca degradar en silencio a "sin datos".
    Una id inválida devuelve ``None`` (no cae al singleton de otra estrategia).
    """
    if strategy_id is None:
        singleton = approvals_dir() / DEFAULT_APPROVAL_FILE
        return singleton if singleton.is_file() else None
    if not is_valid_strategy_id(strategy_id):
        return None

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
    """Lee el artefacto ÍNTEGRO. ``None`` si no existe, excede el cap de tamaño o no
    parsea (fail-closed, mismo criterio que ``readJson`` en el lado TypeScript)."""
    path = resolve_approval_path(strategy_id)
    if path is None:
        return None
    try:
        if path.stat().st_size > MAX_APPROVAL_BYTES:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


# ───────────────────────────────────────────── validación de schema (fail-closed)


def _assert_finite(node: Any, where: str = "$") -> None:
    """``Infinity``/``NaN`` jamás salen del sistema (``strategy-contract.md`` §2)."""
    if isinstance(node, float):
        if math.isnan(node) or math.isinf(node):
            raise ValueError(f"non-finite number at {where} (use null, never NaN/Infinity)")
    elif isinstance(node, dict):
        for k, v in node.items():
            _assert_finite(v, f"{where}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            _assert_finite(v, f"{where}[{i}]")


def validate_approval_document(state: Any) -> Dict[str, Any]:
    """Valida el documento ANTES de publicarlo. Lanza ``ValueError`` si no cumple.

    Un artefacto malformado en esta ruta no es un bug cosmético: el DAG H5-L4b y el
    runner diario deciden con él si hay dinero en riesgo. Se valida lo que gobierna esa
    decisión (estado, identidad, finitud, tamaño), no el documento entero.
    """
    if not isinstance(state, dict):
        raise ValueError("approval state must be a JSON object")
    status = state.get("status")
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status {status!r} (expected one of {VALID_STATUSES})")
    strategy = state.get("strategy")
    if not is_valid_strategy_id(strategy):
        raise ValueError(f"invalid strategy id {strategy!r} in approval state")
    _assert_finite(state)
    payload = json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False)
    if len(payload.encode("utf-8")) > MAX_APPROVAL_BYTES:
        raise ValueError("approval state exceeds the 2 MiB cap shared with the TS reader")
    return state


# ──────────────────────────────── lock interproceso + publicación atómica


def acquire_approval_lock(path: Path, timeout_s: float = _LOCK_WAIT_S):
    """Lock EXCLUSIVO sobre el artefacto, compatible con el lado Node.

    Mismo primitivo que el publicador de interpretabilidad (C-006) y que
    ``lib/approvals/store.ts``: ``O_CREAT|O_EXCL`` es la única operación de fichero
    atómica **y** exclusiva, así que da exactamente-un-ganador sin servicio externo.
    Se usa aquí para que el export de Python y el Voto 2 de Node no puedan publicar a
    la vez sobre el mismo SSOT.

    Devuelve un context manager. Timeout ⇒ ``ApprovalLockTimeout``: **no se escribe**.
    """

    class _Lock:
        def __init__(self) -> None:
            self._lock = Path(str(path) + LOCK_SUFFIX)
            self._fd: Optional[int] = None

        def __enter__(self) -> "_Lock":
            deadline = time.monotonic() + timeout_s
            while True:
                try:
                    self._fd = os.open(str(self._lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(self._fd, f'{{"pid": {os.getpid()}}}'.encode())
                    return self
                except FileExistsError:
                    # Se libera siempre en ``__exit__``: un lock viejo solo puede venir
                    # de un proceso muerto.
                    try:
                        if time.time() - self._lock.stat().st_mtime > _LOCK_STALE_S:
                            self._lock.unlink(missing_ok=True)
                            continue
                    except FileNotFoundError:
                        continue
                    if time.monotonic() >= deadline:
                        raise ApprovalLockTimeout(f"approval state busy: {path.name}")
                    time.sleep(0.015)

        def __exit__(self, *exc: Any) -> None:
            if self._fd is not None:
                os.close(self._fd)
            self._lock.unlink(missing_ok=True)

    return _Lock()


def write_approval(state: Dict[str, Any], strategy_id: Optional[str] = None) -> Path:
    """Publica el artefacto **validado y de forma atómica**. Devuelve la ruta.

    - ``strategy_id`` inválido ⇒ ``ValueError`` (no degrada al singleton).
    - Documento que no cumple el schema / con no-finitos ⇒ ``ValueError``, sin tocar disco.
    - Publicación: temporal en el MISMO directorio → ``fsync`` → ``os.replace``. Un
      crash en cualquier punto deja el artefacto anterior ÍNTEGRO (antes, un
      ``write_text`` a medias truncaba el SSOT y el lector fail-closed lo volvía un 404).
    """
    path = approval_path(strategy_id)
    validate_approval_document(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False)

    with acquire_approval_lock(path):
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
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
