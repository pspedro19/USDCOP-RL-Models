"""
CTR-DEPLOY-CMD-001 — resolución SEGURA del comando de deploy (lado Python).
===========================================================================

Espejo exacto de ``usdcop-trading-dashboard/lib/security/deploy-command.ts``. Ambos
ejecutan el MISMO artefacto (``data/approvals/approval_state*.json → deploy_manifest``):

  * el dashboard, cuando el deploy corre en proceso (``/api/production/deploy``);
  * el DAG ``forecast_h5_l4b_production_deploy``, que es el camino PREFERIDO en
    producción (el contenedor de Node no tiene Python).

Por qué existe este módulo, además del arreglo del shell en el dashboard: el DAG ya
lanzaba con ``subprocess.run([...])`` (sin shell, luego sin inyección de comandos), pero
construía la ruta así::

    script = PROJECT_ROOT / plan['script']

y ``pathlib`` **descarta el operando izquierdo si el derecho es absoluto**
(``Path('/opt/airflow') / '/etc/evil.py'`` → ``/etc/evil.py``). Con ``..`` pasaba lo
mismo por otra vía. Resultado: el manifiesto elegía qué programa ejecuta el deploy.

Reglas (allowlist, nunca blacklist — K-040):

  1. ``script``: relativo POSIX, primer segmento ``scripts``, segmentos
     ``[A-Za-z0-9_][A-Za-z0-9_.-]*`` (``.``/``..`` quedan fuera por forma), extensión
     ``.py``, existente, y su ``realpath`` DENTRO del ``realpath`` de ``<root>/scripts``.
  2. ``args``: cada elemento es bandera (``--phase``) o valor simple (``production``,
     ``1.2.1``). Sin espacios, comillas, ``$``, ``/``, ``\\`` ni metacaracteres.
  3. Fail-closed: cualquier duda ⇒ ``DeployManifestRejected`` (con ``field``/``reason``
     para el rastro de auditoría). El llamador NO ejecuta nada.

Lo que NO promete: que el script permitido sea inocuo. La frontera es "código versionado
bajo ``scripts/``" vs "cadena arbitraria"; un allowlist literal por estrategia exigiría un
registro de entrypoints y está declarado como deuda.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "ALLOWED_SCRIPT_ROOT",
    "ALLOWED_SCRIPT_EXT",
    "LEGACY_SCRIPT",
    "LEGACY_ARGS",
    "MAX_ARGS",
    "MAX_ARG_LEN",
    "MAX_SCRIPT_LEN",
    "DeployCommand",
    "DeployManifestRejected",
    "resolve_deploy_command",
    "validate_script_shape",
    "validate_args",
]

ALLOWED_SCRIPT_ROOT = "scripts"
ALLOWED_SCRIPT_EXT = ".py"

LEGACY_SCRIPT = "scripts/pipeline/train_and_export_smart_simple.py"
LEGACY_ARGS: tuple[str, ...] = ("--phase", "production", "--no-png", "--seed-db")

MAX_ARGS = 24
MAX_ARG_LEN = 128
MAX_SCRIPT_LEN = 200

_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_FLAG_RE = re.compile(r"^--?[A-Za-z0-9][A-Za-z0-9-]*$")
_VALUE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class DeployManifestRejected(ValueError):
    """El manifiesto aprobado pide algo que NO se ejecuta. Incidente de auditoría."""

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"deploy manifest rejected ({field}): {reason}")
        self.field = field
        self.reason = reason


@dataclass(frozen=True)
class DeployCommand:
    interpreter: str
    script_path: Path
    args: list[str]
    script_rel: str
    legacy: bool

    def argv(self) -> list[str]:
        """argv EXPLÍCITO para ``subprocess.run`` — nunca una cadena para un shell."""
        return [self.interpreter, str(self.script_path), *self.args]


def validate_script_shape(script: Any) -> Optional[str]:
    """Valida la forma de la ruta relativa (sin tocar el disco). ``None`` = válida."""
    if not isinstance(script, str):
        return "script ausente o no textual"
    if not script:
        return "script vacío"
    if len(script) > MAX_SCRIPT_LEN:
        return f"script demasiado largo ({len(script)} > {MAX_SCRIPT_LEN})"
    if "\\" in script:
        return "separador de Windows / UNC no permitido (usa `/`)"
    if "\0" in script:
        return "byte nulo en la ruta"
    if script.startswith("/") or re.match(r"^[A-Za-z]:", script):
        return "ruta absoluta no permitida"
    segments = script.split("/")
    if segments[0] != ALLOWED_SCRIPT_ROOT:
        return f"debe estar bajo `{ALLOWED_SCRIPT_ROOT}/`"
    if len(segments) < 2:
        return "falta el nombre del script"
    for seg in segments:
        if not _SEGMENT_RE.match(seg):
            return f"segmento inválido: {seg!r}"
    if not script.endswith(ALLOWED_SCRIPT_EXT):
        return f"extensión no permitida (se exige {ALLOWED_SCRIPT_EXT})"
    return None


def validate_args(args: Any) -> Optional[str]:
    """Valida los argumentos por forma declarada. ``None`` = válidos."""
    if args is None:
        return None
    if isinstance(args, (str, bytes)) or not isinstance(args, (list, tuple)):
        return "args debe ser una lista"
    if len(args) > MAX_ARGS:
        return f"demasiados args ({len(args)} > {MAX_ARGS})"
    for raw in args:
        if not isinstance(raw, str):
            return f"arg no textual: {raw!r}"
        if not raw or len(raw) > MAX_ARG_LEN:
            return f"longitud de arg inválida: {len(raw)}"
        if not _FLAG_RE.match(raw) and not _VALUE_RE.match(raw):
            return f"arg fuera de la forma permitida: {raw!r}"
    return None


def resolve_deploy_command(
    project_root: Path | str,
    manifest: Optional[dict],
    interpreter: Optional[str] = None,
) -> DeployCommand:
    """Devuelve el comando ejecutable o lanza ``DeployManifestRejected``.

    ``manifest`` ausente/vacío ⇒ fallback legado (que pasa por la MISMA validación:
    el fallback tampoco es un permiso para saltarse el contrato).
    """
    root = Path(project_root).resolve()
    legacy = not manifest

    if legacy:
        script: Any = LEGACY_SCRIPT
        args: Any = list(LEGACY_ARGS)
    else:
        if not isinstance(manifest, dict):
            raise DeployManifestRejected("manifest", "manifiesto no es un objeto")
        script = manifest.get("script")
        args = manifest.get("args") or []

    shape_error = validate_script_shape(script)
    if shape_error:
        raise DeployManifestRejected("script", shape_error)

    args_error = validate_args(args)
    if args_error:
        raise DeployManifestRejected("args", args_error)

    allowed_root = root / ALLOWED_SCRIPT_ROOT
    try:
        real_root = allowed_root.resolve(strict=True)
    except OSError:
        raise DeployManifestRejected("script", f"no existe el árbol permitido `{ALLOWED_SCRIPT_ROOT}/`")

    candidate = root.joinpath(*str(script).split("/"))
    try:
        real = candidate.resolve(strict=True)
    except OSError:
        raise DeployManifestRejected("script", f"el script no existe: {script}")

    if real != real_root and real_root not in real.parents:
        raise DeployManifestRejected("script", f"el script resuelve fuera de `{ALLOWED_SCRIPT_ROOT}/`")
    if not real.is_file():
        raise DeployManifestRejected("script", "el script no es un fichero regular")

    return DeployCommand(
        interpreter=interpreter or sys.executable,
        script_path=real,
        args=list(args) if isinstance(args, (list, tuple)) else [],
        script_rel=str(script),
        legacy=legacy,
    )


def describe_rejection(exc: DeployManifestRejected, manifest: Optional[dict]) -> dict:
    """Detalle canónico del incidente para logs / auditoría (nunca ejecuta nada)."""
    manifest = manifest or {}
    return {
        "field": exc.field,
        "reason": exc.reason,
        "manifest_script": manifest.get("script"),
        "manifest_args": manifest.get("args"),
        "executed": False,
    }
