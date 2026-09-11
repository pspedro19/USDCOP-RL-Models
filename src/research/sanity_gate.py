"""Gate fail-closed entre controles sintéticos y experimentos de mercado."""
from __future__ import annotations

import json
from pathlib import Path


def require_sanity_pass(report: Path, *, fixtures: tuple[str, ...] = ("S1", "S2", "S3", "S4")) -> dict:
    """Exige una receta que haya pasado todas las fixtures pre-registradas."""
    if not report.is_file():
        raise RuntimeError(f"sanity gate: falta el informe {report}")
    payload = json.loads(report.read_text(encoding="utf-8"))
    if payload.get("synthetic_only") is not True or payload.get("market_trials_charged") != 0:
        raise RuntimeError("sanity gate: el informe no está marcado como sintético/cero trials")
    if payload.get("selected_probe") is None:
        raise RuntimeError("sanity gate: ninguna receta pasó S1–S4")
    seen = {x.get("fixture") for a in payload.get("attempts", [])
            for x in a.get("fixtures", [])}
    missing = set(fixtures) - seen
    if missing:
        raise RuntimeError(f"sanity gate: faltan fixtures {sorted(missing)}")
    return payload


def require_macro_identity(report: Path) -> dict:
    """Exige reconciliación positiva de todas las fuentes macro declaradas."""
    if not report.is_file():
        raise RuntimeError(f"macro identity gate: falta el informe {report}")
    payload = json.loads(report.read_text(encoding="utf-8"))
    if payload.get("all_declared_identities_honoured") is not True:
        raise RuntimeError("macro identity gate: una o más fuentes no coinciden con el SSOT")
    return payload
