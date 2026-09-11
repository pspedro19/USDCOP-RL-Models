"""Gate fail-closed entre controles sintéticos y experimentos de mercado."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def require_macro_identity(report: Path, *, availability: Path | None = None,
                           clean: Path | None = None) -> dict:
    """Exige reconciliación positiva de todas las fuentes macro declaradas."""
    if not report.is_file():
        raise RuntimeError(f"macro identity gate: falta el informe {report}")
    payload = json.loads(report.read_text(encoding="utf-8"))
    if payload.get("all_declared_identities_honoured") is not True:
        raise RuntimeError("macro identity gate: una o más fuentes no coinciden con el SSOT")
    series = payload.get("series")
    if not isinstance(series, dict) or not series:
        raise RuntimeError("macro identity gate: el informe no contiene series verificadas")
    bad = [name for name, entry in series.items()
           if not isinstance(entry, dict)
           or entry.get("honoured") is not True
           or entry.get("status") != "COINCIDE"]
    if bad:
        raise RuntimeError(f"macro identity gate: series no verificadas {sorted(bad)}")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise RuntimeError("macro identity gate: faltan hashes de los artefactos verificados")
    # Bind the report to the current availability SSOT. A hand-edited report
    # must not be enough to unlock training with a different universe.
    try:
        import yaml
        ssot = availability or (report.parents[1] / "config" / "research" / "macro_availability.yaml")
        clean_path = clean or (report.parents[1] / "data" / "pipeline" / "04_cleaning" / "output" / "MACRO_DAILY_CLEAN.parquet")
        if not ssot.is_file() or not clean_path.is_file():
            raise RuntimeError("macro identity gate: faltan artefactos para validar hashes")
        if inputs.get("availability_sha256") != _sha256(ssot):
            raise RuntimeError("macro identity gate: hash del SSOT no coincide")
        if inputs.get("clean_sha256") != _sha256(clean_path):
            raise RuntimeError("macro identity gate: hash del parquet limpio no coincide")
        declared = yaml.safe_load(ssot.read_text(encoding="utf-8")).get("series", {})
        if set(series) != set(declared):
            raise RuntimeError("macro identity gate: series distintas del SSOT")
        for name, spec in declared.items():
            entry = series[name]
            if entry.get("column") != spec.get("column") or entry.get("declared_source") != spec.get("source"):
                raise RuntimeError(f"macro identity gate: identidad SSOT inconsistente para {name}")
            if spec.get("fallback", "forbidden") == "forbidden" and entry.get("fallback") not in (None, "forbidden"):
                raise RuntimeError(f"macro identity gate: fallback no permitido para {name}")
    except FileNotFoundError as exc:
        raise RuntimeError("macro identity gate: falta el SSOT de disponibilidad") from exc
    return payload
