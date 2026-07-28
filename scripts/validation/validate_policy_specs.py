#!/usr/bin/env python3
"""Valida los specs de política de `config/policies/` (§11 de planes/05, BL-47).

Sin infraestructura: solo lee YAML y compila las políticas. Fail-closed —
cualquier violación devuelve exit 1 con el motivo.

Checks implementados (§11):
  ✓ rule_based implica retrain=never
  ✓ rule_based no declara capability=train
  ✓ rule_based no exige model_snapshot_id
  ✓ toda policy referencia feature_set y resample_policy
  ✓ todo operador declarativo pertenece al whitelist
  ✓ el YAML no contiene eval, SQL libre ni Python arbitrario
  ✓ toda política tiene default/fallback explícito
  ✓ conflictos entre reglas tienen prioridad o resolución declarada
  ✓ toda salida respeta direction y exposure caps
  ✓ mismos inputs + misma policy => misma decisión (determinismo)
  ✓ policy_hash coincide con parámetros y schema (freeze)
  ✓ el módulo de un coded_policy vive bajo el allowlist e implementa Policy

Uso:  python scripts/validation/validate_policy_specs.py [--dir config/policies]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts.policy import Policy, PolicyContext  # noqa: E402
from src.strategies.policies.loader import (  # noqa: E402
    PolicySpecError,
    build_policy,
    canonical_policy_hash,
    load_all_policy_specs,
    policy_specs_dir,
)

#: Un YAML de política nunca contiene código. Se busca en el TEXTO CRUDO para
#: que ni un comentario pueda insinuar un camino de ejecución.
FORBIDDEN_TEXT = re.compile(
    r"(?:^|[^A-Za-z_])(eval\s*\(|exec\s*\(|__import__|os\.system|subprocess|"
    r"SELECT\s+.+\s+FROM\s|lambda\s)", re.IGNORECASE)

#: Cap duro de exposición: ninguna salida declarativa puede superarlo sin ADR.
EXPOSURE_ABS_CAP = 3.0


def _check_text(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    hits = [m.group(0).strip() for m in FORBIDDEN_TEXT.finditer(text)]
    if hits:
        return [f"{path.name}: el YAML insinúa ejecución de código ({hits[:3]})"]
    return []


def _check_outputs(spec: dict) -> list[str]:
    errors: list[str] = []
    rules = (spec.get("policy") or {}).get("rules") or []
    priorities = [r.get("priority") for r in rules]
    if len(rules) > 1 and (None in priorities or len(set(priorities)) != len(priorities)):
        errors.append(
            f"{spec['id']}: con >1 regla cada una necesita 'priority' única "
            "(resolución de conflictos declarada, invariante 9)"
        )
    for rule in rules:
        exposure = (rule.get("output") or {}).get("target_exposure")
        if not isinstance(exposure, (int, float)) or isinstance(exposure, bool):
            errors.append(f"{spec['id']}/{rule.get('id')}: target_exposure debe ser numérico")
        elif abs(float(exposure)) > EXPOSURE_ABS_CAP:
            errors.append(
                f"{spec['id']}/{rule.get('id')}: target_exposure {exposure} supera el cap "
                f"{EXPOSURE_ABS_CAP}"
            )
    default_exposure = (spec["policy"]["resolution"]).get("default_target_exposure")
    if abs(float(default_exposure)) > EXPOSURE_ABS_CAP:
        errors.append(f"{spec['id']}: default_target_exposure supera el cap")
    return errors


def _check_determinism(spec: dict) -> list[str]:
    """Mismo snapshot + misma policy => decisión idéntica (fingerprint incluido)."""
    if spec["migration"]["status"] == "SPEC_ONLY":
        return []
    try:
        policy = build_policy(spec)
    except PolicySpecError as exc:
        return [f"{spec['id']}: no compila ({exc})"]
    if not isinstance(policy, Policy):
        return [f"{spec['id']}: la política no implementa el protocolo Policy"]
    # Snapshot sintético: valores neutros y finitos; solo se comprueba que dos
    # evaluaciones idénticas produzcan el MISMO fingerprint (no economía).
    snapshot = {name: 1.0 for name in policy.required_features()}
    ctx = PolicyContext(as_of="2026-07-28")
    try:
        first = policy.evaluate(snapshot, ctx)
        second = policy.evaluate(snapshot, PolicyContext(as_of="2026-07-28"))
    except Exception as exc:  # noqa: BLE001 - cualquier fallo es un fallo del spec
        return [f"{spec['id']}: evaluate() falló sobre snapshot neutro ({exc})"]
    if first.decision_fingerprint != second.decision_fingerprint:
        return [f"{spec['id']}: NO determinista (fingerprints distintos)"]
    if first.signal_id != second.signal_id:
        return [f"{spec['id']}: NO determinista (signal_id distinto)"]
    return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--print-hashes", action="store_true",
                    help="imprime el policy_hash derivado de cada spec")
    args = ap.parse_args(argv)

    directory = Path(args.dir) if args.dir else policy_specs_dir()
    files = sorted(directory.glob("*.yaml"))
    if not files:
        print(f"[FAIL] no hay specs en {directory}")
        return 1

    errors: list[str] = []
    for path in files:
        errors.extend(_check_text(path))
    try:
        specs = load_all_policy_specs(directory)
    except PolicySpecError as exc:
        print(f"[FAIL] {exc}")
        return 1

    for spec in specs:
        errors.extend(_check_outputs(spec))
        errors.extend(_check_determinism(spec))
        if args.print_hashes:
            print(f"  {spec['id']}: {canonical_policy_hash(spec)}")

    for spec in specs:
        status = spec["migration"]["status"]
        engine = spec["engine"]["type"]
        mode = spec["engine"]["implementation"]["mode"]
        print(f"  {spec['id']:<26} engine={engine:<10} mode={mode:<13} {status}")

    if errors:
        print("\n[FAIL] " + f"{len(errors)} violaciones:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"\n[OK] {len(specs)} specs de política válidos (§11).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
