#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validador del ledger doble FT-/AT- y de las familias transversales (BL-09/BL-10/BL-11).

Contratos que hace cumplir (FABRIC §9.4-§9.7 + ADR-0022):

1. `registries/ledger.jsonl` parsea línea a línea; campos obligatorios presentes;
   `trial_id` con formato FT-####/AT-#### y prefijo coherente con `kind`
   (FT=forecast, AT=action).
2. `trial_id` único en todo el ledger (los dos linajes comparten unicidad).
3. Cadena de hashes íntegra: `line_hash = sha256(json canónico del payload sin line_hash)`,
   `prev_hash` de cada línea == `line_hash` de la anterior (génesis = 64 ceros).
   Append-only: cualquier edición de una línea histórica rompe la cadena.
4. Contadores corrientes N_family / N_cluster / N_global correctos (recomputados).
5. Suma EXACTA por activo == `n_trials_total` del front-matter de su HYPOTHESIS-REGISTRY
   (BL-10: el backfill legacy_estimate NO cambia ningún conteo).
6. N_global final <= N_MAX=989 — cota constitucional de GASTO únicamente (§9.7):
   N_MAX JAMÁS entra en el DSR; el DSR usa el N efectivamente cobrado aquí.
7. `registries/families/*.yaml`: family_id == nombre de archivo; cada celda con
   trial_id/trial_id_range existe en el ledger con family y asset coincidentes; sin
   solapes; `trials_charged` == celdas cubiertas == líneas del ledger de esa familia
   (BL-09/BL-11 CI). Toda línea nueva (env != legacy_backfill) exige familia declarada.

Uso: python scripts/validation/check_trial_ledger.py  (exit 0 = verde, 1 = violaciones)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = ROOT / "registries" / "ledger.jsonl"
FAMILIES_DIR = ROOT / "registries" / "families"

# Cota constitucional de gasto (FABRIC §9.7). Commitment device contra el p-hacking.
# NUNCA entra en el DSR ni en ninguna fórmula estadística.
N_MAX = 989

GENESIS_HASH = "0" * 64
TRIAL_ID_RE = re.compile(r"^(FT|AT)-\d{4}$")
KIND_BY_PREFIX = {"FT": "forecast", "AT": "action"}

REQUIRED_FIELDS = [
    "trial_id", "kind", "family", "cluster", "asset", "variant", "charged_at",
    "cutoff", "env", "label", "code_hash", "data_hash", "result", "source",
    "N_family", "N_cluster", "N_global", "prev_hash", "line_hash",
]

# SSOT de conteo por activo: front-matter de cada HYPOTHESIS-REGISTRY (solo LECTURA).
REGISTRY_PATHS = {
    "usdcop": ROOT / ".claude/specs/assets/usdcop/HYPOTHESIS-REGISTRY.md",
    "xauusd": ROOT / ".claude/specs/assets/xauusd/HYPOTHESIS-REGISTRY.md",
    "btcusdt": ROOT / ".claude/specs/assets/btcusdt/design/HYPOTHESIS-REGISTRY.md",
    "spx500": ROOT / ".claude/specs/assets/spx500/HYPOTHESIS-REGISTRY.md",
}

N_TRIALS_RE = re.compile(r"^n_trials_total:\s*(\d+)\s*(?:#.*)?$", re.MULTILINE)


def read_n_trials_total(path: Path) -> int:
    """Lee n_trials_total del front-matter YAML de un HYPOTHESIS-REGISTRY (solo lectura)."""
    text = path.read_text(encoding="utf-8")
    match = N_TRIALS_RE.search(text)
    if match is None:
        raise ValueError(f"n_trials_total no encontrado en {path}")
    return int(match.group(1))


def load_ledger(path: Path = LEDGER_PATH) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"ledger línea {lineno}: JSON inválido ({exc})") from exc
    return records


def canonical_line_hash(record: dict) -> str:
    payload = {key: value for key, value in record.items() if key != "line_hash"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def check_schema_and_ids(records: list[dict]) -> list[str]:
    errors = []
    seen: Counter = Counter()
    for index, record in enumerate(records, start=1):
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            errors.append(f"línea {index}: faltan campos {missing}")
            continue
        trial_id = record["trial_id"]
        if not TRIAL_ID_RE.match(trial_id):
            errors.append(f"línea {index}: trial_id '{trial_id}' no cumple ^(FT|AT)-\\d{{4}}$")
            continue
        prefix = trial_id[:2]
        if record["kind"] != KIND_BY_PREFIX[prefix]:
            errors.append(
                f"línea {index}: {trial_id} tiene kind='{record['kind']}' "
                f"pero el prefijo {prefix}- exige '{KIND_BY_PREFIX[prefix]}'"
            )
        seen[trial_id] += 1
    for trial_id, count in seen.items():
        if count > 1:
            errors.append(f"trial_id duplicado: {trial_id} aparece {count} veces")
    return errors


def check_hash_chain(records: list[dict]) -> list[str]:
    errors = []
    prev_hash = GENESIS_HASH
    for index, record in enumerate(records, start=1):
        if record.get("prev_hash") != prev_hash:
            errors.append(
                f"línea {index} ({record.get('trial_id')}): prev_hash roto "
                f"(esperado {prev_hash[:12]}…, encontrado {str(record.get('prev_hash'))[:12]}…)"
            )
        expected = canonical_line_hash(record)
        if record.get("line_hash") != expected:
            errors.append(
                f"línea {index} ({record.get('trial_id')}): line_hash no coincide con el "
                f"contenido (append-only violado o línea editada)"
            )
        prev_hash = record.get("line_hash", expected)
    return errors


def check_running_counters(records: list[dict]) -> list[str]:
    errors = []
    n_family: Counter = Counter()
    n_cluster: Counter = Counter()
    n_global = 0
    for index, record in enumerate(records, start=1):
        n_family[record["family"]] += 1
        n_cluster[record["cluster"]] += 1
        n_global += 1
        for field, expected in (
            ("N_family", n_family[record["family"]]),
            ("N_cluster", n_cluster[record["cluster"]]),
            ("N_global", n_global),
        ):
            if record.get(field) != expected:
                errors.append(
                    f"línea {index} ({record['trial_id']}): {field}={record.get(field)} "
                    f"pero el recomputo da {expected}"
                )
    if n_global > N_MAX:
        errors.append(
            f"N_global={n_global} excede la cota de gasto N_MAX={N_MAX} "
            f"(§9.7 — presupuesto agotado; N_MAX no entra en el DSR)"
        )
    return errors


def check_asset_sums(records: list[dict]) -> list[str]:
    """BL-10: suma FT+AT por activo == n_trials_total del registry del activo, EXACTA."""
    errors = []
    per_asset = Counter(record["asset"] for record in records)
    for asset in sorted(per_asset):
        if asset not in REGISTRY_PATHS:
            errors.append(f"activo '{asset}' en el ledger sin HYPOTHESIS-REGISTRY conocido")
    for asset, registry_path in REGISTRY_PATHS.items():
        if not registry_path.exists():
            errors.append(f"{asset}: registry no encontrado en {registry_path}")
            continue
        expected = read_n_trials_total(registry_path)
        actual = per_asset.get(asset, 0)
        if actual != expected:
            errors.append(
                f"{asset}: ledger suma {actual} trials pero n_trials_total={expected} "
                f"en {registry_path.name} — el backfill no puede cambiar el conteo"
            )
    return errors


def _expand_cell_trial_ids(cell: dict, family_id: str) -> tuple[list[str], list[str]]:
    """Devuelve (trial_ids, errores) de una celda con trial_id o trial_id_range."""
    errors: list[str] = []
    if cell.get("trial_id") is not None and cell.get("trial_id_range") is not None:
        return [], [f"{family_id}: celda '{cell.get('variant')}' declara trial_id Y trial_id_range"]
    if cell.get("trial_id") is not None:
        trial_id = cell["trial_id"]
        if not TRIAL_ID_RE.match(str(trial_id)):
            return [], [f"{family_id}: trial_id inválido '{trial_id}' en celda '{cell.get('variant')}'"]
        return [trial_id], []
    id_range = cell.get("trial_id_range")
    if id_range is None:
        return [], []  # celda DECLARED sin cobrar (trial_id null) — válida, no cuenta
    start, end = str(id_range.get("from")), str(id_range.get("to"))
    if not (TRIAL_ID_RE.match(start) and TRIAL_ID_RE.match(end)):
        return [], [f"{family_id}: trial_id_range inválido {id_range} en '{cell.get('variant')}'"]
    if start[:3] != end[:3]:
        return [], [f"{family_id}: trial_id_range mezcla linajes {start}..{end}"]
    lo, hi = int(start[3:]), int(end[3:])
    if lo > hi:
        return [], [f"{family_id}: trial_id_range invertido {start}..{end}"]
    ids = [f"{start[:3]}{i:04d}" for i in range(lo, hi + 1)]
    declared = cell.get("n_trials")
    if declared is not None and declared != len(ids):
        errors.append(
            f"{family_id}: celda '{cell.get('variant')}' declara n_trials={declared} "
            f"pero el rango {start}..{end} cubre {len(ids)}"
        )
    return ids, errors


def check_families(records: list[dict], families_dir: Path = FAMILIES_DIR) -> list[str]:
    errors = []
    by_id = {record["trial_id"]: record for record in records}
    ledger_family_counts = Counter(record["family"] for record in records)
    declared_families = set()

    for yaml_path in sorted(families_dir.glob("*.yaml")):
        family = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        family_id = family.get("family_id")
        declared_families.add(family_id)
        if family_id != yaml_path.stem:
            errors.append(f"{yaml_path.name}: family_id='{family_id}' != nombre de archivo")
            continue
        covered: list[str] = []
        for cell in family.get("cells", []):
            cell_ids, cell_errors = _expand_cell_trial_ids(cell, family_id)
            errors.extend(cell_errors)
            for trial_id in cell_ids:
                record = by_id.get(trial_id)
                if record is None:
                    errors.append(f"{family_id}: celda referencia {trial_id} que NO existe en el ledger")
                    continue
                if record["family"] != family_id:
                    errors.append(
                        f"{family_id}: {trial_id} pertenece a family='{record['family']}' en el ledger"
                    )
                if record["asset"] != cell.get("asset"):
                    errors.append(
                        f"{family_id}: {trial_id} es de asset='{record['asset']}' pero la "
                        f"celda declara '{cell.get('asset')}'"
                    )
            covered.extend(cell_ids)
        duplicates = [trial_id for trial_id, count in Counter(covered).items() if count > 1]
        if duplicates:
            errors.append(f"{family_id}: celdas solapadas sobre {duplicates}")
        trials_charged = family.get("trials_charged")
        if trials_charged != len(covered):
            errors.append(
                f"{family_id}: trials_charged={trials_charged} pero las celdas cubren {len(covered)}"
            )
        if trials_charged != ledger_family_counts.get(family_id, 0):
            errors.append(
                f"{family_id}: trials_charged={trials_charged} pero el ledger tiene "
                f"{ledger_family_counts.get(family_id, 0)} líneas de esa familia"
            )

    # Prospectivo: todo trial NO-legacy debe pertenecer a una familia declarada en YAML.
    for record in records:
        if record.get("env") != "legacy_backfill" and record["family"] not in declared_families:
            errors.append(
                f"{record['trial_id']}: env='{record.get('env')}' exige familia declarada en "
                f"registries/families/ (falta {record['family']}.yaml)"
            )
    return errors


def run_all_checks(ledger_path: Path = LEDGER_PATH,
                   families_dir: Path = FAMILIES_DIR) -> list[str]:
    records = load_ledger(ledger_path)
    if not records:
        return ["ledger vacío"]
    errors = []
    errors += check_schema_and_ids(records)
    errors += check_hash_chain(records)
    errors += check_running_counters(records)
    errors += check_asset_sums(records)
    errors += check_families(records, families_dir)
    return errors


def main() -> int:
    try:
        records = load_ledger()
    except (OSError, ValueError) as exc:
        print(f"ERROR leyendo el ledger: {exc}")
        return 1
    errors = run_all_checks()
    per_asset = Counter(record["asset"] for record in records)
    per_lineage = Counter(record["trial_id"][:2] for record in records)
    print(f"ledger: {len(records)} trials "
          f"(FT={per_lineage.get('FT', 0)}, AT={per_lineage.get('AT', 0)}) "
          f"| N_global={len(records)}/{N_MAX} (cota de gasto, no entra al DSR)")
    for asset in sorted(per_asset):
        print(f"  {asset}: {per_asset[asset]}")
    if errors:
        print(f"\n{len(errors)} VIOLACIONES:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("\nOK: sumas exactas por activo, unicidad, cadena de hashes, contadores y familias.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
