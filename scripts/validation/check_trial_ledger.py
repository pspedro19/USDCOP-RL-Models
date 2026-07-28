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

--- ENDURECIMIENTO ADITIVO 2026-07-28 (BL-09-r2 / BL-11-r2 / BL-12-r3) -------------------
Origen: rechazo de Codex ("8/10 familias legacy exentas; 218/237 cutoff null;
`trend_regime` comprime 94 trials a 8 celdas; guard de prosa evadible por case/orden;
SHA sella 237/109 frente a header 111"). Los checks 1-7 y el contenido BL-10 del ledger
NO se tocan; lo que sigue se AÑADE:

8.  `check_family_declaration_coverage` — **CERO exenciones legacy**: toda familia presente
    en el ledger tiene su YAML en `registries/families/`. La exención por
    `env == legacy_backfill` que sobrevivía en el check 7 deja de existir.
9.  `check_cell_granularity` — cada celda declara `cell_kind`:
    `atomic` (1 trial = 1 mirada), `grid_block` (ejes reales declarados; producto de ejes
    + variantes extra == n_trials) o `estimated_block` (bloque legacy NO descomponible,
    que exige `decomposable: false` + `granularity_note` + `provenance_source` existente
    en disco). Una celda cuyas filas del ledger sean TODAS `label: documented` no puede
    ser `estimated_block` (está documentada: es descomponible por definición).
10. `check_cutoff_classification` — el `cutoff` nulo deja de ser silencio: toda fila con
    `cutoff == null` debe quedar cubierta por una celda con `cutoff_class` del vocabulario
    cerrado {`legacy_irrecoverable`, `documented_in_registry`, `declared_in_cell`} y su
    evidencia verificable (fichero que existe / fecha ISO). `legacy_irrecoverable` está
    PROHIBIDO para filas `label: documented`.
11. `check_provenance_wall` (BL-12, estructural — NO prosa) — toda familia declara bloque
    `provenance` con `crosses_wall` booleano explícito. Si cruza la muralla FT->AT debe
    declarar `inherits_from_family` (familia existente, forecast) y/o `forecast_trial_ids`
    (existentes y de kind=forecast), y el cluster de origen debe coincidir con el suyo —
    así los FT heredados entran de verdad en su N_cluster (ADR-0022 §2/§3).
12. `check_declared_totals` (BL-12) — coherencia ledger<->cabecera verificada por MÁQUINA:
    `registries/README.md` publica un bloque delimitado `<!-- LEDGER-TOTALS ... -->` con
    YAML (n_global/n_ft/n_at/por activo) y el validador exige igualdad EXACTA con el
    recomputo del ledger. No hay regex de prosa que se pueda evadir por mayúsculas u orden.
13. `check_family_siblings` — familias con el mismo `hypothesis_key` (la misma mecánica
    replicada en varios activos) deben declararse mutuamente en `sibling_families` y usar
    `deflation_scope: cluster`. Dividir una familia por activo no lava la multiplicidad.

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
            cell_records: list[dict] = []
            for trial_id in cell_ids:
                record = by_id.get(trial_id)
                if record is None:
                    errors.append(f"{family_id}: celda referencia {trial_id} que NO existe en el ledger")
                    continue
                cell_records.append(record)
                if record["family"] != family_id:
                    errors.append(
                        f"{family_id}: {trial_id} pertenece a family='{record['family']}' en el ledger"
                    )
                if record["asset"] != cell.get("asset"):
                    errors.append(
                        f"{family_id}: {trial_id} es de asset='{record['asset']}' pero la "
                        f"celda declara '{cell.get('asset')}'"
                    )
            if any(record.get("label") == "legacy_estimate" for record in cell_records):
                if "legacy_estimate" not in str(cell.get("note") or ""):
                    errors.append(
                        f"{family_id}: celda '{cell.get('variant')}' con backfill "
                        "debe documentar 'legacy_estimate' en note"
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


# ---------------------------------------------------------------------------------------
# ENDURECIMIENTO ADITIVO BL-09-r2 / BL-11-r2 / BL-12-r3 (2026-07-28)
# Nada de lo anterior se modifica: estos checks se SUMAN a run_all_checks().
# ---------------------------------------------------------------------------------------

README_PATH = ROOT / "registries" / "README.md"

CELL_KINDS = {"atomic", "grid_block", "estimated_block"}
CUTOFF_CLASSES = {"legacy_irrecoverable", "documented_in_registry", "declared_in_cell"}
DEFLATION_SCOPES = {"family", "cluster"}
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

TOTALS_OPEN = "<!-- LEDGER-TOTALS"
TOTALS_CLOSE = "LEDGER-TOTALS -->"


def load_families(families_dir: Path = FAMILIES_DIR) -> dict[str, dict]:
    """Carga todos los YAML de familia (family_id -> dict). Solo lectura."""
    families: dict[str, dict] = {}
    for yaml_path in sorted(families_dir.glob("*.yaml")):
        families[yaml_path.stem] = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    return families


def iter_cells(families: dict[str, dict]):
    """(family_id, family, cell, trial_ids) por cada celda declarada."""
    for family_id, family in sorted(families.items()):
        for cell in family.get("cells", []) or []:
            trial_ids, _ = _expand_cell_trial_ids(cell, family_id)
            yield family_id, family, cell, trial_ids


def check_family_declaration_coverage(records: list[dict],
                                      families_dir: Path = FAMILIES_DIR) -> list[str]:
    """BL-11-r2: CERO exención legacy — toda familia del ledger tiene YAML declarado."""
    errors = []
    declared = set(load_families(families_dir))
    ledger_families = Counter(record["family"] for record in records)
    for family_id in sorted(ledger_families):
        if family_id not in declared:
            errors.append(
                f"familia '{family_id}' tiene {ledger_families[family_id]} líneas en el ledger "
                f"pero NO existe registries/families/{family_id}.yaml — la exención legacy "
                "quedó derogada (BL-11-r2): sin YAML no hay bar pre-firmado ni gate DSR"
            )
    for family_id in sorted(declared - set(ledger_families)):
        errors.append(
            f"familia '{family_id}' declarada en YAML pero sin líneas en el ledger "
            "(familia fantasma: o se cobra o se borra)"
        )
    return errors


def check_cell_granularity(records: list[dict],
                           families_dir: Path = FAMILIES_DIR) -> list[str]:
    """BL-11-r2: la granularidad de celda debe reflejar las miradas reales.

    `trend_regime` comprimía 94 trials en 8 celdas sin decir cuáles eran descomponibles y
    cuáles no. Ahora cada celda lo DECLARA y el validador lo verifica.
    """
    errors = []
    by_id = {record["trial_id"]: record for record in records}
    families = load_families(families_dir)
    for family_id, _family, cell, trial_ids in iter_cells(families):
        variant = cell.get("variant")
        where = f"{family_id}/{variant}"
        cell_kind = cell.get("cell_kind")
        if cell_kind not in CELL_KINDS:
            errors.append(
                f"{where}: cell_kind={cell_kind!r} no está en {sorted(CELL_KINDS)} "
                "(BL-11-r2: la granularidad se declara, no se asume)"
            )
            continue
        n = len(trial_ids)
        cell_records = [by_id[t] for t in trial_ids if t in by_id]
        all_documented = bool(cell_records) and all(
            record.get("label") == "documented" for record in cell_records
        )

        if cell_kind == "atomic":
            if n != 1:
                errors.append(f"{where}: cell_kind=atomic exige exactamente 1 trial, cubre {n}")
        elif cell_kind == "grid_block":
            axes = cell.get("grid_axes") or {}
            if not isinstance(axes, dict) or not axes:
                errors.append(f"{where}: grid_block exige grid_axes con los ejes REALES")
                continue
            product = 1
            for axis_name, values in axes.items():
                if not isinstance(values, list) or not values:
                    errors.append(f"{where}: eje '{axis_name}' vacío o no-lista")
                    product = 0
                    break
                product *= len(values)
            extra = cell.get("extra_variants") or []
            if product and product + len(extra) != n:
                errors.append(
                    f"{where}: grid_axes cubre {product} celdas + {len(extra)} variantes extra "
                    f"= {product + len(extra)}, pero el bloque cobra {n} trials"
                )
        else:  # estimated_block
            if all_documented:
                errors.append(
                    f"{where}: todas sus filas son label=documented => NO puede ser "
                    "estimated_block (lo documentado es descomponible por definición)"
                )
            if cell.get("decomposable") is not False:
                errors.append(
                    f"{where}: estimated_block exige `decomposable: false` explícito "
                    "(criterio objetivo de por qué no se puede desagregar)"
                )
            if not str(cell.get("granularity_note") or "").strip():
                errors.append(f"{where}: estimated_block exige granularity_note por escrito")
            source = cell.get("provenance_source")
            if not source:
                errors.append(f"{where}: estimated_block exige provenance_source verificable")
            elif not (ROOT / str(source).split(":")[0]).exists():
                errors.append(
                    f"{where}: provenance_source '{source}' no existe en el árbol"
                )
        # Prospectivo: una celda nueva (no legacy) tiene que ser atómica.
        if cell_records and all(r.get("env") != "legacy_backfill" for r in cell_records):
            if cell_kind != "atomic":
                errors.append(
                    f"{where}: celda NO-legacy debe ser atomic (1 trial = 1 mirada); "
                    f"los bloques estimados son un privilegio del backfill histórico"
                )
    return errors


def check_cutoff_classification(records: list[dict],
                                families_dir: Path = FAMILIES_DIR) -> list[str]:
    """BL-09-r2: los 218 `cutoff: null` quedan CLASIFICADOS con regla verificable.

    Regla:
      * `cutoff` no-nulo en el ledger  -> nada que declarar.
      * `cutoff` nulo -> la celda que lo cubre declara `cutoff_class`:
          - `legacy_irrecoverable`  : SOLO filas label=legacy_estimate; exige
            `provenance_source` que exista en disco (ya validado en granularidad).
          - `documented_in_registry`: SOLO filas label=documented; exige `cutoff_evidence`
            (fichero existente) y `hypothesis_id` cuyo texto aparezca en ese fichero.
          - `declared_in_cell`      : exige `cutoff` ISO YYYY-MM-DD en la propia celda.
      * Fila NUEVA (env != legacy_backfill) con cutoff nulo -> violación dura, sin clase
        que la salve (fail-closed prospectivo).
    """
    errors = []
    by_id = {record["trial_id"]: record for record in records}
    families = load_families(families_dir)
    covered: set[str] = set()

    for family_id, _family, cell, trial_ids in iter_cells(families):
        variant = cell.get("variant")
        where = f"{family_id}/{variant}"
        cell_records = [by_id[t] for t in trial_ids if t in by_id]
        null_records = [r for r in cell_records if r.get("cutoff") in (None, "", "null")]
        covered.update(r["trial_id"] for r in cell_records)
        if not null_records:
            continue
        cutoff_class = cell.get("cutoff_class")
        if cutoff_class not in CUTOFF_CLASSES:
            errors.append(
                f"{where}: {len(null_records)} filas con cutoff nulo y cutoff_class="
                f"{cutoff_class!r} fuera de {sorted(CUTOFF_CLASSES)} (BL-09-r2: el nulo "
                "se clasifica con regla verificable, no se ignora)"
            )
            continue
        if cutoff_class == "legacy_irrecoverable":
            bad = [r["trial_id"] for r in null_records if r.get("label") != "legacy_estimate"]
            if bad:
                errors.append(
                    f"{where}: cutoff_class=legacy_irrecoverable PROHIBIDO para filas "
                    f"label=documented ({bad[:5]}) — su cutoff es recuperable del registry"
                )
        elif cutoff_class == "documented_in_registry":
            bad = [r["trial_id"] for r in null_records if r.get("label") != "documented"]
            if bad:
                errors.append(
                    f"{where}: cutoff_class=documented_in_registry exige label=documented "
                    f"en todas sus filas nulas (incumplen {bad[:5]})"
                )
            evidence = cell.get("cutoff_evidence")
            hypothesis_id = cell.get("hypothesis_id")
            if not evidence or not hypothesis_id:
                errors.append(
                    f"{where}: documented_in_registry exige cutoff_evidence + hypothesis_id"
                )
            else:
                evidence_path = ROOT / str(evidence)
                if not evidence_path.exists():
                    errors.append(f"{where}: cutoff_evidence '{evidence}' no existe")
                elif str(hypothesis_id) not in evidence_path.read_text(
                    encoding="utf-8", errors="replace"
                ):
                    errors.append(
                        f"{where}: hypothesis_id '{hypothesis_id}' no aparece en {evidence}"
                    )
        else:  # declared_in_cell
            declared = cell.get("cutoff")
            if not (isinstance(declared, str) and ISO_DATE_RE.match(declared)):
                errors.append(
                    f"{where}: declared_in_cell exige `cutoff: YYYY-MM-DD` en la celda "
                    f"(encontrado {declared!r})"
                )

    # fail-closed prospectivo + ninguna fila nula puede quedar huérfana de celda
    for record in records:
        if record.get("cutoff") in (None, "", "null"):
            if record.get("env") != "legacy_backfill":
                errors.append(
                    f"{record['trial_id']}: env='{record.get('env')}' con cutoff nulo — "
                    "toda fila nueva exige cutoff explícito (fail-closed)"
                )
            if record["trial_id"] not in covered:
                errors.append(
                    f"{record['trial_id']}: cutoff nulo y NINGUNA celda lo cubre — "
                    "sin celda no hay clasificación posible"
                )
    return errors


def check_provenance_wall(records: list[dict],
                          families_dir: Path = FAMILIES_DIR) -> list[str]:
    """BL-12-r3: la muralla FT->AT se verifica ESTRUCTURALMENTE (nada de regex de prosa)."""
    errors = []
    by_id = {record["trial_id"]: record for record in records}
    families = load_families(families_dir)
    cluster_of = {record["family"]: record["cluster"] for record in records}

    for family_id, family in sorted(families.items()):
        provenance = family.get("provenance")
        if not isinstance(provenance, dict):
            errors.append(
                f"{family_id}: falta el bloque `provenance` (ADR-0022 §4 — obligatorio, "
                "aunque sea para declarar crosses_wall: false)"
            )
            continue
        crosses = provenance.get("crosses_wall")
        if not isinstance(crosses, bool):
            errors.append(f"{family_id}: provenance.crosses_wall debe ser booleano explícito")
            continue
        if family.get("kind") == "forecast" and crosses:
            errors.append(
                f"{family_id}: una familia kind=forecast no cruza la muralla hacia sí misma"
            )
        if not crosses:
            if provenance.get("inherits_from_family") or provenance.get("forecast_trial_ids"):
                errors.append(
                    f"{family_id}: crosses_wall=false pero declara herencia predictiva — "
                    "heredar FT ES cruzar la muralla (+1 AT, ADR-0022 §2)"
                )
            continue
        inherits = provenance.get("inherits_from_family")
        ft_ids = provenance.get("forecast_trial_ids") or []
        if not inherits and not ft_ids:
            errors.append(
                f"{family_id}: crosses_wall=true exige inherits_from_family y/o "
                "forecast_trial_ids (los FT heredados entran al N_cluster)"
            )
        if inherits:
            if inherits not in families:
                errors.append(f"{family_id}: inherits_from_family '{inherits}' no declarada")
            else:
                if families[inherits].get("kind") != "forecast":
                    errors.append(
                        f"{family_id}: inherits_from_family '{inherits}' no es kind=forecast"
                    )
                if cluster_of.get(inherits) != cluster_of.get(family_id):
                    errors.append(
                        f"{family_id}: hereda de '{inherits}' que vive en cluster "
                        f"'{cluster_of.get(inherits)}' != '{cluster_of.get(family_id)}' — los FT "
                        "heredados NO entrarían en su N_cluster (ADR-0022 §3)"
                    )
        for trial_id in ft_ids:
            record = by_id.get(trial_id)
            if record is None:
                errors.append(f"{family_id}: provenance cita {trial_id} inexistente en el ledger")
            elif record.get("kind") != "forecast":
                errors.append(f"{family_id}: provenance cita {trial_id} que NO es kind=forecast")
    return errors


def parse_declared_totals(readme_path: Path = README_PATH) -> dict:
    """Lee el bloque `<!-- LEDGER-TOTALS ... LEDGER-TOTALS -->` del README (YAML)."""
    text = readme_path.read_text(encoding="utf-8")
    start = text.find(TOTALS_OPEN)
    end = text.find(TOTALS_CLOSE, start + 1) if start != -1 else -1
    if start == -1 or end == -1:
        raise ValueError(
            "registries/README.md no publica el bloque LEDGER-TOTALS "
            "(BL-12-r3: la cabecera debe ser maquinal, no prosa)"
        )
    body = text[start + len(TOTALS_OPEN):end]
    return yaml.safe_load(body) or {}


def check_declared_totals(records: list[dict], readme_path: Path = README_PATH) -> list[str]:
    """BL-12-r3: coherencia ledger <-> cabecera publicada, verificada por máquina."""
    try:
        declared = parse_declared_totals(readme_path)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    errors = []
    lineage = Counter(record["trial_id"][:2] for record in records)
    per_asset = Counter(record["asset"] for record in records)
    actual = {
        "n_global": len(records),
        "n_ft": lineage.get("FT", 0),
        "n_at": lineage.get("AT", 0),
    }
    for key, value in actual.items():
        if declared.get(key) != value:
            errors.append(
                f"README LEDGER-TOTALS.{key}={declared.get(key)} pero el ledger da {value} "
                "(la cabecera nunca puede sellar un total distinto al ledger)"
            )
    declared_assets = declared.get("per_asset") or {}
    if set(declared_assets) != set(per_asset):
        errors.append(
            f"README LEDGER-TOTALS.per_asset cubre {sorted(declared_assets)} "
            f"pero el ledger tiene {sorted(per_asset)}"
        )
    for asset, count in sorted(per_asset.items()):
        if declared_assets.get(asset) != count:
            errors.append(
                f"README LEDGER-TOTALS.per_asset.{asset}={declared_assets.get(asset)} "
                f"pero el ledger da {count}"
            )
    declared_families = declared.get("per_family") or {}
    ledger_families = Counter(record["family"] for record in records)
    if declared_families and set(declared_families) != set(ledger_families):
        errors.append(
            f"README LEDGER-TOTALS.per_family cubre {sorted(declared_families)} "
            f"pero el ledger tiene {sorted(ledger_families)}"
        )
    for family_id, count in sorted(ledger_families.items()):
        if declared_families and declared_families.get(family_id) != count:
            errors.append(
                f"README LEDGER-TOTALS.per_family.{family_id}="
                f"{declared_families.get(family_id)} pero el ledger da {count}"
            )
    return errors


def check_family_siblings(records: list[dict],
                          families_dir: Path = FAMILIES_DIR) -> list[str]:
    """BL-11-r2: la misma mecánica en N activos no se lava partiéndola en N familias."""
    errors = []
    families = load_families(families_dir)
    by_key: dict[str, list[str]] = {}
    for family_id, family in sorted(families.items()):
        key = family.get("hypothesis_key")
        if not key:
            errors.append(
                f"{family_id}: falta `hypothesis_key` (identifica la MISMA mecánica "
                "replicada en varios activos — FABRIC §31)"
            )
            continue
        by_key.setdefault(key, []).append(family_id)
    for family_id, family in sorted(families.items()):
        scope = family.get("deflation_scope")
        if scope not in DEFLATION_SCOPES:
            errors.append(
                f"{family_id}: deflation_scope={scope!r} fuera de {sorted(DEFLATION_SCOPES)}"
            )
    for key, members in sorted(by_key.items()):
        if len(members) < 2:
            continue
        for family_id in members:
            family = families[family_id]
            expected_siblings = sorted(set(members) - {family_id})
            declared_siblings = sorted(family.get("sibling_families") or [])
            if declared_siblings != expected_siblings:
                errors.append(
                    f"{family_id}: hypothesis_key='{key}' compartido con {expected_siblings} "
                    f"pero sibling_families declara {declared_siblings}"
                )
            if family.get("deflation_scope") != "cluster":
                errors.append(
                    f"{family_id}: comparte mecánica con {expected_siblings} => "
                    "deflation_scope debe ser 'cluster' (el N por familia subestima la "
                    "deflación cruzada; partir familias no lava multiplicidad)"
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
    # --- endurecimiento aditivo 2026-07-28 ---
    errors += check_family_declaration_coverage(records, families_dir)
    errors += check_cell_granularity(records, families_dir)
    errors += check_cutoff_classification(records, families_dir)
    errors += check_provenance_wall(records, families_dir)
    errors += check_declared_totals(records)
    errors += check_family_siblings(records, families_dir)
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
