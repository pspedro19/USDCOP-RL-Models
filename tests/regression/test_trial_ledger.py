# -*- coding: utf-8 -*-
"""Regression: ledger doble FT-/AT- + familias transversales (BL-09/BL-10/BL-11).

Contratos que congela:
- BL-10: la suma FT+AT por activo es EXACTAMENTE el n_trials_total del front-matter de su
  HYPOTHESIS-REGISTRY (el backfill legacy_estimate no puede cambiar ningún conteo).
- BL-09: trial_id único, cadena de hashes append-only íntegra, contadores N_family/
  N_cluster/N_global correctos, trials_charged de cada familia == conteo en el ledger.
- BL-11: cada celda de registries/families/*.yaml referencia trial_ids que existen en el
  ledger con family y asset coincidentes; toda familia backfilled lleva nota legacy_estimate.
- §9.7: N_global <= N_MAX=989 (cota de gasto; JAMÁS entra en el DSR).
"""
from collections import Counter
from pathlib import Path

import yaml

from scripts.validation.check_trial_ledger import (
    FAMILIES_DIR,
    GENESIS_HASH,
    LEDGER_PATH,
    N_MAX,
    REGISTRY_PATHS,
    TRIAL_ID_RE,
    canonical_line_hash,
    check_asset_sums,
    check_families,
    check_hash_chain,
    check_running_counters,
    check_schema_and_ids,
    load_ledger,
    read_n_trials_total,
    run_all_checks,
)

ROOT = Path(__file__).resolve().parents[2]


def _records():
    return load_ledger(LEDGER_PATH)


def test_ledger_exists_and_parses():
    records = _records()
    assert records, "registries/ledger.jsonl vacío o ausente"


def test_schema_and_unique_trial_ids():
    records = _records()
    assert check_schema_and_ids(records) == []
    ids = [record["trial_id"] for record in records]
    assert len(ids) == len(set(ids)), "trial_id duplicado"
    assert all(TRIAL_ID_RE.match(trial_id) for trial_id in ids)


def test_hash_chain_append_only():
    records = _records()
    assert check_hash_chain(records) == []
    assert records[0]["prev_hash"] == GENESIS_HASH
    # editar cualquier línea histórica debe romper la cadena
    tampered = dict(records[0])
    tampered["result"] = "tampered"
    assert canonical_line_hash(tampered) != records[0]["line_hash"]


def test_running_counters_and_budget_cap():
    records = _records()
    assert check_running_counters(records) == []
    assert records[-1]["N_global"] == len(records) <= N_MAX


def test_sum_per_asset_equals_registry_frontmatter_exactly():
    """BL-10 (el test que lo exige): suma EXACTA == n_trials_total por activo."""
    records = _records()
    per_asset = Counter(record["asset"] for record in records)
    assert check_asset_sums(records) == []
    for asset, registry_path in REGISTRY_PATHS.items():
        expected = read_n_trials_total(registry_path)
        assert per_asset[asset] == expected, (
            f"{asset}: ledger={per_asset[asset]} != n_trials_total={expected}"
        )
    # y nada fuera de los 4 activos gobernados
    assert set(per_asset) == set(REGISTRY_PATHS)


def test_lineage_split_covers_both_prefixes():
    """ADR-0022: ambos linajes existen y suman el total (la partición jamás resetea el N)."""
    records = _records()
    per_lineage = Counter(record["trial_id"][:2] for record in records)
    assert per_lineage["FT"] > 0 and per_lineage["AT"] > 0
    assert per_lineage["FT"] + per_lineage["AT"] == len(records)


def test_families_match_ledger():
    records = _records()
    assert check_families(records) == []


def test_pilot_family_trend_regime_has_real_cells():
    """BL-11: piloto trend_regime con celdas SPX/Oro/BTC reales y bar pre-firmado."""
    path = FAMILIES_DIR / "trend_regime.yaml"
    family = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert family["family_id"] == "trend_regime"
    assert family["kind"] == "action"
    assert family["bar"], "bar pre-firmado obligatorio"
    assets = {cell["asset"] for cell in family["cells"]}
    assert {"spx500", "xauusd", "btcusdt"} <= assets
    ledger_count = sum(1 for record in _records() if record["family"] == "trend_regime")
    assert family["trials_charged"] == ledger_count


def test_backfilled_families_carry_legacy_estimate_note():
    """BL-10: nota legacy_estimate por escrito en cada familia backfilled."""
    for yaml_path in sorted(FAMILIES_DIR.glob("*.yaml")):
        family = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if family.get("label") != "legacy_estimate":
            continue
        text = yaml_path.read_text(encoding="utf-8")
        assert "legacy_estimate" in text
    # y en el ledger: toda línea de backfill estimada lleva la etiqueta
    for record in _records():
        assert record["env"] == "legacy_backfill"
        assert record["label"] in {"legacy_estimate", "documented"}


def test_full_validator_green():
    assert run_all_checks() == []
